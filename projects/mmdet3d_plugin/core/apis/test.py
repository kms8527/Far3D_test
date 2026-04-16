# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import os.path as osp
import pickle
import re
import shutil
import tempfile
import time
import warnings

import cv2
import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results


import mmcv
import numpy as np
import pycocotools.mask as mask_util
from matplotlib import pyplot as plt

def _to_visual_results(result):
    if isinstance(result, dict):
        if 'bbox_results' in result:
            return result['bbox_results']
        if 'pts_bbox' in result:
            return [result]
        return None
    return result


def _unwrap_batch_item(item):
    value = item
    while True:
        if hasattr(value, 'data') and not isinstance(value, torch.Tensor):
            value = value.data
            continue
        if isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
            continue
        break
    return value


def _to_numpy_image(img_tensor):
    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def _camera_name_from_filename(filename, view_idx):
    if not isinstance(filename, str):
        return f'camera_{view_idx}'

    match = re.search(r'(CAM_(?:FRONT_LEFT|FRONT_RIGHT|FRONT|BACK_LEFT|BACK_RIGHT|BACK))', filename)
    if match is not None:
        return match.group(1)

    match = re.search(r'(ring_(?:rear_left|side_left|front_left|front_center|front_right|side_right|rear_right))', filename)
    if match is not None:
        return match.group(1)

    parent = osp.basename(osp.dirname(filename))
    if parent:
        return parent
    return f'camera_{view_idx}'


def _class_color(class_id):
    palette = [
        (255, 99, 71),
        (60, 179, 113),
        (30, 144, 255),
        (255, 215, 0),
        (186, 85, 211),
        (255, 140, 0),
        (46, 139, 87),
        (70, 130, 180),
        (220, 20, 60),
        (154, 205, 50),
        (0, 191, 255),
        (199, 21, 133),
    ]
    return palette[int(class_id) % len(palette)]


def _class_name(class_names, class_id):
    if class_names is not None and 0 <= int(class_id) < len(class_names):
        return str(class_names[int(class_id)])
    return f'class_{int(class_id)}'


def _sample_token_from_meta(sample_meta):
    for key in ('sample_idx', 'token'):
        value = sample_meta.get(key)
        if value is not None:
            return str(value)
    return None


def _normalize_filename_key(filenames):
    if filenames is None:
        return None
    if isinstance(filenames, str):
        filenames = [filenames]
    if not isinstance(filenames, (list, tuple)):
        return None

    normalized = []
    for filename in filenames:
        if filename is None:
            continue
        normalized.append(osp.normpath(str(filename)))
    if not normalized:
        return None
    return tuple(normalized)


def _dataset_index_from_token(dataset, sample_token):
    if dataset is None or sample_token is None:
        return None

    token2idx = getattr(dataset, '_vis_token2idx', None)
    if token2idx is None:
        token2idx = {}
        for index, info in enumerate(getattr(dataset, 'data_infos', [])):
            token = info.get('token')
            if token is not None:
                token2idx[str(token)] = index
        dataset._vis_token2idx = token2idx

    return token2idx.get(str(sample_token))


def _dataset_index_from_filenames(dataset, filenames):
    key = _normalize_filename_key(filenames)
    if dataset is None or key is None:
        return None

    filename2idx = getattr(dataset, '_vis_filename2idx', None)
    if filename2idx is None:
        filename2idx = {}
        for index, info in enumerate(getattr(dataset, 'data_infos', [])):
            cams = info.get('cams', {})
            ordered = []
            for _, cam_info in sorted(cams.items()):
                data_path = cam_info.get('data_path')
                if data_path is not None:
                    ordered.append(osp.normpath(str(data_path)))
            if ordered:
                filename2idx[tuple(ordered)] = index
        dataset._vis_filename2idx = filename2idx

    return filename2idx.get(key)


def _to_numpy_labels(labels):
    if labels is None:
        return None
    if isinstance(labels, torch.Tensor):
        return labels.detach().cpu().numpy()
    return np.asarray(labels)


def _to_numpy_scores(scores):
    if scores is None:
        return None
    if isinstance(scores, torch.Tensor):
        return scores.detach().cpu().numpy()
    return np.asarray(scores)


def _project_boxes_to_views(boxes_3d, lidar2img):
    if boxes_3d is None or lidar2img is None:
        return None

    corners = boxes_3d.corners
    if isinstance(corners, torch.Tensor):
        corners = corners.detach().cpu().numpy()
    else:
        corners = np.asarray(corners)

    if corners.size == 0:
        return np.empty((0, lidar2img.shape[0], 8, 4), dtype=np.float32)

    pts = corners.reshape(-1, 3)
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=corners.dtype)], axis=1)
    projected = []
    for proj in lidar2img:
        proj_pts = pts_h @ proj.T
        proj_pts[:, :2] /= np.clip(proj_pts[:, 2:3], 1e-5, None)
        projected.append(proj_pts.reshape(-1, 8, 4))
    return np.stack(projected, axis=1)


def _draw_line_with_outline(canvas, start_point, end_point, color, thickness):
    cv2.line(canvas, start_point, end_point, (0, 0, 0), thickness + 2)
    cv2.line(canvas, start_point, end_point, color, thickness)


def _draw_text_with_outline(canvas, text, origin, color, scale=0.45, thickness=1):
    cv2.putText(
        canvas,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _draw_legend_overlay(canvas, visible_classes):
    if not visible_classes:
        return

    lines = []
    for is_gt, class_id in sorted(visible_classes):
        cls_name, color = visible_classes[(is_gt, class_id)]
        prefix = 'GT' if is_gt else 'Pred'
        lines.append((f'{prefix} {class_id}: {cls_name}', color))

    line_height = 20
    box_width = 0
    for text, _ in lines:
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        box_width = max(box_width, text_size[0])

    origin_x = 8
    origin_y = 10
    box_width = min(box_width + 34, max(1, canvas.shape[1] - origin_x - 8))
    box_height = min(10 + line_height * len(lines), max(1, canvas.shape[0] - origin_y - 8))

    overlay = canvas.copy()
    cv2.rectangle(
        overlay,
        (origin_x, origin_y),
        (origin_x + box_width, origin_y + box_height),
        (20, 20, 20),
        -1,
    )
    cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, dst=canvas)

    legend_y = origin_y + 18
    for text, color in lines:
        cv2.rectangle(canvas, (origin_x + 6, legend_y - 10), (origin_x + 22, legend_y + 4), color, -1)
        _draw_text_with_outline(canvas, text, (origin_x + 28, legend_y), color)
        legend_y += line_height


def _resolve_view_shape(shape_meta, view_idx):
    if shape_meta is None:
        return None
    if isinstance(shape_meta, np.ndarray):
        shape_meta = shape_meta.tolist()
    if isinstance(shape_meta, (list, tuple)):
        if len(shape_meta) >= 2 and isinstance(shape_meta[0], (int, np.integer)):
            return int(shape_meta[0]), int(shape_meta[1])
        if view_idx < len(shape_meta):
            return _resolve_view_shape(shape_meta[view_idx], view_idx)
    return None


def _crop_to_unpadded_view(view_img, sample_meta, view_idx):
    view_shape = _resolve_view_shape(sample_meta.get('ori_shape'), view_idx)
    if view_shape is None:
        return np.ascontiguousarray(view_img)

    view_h = max(1, min(view_shape[0], view_img.shape[0]))
    view_w = max(1, min(view_shape[1], view_img.shape[1]))
    return np.ascontiguousarray(view_img[:view_h, :view_w])


def _trim_black_borders(view_img, threshold=2):
    if view_img.size == 0:
        return np.ascontiguousarray(view_img), 0, 0

    non_black = np.any(view_img > threshold, axis=2)
    valid_rows = np.where(non_black.any(axis=1))[0]
    valid_cols = np.where(non_black.any(axis=0))[0]
    if len(valid_rows) == 0 or len(valid_cols) == 0:
        return np.ascontiguousarray(view_img), 0, 0

    top = int(valid_rows[0])
    bottom = int(valid_rows[-1]) + 1
    left = int(valid_cols[0])
    right = int(valid_cols[-1]) + 1
    trimmed = np.ascontiguousarray(view_img[top:bottom, left:right])
    return trimmed, left, top


def _shift_projected_boxes(boxes_2d, offset_x, offset_y):
    if boxes_2d is None:
        return None

    shifted = boxes_2d.copy()
    shifted[..., 0] -= offset_x
    shifted[..., 1] -= offset_y
    return shifted


def _draw_projected_boxes(canvas, boxes_2d, labels, edges, class_names=None, scores=None, gt=False):
    if boxes_2d is None or labels is None or len(labels) == 0:
        return {}

    visible_classes = {}
    for box_idx, box in enumerate(boxes_2d):
        class_id = int(labels[box_idx])
        cls_name = _class_name(class_names, class_id)
        color = (255, 255, 255) if gt else _class_color(class_id)
        visible_classes[(gt, class_id)] = (cls_name, color)
        visible = []
        for start, end in edges:
            if box[start][2] > 0 and box[end][2] > 0:
                visible.append(start)
                visible.append(end)
                _draw_line_with_outline(
                    canvas,
                    (int(box[start][0]), int(box[start][1])),
                    (int(box[end][0]), int(box[end][1])),
                    color,
                    2,
                )

        if visible:
            vis_points = box[visible, :2]
            text_x = int(np.min(vis_points[:, 0]))
            text_y = max(15, int(np.min(vis_points[:, 1])) - 4)
            if gt:
                label_text = f'GT:{cls_name}'
            else:
                label_text = f'{cls_name}:{float(scores[box_idx]):.2f}' if scores is not None else cls_name
            _draw_text_with_outline(canvas, label_text, (text_x, text_y), color)

    return visible_classes


def _load_gt_annotations(dataset, sample_meta):
    sample_token = _sample_token_from_meta(sample_meta)
    dataset_index = _dataset_index_from_token(dataset, sample_token)
    if dataset_index is None:
        dataset_index = _dataset_index_from_filenames(dataset, sample_meta.get('filename'))
    if dataset_index is None:
        return None, None

    ann_info = dataset.get_ann_info(dataset_index)
    if not isinstance(ann_info, dict):
        return None, None

    return ann_info.get('gt_bboxes_3d'), _to_numpy_labels(ann_info.get('gt_labels_3d'))


def _load_gt_annotations_by_index(dataset, dataset_index):
    if dataset is None or dataset_index is None:
        return None, None
    if dataset_index < 0 or dataset_index >= len(getattr(dataset, 'data_infos', [])):
        return None, None

    ann_info = dataset.get_ann_info(dataset_index)
    if not isinstance(ann_info, dict):
        return None, None

    return ann_info.get('gt_bboxes_3d'), _to_numpy_labels(ann_info.get('gt_labels_3d'))


def _save_projected_3d_boxes(data, vis_result, show_dir, rank, class_names=None, dataset=None, show_gt=False, dataset_start_index=None):
    if show_dir is None:
        return False

    img_raw = _unwrap_batch_item(data.get('img'))
    img_metas_raw = _unwrap_batch_item(data.get('img_metas'))
    if img_raw is None or img_metas_raw is None:
        return False

    if isinstance(img_metas_raw, dict):
        img_metas = [img_metas_raw]
    else:
        img_metas = list(img_metas_raw)

    if not isinstance(img_raw, torch.Tensor):
        return False
    if img_raw.dim() == 5:
        batch_imgs = [img_raw[idx] for idx in range(img_raw.shape[0])]
    elif img_raw.dim() == 4:
        batch_imgs = [img_raw]
    else:
        return False

    rank_dir = osp.join(show_dir, f'rank_{rank}')
    mmcv.mkdir_or_exist(rank_dir)
    edges = [(0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7), (4, 5), (4, 7), (2, 6), (5, 6), (6, 7)]

    lidar2img_raw = _unwrap_batch_item(data.get('lidar2img'))

    sample_count = min(len(vis_result), len(batch_imgs), len(img_metas))
    for sample_idx in range(sample_count):
        sample_meta = img_metas[sample_idx]
        sample_result = vis_result[sample_idx]
        if not isinstance(sample_result, dict) or 'pts_bbox' not in sample_result:
            continue

        pts_bbox = sample_result['pts_bbox']
        if 'boxes_3d' not in pts_bbox:
            continue

        scores = _to_numpy_scores(pts_bbox.get('scores_3d'))
        labels = _to_numpy_labels(pts_bbox.get('labels_3d'))
        pred_boxes_3d = pts_bbox['boxes_3d']
        if scores is not None:
            score_mask = scores > 0.2
            pred_boxes_3d = pred_boxes_3d[score_mask]
            scores = scores[score_mask]
            if labels is None:
                labels = np.zeros_like(scores, dtype=np.int64)
            else:
                labels = labels[score_mask]
        elif labels is None:
            labels = np.zeros((len(pred_boxes_3d),), dtype=np.int64)

        img_tensor = batch_imgs[sample_idx]
        if img_tensor.dim() != 4:
            continue

        if isinstance(sample_meta.get('img_norm_cfg'), dict):
            view_images = tensor2imgs(img_tensor, **sample_meta['img_norm_cfg'])
        else:
            view_images = [_to_numpy_image(view) for view in img_tensor]

        lidar2img = sample_meta.get('lidar2img')
        if lidar2img is None and lidar2img_raw is not None:
            if isinstance(lidar2img_raw, torch.Tensor):
                if lidar2img_raw.dim() == 4:
                    lidar2img = lidar2img_raw[sample_idx].detach().cpu().numpy()
                elif lidar2img_raw.dim() == 3:
                    lidar2img = lidar2img_raw.detach().cpu().numpy()
            elif isinstance(lidar2img_raw, (list, tuple)) and sample_idx < len(lidar2img_raw):
                current = lidar2img_raw[sample_idx]
                if isinstance(current, torch.Tensor):
                    lidar2img = current.detach().cpu().numpy()
                else:
                    lidar2img = np.asarray(current)

        lidar2img = np.asarray(lidar2img) if lidar2img is not None else None
        if lidar2img.ndim != 3:
            continue

        pred_boxes_2d = _project_boxes_to_views(pred_boxes_3d, lidar2img)
        gt_boxes_3d, gt_labels = _load_gt_annotations(dataset, sample_meta) if show_gt else (None, None)
        if show_gt and gt_boxes_3d is None and dataset_start_index is not None:
            gt_boxes_3d, gt_labels = _load_gt_annotations_by_index(dataset, dataset_start_index + sample_idx)
        gt_boxes_2d = _project_boxes_to_views(gt_boxes_3d, lidar2img) if gt_boxes_3d is not None else None
        if (pred_boxes_2d is None or len(pred_boxes_2d) == 0) and (gt_boxes_2d is None or len(gt_boxes_2d) == 0):
            continue

        sample_name = str(sample_meta.get('sample_idx') or sample_meta.get('scene_token') or sample_meta.get('lidar_timestamp') or sample_idx)
        filenames = sample_meta.get('filename')
        for view_idx, view_img in enumerate(view_images):
            if view_idx >= lidar2img.shape[0]:
                break
            # # plot raw image[0]
            # plt.imshow(sample_meta.get('ori_img', view_img))
            # plt.show()

            unpadded_view = _crop_to_unpadded_view(view_img, sample_meta, view_idx)
            canvas, offset_x, offset_y = _trim_black_borders(unpadded_view)
            canvas = canvas.copy()
            pred_view_boxes = _shift_projected_boxes(
                pred_boxes_2d[:, view_idx] if pred_boxes_2d is not None else None,
                offset_x,
                offset_y,
            )
            gt_view_boxes = _shift_projected_boxes(
                gt_boxes_2d[:, view_idx] if gt_boxes_2d is not None else None,
                offset_x,
                offset_y,
            )
            visible_classes = {}
            pred_visible = _draw_projected_boxes(
                canvas,
                pred_view_boxes,
                labels,
                edges,
                class_names=class_names,
                scores=scores,
                gt=False,
            )
            gt_visible = _draw_projected_boxes(
                canvas,
                gt_view_boxes,
                gt_labels,
                edges,
                class_names=class_names,
                gt=True,
            )
            visible_classes.update(pred_visible)
            visible_classes.update(gt_visible)
            _draw_legend_overlay(canvas, visible_classes)

            camera_name = f'camera_{view_idx}'
            if isinstance(filenames, (list, tuple)) and view_idx < len(filenames):
                stem = osp.splitext(osp.basename(filenames[view_idx]))[0]
                camera_name = _camera_name_from_filename(filenames[view_idx], view_idx)
                out_name = f'{sample_name}_{stem}_3dbox.jpg'
            else:
                out_name = f'{sample_name}_view{view_idx}_3dbox.jpg'
            camera_dir = osp.join(rank_dir, camera_name)
            mmcv.mkdir_or_exist(camera_dir)
            mmcv.imwrite(canvas, osp.join(camera_dir, out_name))

    return True


def _draw_vehicle_kinematics_bev(canvas, raw_xy, propagated_xy, heading, labels, speed, yaw_rate, slot_indices, class_names, pc_range):
    height, width = canvas.shape[:2]
    x_min, y_min, _, x_max, y_max, _ = pc_range
    span_x = max(float(x_max - x_min), 1e-6)
    span_y = max(float(y_max - y_min), 1e-6)

    def to_canvas(point_xy):
        px = int(np.clip((point_xy[0] - x_min) / span_x * (width - 1), 0, width - 1))
        py = int(np.clip((1.0 - (point_xy[1] - y_min) / span_y) * (height - 1), 0, height - 1))
        return px, py

    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (70, 70, 70), 1)
    axis_x0, axis_y0 = to_canvas((0.0, 0.0))
    cv2.line(canvas, (axis_x0, 0), (axis_x0, height - 1), (55, 55, 55), 1)
    cv2.line(canvas, (0, axis_y0), (width - 1, axis_y0), (55, 55, 55), 1)

    for idx in range(raw_xy.shape[0]):
        class_id = int(labels[idx])
        color = _class_color(class_id)
        raw_pt = to_canvas(raw_xy[idx])
        prop_pt = to_canvas(propagated_xy[idx])
        cv2.circle(canvas, raw_pt, 4, (200, 200, 200), -1)
        cv2.arrowedLine(canvas, raw_pt, prop_pt, color, 2, tipLength=0.25)
        heading_tip = propagated_xy[idx] + heading[idx] * 1.5
        cv2.arrowedLine(canvas, prop_pt, to_canvas(heading_tip), color, 1, tipLength=0.3)
        label_text = f'#{int(slot_indices[idx])} {_class_name(class_names, class_id)} v={float(speed[idx]):.1f} yaw={float(yaw_rate[idx]):.2f}'
        cv2.putText(canvas, label_text, (prop_pt[0] + 4, max(15, prop_pt[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)


def _save_vehicle_kinematics_debug(model, data, show_dir, rank, class_names=None):
    if show_dir is None:
        return False

    module = model.module if hasattr(model, 'module') else model
    pts_bbox_head = getattr(module, 'pts_bbox_head', None)
    debug_info = getattr(pts_bbox_head, 'debug_last_vehicle_kinematics', None)
    if not isinstance(debug_info, dict):
        return False

    entries = debug_info.get('entries')
    pc_range = np.asarray(debug_info.get('pc_range'))
    if not isinstance(entries, list) or pc_range.shape[0] < 6:
        return False

    img_metas_raw = _unwrap_batch_item(data.get('img_metas'))
    if img_metas_raw is None:
        return False
    if isinstance(img_metas_raw, dict):
        img_metas = [img_metas_raw]
    else:
        img_metas = list(img_metas_raw)

    rank_dir = osp.join(show_dir, f'rank_{rank}', 'vehicle_kinematics_debug')
    mmcv.mkdir_or_exist(rank_dir)
    saved = False
    sample_count = min(len(entries), len(img_metas))
    for sample_idx in range(sample_count):
        entry = entries[sample_idx]
        if entry is None or len(entry.get('raw_xy', [])) == 0:
            continue

        raw_xy = np.nan_to_num(np.asarray(entry['raw_xy']), nan=0.0, posinf=0.0, neginf=0.0)
        propagated_xy = np.nan_to_num(np.asarray(entry['propagated_xy']), nan=0.0, posinf=0.0, neginf=0.0)
        heading = np.nan_to_num(np.asarray(entry['heading']), nan=0.0, posinf=0.0, neginf=0.0)
        labels = np.asarray(entry['labels'])
        dt = np.nan_to_num(np.asarray(entry['dt']), nan=0.0, posinf=0.0, neginf=0.0)
        speed = np.nan_to_num(np.asarray(entry['speed']), nan=0.0, posinf=0.0, neginf=0.0)
        yaw_rate = np.nan_to_num(np.asarray(entry['yaw_rate']), nan=0.0, posinf=0.0, neginf=0.0)
        slot_indices = np.asarray(entry['slot_indices'])

        canvas = np.full((1024, 1024, 3), 18, dtype=np.uint8)
        _draw_vehicle_kinematics_bev(
            canvas,
            raw_xy,
            propagated_xy,
            heading,
            labels,
            speed,
            yaw_rate,
            slot_indices,
            class_names,
            pc_range,
        )

        sample_meta = img_metas[sample_idx]
        sample_name = str(sample_meta.get('sample_idx') or sample_meta.get('scene_token') or sample_meta.get('lidar_timestamp') or sample_idx)
        image_path = osp.join(rank_dir, f'{sample_name}_vehicle_kinematics_bev.jpg')
        mmcv.imwrite(canvas, image_path)
        np.savez(
            osp.join(rank_dir, f'{sample_name}_vehicle_kinematics_bev.npz'),
            raw_xy=raw_xy,
            propagated_xy=propagated_xy,
            heading=heading,
            labels=labels,
            dt=dt,
            speed=speed,
            yaw_rate=yaw_rate,
            slot_indices=slot_indices,
            pc_range=pc_range,
        )
        saved = True

    return saved


def _run_show_results(model, data, result, show, show_dir, rank, class_names=None, dataset=None, show_gt=False, dataset_start_index=None):
    if not (show or show_dir):
        return

    debug_saved = _save_vehicle_kinematics_debug(model, data, show_dir, rank, class_names=class_names)

    vis_result = _to_visual_results(result)
    if vis_result is None:
        if debug_saved:
            return
        warnings.warn('Unable to map prediction result to visualization format.')
        return

    module = model.module if hasattr(model, 'module') else model
    show_fn = getattr(module, 'show_results', None)
    if show_fn is None:
        if debug_saved or _save_projected_3d_boxes(
            data,
            vis_result,
            show_dir,
            rank,
            class_names=class_names,
            dataset=dataset,
            show_gt=show_gt,
            dataset_start_index=dataset_start_index,
        ):
            return
        warnings.warn('`--show`/`--show-dir` requested but model has no show_results method.')
        return

    last_error = None
    call_patterns = [
        lambda: show_fn(data, vis_result, show=show, out_dir=show_dir),
        lambda: show_fn(data, vis_result, out_dir=show_dir),
        lambda: show_fn(data, vis_result, show=show),
        lambda: show_fn(data, vis_result),
    ]
    for call in call_patterns:
        try:
            call()
            _save_projected_3d_boxes(
                data,
                vis_result,
                show_dir,
                rank,
                class_names=class_names,
                dataset=dataset,
                show_gt=show_gt,
                dataset_start_index=dataset_start_index,
            )
            return
        except TypeError as err:
            last_error = err
        except KeyError as err:
            if str(err).strip("'") == 'points':
                if _save_projected_3d_boxes(
                    data,
                    vis_result,
                    show_dir,
                    rank,
                    class_names=class_names,
                    dataset=dataset,
                    show_gt=show_gt,
                    dataset_start_index=dataset_start_index,
                ) or debug_saved:
                    return
            raise

    if _save_projected_3d_boxes(
        data,
        vis_result,
        show_dir,
        rank,
        class_names=class_names,
        dataset=dataset,
        show_gt=show_gt,
        dataset_start_index=dataset_start_index,
    ) or debug_saved:
        return

    warnings.warn(
        '`--show`/`--show-dir` was requested, but all show_results call signatures failed. '
        f'Last error: {last_error}')

def custom_encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code. Semantic Masks only
    Args:
        mask_results (list | tuple[list]): bitmap mask results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).
    Returns:
        list | tuple: RLE encoded mask.
    """
    cls_segms = mask_results
    num_classes = len(cls_segms)
    encoded_mask_results = []
    for i in range(len(cls_segms)):
        encoded_mask_results.append(
            mask_util.encode(
                np.array(
                    cls_segms[i][:, :, np.newaxis], order='F',
                        dtype='uint8'))[0])  # encoded with RLE
    return [encoded_mask_results]

def custom_multi_gpu_test(model,
                          data_loader,
                          tmpdir=None,
                          gpu_collect=False,
                          show=False,
                          show_dir=None,
                          show_gt=False):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """
    model.eval()
    if show_dir is not None:
        mmcv.mkdir_or_exist(show_dir)
    bbox_results = []
    mask_results = []
    dataset = data_loader.dataset
    class_names = getattr(model.module if hasattr(model, 'module') else model, 'CLASSES', None)
    if class_names is None:
        class_names = getattr(dataset, 'CLASSES', None)
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    have_mask = False
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            batch_size = 0

            _run_show_results(
                model,
                data,
                result,
                show=show,
                show_dir=show_dir,
                rank=rank,
                class_names=class_names,
                dataset=dataset,
                show_gt=show_gt,
                dataset_start_index=i * world_size,
            )

            # encode mask results
            if isinstance(result, dict):
                if 'bbox_results' in result.keys():
                    bbox_result = result['bbox_results']
                    batch_size = len(result['bbox_results'])
                    bbox_results.extend(bbox_result)
                elif 'pts_bbox' in result.keys():
                    batch_size = 1
                    bbox_results.append(result)
                if 'mask_results' in result.keys() and result['mask_results'] is not None:
                    mask_result = custom_encode_mask_results(result['mask_results'])
                    mask_results.extend(mask_result)
                    have_mask = True
            else:
                batch_size = len(result)
                bbox_results.extend(result)

            #if isinstance(result[0], tuple):
            #    assert False, 'this code is for instance segmentation, which our code will not utilize.'
            #    result = [(bbox_results, encode_mask_results(mask_results))
            #              for bbox_results, mask_results in result]
        if rank == 0:
            
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        bbox_results = collect_results_gpu(bbox_results, len(dataset))
        if have_mask:
            mask_results = collect_results_gpu(mask_results, len(dataset))
        else:
            mask_results = None
    else:
        bbox_results = collect_results_cpu(bbox_results, len(dataset), tmpdir)
        tmpdir = tmpdir+'_mask' if tmpdir is not None else None
        if have_mask:
            mask_results = collect_results_cpu(mask_results, len(dataset), tmpdir)
        else:
            mask_results = None

    if mask_results is None:
        return bbox_results
    return {'bbox_results': bbox_results, 'mask_results': mask_results}


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        '''
        bacause we change the sample of the evaluation stage to make sure that each gpu will handle continuous sample,
        '''
        #for res in zip(*part_list):
        for res in part_list:  
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    collect_results_cpu(result_part, size)
