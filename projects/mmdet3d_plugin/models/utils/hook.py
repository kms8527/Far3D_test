from os import path as osp
from math import ceil

from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook

@HOOKS.register_module()
class UseGtDepthHook(Hook):
    def __init__(
        self,
        stop_gt_depth_iter=0,
        stop_iter=0,
    ):
        self.stop_gt_depth_iter = stop_gt_depth_iter
        self.stop_iter = stop_iter

    def before_train_iter(self, runner):
        cur_iter = runner.iter
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        if cur_iter >= self.stop_gt_depth_iter:
            model.pts_bbox_head.flag_disable_gt_depth = True
        if cur_iter >= self.stop_iter:
            model.pts_bbox_head.loss_flag = False


@HOOKS.register_module()
class PeriodicCkptHook(Hook):
    def __init__(self, interval=1000, save_dir='ckpts'):
        self.interval = interval
        self.save_dir = save_dir

    def after_train_iter(self, runner):
        cur_iter = runner.iter + 1
        if self.interval <= 0 or cur_iter % self.interval != 0:
            return

        exp_name = osp.basename(osp.normpath(runner.work_dir))
        target_dir = osp.join(self.save_dir, exp_name)
        iters_per_epoch = max(1, len(runner.data_loader))
        cur_epoch = cur_iter // iters_per_epoch
        if cur_iter % iters_per_epoch != 0:
            cur_epoch += 1

        max_iters = getattr(runner, '_max_iters', None)
        if max_iters is None:
            max_iters = getattr(runner, 'max_iters', cur_iter)
        total_epoch = max(1, ceil(max_iters / iters_per_epoch))

        runner.save_checkpoint(
            target_dir,
            filename_tmpl=(
                f'iter_{cur_iter}_epoch_{cur_epoch}_of_{total_epoch}.pth'),
            create_symlink=False)
