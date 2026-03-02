from .nuscenes_dataset import CustomNuScenesDataset
from .builder import custom_build_dataset

try:
    from .argoverse2_dataset import Argoverse2Dataset
    from .argoverse2_dataset_t import Argoverse2DatasetT
except ModuleNotFoundError:
    Argoverse2Dataset = None
    Argoverse2DatasetT = None

__all__ = [
    'CustomNuScenesDataset',
    'custom_build_dataset',
]

if Argoverse2Dataset is not None:
    __all__.append('Argoverse2Dataset')
if Argoverse2DatasetT is not None:
    __all__.append('Argoverse2DatasetT')
