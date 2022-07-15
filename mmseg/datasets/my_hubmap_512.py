from .builder import DATASETS
from .custom import CustomDataset
import os.path as osp

@DATASETS.register_module()
class MRDDataset(CustomDataset):
    CLASSES = ("background","ftu")
    PALETTE = [[0,0,0],[255,255,255]]
    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='_ann.png', 
                     split=split, **kwargs)
                     
        assert osp.exists(self.img_dir) and self.split is not None
