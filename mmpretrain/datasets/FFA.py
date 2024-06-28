
from .multi_label import MultiLabelDataset
from typing import List

from mmpretrain.registry import DATASETS
from .base_dataset import BaseDataset


@DATASETS.register_module()
class Ffa(MultiLabelDataset):
    
    
     def get_cat_ids(self, idx: int) -> List[int]:
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            cat_ids (List[int]): Image categories of specified index.
        """
        return self.get_data_info(idx)['gt_score']