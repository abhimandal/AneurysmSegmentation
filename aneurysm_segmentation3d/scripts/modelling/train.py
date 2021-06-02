# print(
#     "__file__={0:<35} | __name__={1:<20} | __package__={2:<20}".format(
#         __file__, __name__, str(__package__)
#     )
# )

# from ..data import AneurysmDataset
# from aneurysm_segmentation3d.scripts.data import AneurysmDataset
import os, sys

# print(os.getcwd())

# sys.path.append(os.getcwd())
# from src.data.AneurysmDataset import AneurysmDataset
from aneurysm_segmentation3d.scripts.data import AneurysmDataset
from torch.utils.data import Dataset
