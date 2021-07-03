import os, sys
import glob
import torch
import pyvista as pv
import pandas as pd
import numpy as np

sys.path.append(os.getcwd())

from omegaconf import OmegaConf
from torch.utils.data import Dataset
from torch_points3d.core.common_modules.base_modules import (
    MultiHeadClassifier,
)

# Own Libs
from aneurysm_segmentation3d.scripts.data import AneurysmDataset
from aneurysm_segmentation3d.scripts.modelling.model import (
    PartSegKPConv,
)
from aneurysm_segmentation3d.scripts.modelling.trainer_chkp import (
    Trainer,
)


BASE_DIR = "D:\\Workspace\\Python\AneurysmSegmentation\\aneurysm_segmentation3d"
CONF_DIR = os.path.join(
    BASE_DIR, "scripts\\data\conf\\conf_test.yaml"
)
DATAROOT = os.path.join(BASE_DIR, "datasets\\data")
PROCESSED_DIR = os.listdir(
    os.path.join(DATAROOT, "aneurysm\\processed")
)

# BASE_DIR = "/workspace/Storage_fast/AneurysmSegmentation/aneurysm_segmentation3d"
# CONF_DIR = os.path.join(
#     BASE_DIR, "scripts/data/conf/conf_base.yaml"
# )
# DATAROOT = os.path.join(BASE_DIR, "datasets/data")
# PROCESSED_DIR = os.listdir(
#     os.path.join(DATAROOT, "aneurysm/processed")
# )

NUM_WORKERS = 2
BATCH_SIZE = 3
PARTS_TO_SEGMENT = 5
DELETE_OLD_FILES = False

# Load Config file
params = OmegaConf.load(CONF_DIR)
params.dataroot = DATAROOT
params.parts_to_segment = PARTS_TO_SEGMENT

# create category counts
# category_to_seg = {"aneur": np.arange(PARTS_TO_SEGMENT)}

# Delete older files
files = glob.glob(DATAROOT)
if DELETE_OLD_FILES:
    for f in files:
        os.remove(f)
    print("Finished deleting older processed files")


# Create Dataset

# NOTE: Since creating new processed file, throws error, exit and run again.
# If the files are already exit for a given config inside the "processed" dir,
# then the files are loaded.


if (len(PROCESSED_DIR)) == 0:
    # Create Dataset
    dataset = AneurysmDataset.AneurysmDataset(params)

    print(
        "\n",
        "*" * 20,
        "Finished creating fresh processed dataset files.",
        "Exiting. Please run the script again",
        "*" * 20,
    )

    # to avoid multiprocessing error, run exit and run the script again
    sys.exit()
else:
    # Create Dataset
    dataset = AneurysmDataset.AneurysmDataset(params)

# Create the Model
model = PartSegKPConv(
    dataset.cat_to_seg,
    input_nc=dataset.train_dataset[1].x.shape[1] - 1,
)


if __name__ == "__main__":

    # Create dataloaders
    # dataset.create_dataloaders(
    #     model,
    #     batch_size=BATCH_SIZE,
    #     num_workers=NUM_WORKERS,
    #     shuffle=True,
    #     precompute_multi_scale=True,
    # )
    # sample = next(iter(dataset.train_dataloader))

    trainer = Trainer(
        params, dataset, PARTS_TO_SEGMENT, device=torch.device("cpu"),
    )
    trainer.fit()