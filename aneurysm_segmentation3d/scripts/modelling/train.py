import os, sys
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
from aneurysm_segmentation3d.scripts.modelling.trainer import Trainer


BASE_DIR = "D:\\Workspace\\Python\AneurysmSegmentation\\aneurysm_segmentation3d"
NUM_WORKERS = 2
BATCH_SIZE = 3
PARTS_TO_SEGMENT = 5

CONF_DIR = os.path.join(
    BASE_DIR, "scripts\\data\conf\\conf_base.yaml"
)

# Load Config file
params = OmegaConf.load(CONF_DIR)
params.dataroot = os.path.join(BASE_DIR, "datasets\\data")

# create category counts
category_to_seg = {"aneur": np.arange(PARTS_TO_SEGMENT)}

# Create Dataset
dataset = AneurysmDataset.AneurysmDataset(params, category_to_seg)

# Create the Model
model = PartSegKPConv(
    dataset.class_to_segments,
    input_nc=dataset.train_dataset[1].x.shape[1] - 1,
)


if __name__ == "__main__":

    # Create dataloaders
    dataset.create_dataloaders(
        model,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        precompute_multi_scale=True,
    )
    # sample = next(iter(dataset.train_dataloader))

    trainer = Trainer(
        model, dataset, num_epoch=1, device=torch.device("cpu")
    )
    trainer.fit()
