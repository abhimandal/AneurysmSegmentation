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

CONF_DIR = "D:\\Workspace\\Python\AneurysmSegmentation\\aneurysm_segmentation3d\\scripts\\data\conf\\conf_base.yaml"
PARTS_TO_SEGMENT = 5

# category_to_seg = {'aneur': np.arange(PARTS_TO_SEGMENT)}
category_to_seg = {"aneur": [0, 1, 2, 3, 4]}

# Load Config file
params = OmegaConf.load(CONF_DIR)

params.dataroot = (
    "D:\\Workspace\\Python\\aneurysm-segmentation\\datasets\\data"
)
dataset = AneurysmDataset.AneurysmDataset(params, category_to_seg)

model = PartSegKPConv(dataset.class_to_segments)

NUM_WORKERS = 0
BATCH_SIZE = 3
dataset.create_dataloaders(
    model,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=True,
    precompute_multi_scale=True,
)

# sample = next(iter(dataset.train_dataloader))
trainer = Trainer(model, dataset)
trainer.fit()
