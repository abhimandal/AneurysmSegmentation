import os, sys
import torch
import pyvista as pv
import pandas as pd
import numpy as np

sys.path.append(os.getcwd())

from torch_points3d.applications.kpconv import KPConv
from torch_points3d.core.common_modules.base_modules import (
    MultiHeadClassifier,
)


class PartSegKPConv(torch.nn.Module):
    def __init__(self, cat_to_seg, input_nc):
        super().__init__()
        self.unet = KPConv(
            architecture="unet",
            input_nc=input_nc,
            num_layers=4,
            in_grid_size=0.02,
        )
        self.classifier = MultiHeadClassifier(
            self.unet.output_nc, cat_to_seg
        )

    @property
    def conv_type(self):
        """ This is needed by the dataset to infer which batch collate should be used"""
        return self.unet.conv_type

    def get_batch(self):
        return self.batch

    def get_output(self):
        """ This is needed by the tracker to get access to the ouputs of the network"""
        return self.output

    def get_labels(self):
        """ Needed by the tracker in order to access ground truth labels"""
        return self.labels

    def get_current_losses(self):
        """ Entry point for the tracker to grab the loss """
        return {"loss_seg": float(self.loss_seg)}

    def forward(self, data):
        self.labels = data.y
        self.batch = data.batch

        # Forward through unet and classifier
        data_features = self.unet(data)
        self.output = self.classifier(data_features.x, data.category)

        # Set loss for the backward pass
        self.loss_seg = torch.nn.functional.nll_loss(
            self.output, self.labels
        )
        return self.output

    def get_spatial_ops(self):
        return self.unet.get_spatial_ops()

    def backward(self):
        self.loss_seg.backward()

