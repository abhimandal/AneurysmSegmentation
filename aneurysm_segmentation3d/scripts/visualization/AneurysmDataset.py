import torch
import glob
import os
import sys

sys.path.append(os.getcwd())

import shutil
import warnings

warnings.filterwarnings("ignore")

from torch_geometric.io import read_txt_array
from torch_geometric.data.data import Data
import torch_geometric.transforms as T
from torch_geometric.nn import knn_interpolate
import numpy as np
import logging

from torch_points3d.core.data_transform import SaveOriginalPosId
from torch_points3d.utils import is_list
from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.metrics.shapenet_part_tracker import (
    ShapenetPartTracker,
)
from torch_points3d.datasets.segmentation.shapenet import ShapeNet

# Aneurysm files
from aneurysm_segmentation3d.scripts.dataset import AneurysmDataset
from aneurysm_segmentation3d.scripts.dataset.AneurysmDataset import (
    read_mesh_vertices,
)
from torch_points3d.metrics.segmentation_tracker import (
    SegmentationTracker,
)
from plyfile import PlyData, PlyElement, PlyProperty, PlyListProperty
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
)

log = logging.getLogger(__name__)


class _ForwardAneurysm(torch.utils.data.Dataset):
    """Dataset to run forward inference on Shapenet kind of data data. Runs on a whole folder.
    Arguments:
        path: folder that contains a set of files of a given category
        category: index of the category to use for forward inference. This value depends on how many categories the model has been trained on.
        transforms: transforms to be applied to the data
    """

    def __init__(
        self,
        path,
        category: int,
        transforms=None,
        predict_sample_pid="18_EM",
        raw_file_identifiers=None,
        custom_features_dict={"shot": True},
        num_parts_to_segment=5,
        shuffled_splits=None,
        scaler_type="global",
    ):
        super().__init__()
        self.shuffled_splits = shuffled_splits
        self.raw_file_identifiers = raw_file_identifiers
        self.custom_features_dict = custom_features_dict
        self.num_parts_to_segment = num_parts_to_segment
        self.scaler_type = scaler_type

        self._category = category
        self._path = path
        self.predict_sample_PID = predict_sample_pid
        self._transforms = transforms
        assert os.path.exists(self._path)
        if self.__len__() == 0:
            raise ValueError("Empty folder %s" % path)

    def __len__(self):
        return len(self.predict_sample_PID)

    def _read_file(self, P_ID, index):
        filepath = os.path.join(self._path, f"{P_ID}_pca.ply")

        data = torch.from_numpy(
            read_mesh_vertices(
                filepath,
                self.custom_features_dict,
                self.num_parts_to_segment,
                self.scaler_type,
            )
        )
        pos = data[:, :3]
        x = data[:, 3:-1]
        y = data[:, -1].type(torch.long)
        # Re-assign negative classes (floating point error) to zero
        y = torch.where(y < 0, 0, y)
        data = Data(
            pos=pos,
            x=x,
            y=y,
        )
        return data

    def get_raw(self, index):
        """returns the untransformed data associated with an element"""
        return self._read_file(self.predict_sample_PID, index)

    @property
    def num_features(self):
        feats = self[0].x
        if feats is not None:
            return feats.shape[-1]
        return 0

    def get_filename(self, index):
        P_ID = self.predict_sample_PID
        return os.path.basename(
            os.path.join(self._path, f"{P_ID}_pca.ply")
        )

    def __getitem__(self, index):
        data = self._read_file(self.predict_sample_PID, index)
        category = torch.ones(data.pos.shape[0], dtype=torch.long) * 0
        setattr(data, "category", category)
        setattr(data, "sampleid", torch.tensor([index]))

        if self._transforms is not None:
            data = self._transforms(data)
        return data


class ForwardAneurysmDataset(BaseDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        # dataset specific
        is_test = dataset_opt.get("is_test", False)
        shuffled_splits = dataset_opt.get("shuffled_splits", False)
        raw_file_identifiers = dataset_opt.get(
            "raw_file_identifiers", False
        )
        custom_features_dict = dataset_opt.get(
            "features_to_include", False
        )
        num_parts_to_segment = dataset_opt.get(
            "parts_to_segment", False
        )
        self.cat_to_seg = {"aneur": np.arange(num_parts_to_segment)}
        scaler_type = dataset_opt.get("scaler_type", False)
        predict_sample_pid = dataset_opt.get("forward_pid", False)

        # forward specific
        forward_category = dataset_opt.forward_category
        if not isinstance(forward_category, str):
            raise ValueError(
                "dataset_opt.forward_category is not set or is not a string. Current value: {}".format(
                    dataset_opt.forward_category
                )
            )

        # modified from dataset.category
        self._train_categories = dataset_opt.forward_category

        if not is_list(self._train_categories):
            self._train_categories = [self._train_categories]

        # Sets the index of the category with respect to the categories in the trained model
        self._cat_idx = None
        for i, train_category in enumerate(self._train_categories):
            if forward_category.lower() == train_category.lower():
                self._cat_idx = i
                break
        if self._cat_idx is None:
            raise ValueError(
                "Cannot run an inference on category {} with a network trained on {}".format(
                    forward_category, self._train_categories
                )
            )
        log.info(
            "Running an inference on category {} with a network trained on {}".format(
                forward_category, self._train_categories
            )
        )

        self._data_path = dataset_opt.dataroot

        transforms = SaveOriginalPosId()
        for t in [self.pre_transform, self.test_transform]:
            if t:
                transforms = T.Compose([transforms, t])
        self.test_dataset = _ForwardAneurysm(
            path=self._data_path,
            category=self._cat_idx,
            transforms=transforms,
            predict_sample_pid=predict_sample_pid,
            raw_file_identifiers=raw_file_identifiers,
            custom_features_dict=custom_features_dict,
            num_parts_to_segment=num_parts_to_segment,
            scaler_type=scaler_type,
            shuffled_splits=shuffled_splits,
        )

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        # return ShapenetPartTracker(
        #     self, wandb_log=wandb_log, use_tensorboard=tensorboard_log
        # )
        return SegmentationTracker(
            self, wandb_log=wandb_log, use_tensorboard=tensorboard_log
        )

    def predict_original_samples(self, batch, conv_type, output):
        """Takes the output generated by the NN and upsamples it to the original data
        Arguments:
            batch -- processed batch
            conv_type -- Type of convolutio (DENSE, PARTIAL_DENSE, etc...)
            output -- output predicted by the model
        """
        full_res_results = {}
        num_sample = BaseDataset.get_num_samples(batch, conv_type)
        if conv_type == "DENSE":
            output = output.reshape(
                num_sample, -1, output.shape[-1]
            )  # [B,N,L]

        setattr(batch, "_pred", output)
        for b in range(num_sample):
            sampleid = batch.sampleid[b]
            sample_raw_pos = (
                self.test_dataset[0]
                .get_raw(sampleid)
                .pos.to(output.device)
            )
            predicted = BaseDataset.get_sample(
                batch, "_pred", b, conv_type
            )
            origindid = BaseDataset.get_sample(
                batch, SaveOriginalPosId.KEY, b, conv_type
            )
            full_prediction = knn_interpolate(
                predicted,
                sample_raw_pos[origindid],
                sample_raw_pos,
                k=3,
            )
            labels = full_prediction.max(1)[1].unsqueeze(-1)
            full_res_results[
                self.test_dataset[0].get_filename(sampleid)
            ] = np.hstack(
                (
                    sample_raw_pos.cpu().numpy(),
                    labels.cpu().numpy(),
                )
            )
        return full_res_results

    @property
    def class_to_segments(self):
        classes_to_segment = {}
        classes_to_segment["aneur"] = self.cat_to_seg["aneur"]
        return classes_to_segment

    @property
    def num_classes(self):
        segments = self.class_to_segments.values()
        num = 0
        for seg in segments:
            num = max(num, max(seg))
        return num + 1
