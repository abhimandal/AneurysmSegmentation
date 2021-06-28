import os
import os.path as osp
import shutil
import json
from tqdm.auto import tqdm as tq
from itertools import repeat, product
import numpy as np
import pandas as pd
import torch

from torch_geometric.data import Data, InMemoryDataset, extract_zip
from torch_geometric.io import read_txt_array
import torch_geometric.transforms as T
from torch_points3d.core.data_transform import SaveOriginalPosId
from torch_points3d.metrics.shapenet_part_tracker import (
    ShapenetPartTracker,
)
from torch_points3d.datasets.base_dataset import (
    BaseDataset,
    save_used_properties,
)
from torch_points3d.utils.download import download_url
from plyfile import PlyData, PlyElement, PlyProperty, PlyListProperty
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
)

########################################################################################
#                                                                                      #
#                                      UTILS                                           #
#                                                                                      #
########################################################################################


def scale_data(col: pd.Series, scaler) -> pd.Series:
    """
    Scale a column
    """
    X = col.values.reshape(-1, 1).copy()
    scaled_array = scaler.fit_transform(X)
    scaled_column = pd.Series(scaled_array.tolist()).explode()
    return scaled_column


def convert_mesh_to_dataframe(meshply, feat_dict):
    """
    Convert mesh values into a dataframe and add feature_names according 
    to the dictionary passed and scale them between 0 and 1.
    
    Args:
        meshply (meshply obj): Mesh obj that should be converted to a 
            dataframe
        features_dict (dictionary,optional): Custom features to be 
            used to make Dataset. Values should be True or False or 
            1 or 0 for each key.
            Accepted keys - "mean_curvature", "gauss_curvature", "fpfh",
                "shot", "rf", "ones"
    """

    df = pd.DataFrame()

    # Scaler for the dataframe
    min_max_scaler = MinMaxScaler()

    df["x"] = pd.Series(meshply.elements[0].data["x"])
    df["y"] = pd.Series(meshply.elements[0].data["y"])
    df["z"] = pd.Series(meshply.elements[0].data["z"])

    ###### FEATURE: MEAN CURVATURE ######
    if [v for k, v in feat_dict.items() if k == "mean_curvature"][0]:
        df["mean_curv"] = pd.Series(
            meshply.elements[0].data["mean_curv"]
        )
        # Scaler for the feature
        df["mean_curv"] = scale_data(df["mean_curv"], min_max_scaler)

    ###### FEATURE: GAUSSIAN CURVATURE ######
    if [v for k, v in feat_dict.items() if k == "gauss_curvature"][0]:
        df["gauss_curv"] = pd.Series(
            meshply.elements[0].data["gauss_curv"]
        )
        # Scaler for the feature
        df["gauss_curv"] = scale_data(
            df["gauss_curv"], min_max_scaler
        )

    ###### FEATURE: FPFH DESCRIPTOR ######
    if [v for k, v in feat_dict.items() if k == "fpfh"][0]:
        df[["fpfh_1", "fpfh_2"]] = pd.DataFrame(
            meshply.elements[0].data["fpfh"].tolist()
        )
        # Scaler for the feature
        df["fpfh_1"] = scale_data(df["fpfh_1"], min_max_scaler)
        df["fpfh_2"] = scale_data(df["fpfh_2"], min_max_scaler)

    ###### FEATURE: SHOT DESCRIPTOR ######
    if [v for k, v in feat_dict.items() if k == "shot"][0]:
        df[["shot_1", "shot_2", "shot_3"]] = pd.DataFrame(
            meshply.elements[0].data["shot"].tolist()
        )
        # Scaler for the feature
        df["shot_1"] = scale_data(df["shot_1"], min_max_scaler)
        df["shot_2"] = scale_data(df["shot_2"], min_max_scaler)
        df["shot_3"] = scale_data(df["shot_3"], min_max_scaler)

    ###### FEATURE: SHOT_RF DESCRIPTOR ######
    if [v for k, v in feat_dict.items() if k == "rf"][0]:
        df[["rf_1", "rf_2", "rf_3"]] = pd.DataFrame(
            meshply.elements[0].data["rf"].tolist()
        )
        # Scaler for the feature
        df["rf_1"] = scale_data(df["rf_1"], min_max_scaler)
        df["rf_2"] = scale_data(df["rf_2"], min_max_scaler)
        df["rf_3"] = scale_data(df["rf_3"], min_max_scaler)

    ###### FEATURE: ADD ONES ######
    if [v for k, v in feat_dict.items() if k == "ones"][0]:
        df["ones"] = int(1)

    ###### LABEL: WSS - FOR PREDICTION ######
    df["WSS"] = pd.Series(meshply.elements[0].data["WSS"])
    # Scaler for the feature
    df["WSS"] = scale_data(df["WSS"], min_max_scaler)

    return df


def read_mesh_vertices(
    filepath, custom_features_dict, num_parts_to_segment
):
    """read XYZ and features for each vertex in numpy ndarray
    
    Example - 
    If only XYZ to be populated then for 5 points, vertices will be:
    
    vertices = array([[ 0.02699408, -0.16551971, -0.12976472],
                    [ 0.02701367, -0.16554399, -0.12981543],
                    [ 0.02698801, -0.16551463, -0.12982164],
                    [ 0.02702969, -0.16545248, -0.12979169],
                    [ 0.02706531, -0.16538525, -0.12981866]], dtype=float32)
    
    """
    assert os.path.isfile(filepath)

    with open(filepath, "rb") as f:
        meshplydata = PlyData.read(f)
        num_verts = meshplydata["vertex"].count
        df = convert_mesh_to_dataframe(
            meshplydata, custom_features_dict
        )

        # Categorize the WSS to different parts for part-segmentation
        df["WSS"] = pd.cut(
            df["WSS"],
            bins=np.linspace(0, 1, num_parts_to_segment + 1),
            labels=np.arange(0, num_parts_to_segment),
        )

        vertices = np.empty((0, df.shape[1]), dtype=np.float32)
        # Stack all the vertices
        vertices = np.vstack(
            (vertices, df.iloc[:, :].astype(np.float32))
        )

    return vertices


class Aneurysm(InMemoryDataset):
    r""" Aneurysm dataset for part-level segmentation which corresponds to 
    different WSS regions in one patient file
    
    
    Each file has with 2 to 5 parts/regions based on WSS values. 

    Args:
        root (string): Root directory where the dataset should be saved.
        split (string, optional): If :obj:`"train"`, loads the training
            dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"trainval"`, loads the training and validation dataset.
            If :obj:`"test"`, loads the test dataset.
            (default: :obj:`"trainval"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    def __init__(
        self,
        root,
        split="trainval",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        is_test=False,
        raw_file_identifiers=None,
        custom_features_dict={"shot": True},
        num_parts_to_segment=5,
        shuffled_splits=None,
    ):
        self.is_test = is_test
        self.shuffled_splits = shuffled_splits
        self.raw_file_identifiers = raw_file_identifiers
        self.custom_features_dict = custom_features_dict
        self.num_parts_to_segment = num_parts_to_segment
        super(Aneurysm, self).__init__(
            root, transform, pre_transform, pre_filter
        )

        if split == "train":
            path = self.processed_paths[0]
            raw_path = self.processed_raw_paths[0]
        elif split == "val":
            path = self.processed_paths[1]
            raw_path = self.processed_raw_paths[1]
        elif split == "test":
            path = self.processed_paths[2]
            raw_path = self.processed_raw_paths[2]
        elif split == "trainval":
            path = self.processed_paths[3]
            raw_path = self.processed_raw_paths[3]
        else:
            raise ValueError(
                (
                    f"Split {split} found, but expected either "
                    "train, val, trainval or test"
                )
            )

        self.data, self.slices = torch.load(path)

    def load_data(self, path, include_normals):
        """This function is used twice to load data for both raw and pre_transformed
        """
        data, slices = torch.load(path)
        data.x = data.x if include_normals else None

        y_mask = torch.zeros(
            (len(self.seg_classes.keys()), 50), dtype=torch.bool
        )
        for i, labels in enumerate(self.seg_classes.values()):
            y_mask[i, labels] = 1

        return data, slices, y_mask

    @property
    def raw_file_names(self):
        return [x + "_pca.ply" for x in self.raw_file_identifiers]

    @property
    def processed_raw_paths(self):
        processed_raw_paths = [
            os.path.join(self.processed_dir, "raw_{}".format(s))
            for s in ["train", "val", "test", "trainval"]
        ]
        return processed_raw_paths

    @property
    def processed_file_names(self):
        return [
            os.path.join("{}.pt".format(split))
            for split in ["train", "val", "test", "trainval"]
        ]

    def download(self):
        pass

    def _process_filenames(self, filepaths):
        data_raw_list = []
        data_list = []

        has_pre_transform = self.pre_transform is not None

        id_scan = -1
        for filepath in tq(filepaths):
            id_scan += 1
            data = torch.from_numpy(
                read_mesh_vertices(
                    filepath,
                    self.custom_features_dict,
                    self.num_parts_to_segment,
                )
            )
            pos = data[:, :3]
            x = data[:, 3:-1]
            y = data[:, -1].type(torch.long)
            # Re-assign negative classes (floating point error) to zero
            y = torch.where(y < 0, 0, y)

            # from collections import Counter
            # print(f"data: {Counter(y.cpu().detach().numpy())}")

            category = torch.ones(x.shape[0], dtype=torch.long) * 0
            id_scan_tensor = torch.from_numpy(
                np.asarray([id_scan])
            ).clone()
            data = Data(
                pos=pos,
                x=x,
                y=y,
                category=category,
                id_scan=id_scan_tensor,
            )

            data = SaveOriginalPosId()(data)

            if self.pre_filter is not None and not self.pre_filter(
                data
            ):
                continue

            data_raw_list.append(
                data.clone() if has_pre_transform else data
            )
            if has_pre_transform:
                data = self.pre_transform(data)
                data_list.append(data)
            #     print(
            #         f"data_list:{Counter(data_list[0].y.cpu().detach().numpy())}"
            #     )
            # print(
            #     f"data_raw_list:{Counter(data_raw_list[0].y.cpu().detach().numpy())}"
            # )

        if not has_pre_transform:
            return [], data_raw_list
        return data_raw_list, data_list

    def _save_data_list(self, datas, path_to_datas, save_bool=True):
        if save_bool:
            torch.save(self.collate(datas), path_to_datas)

    def _re_index_trainval(self, trainval):
        if len(trainval) == 0:
            return trainval
        train, val = trainval
        for v in val:
            v.id_scan += len(train)
        assert (train[-1].id_scan + 1 == val[0].id_scan).item(), (
            train[-1].id_scan,
            val[0].id_scan,
        )
        return train + val

    def process(self):
        if self.is_test:
            return
        raw_trainval = []
        trainval = []
        for i, split in enumerate(["train", "val", "test"]):

            # Get the patient Id from shuffled_splits
            PATIENT_ID_LIST = [
                v
                for k, v in self.shuffled_splits.items()
                if k == split
            ]
            # Flatten
            PATIENT_ID_LIST = list(
                pd.core.common.flatten(PATIENT_ID_LIST)
            )

            filenames = [
                os.path.join(self.raw_dir, f"{P_ID}_pca.ply")
                for P_ID in PATIENT_ID_LIST
            ]

            data_raw_list, data_list = self._process_filenames(
                sorted(filenames)
            )

            if split == "train" or split == "val":
                if len(data_raw_list) > 0:
                    raw_trainval.append(data_raw_list)
                trainval.append(data_list)

            self._save_data_list(data_list, self.processed_paths[i])
            self._save_data_list(
                data_raw_list,
                self.processed_raw_paths[i],
                save_bool=len(data_raw_list) > 0,
            )

        self._save_data_list(
            self._re_index_trainval(trainval), self.processed_paths[3]
        )
        self._save_data_list(
            self._re_index_trainval(raw_trainval),
            self.processed_raw_paths[3],
            save_bool=len(raw_trainval) > 0,
        )


class AneurysmDataset(BaseDataset):
    """ Wrapper around Aneurysm that creates train and test datasets.

    Parameters
    ----------
    dataset_opt: omegaconf.DictConfig
        Config dictionary that should contain

            - dataroot
            - pre_transforms
            - train_transforms
            - test_transforms
            - val_transforms
            - shuffled_splits : dict for train, test, val, trainval
            - raw_file_identifiers : List of Patient IDs for Filenames
            - custom_features_dict : dict for including features
    """

    FORWARD_CLASS = "forward.shapenet.ForwardShapenetDataset"

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)
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
        # self.cat_to_seg = dataset_opt.get("category_to_seg", False)
        # self.cat_to_seg = cat_to_seg
        self.cat_to_seg = {"aneur": np.arange(num_parts_to_segment)}

        self.train_dataset = Aneurysm(
            self._data_path,
            raw_file_identifiers=raw_file_identifiers,
            custom_features_dict=custom_features_dict,
            num_parts_to_segment=num_parts_to_segment,
            shuffled_splits=shuffled_splits,
            split="train",
            pre_transform=self.pre_transform,
            transform=self.train_transform,
            is_test=is_test,
        )

        self.val_dataset = Aneurysm(
            self._data_path,
            raw_file_identifiers=raw_file_identifiers,
            custom_features_dict=custom_features_dict,
            num_parts_to_segment=num_parts_to_segment,
            shuffled_splits=shuffled_splits,
            split="val",
            pre_transform=self.pre_transform,
            transform=self.val_transform,
            is_test=is_test,
        )

        self.test_dataset = Aneurysm(
            self._data_path,
            raw_file_identifiers=raw_file_identifiers,
            custom_features_dict=custom_features_dict,
            num_parts_to_segment=num_parts_to_segment,
            shuffled_splits=shuffled_splits,
            split="test",
            transform=self.test_transform,
            pre_transform=self.pre_transform,
            is_test=is_test,
        )
        # self._categories = self.train_dataset.categories

    @property  # type: ignore
    @save_used_properties
    def class_to_segments(self):
        classes_to_segment = {}
        classes_to_segment["aneur"] = self.cat_to_seg["aneur"]
        return classes_to_segment

    # @property
    # def is_hierarchical(self):
    #     return len(self._categories) > 1

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        return ShapenetPartTracker(
            self, wandb_log=wandb_log, use_tensorboard=tensorboard_log
        )

