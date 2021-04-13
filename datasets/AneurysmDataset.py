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


# def represents_int(s):
#     """ if string s represents an int. """
#     try:
#         int(s)
#         return True
#     except ValueError:
#         return False


# def read_label_mapping(filename, label_from="raw_category", label_to="nyu40id"):
#     assert os.path.isfile(filename)
#     mapping = dict()
#     with open(filename) as csvfile:
#         reader = csv.DictReader(csvfile, delimiter="\t")
#         for row in reader:
#             mapping[row[label_from]] = int(row[label_to])
#     if represents_int(list(mapping.keys())[0]):
#         mapping = {int(k): v for k, v in mapping.items()}
#     return mapping

NUM_OF_PARTS = 5
# TODO: Add correct shuffled splits
SHUFFLED_SPLITS = {
    "train": ["4_BC", "5_BP"],
    "val": ["2_BC", "9_BP"],
    "test": ["13_CM", "18_BC"],
}


def scale_data(col: pd.Series, scaler) -> pd.Series:
    """
    Scale a column
    """
    X = col.values.reshape(-1, 1).copy()
    scaled_array = scaler.fit_transform(X)

    scaled_column = pd.Series(scaled_array.tolist()).explode()

    return scaled_column


def convert_mesh_to_dataframe(meshply):
    """
    Convert mesh values into a dataframe and add feature_names
    """

    df = pd.DataFrame()
    df["x"] = pd.Series(meshply.elements[0].data["x"])
    df["y"] = pd.Series(meshply.elements[0].data["y"])
    df["z"] = pd.Series(meshply.elements[0].data["z"])

    df["mean_curv"] = pd.Series(meshply.elements[0].data["mean_curv"])
    df["gauss_curv"] = pd.Series(
        meshply.elements[0].data["gauss_curv"]
    )

    df[["fpfh_1", "fpfh_2"]] = pd.DataFrame(
        meshply.elements[0].data["fpfh"].tolist()
    )
    df[["shot_1", "shot_2", "shot_3"]] = pd.DataFrame(
        meshply.elements[0].data["shot"].tolist()
    )
    df[["rf_1", "rf_2", "rf_3"]] = pd.DataFrame(
        meshply.elements[0].data["rf"].tolist()
    )
    # WSS - Feature for Prediction
    df["WSS"] = pd.Series(meshply.elements[0].data["WSS"])

    # Process the dataframe
    min_max_scaler = MinMaxScaler()
    df["mean_curv"] = scale_data(df["mean_curv"], min_max_scaler)
    df["gauss_curv"] = scale_data(df["gauss_curv"], min_max_scaler)
    df["fpfh_1"] = scale_data(df["fpfh_1"], min_max_scaler)
    df["fpfh_2"] = scale_data(df["fpfh_2"], min_max_scaler)
    df["shot_1"] = scale_data(df["shot_1"], min_max_scaler)
    df["shot_2"] = scale_data(df["shot_2"], min_max_scaler)
    df["shot_3"] = scale_data(df["shot_3"], min_max_scaler)
    df["rf_1"] = scale_data(df["rf_1"], min_max_scaler)
    df["rf_2"] = scale_data(df["rf_2"], min_max_scaler)
    df["rf_3"] = scale_data(df["rf_3"], min_max_scaler)
    df["WSS"] = scale_data(df["WSS"], min_max_scaler)

    return df


def read_mesh_vertices(filepath):
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
        df = convert_mesh_to_dataframe(meshplydata)

        # Categorize the WSS to different parts for part-segmentation
        df["WSS"] = pd.cut(
            df["WSS"],
            bins=np.linspace(0, 1, NUM_OF_PARTS + 1),
            labels=np.arange(0, NUM_OF_PARTS),
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
        categories (string or [string], optional): The category of the CAD
            models (one or a combination of :obj:`"Airplane"`, :obj:`"Bag"`,
            :obj:`"Cap"`, :obj:`"Car"`, :obj:`"Chair"`, :obj:`"Earphone"`
            (default: :obj:`None`)
        include_normals (bool, optional): If set to :obj:`False`, will not
            include normal vectors as input features. (default: :obj:`True`)
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

    url = (
        "https://shapenet.cs.stanford.edu/media/"
        "shapenetcore_partanno_segmentation_benchmark_v0_normal.zip"
    )

    category_ids = {
        "Airplane": "02691156",
    }

    seg_classes = {"Airplane": [0, 1, 2, 3]}

    def __init__(
        self,
        root,
        categories=None,
        include_normals=False,
        split="trainval",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        is_test=False,
        raw_file_identifiers=None,
    ):
        if categories is None:
            categories = list(self.category_ids.keys())
        if isinstance(categories, str):
            categories = [categories]
        assert all(
            category in self.category_ids for category in categories
        )
        self.categories = categories
        self.is_test = is_test

        super(Aneurysm, self).__init__(
            None, transform, pre_transform, pre_filter
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

        self.data, self.slices, self.y_mask = self.load_data(
            path, include_normals
        )

        # We have perform a slighly optimzation on memory space if no pre-transform was used.
        # c.f self._process_filenames
        if os.path.exists(raw_path):
            self.raw_data, self.raw_slices, _ = self.load_data(
                raw_path, include_normals
            )
        else:
            self.get_raw_data = self.get

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
        cats = "_".join([cat[:3].lower() for cat in self.categories])
        processed_raw_paths = [
            os.path.join(
                self.processed_dir, "raw_{}_{}".format(cats, s)
            )
            for s in ["train", "val", "test", "trainval"]
        ]
        return processed_raw_paths

    @property
    def processed_file_names(self):
        cats = "_".join([cat[:3].lower() for cat in self.categories])
        return [
            os.path.join("{}_{}.pt".format(cats, split))
            for split in ["train", "val", "test", "trainval"]
        ]

    def download(self):
        if self.is_test:
            return
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        name = self.url.split("/")[-1].split(".")[0]
        os.rename(osp.join(self.root, name), self.raw_dir)

    # def get_raw_data(self, idx, **kwargs):
    # data = self.raw_data.__class__()

    # if hasattr(self.raw_data, "__num_nodes__"):
    #     data.num_nodes = self.raw_data.__num_nodes__[idx]

    # for key in self.raw_data.keys:
    #     item, slices = self.raw_data[key], self.raw_slices[key]
    #     start, end = slices[idx].item(), slices[idx + 1].item()
    #     # print(slices[idx], slices[idx + 1])
    #     if torch.is_tensor(item):
    #         s = list(repeat(slice(None), item.dim()))
    #         s[self.raw_data.__cat_dim__(key, item)] = slice(start, end)
    #     elif start + 1 == end:
    #         s = slices[start]
    #     else:
    #         s = slice(start, end)
    #     data[key] = item[s]
    # return data

    def _process_filenames(self, filepaths):
        data_raw_list = []
        data_list = []
        categories_ids = [
            self.category_ids[cat] for cat in self.categories
        ]
        cat_idx = {
            categories_ids[i]: i for i in range(len(categories_ids))
        }

        has_pre_transform = self.pre_transform is not None

        id_scan = -1
        for filepath in tq(filepaths):
            cat = filepath.split(osp.sep)[0]
            if cat not in categories_ids:
                continue
            id_scan += 1
            data = torch.from_numpy(read_mesh_vertices(filepath))
            pos = data[:, :3]
            x = data[:, 3:-1]
            y = data[:, -1].type(torch.long)
            category = (
                torch.ones(x.shape[0], dtype=torch.long)
                * cat_idx[cat]
            )
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

            # Get the patient Id from SHUFFLED_SPLITS
            PATIENT_ID_LIST = [
                v for k, v in SHUFFLED_SPLITS.items() if k == split
            ]
            # Flatten
            PATIENT_ID_LIST = list(
                pd.core.common.flatten(PATIENT_ID_LIST)
            )

            filenames = [
                os.path.join(self.raw_dir, f"{P_ID}{PCA_FILENAME}")
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

    def __repr__(self):
        return "{}({}, categories={})".format(
            self.__class__.__name__, len(self), self.categories
        )


class AneurysmDataset(BaseDataset):
    """ Wrapper around Aneurysm that creates train and test datasets.

    Parameters
    ----------
    dataset_opt: omegaconf.DictConfig
        Config dictionary that should contain

            - dataroot
            - category: List of categories or All
            - normal: bool, include normals or not
            - pre_transforms
            - train_transforms
            - test_transforms
            - val_transforms
    """

    FORWARD_CLASS = "forward.shapenet.ForwardShapenetDataset"

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)
        try:
            self._category = dataset_opt.category
            is_test = dataset_opt.get("is_test", False)
        except KeyError:
            self._category = None

        self.train_dataset = Aneurysm(
            self._data_path,
            self._category,
            include_normals=dataset_opt.normal,
            split="train",
            pre_transform=self.pre_transform,
            transform=self.train_transform,
            is_test=is_test,
        )

        self.val_dataset = Aneurysm(
            self._data_path,
            self._category,
            include_normals=dataset_opt.normal,
            split="val",
            pre_transform=self.pre_transform,
            transform=self.val_transform,
            is_test=is_test,
        )

        self.test_dataset = Aneurysm(
            self._data_path,
            self._category,
            include_normals=dataset_opt.normal,
            split="test",
            transform=self.test_transform,
            pre_transform=self.pre_transform,
            is_test=is_test,
        )
        self._categories = self.train_dataset.categories

    @property  # type: ignore
    @save_used_properties
    def class_to_segments(self):
        classes_to_segment = {}
        for key in self._categories:
            classes_to_segment[key] = Aneurysm.seg_classes[key]
        return classes_to_segment

    @property
    def is_hierarchical(self):
        return len(self._categories) > 1

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

