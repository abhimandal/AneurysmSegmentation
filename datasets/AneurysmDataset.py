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

    # Catrgorize the WSS for segmentation
    df["WSS"] = pd.cut(
        df["WSS"],
        bins=np.linspace(0, 1, NUM_OF_PARTS + 1),
        labels=np.arange(0, NUM_OF_PARTS),
    )
    return df


def read_mesh_vertices(filename):
    """read XYZ and features for each vertex in numpy ndarray
    
    Example - 
    If only XYZ to be populated then for 5 points, vertices will be:
    
    vertices = array([[ 0.02699408, -0.16551971, -0.12976472],
                    [ 0.02701367, -0.16554399, -0.12981543],
                    [ 0.02698801, -0.16551463, -0.12982164],
                    [ 0.02702969, -0.16545248, -0.12979169],
                    [ 0.02706531, -0.16538525, -0.12981866]], dtype=float32)
    
    """
    assert os.path.isfile(filename)
    with open(filename, "rb") as f:
        meshplydata = PlyData.read(f)
        num_verts = meshplydata["vertex"].count
        df = convert_mesh_to_dataframe(meshplydata)

        vertices = np.empty((0, df.shape[1]), dtype=np.float32)
        # Stack all the vertices
        vertices = np.vstack(
            (vertices, df.iloc[:, :].astype(np.float32))
        )

    return vertices

