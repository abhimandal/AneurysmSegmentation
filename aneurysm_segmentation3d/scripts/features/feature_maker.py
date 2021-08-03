# Essential imports
import os, sys

sys.path.append(os.getcwd())
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Mesh Imports
import trimesh
import open3d as o3d
import pymeshlab as pyml
import pclpy
from pclpy import pcl
from plyfile import PlyData, PlyElement, PlyProperty, PlyListProperty

# ML
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree


# Define CONSTANTS
os.chdir("D:\\Workspace\\Python\\Thesis Data\\")
INPUT_PATH = os.getcwd() + "\\Save data\\Processed_data"
OUTPUT_TEMP_PATH = (
    os.getcwd() + "\\Save data\\Processed_data\\Output_temp"
)

INPUT_TEMP_PATH = (
    os.getcwd() + "\\Save data\\Processed_data\\Output_temp"
)
OUTPUT_PATH = os.getcwd() + "\\Save data\\Processed_data\\Output"
OUTPUT_PCA_PATH = (
    os.getcwd() + "\\Save data\\Processed_data\\Output\\PCA"
)


ORIGINAL_FILENAME = "_PLY0.ply"
WSS_FILENAME = "_WSS.csv"
WSS_DOWN_FILENME = "_WSS_down.csv"
CURV_FILENAME = "_curv_down.csv"

PYMESH_FILENAME = "_pymesh.ply"
DESC_1_FILENAME = "_fpfh.ply"
DESC_2_FILENAME = "_shot.ply"

RADIUS_SEARCH = 0.00024
RADIUS = RADIUS_SEARCH * 0.6

OUTPUT_FILENAME = "_output.ply"
PCA_FILENAME = "_pca.ply"

# For Downsampling - Gives around 15.2k Points
TARGET_FACE_COUNT = 30000

FPFH_N_COMPONENTS = 2
SHOT_N_COMPONENTS = 3

# patient_id_list_new = "2_BC	3_BC	4_BC	5_BM	6_BM	7_BP	8_BP	9_KBW	10_SUM	11_DHM	12_GAW	13_PMM	14_TR	15_TR	16_TR	18_EM	19_EM	20_EM	21_FA	22_FA	23_HJ	24_HJ	25_HM	26_HM	27_HM	28_JM	29_JM	30_JM	31_JM	32_JM	33_KBB	34_KBB	35_KBB	36_KBB	37_KBB	38_KBB	39_KBB	40_KBB	41_KBB	42_KBB	43_KBB	44_AC	45_AC	46_LE	47_LE	48_LE	50_LE	51_LE	52_LE	54_MR	55_MR	56_WA	57_SF	58_SI	59_SI"
# patient_id_list_new = patient_id_list_new.split()
patient_id_list_new = ["49_LE", "53_LS"]


# PIPELINE
for PATIENT_ID in patient_id_list_new:
    downsample_mesh(PATIENT_ID)
    calculate_downsampled_WSS(PATIENT_ID)
    calculate_curvature_values(PATIENT_ID)
    calculate_local_descriptors(PATIENT_ID)


############################################
############  Downsample PLY    ############
############################################


def downsample_mesh(PATIENT_ID):
    print(f"\nProcessing File at:")
    print(
        os.path.join(INPUT_PATH, f"{PATIENT_ID}{ORIGINAL_FILENAME}")
    )

    ms = pyml.MeshSet()

    ms.load_new_mesh(
        os.path.join(INPUT_PATH, f"{PATIENT_ID}{ORIGINAL_FILENAME}")
    )
    ms.apply_filter("repair_non_manifold_edges_by_removing_faces")
    ms.apply_filter(
        "simplification_quadric_edge_collapse_decimation",
        targetfacenum=TARGET_FACE_COUNT,
        preserveboundary=True,
        preservenormal=True,
        preservetopology=True,
    )

    # ms.save_current_mesh(os.path.join(path,"bm_pymesh.ply"),format = 'ascii' )
    ms.save_current_mesh(
        os.path.join(
            OUTPUT_TEMP_PATH, f"{PATIENT_ID}_binary{PYMESH_FILENAME}"
        )
    )

    # Convert BINARY to ASCII
    plydata = PlyData.read(
        os.path.join(
            OUTPUT_TEMP_PATH, f"{PATIENT_ID}_binary{PYMESH_FILENAME}"
        ),
        mmap=False,
    )

    # Convert float64 to float32 as PCLPY features dont work on f8
    plydata.elements[0].properties = (
        PlyProperty("x", "f4"),
        PlyProperty("y", "f4"),
        PlyProperty("z", "f4"),
    )

    pl_wr = PlyData(
        plydata.elements,
        text=True,
        comments=["PLYFILE BINARY TO ASCII CONVERTED"],
    )
    pl_wr.write(
        os.path.join(
            OUTPUT_TEMP_PATH, f"{PATIENT_ID}{PYMESH_FILENAME}"
        )
    )

    os.remove(
        os.path.join(
            OUTPUT_TEMP_PATH, f"{PATIENT_ID}_binary{PYMESH_FILENAME}"
        )
    )


############################################
############    Runtime calc    ############
############################################


def get_total_runtime(start, end, return_time=False):
    temp = end - start
    print("Total seconds = %d" % temp)
    hours = temp // 3600
    temp = temp - 3600 * hours
    minutes = temp // 60
    seconds = temp - 60 * minutes

    if return_time:
        return hours, minutes, seconds
    else:
        print(
            "Current file finished in %d hrs %d mins %d secs"
            % (hours, minutes, seconds)
        )


############################################
############  Downsampled WSS   ############
############################################
def idx_to_WSS(df_org, row, kd_idxs):
    # Get list of nearest original indexes for the current downsampled index
    idxs = kd_idxs[int(row["index"])]

    # calculate median from the
    WSS_down = np.median(df_org.loc[idxs].WSS)
    return WSS_down


def calculate_downsampled_WSS(PATIENT_ID):
    print(f"\nProcessing File at:")
    print(
        os.path.join(INPUT_PATH, f"{PATIENT_ID}{ORIGINAL_FILENAME}")
    )

    start = time.time()

    # Read Original and Downsampled point cloud
    pc_org = o3d.io.read_point_cloud(
        os.path.join(INPUT_PATH, f"{PATIENT_ID}{ORIGINAL_FILENAME}")
    )
    pc_down = o3d.io.read_point_cloud(
        os.path.join(
            OUTPUT_TEMP_PATH, f"{PATIENT_ID}{PYMESH_FILENAME}"
        )
    )

    # Read original file with WSS
    df_wss = pd.read_csv(
        os.path.join(INPUT_PATH, f"{PATIENT_ID}{WSS_FILENAME}")
    )
    # Round the points
    df_wss = np.around(df_wss, 8).copy()

    df_down = pd.DataFrame(
        np.around(np.array(pc_down.points), decimals=8),
        columns=df_wss.columns.values[:3],
    )
    df_down = df_down.reset_index()

    # Build Tree
    tree = cKDTree(np.around(np.array(pc_org.points), decimals=8))
    search_pt = np.around(np.array(pc_down.points), decimals=8)
    kd_idxs = tree.query_ball_point(search_pt, RADIUS_SEARCH)
    # Get downsampled WSS
    df_down["WSS"] = df_down.apply(
        lambda row: idx_to_WSS(df_wss, row, kd_idxs), axis=1
    )

    # drop index column
    df_down = df_down.drop(["index"], 1)

    end = time.time()
    get_total_runtime(start, end)

    # Round up the file to clean it.
    df_down = np.around(df_down, 8)
    # Save the csv file
    df_down.to_csv(
        os.path.join(
            OUTPUT_TEMP_PATH, f"{PATIENT_ID}{WSS_DOWN_FILENME}"
        )
    )


############################################
############  Curvature Calc    ############
############################################


def calculate_curvature_values(PATIENT_ID):
    start = time.time()
    print(f"\nProcessing File at:")
    print(
        os.path.join(
            OUTPUT_TEMP_PATH, f"{PATIENT_ID}{PYMESH_FILENAME}"
        )
    )

    # Load Mesh with Trimesh
    # NOTE: The order of the points change compared to the original file
    # In order to concatenate files correctly, original point order needs to be loaded
    # and the curvature values should be added to that
    tr_mesh = trimesh.load(
        os.path.join(
            OUTPUT_TEMP_PATH, f"{PATIENT_ID}{PYMESH_FILENAME}"
        )
    )

    # Load downsampled mesh with original point order
    pc_down = o3d.io.read_point_cloud(
        os.path.join(
            OUTPUT_TEMP_PATH, f"{PATIENT_ID}{PYMESH_FILENAME}"
        )
    )

    # Make a df with correct order of points - to be used for concatenation of all the features
    curv_df = pd.DataFrame(
        np.array(pc_down.points), columns=["x", "y", "z"]
    )
    curv_df = np.around(curv_df, decimals=8)

    # Make df to store the calculated curvature values from trimesh against the vertex
    tri_curv_df = pd.DataFrame(
        tr_mesh.vertices, columns=["x", "y", "z"]
    )

    # Compute Curvatures
    mean_curv = trimesh.curvature.discrete_mean_curvature_measure(
        mesh=tr_mesh, points=tr_mesh.vertices, radius=RADIUS_SEARCH
    )
    gauss_curv = trimesh.curvature.discrete_gaussian_curvature_measure(
        mesh=tr_mesh, points=tr_mesh.vertices, radius=RADIUS_SEARCH
    )

    # Assign curvature values for
    tri_curv_df["mean_curv"] = mean_curv
    tri_curv_df["gauss_curv"] = gauss_curv

    # Round it for better merging
    tri_curv_df = np.around(tri_curv_df, decimals=8)

    # Merge the curv values with ordered pointset
    curv_df = curv_df.merge(tri_curv_df, on=["x", "y", "z"])

    # Save the file with mean and gauss curv vales
    curv_df.to_csv(
        os.path.join(
            OUTPUT_TEMP_PATH, f"{PATIENT_ID}{CURV_FILENAME}"
        ),
        index=False,
    )

    end = time.time()

    get_total_runtime(start, end)


############################################
############  Local Descriptors ############
############################################


def calculate_local_descriptors(PATIENT_ID):
    fullstart = time.time()
    print(f"\nProcessing File at:")
    print(
        os.path.join(
            OUTPUT_TEMP_PATH, f"{PATIENT_ID}{PYMESH_FILENAME}"
        )
    )

    start = time.time()
    print("--------------> Starting Local descriptor computation")

    pcl_pc_obj = pclpy.pcl.PointCloud.PointXYZ()

    # Store in pcl_pc_obj
    pc = pcl.io.loadPLYFile(
        file_name=os.path.join(
            OUTPUT_TEMP_PATH, f"{PATIENT_ID}{PYMESH_FILENAME}"
        ),
        cloud=pcl_pc_obj,
    )

    data_normals = pcl_pc_obj.compute_normals(
        radius=RADIUS, num_threads=8
    )

    # FPFH
    fpfh = (
        pcl.features.FPFHEstimation.PointXYZ_Normal_FPFHSignature33()
    )
    fpfh.setInputCloud(cloud=pcl_pc_obj)
    fpfh.setInputNormals(data_normals)
    fpfh.setRadiusSearch(RADIUS_SEARCH)

    fpfh_desc = pcl.PointCloud.FPFHSignature33()
    fpfh.compute(fpfh_desc)

    # SHOT
    shot = (
        pcl.features.SHOTEstimation.PointXYZ_Normal_SHOT352_ReferenceFrame()
    )
    shot.setInputCloud(cloud=pcl_pc_obj)
    shot.setInputNormals(data_normals)
    shot.setRadiusSearch(RADIUS_SEARCH)

    shot_desc = pcl.PointCloud.SHOT352()
    shot.compute(shot_desc)

    # Save files
    pcl.io.savePLYFileASCII(
        file_name=os.path.join(
            OUTPUT_TEMP_PATH, f"{PATIENT_ID}{DESC_1_FILENAME}"
        ),
        cloud=fpfh_desc,
    )
    pcl.io.savePLYFileASCII(
        file_name=os.path.join(
            OUTPUT_TEMP_PATH, f"{PATIENT_ID}{DESC_2_FILENAME}"
        ),
        cloud=shot_desc,
    )

    end = time.time()

    # Compute total time
    hours, minutes, seconds = get_total_runtime(
        start, end, return_time=True
    )
    print(
        f"Finished computing and saving Local descriptors in {hours} hrs {minutes} mins {round(seconds)} secs"
    )

    ############-------------------#############
    ############ Concatenate Files #############
    ############-------------------#############
    # https://github.com/dranjan/python-plyfile/issues/26

    start = time.time()
    print("--------------> Starting File Concatenation")

    # Load the original downsampled mesh
    meshply = PlyData.read(
        os.path.join(
            OUTPUT_TEMP_PATH, f"{PATIENT_ID}{PYMESH_FILENAME}"
        )
    )

    # Load curvature dataframe
    curv_df = pd.read_csv(
        os.path.join(OUTPUT_TEMP_PATH, f"{PATIENT_ID}{CURV_FILENAME}")
    )

    # Load WSS dataframe - downsampled
    wss_df = pd.read_csv(
        os.path.join(
            OUTPUT_TEMP_PATH, f"{PATIENT_ID}{WSS_DOWN_FILENME}"
        )
    )

    # Divide the elements in different variables for concat
    v = meshply.elements[0]
    f = meshply.elements[1]

    vert, f = get_concatenated_properties(
        v,
        f,
        wss_df,
        curv_df,
        fpfh_desc.histogram,
        shot_desc.descriptor,
        shot_desc.rf,
    )

    # Recreate the PlyData instance with added features
    meshply = PlyData([vert, f], text=True)

    # Save the Mesh file
    meshply.write(
        (os.path.join(OUTPUT_PATH, f"{PATIENT_ID}{OUTPUT_FILENAME}"))
    )

    (
        fpfh_desc_pca,
        shot_desc_main_pca,
        shot_desc_rf_pca,
    ) = compute_PCA_transform(
        fpfh_desc.histogram,
        shot_desc.descriptor,
        shot_desc.rf,
        FPFH_N_COMPONENTS,
        SHOT_N_COMPONENTS,
    )

    vert, f = get_concatenated_properties(
        v,
        f,
        wss_df,
        curv_df,
        fpfh_desc_pca,
        shot_desc_main_pca,
        shot_desc_rf_pca,
    )

    # Recreate the PlyData instance with added features
    meshply = PlyData([vert, f], text=True)

    # Save the Mesh file
    meshply.write(
        (os.path.join(OUTPUT_PCA_PATH, f"{PATIENT_ID}{PCA_FILENAME}"))
    )

    end = time.time()

    # Compute total time
    hours, minutes, seconds = get_total_runtime(
        start, end, return_time=True
    )
    print(
        f"Finished concatenating files in {hours} hrs {minutes} mins {round(seconds)} secs"
    )

    fullend = time.time()
    hours, minutes, seconds = get_total_runtime(
        fullstart, fullend, return_time=True
    )
    print(
        f"Total time taken for PATIENT_ID {PATIENT_ID}: {hours} hrs {minutes} mins {round(seconds)} secs"
    )
    print(f"\n {'-'*100}")


############################################
############ PCA Transformation ############
############################################


def compute_PCA_transform(
    fpfh_desc,
    shot_desc_main,
    shot_desc_rf,
    fpfh_N_COMPONENTS,
    shot_N_components,
):

    fp = np.nan_to_num(fpfh_desc)
    sh_main = np.nan_to_num(shot_desc_main)
    sh_rf = np.nan_to_num(shot_desc_rf)

    # Instantiate PCA for FPFH
    pca = PCA(n_components=fpfh_N_COMPONENTS)
    pca.fit(fp)
    fpfh = pca.transform(fp)

    # Instantiate PCA for SHOT
    pca = PCA(n_components=shot_N_components)
    pca.fit(sh_main)
    shot_main = pca.transform(sh_main)

    pca.fit(sh_rf)
    shot_rf = pca.transform(sh_rf)

    return fpfh, shot_main, shot_rf


############################################
############  Concatenate Props ############
############################################


def get_concatenated_properties(
    vertex,
    face,
    wss_df,
    curv_df,
    fpfh_desc,
    shot_desc_main,
    shot_desc_rf,
):

    # Create the new vertex data with a new name with required dtype
    a = np.empty(
        len(vertex.data),
        vertex.data.dtype.descr
        + [("WSS", "f4")]
        + [("mean_curv", "f4")]
        + [("gauss_curv", "f4")]
        + [("fpfh", "|O")]
        + [("shot", "|O")]
        + [("rf", "|O")],
    )

    for name in vertex.data.dtype.fields:
        # re-assign
        a[name] = vertex[name]
    a["WSS"] = wss_df["WSS"].values
    a["mean_curv"] = curv_df["mean_curv"].values
    a["gauss_curv"] = curv_df["gauss_curv"].values

    # Should be list of list for saving using PlyListProperty
    a["fpfh"] = fpfh_desc.tolist()
    a["shot"] = shot_desc_main.tolist()
    a["rf"] = shot_desc_rf.tolist()

    # Recreate the PlyElement instance
    vert = PlyElement.describe(a, "vertex")

    # Redefine properties - to save correct file format
    vert.properties = vertex.properties + (
        PlyProperty("WSS", "f4"),
        PlyProperty("mean_curv", "f4"),
        PlyProperty("gauss_curv", "f4"),
        PlyListProperty("fpfh", "uint", "float"),
        PlyListProperty("shot", "uint", "float"),
        PlyListProperty("rf", "uint", "float"),
    )

    return vert, face
