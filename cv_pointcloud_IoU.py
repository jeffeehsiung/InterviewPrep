import pandas as pd
import numpy as np
import glob
import os
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import open3d as o3d
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rest of your code

def load_ir_camera_data(file_path):
    if not isinstance(file_path, str):
        logger.error(f"Expected file_path to be a string, but got {type(file_path)}")
        logger.info(f"Contents of file_path: {file_path}")

    # Load CSV with the correct column naming
    columns = ['frame', 'sub_frame', 
               'head_rx', 'head_ry', 'head_rz', 'head_tx', 'head_ty', 'head_tz',
               'left_hand_rx', 'left_hand_ry', 'left_hand_rz', 'left_hand_tx', 'left_hand_ty', 'left_hand_tz',
               'left_leg_rx', 'left_leg_ry', 'left_leg_rz', 'left_leg_tx', 'left_leg_ty', 'left_leg_tz',
               'right_hand_rx', 'right_hand_ry', 'right_hand_rz', 'right_hand_tx', 'right_hand_ty', 'right_hand_tz',
               'right_leg_rx', 'right_leg_ry', 'right_leg_rz', 'right_leg_tx', 'right_leg_ty', 'right_leg_tz', 
               'video_frame_timestamps', 'start_date', 'start_timestamp', 'end_date', 'end_timestamp', 'video_frame_sequence_number']
    
    # Read the CSV file
    data = pd.read_csv(file_path, names=columns, skiprows=1)
    
    # Extract only the position (translation) columns and drop rows with NaN values
    position_columns = ['frame', 'sub_frame', 
        'head_tx', 'head_ty', 'head_tz',
        'left_hand_tx', 'left_hand_ty', 'left_hand_tz',
        'left_leg_tx', 'left_leg_ty', 'left_leg_tz',
        'right_hand_tx', 'right_hand_ty', 'right_hand_tz',
        'right_leg_tx', 'right_leg_ty', 'right_leg_tz', 
        'video_frame_timestamps',
        'video_frame_sequence_number'
    ]
    
    data = data[position_columns]
    # Drop rows with missing values
    data.dropna(inplace=True)
    # Calculate centroid of hands and legs
    data['hands_centroid_x'] = (data['left_hand_tx'] + data['right_hand_tx']) / 2
    data['hands_centroid_y']= (data['left_hand_ty'] + data['right_hand_ty']) / 2
    data['hands_centroid_z'] = (data['left_hand_tz'] + data['right_hand_tz']) / 2
    data['legs_centroid_x'] = (data['left_leg_tx'] + data['right_leg_tx']) / 2
    data['legs_centroid_y'] = (data['left_leg_ty'] + data['right_leg_ty']) / 2
    data['legs_centroid_z'] = (data['left_leg_tz'] + data['right_leg_tz']) / 2
    # Calculate centroid for the entire body parts
    data['body_centroid_x'] = (data['head_tx'] + data['hands_centroid_x'] + data['legs_centroid_x']) / 3
    data['body_centroid_y'] = (data['head_ty'] + data['hands_centroid_y'] + data['legs_centroid_y']) / 3
    data['body_centroid_z'] = (data['head_tz'] + data['hands_centroid_z'] + data['legs_centroid_z']) / 3
    # Calcualte the centroid between right leg and body centroid
    data['right_leg_body_centroid_x'] = (data['right_leg_tx'] + data['body_centroid_x']) / 2
    data['right_leg_body_centroid_y'] = (data['right_leg_ty'] + data['body_centroid_y']) / 2
    data['right_leg_body_centroid_z'] = (data['right_leg_tz'] + data['body_centroid_z']) / 2
    # Calcualte the centroid between left leg and body centroid
    data['left_leg_body_centroid_x'] = (data['left_leg_tx'] + data['body_centroid_x']) / 2
    data['left_leg_body_centroid_y'] = (data['left_leg_ty'] + data['body_centroid_y']) / 2
    data['left_leg_body_centroid_z'] = (data['left_leg_tz'] + data['body_centroid_z']) / 2
    # Calcualte the centroid between right hand and body centroid
    data['right_hand_body_centroid_x'] = (data['right_hand_tx'] + data['body_centroid_x']) / 2
    data['right_hand_body_centroid_y'] = (data['right_hand_ty'] + data['body_centroid_y']) / 2
    data['right_hand_body_centroid_z'] = (data['right_hand_tz'] + data['body_centroid_z']) / 2
    # Calcualte the centroid between left hand and body centroid
    data['left_hand_body_centroid_x'] = (data['left_hand_tx'] + data['body_centroid_x']) / 2
    data['left_hand_body_centroid_y'] = (data['left_hand_ty'] + data['body_centroid_y']) / 2
    data['left_hand_body_centroid_z'] = (data['left_hand_tz'] + data['body_centroid_z']) / 2
    # Calculate the centroid between head and body centroid
    data['head_body_centroid_x'] = (data['head_tx'] + data['body_centroid_x']) / 2
    data['head_body_centroid_y'] = (data['head_ty'] + data['body_centroid_y']) / 2
    data['head_body_centroid_z'] = (data['head_tz'] + data['body_centroid_z']) / 2
    # Compute the centroid between hands and body centroid
    data['hands_body_centroid_x'] = (data['hands_centroid_x'] + data['body_centroid_x']) / 2
    data['hands_body_centroid_y'] = (data['hands_centroid_y'] + data['body_centroid_y']) / 2
    data['hands_body_centroid_z'] = (data['hands_centroid_z'] + data['body_centroid_z']) / 2
    # Compute the centroid between legs and head centroid
    data['legs_head_centroid_x'] = (data['legs_centroid_x'] + data['head_tx']) / 2
    data['legs_head_centroid_y'] = (data['legs_centroid_y'] + data['head_ty']) / 2
    data['legs_head_centroid_z'] = (data['legs_centroid_z'] + data['head_tz']) / 2
    # Compute the centroid between legs and body centroid
    data['legs_body_centroid_x'] = (data['legs_centroid_x'] + data['body_centroid_x']) / 2
    data['legs_body_centroid_y'] = (data['legs_centroid_y'] + data['body_centroid_y']) / 2
    data['legs_body_centroid_z'] = (data['legs_centroid_z'] + data['body_centroid_z']) / 2
    
    return data

def load_radar_data(npz_file_path):
    radar_data = np.load(npz_file_path, allow_pickle=True)
    # print the keys in the npz file
    logger.info(f"Keys in the radar npz file: {radar_data.files}")
    video_frame_timestamps = radar_data['video_frames_timestamps']
    video_frame_sequence_numbers = radar_data['video_frame_sequence_numbers']
    video_frames = radar_data['video_frames']
    radar_targets_per_frame = radar_data['targets_per_frame']
    radar_spatial_point_cloud = radar_data['spatial_point_cloud']
    logger.info(f"spatial_point_cloud shape: {radar_spatial_point_cloud.shape}, radar_targets_per_frame shape: {radar_targets_per_frame.shape}")

    # Extract frame IDs from the radar spatial point cloud
    frame_ids = radar_spatial_point_cloud[:, 0].astype(int)
    
    # Get unique frame IDs
    unique_frame_ids = np.unique(frame_ids)
    
    # Create a mapping from frame_id to sequence number and timestamp
    frame_id_to_sequence_number = {frame_id: video_frame_sequence_numbers[frame_id] for frame_id in unique_frame_ids}
    frame_id_to_timestamp = {frame_id: video_frame_timestamps[frame_id] for frame_id in unique_frame_ids}
    
    # Map the sequence numbers and timestamps to each point using the frame_id
    video_frame_sequence_numbers_mapped = np.array([frame_id_to_sequence_number[frame_id] for frame_id in frame_ids])
    video_frame_timestamps_mapped = np.array([frame_id_to_timestamp[frame_id] for frame_id in frame_ids])
    
    radar_points = pd.DataFrame({
        'frame_id': frame_ids,
        'x': radar_spatial_point_cloud[:, 1],
        'y': radar_spatial_point_cloud[:, 2],
        'z': radar_spatial_point_cloud[:, 3],
        'vx': radar_spatial_point_cloud[:, 4],
        'vy': radar_spatial_point_cloud[:, 5],
        'vz': radar_spatial_point_cloud[:, 6],
        'video_frame_sequence_number': video_frame_sequence_numbers_mapped,
        'video_frame_timestamps': video_frame_timestamps_mapped
    })
    
    # Create dictionary to link unique video frame sequence numbers to video frames
    video_frames_dict = {}
    for i, seq_num in enumerate(video_frame_sequence_numbers):
        if seq_num not in video_frames_dict:
            video_frames_dict[seq_num] = {
                'frame': video_frames[i],
                'timestamp': video_frame_timestamps[i] if video_frame_timestamps is not None else None
            }
    
    return radar_points, video_frames_dict

def remove_outliers_and_noise(points, nb_neighbors=20, std_ratio=2.0, voxel_size=0.05):
    """
    Removes outliers and noise from a point cloud.

    Parameters:
    - points: (N, 3) numpy array of point coordinates
    - nb_neighbors: Number of neighbors to consider for statistical outlier removal
    - std_ratio: Standard deviation multiplier for statistical outlier removal
    - voxel_size: Voxel size for voxel grid filtering

    Returns:
    - filtered_points: (M, 3) numpy array of filtered point coordinates
    """
    # Create an Open3D PointCloud object
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    
    # Remove statistical outliers
    cloud, ind = cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    
    # Voxel grid filtering for noise reduction
    cloud = cloud.voxel_down_sample(voxel_size=voxel_size)
    
    # Convert back to numpy array
    filtered_points = np.asarray(cloud.points)
    
    return filtered_points

def find_common_frames_by_timestamp(ir_camera_data, radar_data, tolerance=pd.Timedelta(seconds=0.5/30.0)):
    """
    Example usage:
    common_frames = find_common_frames(radar_points, ir_camera_data)
    """
    # Ensure 'video_frame_timestamps' is of datetime type and set as the index
    ir_camera_data['video_frame_timestamps'] = pd.to_datetime(ir_camera_data['video_frame_timestamps'])
    radar_data['video_frame_timestamps'] = pd.to_datetime(radar_data['video_frame_timestamps'])

    common_frames = []
    for _, row in ir_camera_data.iterrows():
        timestamp = row['video_frame_timestamps']
        matching_radar_points = radar_data[np.abs(radar_data['video_frame_timestamps'] - timestamp) <= tolerance]
        if not matching_radar_points.empty:
            common_frames.append((row, matching_radar_points))
    return common_frames

def find_common_frames_by_vid_sequence(ir_camera_data, radar_data):
    """
    Example usage:
    common_frames = find_common_frames(radar_points, ir_camera_data)
    """
    common_frames = []
    for _, row in ir_camera_data.iterrows():
        sequence_number = row['video_frame_sequence_number']
        matching_radar_points = radar_data[radar_data['video_frame_sequence_number'] == sequence_number]
        if not matching_radar_points.empty:
            common_frames.append((row, matching_radar_points))
    return common_frames

def extract_ir_points(ir_frame):
    """
    Extracts the tx, ty, tz coordinates for head, hands, and legs from a single IR frame.
    
    Parameters:
    ir_frame (pd.Series): A single row of IR camera data with various body part coordinates.
    
    Returns:
    np.ndarray: An array where each row is a point with x, y, z coordinates.
    """
    # Convert to DataFrame for easier manipulation
    ir_frame_df = ir_frame.to_frame().T
    
    # Convert all columns to numeric, coercing errors to NaN
    ir_frame_numeric = ir_frame_df.apply(pd.to_numeric, errors='coerce')
    
    # Filter to include only numeric columns
    ir_frame_numeric = ir_frame_numeric.dropna(axis=1, how='any')
    
    # Extract the coordinates for each body part
    try:
        head = ir_frame_numeric[['head_tx', 'head_ty', 'head_tz']].values
        left_hand = ir_frame_numeric[['left_hand_tx', 'left_hand_ty', 'left_hand_tz']].values
        right_hand = ir_frame_numeric[['right_hand_tx', 'right_hand_ty', 'right_hand_tz']].values
        left_leg = ir_frame_numeric[['left_leg_tx', 'left_leg_ty', 'left_leg_tz']].values
        right_leg = ir_frame_numeric[['right_leg_tx', 'right_leg_ty', 'right_leg_tz']].values
        # centroids alogn the z-axis
        body_centroid = ir_frame_numeric[['body_centroid_x', 'body_centroid_y', 'body_centroid_z']].values
        head_body_centroid = ir_frame_numeric[['head_body_centroid_x', 'head_body_centroid_y', 'head_body_centroid_z']].values
        hands_body_centroid = ir_frame_numeric[['hands_body_centroid_x', 'hands_body_centroid_y', 'hands_body_centroid_z']].values
        legs_head_centroid = ir_frame_numeric[['legs_head_centroid_x', 'legs_head_centroid_y', 'legs_head_centroid_z']].values
        legs_body_centroid = ir_frame_numeric[['legs_body_centroid_x', 'legs_body_centroid_y', 'legs_body_centroid_z']].values

    except KeyError as e:
        logger.error(f"Missing expected columns in ir_frame: {e}")
        raise

    # Combine all the coordinates into a single array
    ir_points = np.vstack([head, left_hand, right_hand, left_leg, right_leg, body_centroid, head_body_centroid, hands_body_centroid, legs_head_centroid, legs_body_centroid])
    
    # Check for non-numeric values
    if not np.issubdtype(ir_points.dtype, np.number):
        logger.error("ir_points contains non-numeric values")
        raise ValueError("ir_points contains non-numeric values")
    
    return ir_points

def extract_radar_points(radar_frame):
    """
    Extracts and reshapes radar points from a given frame.
    
    Parameters:
    frame (tuple): A tuple containing the IR camera data row and radar points.
    
    Returns:
    np.ndarray: An array where each row is a point with x, y, z coordinates.
    """
    radar_points = radar_frame[['x', 'y', 'z']].values
    return radar_points

def normalize_points(points):
    scaler = StandardScaler()
    normalized_points = scaler.fit_transform(points)
    return normalized_points, scaler

def bin_points(points, target_size):
    if len(points) > target_size:
        indices = np.random.choice(len(points), target_size, replace=False)
        return points[indices]
    else:
        return points  # If not enough points to sample, return as is

def mean_squared_error(points1, points2):
    assert points1.shape == points2.shape, "Point clouds must have the same shape"
    return np.mean(np.sum((points1 - points2) ** 2, axis=1))

def compute_chamfer_distance(points1, points2):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2)

    dists1 = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
    dists2 = np.asarray(pcd2.compute_point_cloud_distance(pcd1))

    return np.mean(dists1) + np.mean(dists2)

def transform_points(points, rotation, translation):
    R = o3d.geometry.get_rotation_matrix_from_xyz(rotation)
    return np.dot(points, R.T) + translation

def cost_function(params, radar_points, ir_points, alpha, beta, gamma=2.0):
    """
    Cost function to be minimized with Z-axis rotation constraint.

    Parameters:
    - params: Rotation and translation parameters (6,)
    - radar_points: Radar point cloud
    - ir_points: IR point cloud
    - alpha: Weight for MSE
    - beta: Weight for Chamfer Distance
    - gamma: Weight for Z-axis rotation constraint

    Returns:
    - Combined cost value
    """
    rotation = params[:3]
    translation = params[3:6]

    # Apply the Z-axis constraint
    z_rotation = rotation[2]  # Extract Z-axis rotation
    constrained_rotation = np.array([rotation[0], rotation[1], 0.0])  # Fix Z-axis rotation

    # Transform radar points
    transformed_points = transform_points(radar_points, constrained_rotation, translation)

    # Compute MSE and Chamfer Distance
    mse = mean_squared_error(transformed_points, ir_points)
    chamfer_dist = compute_chamfer_distance(transformed_points, ir_points)

    # Combined cost with Z-axis rotation penalty
    return alpha * mse + beta * chamfer_dist + gamma * abs(z_rotation)

def iterative_refinement(radar_points, ir_points, alpha=0.5, beta=0.5, gamma=2.0):
    initial_params = np.zeros(6)
    result = minimize(cost_function, initial_params, args=(radar_points, ir_points, alpha, beta, gamma), 
                      method='BFGS', options={'maxiter': 1000})
    
    optimized_params = result.x
    optimized_rotation = optimized_params[:3]
    optimized_translation = optimized_params[3:6]
    
    # Apply Z-axis constraint if necessary
    constrained_rotation = np.array([optimized_rotation[0], optimized_rotation[1], 0.0])
    
    aligned_radar_points = transform_points(radar_points, constrained_rotation, optimized_translation)
    return aligned_radar_points, constrained_rotation, optimized_translation


def align_with_pca(ir_points, radar_points):
    # ensure that the shape of the ir_points and radar_points has columns x, y, z 
    assert np.array(ir_points).shape[1] == 3, f"Expected ir_points to have 3 columns, but got {np.array(ir_points).shape[1]}"
    assert np.array(radar_points).shape[1] == 3, f"Expected radar_points to have 3 columns, but got {np.array(radar_points).shape[1]}"
    
    # Check for non-numeric values
    if not np.issubdtype(ir_points.dtype, np.number):
        raise ValueError("ir_points contains non-numeric values")
    if not np.issubdtype(radar_points.dtype, np.number):
        raise ValueError("radar_points contains non-numeric values")
    
    # Normalize the points
    ir_points, ir_scaler = normalize_points(ir_points)
    radar_points, radar_scaler = normalize_points(radar_points)
    
    # Remove outliers
    radar_points = remove_outliers_and_noise(radar_points)
    
    # Bin the radar points to match the number of IR points
    radar_points = bin_points(radar_points, len(ir_points))
    
    # Apply PCA
    pca_ir = PCA(n_components=3)
    pca_radar = PCA(n_components=3)
    
    pca_ir.fit(ir_points)
    pca_radar.fit(radar_points)
    
    pca_ir.fit(ir_points)
    pca_radar.fit(radar_points)
    
    ir_aligned = pca_ir.transform(ir_points)
    radar_aligned = pca_radar.transform(radar_points)
    
    # Align the radar PCA axes to the IR PCA axes
    R, _ = np.linalg.qr(np.dot(radar_aligned.T, ir_aligned))
    radar_aligned = np.dot(radar_points, R.T)
    
    # Ensure the Z-axis rotation is zero
    R_z_constraint = np.eye(3)
    R_z_constraint[2, 2] = 1
    radar_aligned = np.dot(radar_points, R_z_constraint.T)

    # Inverse transform to original scale
    ir_aligned = ir_scaler.inverse_transform(ir_aligned)
    radar_aligned = radar_scaler.inverse_transform(radar_aligned)
    
    # Calculate the centroid translations
    ir_centroid = np.mean(ir_aligned, axis=0)
    radar_centroid = np.mean(radar_aligned, axis=0)
    translation = ir_centroid - radar_centroid
    
    # Apply the initial alignment
    radar_aligned = radar_aligned + translation
    
    # Perform iterative refinement
    radar_aligned, optimized_rotation, optimized_translation = iterative_refinement(radar_aligned, ir_aligned)

    return ir_aligned, radar_aligned

def compute_bounding_box(points):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    bbox = cloud.get_axis_aligned_bounding_box()
    return bbox

def compute_oriented_bounding_box(points):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    obb = cloud.get_oriented_bounding_box()
    return obb

def kabsch_algorithm_constrained(P, Q):
    """
    Kabsch algorithm to find the optimal rotation matrix that aligns P to Q,
    with a constraint that the Z-axis rotation is fixed.

    Parameters:
    - P (np.ndarray): A set of points (Nx3).
    - Q (np.ndarray): Another set of points (Nx3).
    
    Returns:
    - np.ndarray: The optimal rotation matrix (3x3).
    - np.ndarray: The translation vector (3,)
    """
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    H = P_centered.T @ Q_centered
    
    U, S, Vt = np.linalg.svd(H)
    
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
        
    # Ensure the Z-axis rotation is zero
    R[:, 2] = np.array([0, 0, 1])
    
    t = centroid_Q - R @ centroid_P
    
    return R, t

def kabsch_algorithm(P, Q):
    """
    Kabsch algorithm to find the optimal rotation matrix that aligns P to Q.
    
    Parameters:
    P (np.ndarray): A set of points (Nx3).
    Q (np.ndarray): Another set of points (Nx3).
    
    Returns:
    np.ndarray: The optimal rotation matrix (3x3).
    """
    # Step 1: Compute the centroids of P and Q
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    
    # Step 2: Center the points
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    # Step 3: Compute the covariance matrix H (Singular Value Decomposition)
    H = P_centered.T @ Q_centered
    
    # Step 4: Perform SVD on the covariance matrix H
    U, S, Vt = np.linalg.svd(H)
    
    # Step 5: Compute the optimal rotation matrix R
    R = Vt.T @ U.T

    # Ensure a right-handed coordinate system
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
        
    # Compute the translation vector t
    t = centroid_Q - R @ centroid_P
    
    return R, t

def align_bounding_boxes_constrained(obb_source, obb_target):
    source_points = np.asarray(obb_source.get_box_points())
    target_points = np.asarray(obb_target.get_box_points())
    
    R, t = kabsch_algorithm_constrained(source_points, target_points)
    
    initial_transform = np.eye(4)
    initial_transform[:3, :3] = R
    initial_transform[:3, 3] = t
    
    return initial_transform


def align_bounding_boxes(obb_source, obb_target):
    source_points = np.asarray(obb_source.get_box_points())
    target_points = np.asarray(obb_target.get_box_points())
    
    R, t = kabsch_algorithm(source_points, target_points)
    
    initial_transform = np.eye(4)
    initial_transform[:3, :3] = R
    initial_transform[:3, 3] = t
    
    return initial_transform

def refine_alignment_with_icp_constrained(P, Q, initial_transform):
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    
    source.points = o3d.utility.Vector3dVector(P)
    target.points = o3d.utility.Vector3dVector(Q)
    
    threshold_meter = 0.025
    threshold = convert_m_to_mm(threshold_meter)
    
    icp_result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
    # Copy the transformation matrix to ensure it's writable
    refined_transform = np.copy(icp_result.transformation)
    
    # Ensure Z-axis rotation is fixed
    refined_transform[0, 2] = 0
    refined_transform[1, 2] = 0
    refined_transform[2, 2] = 1
    
    return refined_transform

def refine_alignment_with_icp(P, Q, initial_transform):
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    
    source.points = o3d.utility.Vector3dVector(P)
    target.points = o3d.utility.Vector3dVector(Q)
    
    threshold_meter = 0.025 # Threshold in meters
    threshold = convert_m_to_mm(threshold_meter)
    icp_result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
    return icp_result.transformation

def filter_frames_by_z_distribution(radar_points_list, min_z_distribution=1.5):
    filtered_frames = []
    for points in radar_points_list:
        z_range = np.max(points[:, 2]) - np.min(points[:, 2])
        if z_range >= min_z_distribution:
            filtered_frames.append(points)
        else:
            filtered_frames.append(None)
    return filtered_frames

def ransac_icp_constrained(P, Q, initial_transform, num_iterations=100, threshold=0.02):
    best_transform = initial_transform
    best_inliers = -1

    if len(P) > len(Q):
        P = bin_points(P, len(Q))
    
    num_samples = min(len(P), len(Q), 4)
    
    for _ in range(num_iterations):
        P_indices = np.random.choice(len(P), size=num_samples, replace=False)
        Q_indices = np.random.choice(len(Q), size=num_samples, replace=False)
        
        P_sample = P[P_indices]
        Q_sample = Q[Q_indices]
        
        R, t = kabsch_algorithm_constrained(P_sample, Q_sample)
        sample_transform = np.eye(4)
        sample_transform[:3, :3] = R
        sample_transform[:3, 3] = t
        
        refined_transform = refine_alignment_with_icp_constrained(P, Q, sample_transform)
        
        P_transformed = (refined_transform[:3, :3] @ P.T).T + refined_transform[:3, 3]

        tree = cKDTree(P_transformed)
        distances, _ = tree.query(Q)
        
        inliers = np.sum(distances < threshold)
        
        if inliers > best_inliers:
            best_inliers = inliers
            best_transform = refined_transform
    
    return best_transform

def ransac_icp(P, Q, initial_transform, num_iterations=100, threshold=0.02):
    best_transform = initial_transform
    best_inliers = -1

    # Bin the radar points to match the number of IR points if necessary
    if len(P) > len(Q):
        P = bin_points(P, len(Q))
    
    num_samples = min(len(P), len(Q), 4)  # Ensure we don't sample more points than available
    
    for _ in range(num_iterations):
        P_indices = np.random.choice(len(P), size=num_samples, replace=False)
        Q_indices = np.random.choice(len(Q), size=num_samples, replace=False)
        
        P_sample = P[P_indices]
        Q_sample = Q[Q_indices]
        
        R, t = kabsch_algorithm(P_sample, Q_sample)
        sample_transform = np.eye(4)
        sample_transform[:3, :3] = R
        sample_transform[:3, 3] = t
        
        refined_transform = refine_alignment_with_icp(P, Q, sample_transform)
        
        # Assuming refined_transform is already defined and P is the radar points
        P_transformed = (refined_transform[:3, :3] @ P.T).T + refined_transform[:3, 3]

        # Build a KD-tree with the transformed radar points
        tree = cKDTree(P_transformed)
        # Query the KD-tree to find the nearest neighbor for each IR point in Q
        distances, _ = tree.query(Q)
        
        inliers = np.sum(distances < threshold)
        
        if inliers > best_inliers:
            best_inliers = inliers
            best_transform = refined_transform
    
    return best_transform

def compute_iou(bbox1, bbox2, scaling_factor=1.8, return_percentage=False):
    """
    Compute the Intersection over Union (IoU) between two bounding boxes, 
    adjusting for different volumes using a scaling factor.

    Parameters:
    - bbox1: o3d.geometry.AxisAlignedBoundingBox for the IR camera.
    - bbox2: o3d.geometry.AxisAlignedBoundingBox for the radar point cloud.
    - scaling_factor: Factor to scale bbox2's volume to match bbox1's relative scale.
    - return_percentage: If True, return IoU as a percentage.

    Returns:
    - IoU: Intersection over Union value (percentage if return_percentage is True).
    
    To estimate a scaling factor:

    Measure Individual Body Parts:
    Forehead to Top of Head: Typically about 10 cm (this is a rough estimate).
    From Bottom of Feet to Ankle: Typically around 15-20 cm for an average adult (though this can vary widely).
    
    Estimate the Bounding Box Sizes:
    - IR Camera Bounding Box: Suppose the average volume captured by the IR camera (from head-top - forehead = 10 cm , palm length = 16-20cm, to ankles to botton fo the feet = 15 cm) is around (1.8 - 0.1 - 0.2)) m * 0.6 m * 0.8 m = 0.72 m³.
    - Radar Bounding Box: This captures the whole body, which might be around ( 1.8 m * 0.6 m * 1.2 m ) =  1.296 m³. depending on the size of the person.
    
    Determine the Scaling Factor:
    To match the IR camera's volume to the radar's, you can calculate the scaling factor as the ratio of the radar volume to the IR volume.
    For instance: scaling_factor = 1.296 / 0.72 = 1.8
    """
    bbox1_min = bbox1.get_min_bound()
    bbox1_max = bbox1.get_max_bound()
    bbox2_min = bbox2.get_min_bound()
    bbox2_max = bbox2.get_max_bound()
    
    intersection_min = np.maximum(bbox1_min, bbox2_min)
    intersection_max = np.minimum(bbox1_max, bbox2_max)
    
    # Check if there is an intersection
    if np.any(intersection_min >= intersection_max):
        return 0.0 if not return_percentage else 0.0
    
    intersection_bbox = o3d.geometry.AxisAlignedBoundingBox(intersection_min, intersection_max)
    intersection_vol = intersection_bbox.volume()
    
    bbox1_vol = bbox1.volume()
    bbox2_vol = bbox2.volume() * scaling_factor  # Scale the volume of bbox2
    
    # Scale the intersection volume to match the scaling factor of bbox2
    intersection_vol_scaled = intersection_vol * scaling_factor
    
    union_vol = bbox1_vol + bbox2_vol - intersection_vol_scaled
    
    iou = intersection_vol_scaled / union_vol if union_vol > 0 else 0.0
    
    return iou * 100 if return_percentage else iou

def convert_m_to_mm(values):
    return values * 1000

def visualize_point_clouds(ir_points, radar_points, radar_aligned, frame_number, output_dir):
    fig = plt.figure(figsize=(20, 5))
    
    # Before Alignment
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(ir_points[:, 0], ir_points[:, 1], ir_points[:, 2], c='r', marker='^', label='IR Camera Points')
    ax1.scatter(radar_points[:, 0], radar_points[:, 1], radar_points[:, 2], c='b', marker='o', label='Radar Points')
    ax1.set_title(f'Before Alignment\nFrame {frame_number}')
    ax1.legend()
    
    # After Alignment
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(ir_points[:, 0], ir_points[:, 1], ir_points[:, 2], c='r', marker='^', label='IR Camera Points')
    ax2.scatter(radar_aligned[:, 0], radar_aligned[:, 1], radar_aligned[:, 2], c='g', marker='o', label='Radar Aligned Points')
    ax2.set_title(f'After Alignment\nFrame {frame_number}')
    ax2.legend()
    
    # Combined View
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(ir_points[:, 0], ir_points[:, 1], ir_points[:, 2], c='r', marker='^', label='IR Camera Points')
    ax3.scatter(radar_points[:, 0], radar_points[:, 1], radar_points[:, 2], c='b', marker='o', label='Radar Points')
    ax3.scatter(radar_aligned[:, 0], radar_aligned[:, 1], radar_aligned[:, 2], c='g', marker='o', label='Radar Aligned Points')
    ax3.set_title(f'Combined View\nFrame {frame_number}')
    ax3.legend()
    
    # if the output directory does not exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # save the figure to folder output_dir with the frame number
    plt.savefig(os.path.join(output_dir, f"frame_{frame_number}.png"))
    # close the figure
    plt.close(fig)

def main():
    """
    Pipeline:
    1. Load the two system coordinates (ir camera and radar) data frames
        - extract point clouds from corresponding Frames: 
            - Identify and extract the point clouds from the same frame (or time instance) for both the IR camera and radar recordings.
        - extract Key Points from IR Camera:
            - for each frame, extract the five key points (head, wrists, ankles) from the IR camera point cloud.
    2. Find common frames based on timestamps within a tolerance
    3. Extract Corresponding Points
        - Extract the key points from the IR camera data and the radar point cloud for the common frames
    4. Align the Point Clouds
        - Principal Component Analysis (PCA) for Initial Rough Alignment: Align the principal axes of the point clouds.
        - Kabsch Algorithm: Compute Bounding Boxes and Oriented Bounding Boxes (OBBs) for the point clouds and align them using the Kabsch algorithm.
        - RANSAC + ICP: Refine alignment and make it robust to outliers. ICP algorithm to minimize the distance between points in the two point clouds
    5. Evaluating Alignment: Compute IoU between the aligned radar bounding box and the IR camera bounding box.

    Next Steps
    1. Test the Pipeline: Ensure the pipeline runs without errors and prints IoU values.
    2. Visualization: Optionally, visualize the aligned point clouds to inspect the alignment quality.
    3. Parameter Tuning: Adjust parameters like num_iterations and threshold for RANSAC and ICP based on your data characteristics.
    """
    # Load the two system coordinates (ir camera and radar) data frames
    ir_cam_npz_path = "/Users/jeffeehsiung/Desktop/surveillance_holst"
    radar_npz_path = "/Users/jeffeehsiung/Desktop/surveillance_holst"
    radar_files = glob.glob(os.path.join(radar_npz_path, "video_and_pointcloud_capture*_p*.npz"))
    # process each corresponding radar and ir camera npz files based on the same capture number
    for radar_file in radar_files:
        capture_number = None
        person_number = None
        session_number = None
        try:
            # pattern: video_and_pointcloud_capture{capture_number}_p{person_number}.npz
            logger.info(f"Processing file: {radar_file}")
            match = re.search(r'video_and_pointcloud_capture(\d+)_p(\d+)\.npz', radar_file)
            if not match:
                logger.error(f"Invalid file name format: {radar_file}")
                continue

            capture_number_str, person_number_str = match.groups()

            # Log the extracted parts
            logger.info(f"Extracted capture_number_str: {capture_number_str}")
            logger.info(f"Extracted person_number_str: {person_number_str}")

            # Validate that the extracted parts are digits
            if not (capture_number_str.isdigit() and person_number_str.isdigit()):
                logger.error(f"Invalid file name format: {radar_file}")
                continue

            capture_number = int(capture_number_str)
            session_number = capture_number + 1
            person_number = int(person_number_str)

            logger.info(f"Capture number: {capture_number}, Session number: {session_number}, Person number: {person_number}")
        except IndexError as e:
            logger.error(f"Error processing file {radar_file}: {e}")
        except ValueError as e:
            logger.error(f"Error converting to integer for file {radar_file}: {e}")
        
        if capture_number is None or person_number is None or session_number is None:
            logger.error(f"Skipping file {radar_file}")
            continue
        # pattern: P{person_number}_S{session_number}.csv
        ir_cam_filename = f"P{person_number}_S{session_number}.csv"
        ir_cam_file_path = glob.glob(os.path.join(ir_cam_npz_path, ir_cam_filename))

        # Log the IR camera file path
        logger.info(f"IR camera file path: {ir_cam_file_path}")
        # Step 1: Load and Pre-process the Data
        if ir_cam_file_path:
            # file folder inlcudes person number, capture number, and session number
            output_dir = os.path.join(ir_cam_npz_path, f"person_{person_number}_capture_{capture_number}_session_{session_number}")
            # Load radar data
            radar_data, video_frames_dict = load_radar_data(radar_file)          
            
            # Convert radar data to millimeters
            radar_data[['x', 'y', 'z']] = convert_m_to_mm(radar_data[['x', 'y', 'z']])
            
            # Load ir camera data
            ir_camera_data = load_ir_camera_data(ir_cam_file_path[0])
            
            # Step 2: Find the Common Frames Based on Timestamps
            frame_rate = 30.0  # Frame rate of the ir_camera
            tolerance = pd.Timedelta(seconds=0.5 / frame_rate)  # Tolerance for matching timestamps (tolerance in seconds)
            common_frames = find_common_frames_by_timestamp(ir_camera_data, radar_data, tolerance=tolerance)
            # common_frames = find_common_frames_by_vid_sequence(ir_camera_data, radar_data)
            logger.info(f"Found {len(common_frames)} common frames for capture {capture_number}")
            
            if not common_frames:
                logger.info(f"No common frames found for capture {capture_number}")
                continue
            
            # Step 3: Extract Corresponding Points
            ir_points_list = [extract_ir_points(frame[0]) for frame in common_frames]
            # # Extract and reshape radar points for each common frame
            radar_points_list = [remove_outliers_and_noise(extract_radar_points(frame[1])) for frame in common_frames]
            
            # Filter frames based on z-distribution
            min_z_distribution_meter = 1.7
            min_z_distribution = convert_m_to_mm(min_z_distribution_meter)
            filtered_ir_points_list = []
            filtered_radar_points_list = []
            for i, radar_points in enumerate(radar_points_list):
                if radar_points is not None and (np.max(radar_points[:, 2]) - np.min(radar_points[:, 2])) >= min_z_distribution:
                    filtered_ir_points_list.append(ir_points_list[i])
                    filtered_radar_points_list.append(radar_points)
            
            logger.info(f"Filtered {len(filtered_ir_points_list)} frames based on z-distribution for capture {capture_number}")
            
            # Dictionary to store IoU values
            iou_results = {}
            valid_ious = []
            # Step 4: Align the Point Clouds
            for i, (ir_points, radar_points) in enumerate(zip(filtered_ir_points_list, filtered_radar_points_list)):
                # Initial alignment using PCA
                ir_pca, radar_pca = align_with_pca(ir_points, radar_points)
                
                # Compute oriented bounding boxes
                obb_ir = compute_oriented_bounding_box(ir_pca)
                obb_radar = compute_oriented_bounding_box(radar_pca)
                
                # Align bounding boxes using Kabsch algorithm
                initial_transform = align_bounding_boxes_constrained(obb_ir, obb_radar)
                
                # Refine alignment using RANSAC and ICP
                threshold_meter = 0.05
                threshold = convert_m_to_mm(threshold_meter)
                best_transform = ransac_icp_constrained(radar_points, ir_points, initial_transform, num_iterations=1000, threshold=threshold)
                
                # Apply the transformation to radar points
                radar_aligned = (best_transform[:3, :3] @ radar_points.T).T + best_transform[:3, 3]
                
                # Compute bounding boxes after alignment
                obb_radar_aligned = compute_oriented_bounding_box(radar_aligned)
                
                # Compute IoU
                iou = compute_iou(obb_radar_aligned, obb_ir)
                iou_results[f"capture_{capture_number}_frame_{i}"] = iou
                
                if iou > 0:
                    valid_ious.append(iou)
                
                # Visualize the point clouds
                visualize_point_clouds(ir_points, radar_points, radar_aligned, frame_number=i, output_dir=output_dir)
            # Compute average IoU excluding frames with IoU = 0
            if valid_ious:
                average_iou = sum(valid_ious) / len(valid_ious)
            else:
                average_iou = 0

            # Add average IoU to the results
            iou_results["average_iou"] = average_iou
            
            # Save IoU results to a JSON file
            with open(f"iou_results_capture_{capture_number}.json", "w") as f:
                json.dump(iou_results, f, indent=4)

            logger.info(f"IoU results saved for capture {capture_number}")
        else:
            logger.info(f"No ir_camera data found for capture {capture_number}")
    logger.info("Done")

    
if __name__ == '__main__':
    main()


# def transform_points(points, rotation, translation):
#     R = o3d.geometry.get_rotation_matrix_from_xyz(rotation)
#     return np.dot(points, R.T) + translation

# Cost function to be minimized
# def cost_function(params, radar_points, ir_points, alpha, beta):
#     # Extract rotation and translation from params
#     rotation = params[:3]
#     translation = params[3:6]

#     # Transform radar points
#     transformed_points = transform_points(radar_points, rotation, translation)

#     # Compute MSE and Chamfer Distance
#     mse = mean_squared_error(transformed_points, ir_points)
#     chamfer_dist = compute_chamfer_distance(transformed_points, ir_points)

#     # Combined cost
#     return alpha * mse + beta * chamfer_dist

# def iterative_refinement(radar_points, ir_points, alpha=0.5, beta=0.5):
#     initial_params = np.zeros(6)
#     result = minimize(cost_function, initial_params, args=(radar_points, ir_points, alpha, beta), 
#                       method='BFGS', options={'maxiter': 1000})
    
#     optimized_params = result.x
#     optimized_rotation = optimized_params[:3]
#     optimized_translation = optimized_params[3:6]
    
#     aligned_radar_points = transform_points(radar_points, optimized_rotation, optimized_translation)
#     return aligned_radar_points, optimized_rotation, optimized_translation

# def compute_iou(bbox1, bbox2, scaling_factor=7.0):
#     """
#     Based on this estimation, a default scaling factor of 7.0 could be a good starting point. 
#     This factor assumes that the combined volume of the forehead, wrists, and ankles is about 1/7th of the entire body volume.
#     """
#     bbox1_min = bbox1.get_min_bound()
#     bbox1_max = bbox1.get_max_bound()
#     bbox2_min = bbox2.get_min_bound()
#     bbox2_max = bbox2.get_max_bound()
    
#     intersection_min = np.maximum(bbox1_min, bbox2_min)
#     intersection_max = np.minimum(bbox1_max, bbox2_max)
    
#     # Check if there is an intersection
#     if np.any(intersection_min >= intersection_max):
#         return 0.0
    
#     intersection_bbox = o3d.geometry.AxisAlignedBoundingBox(intersection_min, intersection_max)
#     intersection_vol = intersection_bbox.volume()
    
#     bbox1_vol = bbox1.volume()
#     bbox2_vol = bbox2.volume() * scaling_factor  # Scale the volume of bbox2
    
#     union_vol = bbox1_vol + bbox2_vol - intersection_vol
    
#     iou = intersection_vol / union_vol if union_vol > 0 else 0.0
#     return iou