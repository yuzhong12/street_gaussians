import os,sys
import numpy as np
import cv2
import torch
import json
import open3d as o3d
import math
from tqdm import tqdm 
from lib.config import cfg
from lib.utils.box_utils import bbox_to_corner3d, inbbox_points, get_bound_2d_mask
from lib.utils.colmap_utils import read_points3D_binary
from lib.utils.general_utils import matrix_to_quaternion, quaternion_to_matrix_numpy
from lib.datasets.base_readers import storePly, get_Sphere_Norm
from scipy.spatial.transform import Rotation as R

_camera2label = {
    'cam_front_left': 0,
    'cam_side_left_front': 1,
    'cam_side_right_front': 2,
    'cam_back': 3,
    'cam_side_left_back': 4,
    'cam_side_right_back': 5
}

_label2camera = {
    0: 'cam_front_left',
    1: 'cam_side_left_front',
    2: 'cam_side_right_front',
    3: 'cam_back',
    4: 'cam_side_left_back',
    5: 'cam_side_right_back'

}

image_heights = [1080, 1280, 1280, 1080, 1280, 1280]
image_widths = [1920, 1920, 1920, 1920, 1920, 1920]

# ------------- 一些小工具 -------------

def getRotMat(roll, pitch, yaw):
    sx = np.sin(roll)
    cx = np.cos(roll)
    sy = np.sin(pitch)
    cy = np.cos(pitch)
    sz = np.sin(yaw)
    cz = np.cos(yaw)

    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx 

def getMatrix(pos, rot):
    matrix = np.eye(4)
    matrix[:3,3] = np.array(pos, np.float32)
    matrix[:3,:3] = getRotMat(rot[0], rot[1], rot[2])
    # matrix[:3,:3] = tf3.euler.euler2mat(rot[0], rot[1], rot[2])
    return matrix

def translation_euler(entry):
    t = np.array(entry['translation'])
    e_XYZ = np.array(entry['euler'])
    mat = getMatrix(t, e_XYZ)
    return mat

trans = np.eye(4)
trans[:3,:3] = np.array([[0,-1,0],[0,0,-1],[1,0,0]])

def ego2global_to_matrix(x, y, z, qx, qy, qz, qw):
    # 四元数 -> 3×3 旋转矩阵
    rot = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [    2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qw*qx)],
        [    2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])
    # 拼成 4×4
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3]  = [x, y, z]
    return T

def load_camera_info_cosmos(datadir, cameras):
    selected_frames = cfg.data.get('selected_frames', None)
    start_frame, end_frame = selected_frames[0], selected_frames[1]
    intrinsics = {}
    extrinsics = {}
    ego_frame_poses = []
    ego_cam_poses = {i:[] for i in cameras}
    frame_count = len(os.listdir(os.path.join(datadir, 'autolabel')))

    for idx, cam in enumerate(cameras):
        for frame_id in range(frame_count):
            json_name = frame_id + start_frame
            if json_name > end_frame:
                break
            json_path = os.path.join(datadir, 'autolabel', f"{json_name}.json")
            with open(json_path, 'r') as f:
                js = json.load(f)

            # 内参、外参只需要一帧
            if frame_id == 0:
                intrinsics[cam] = js['global_properties']['camera_intrinsic'][_label2camera[cam]]['intr']
                extrinsics[cam] = np.linalg.inv(js['global_properties']['vehicle2camera_extrinsic'][_label2camera[cam]])
                # extrinsics[cam] = np.linalg.inv(trans @ js['global_properties']['vehicle2camera_extrinsic'][_label2camera[cam]])

            #主车位姿只需要一个视角
            if idx == 0:
                ego_frame_poses.append(ego2global_to_matrix(**js['global_properties']['ego_info']['ego2global']))

            ego_cam_poses[cam].append(ego2global_to_matrix(**js['global_properties']['ego_info']['ego2global']))


    # ego_frame_poses = np.array(ego_frame_poses)
    # center_point = np.mean(ego_frame_poses[:, :3, 3], axis=0)
    # ego_frame_poses[:, :3, 3] -= center_point # [num_frames, 4, 4]
    

    # ego_cam_poses = [np.array(ego_cam_poses[i]) for i in cameras]
    # ego_cam_poses = np.array(ego_cam_poses)
    # ego_cam_poses[:, :, :3, 3] -= center_point # [num_cameras, num_frames, 4, 4]

    ego_cam_poses_cp = {}
    for i, cam in enumerate(cameras):
        ego_cam_poses_cp[cam] = ego_cam_poses[i]

    return intrinsics, extrinsics, ego_frame_poses, ego_cam_poses_cp


# ------------- 分割线 -------------

def make_obj_pose(ego_pose, box_info):
    tx, ty, tz, heading = box_info
    c = math.cos(heading)
    s = math.sin(heading)
    rotz_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    obj_pose_vehicle = np.eye(4)
    obj_pose_vehicle[:3, :3] = rotz_matrix
    obj_pose_vehicle[:3, 3] = np.array([tx, ty, tz])
    obj_pose_world = np.matmul(ego_pose, obj_pose_vehicle)

    obj_rotation_vehicle = torch.from_numpy(obj_pose_vehicle[:3, :3]).float().unsqueeze(0)
    obj_quaternion_vehicle = matrix_to_quaternion(obj_rotation_vehicle).squeeze(0).numpy()
    obj_quaternion_vehicle = obj_quaternion_vehicle / np.linalg.norm(obj_quaternion_vehicle)
    obj_position_vehicle = obj_pose_vehicle[:3, 3]
    obj_pose_vehicle = np.concatenate([obj_position_vehicle, obj_quaternion_vehicle])

    obj_rotation_world = torch.from_numpy(obj_pose_world[:3, :3]).float().unsqueeze(0)
    obj_quaternion_world = matrix_to_quaternion(obj_rotation_world).squeeze(0).numpy()
    obj_quaternion_world = obj_quaternion_world / np.linalg.norm(obj_quaternion_world)
    obj_position_world = obj_pose_world[:3, 3]
    obj_pose_world = np.concatenate([obj_position_world, obj_quaternion_world])
    
    return obj_pose_vehicle, obj_pose_world

def get_obj_pose_tracking(root_dir, ego_poses, cameras=[0, 1, 2], selected_frames=None):
    if selected_frames is None:
        selected_frames = (0, len(ego_poses) - 1)
    start_frame, end_frame = selected_frames
    num_frames = end_frame - start_frame + 1

    objects_info = {}
    tracklets_ls = []
    n_obj_in_frame = np.zeros(num_frames)

    for frame_id in range(start_frame, end_frame + 1):
        anno_dir = os.path.join(root_dir, 'autolabel')
        json_path = os.path.join(anno_dir, f"{frame_id}.json")
        if not os.path.exists(json_path):
            continue

        with open(json_path, 'r') as f:
            data = json.load(f)

        for obj in data.get("object", []):
            track_id = obj["track_id"]
            l, w, h = (obj["psr"]["scale"][k] for k in ("x", "y", "z"))
            center = obj["psr"]["position"]       
            center_relativetoego = np.array([center["x"], center["y"], center["z"]], dtype=np.float64)
            class_name = obj["obj_category"]
            euler = obj["psr"]["rotation"]
            euler_relativetoego = np.array([euler["x"], euler["y"], euler["z"]], dtype=np.float64)

            # 更新 objects_info
            if track_id not in objects_info:
                objects_info[track_id] = {
                    "track_id": track_id,
                    "class": class_name,
                    "class_label": class_name,
                    "height": h,
                    "width": w,
                    "length": l
                }
            else:
                objects_info[track_id]["height"] = max(objects_info[track_id]["height"], h)
                objects_info[track_id]["width"] = max(objects_info[track_id]["width"], w)
                objects_info[track_id]["length"] = max(objects_info[track_id]["length"], l)

            # 记录每一帧跟踪物体信息
            trackle_info = np.array([frame_id, track_id, 0, l, w, h,
                                    center_relativetoego[0], center_relativetoego[1], center_relativetoego[2],
                                    euler_relativetoego[2]], dtype=np.float64)
            tracklets_ls.append(trackle_info)

            n_obj_in_frame[frame_id - start_frame] += 1

    # ---------- 构建输出数组 ----------
    tracklets = np.stack(tracklets_ls, axis=0)          # shape (N, 10)
    # 按第 0 列（frame_id）升序排列
    order = np.argsort(tracklets[:, 0])
    tracklets = tracklets[order]
    # 构造唯一键：frame_id + track_id
    keys = tracklets[:, :2]          # (N, 2)
    _, idx = np.unique(keys, axis=0, return_index=True)
    # 保留唯一键对应的行（保持升序）
    tracklets_array = tracklets[np.sort(idx)]
    max_obj_per_frame = int(n_obj_in_frame[0:num_frames - 1].max())
    visible_objects_ids = np.ones([num_frames, max_obj_per_frame]) * -1.0
    visible_objects_pose_world = np.ones([num_frames, max_obj_per_frame, 7]) * -1.0
    visible_objects_pose_vehicle = np.ones([num_frames, max_obj_per_frame, 7]) * -1.0

    for tr in tracklets_array:
        frame_id = int(tr[0])
        if not (start_frame <= frame_id <= end_frame):
            continue
        frame_idx = frame_id - start_frame
        track_id = int(tr[1])

        obj_column = np.argwhere(visible_objects_ids[frame_idx, :] < 0).min()

        obj_pose_vehicle, obj_pose_world = make_obj_pose(ego_poses[frame_id - start_frame], tr[6:10])

        visible_objects_ids[frame_idx, obj_column] = track_id
        visible_objects_pose_world[frame_idx, obj_column] = obj_pose_world
        visible_objects_pose_vehicle[frame_idx, obj_column] = obj_pose_vehicle

    for frame_idx in range(len(visible_objects_ids)):
        # 获取当前帧的ID和对应的姿态数据
        frame_ids = visible_objects_ids[frame_idx]
        frame_poses_world = visible_objects_pose_world[frame_idx]
        frame_poses_vehicle = visible_objects_pose_vehicle[frame_idx]
        
        # 获取排序的索引（从小到大）
        sorted_indices = np.argsort(frame_ids)
        
        # 按照排序索引重新排列
        visible_objects_ids[frame_idx] = frame_ids[sorted_indices]
        visible_objects_pose_world[frame_idx] = frame_poses_world[sorted_indices]
        visible_objects_pose_vehicle[frame_idx] = frame_poses_vehicle[sorted_indices]

    # ---------- 剔除静态物体 ----------
    print("Removing static objects")
    for key in list(objects_info.keys()):
        idx = np.where(visible_objects_ids == key)
        if idx[0].size:
            pos_world = visible_objects_pose_world[idx][:, :3]
            distance = np.linalg.norm(pos_world[0] - pos_world[-1])
            dynamic = np.any(np.std(pos_world, axis=0) > 0.5) or distance > 2
            if not dynamic:
                visible_objects_ids[idx] = -1.
                visible_objects_pose_vehicle[idx] = -1.
                visible_objects_pose_world[idx] = -1.
                objects_info.pop(key)
        else:
            objects_info.pop(key)
 
    # Clip max_num_obj
    mask = visible_objects_ids >= 0
    max_obj_per_frame_new = np.sum(mask, axis=1).max()
    print("Max obj per frame:", max_obj_per_frame_new)

    if max_obj_per_frame_new == 0:
        print("No moving obj in current sequence, make dummy visible objects")
        visible_objects_ids = np.ones([num_frames, 1]) * -1.0
        visible_objects_pose_world = np.ones([num_frames, 1, 7]) * -1.0
        visible_objects_pose_vehicle = np.ones([num_frames, 1, 7]) * -1.0    
    elif max_obj_per_frame_new < max_obj_per_frame:
        visible_objects_ids_new = np.ones([num_frames, max_obj_per_frame_new]) * -1.0
        visible_objects_pose_vehicle_new = np.ones([num_frames, max_obj_per_frame_new, 7]) * -1.0
        visible_objects_pose_world_new = np.ones([num_frames, max_obj_per_frame_new, 7]) * -1.0
        for frame_idx in range(num_frames):
            for y in range(max_obj_per_frame):
                obj_id = visible_objects_ids[frame_idx, y]
                if obj_id >= 0:
                    obj_column = np.argwhere(visible_objects_ids_new[frame_idx, :] < 0).min()
                    visible_objects_ids_new[frame_idx, obj_column] = obj_id
                    visible_objects_pose_vehicle_new[frame_idx, obj_column] = visible_objects_pose_vehicle[frame_idx, y]
                    visible_objects_pose_world_new[frame_idx, obj_column] = visible_objects_pose_world[frame_idx, y]

        visible_objects_ids = visible_objects_ids_new
        visible_objects_pose_vehicle = visible_objects_pose_vehicle_new
        visible_objects_pose_world = visible_objects_pose_world_new

    box_scale = cfg.data.get('box_scale', 1.0)
    print('box scale: ', box_scale)
    
    frames = list(range(start_frame, end_frame + 1))
    frames = np.array(frames).astype(np.int32)

    # postprocess object_info   
    for key in objects_info.keys():
        obj = objects_info[key]
        if obj['class'] == 'pedestrian':
            obj['deformable'] = True
        else:
            obj['deformable'] = False
        
        obj['width'] = obj['width'] * box_scale
        obj['length'] = obj['length'] * box_scale
        obj['height'] = obj['height'] * box_scale
        
        obj_frame_idx = np.argwhere(visible_objects_ids == key)[:, 0]
        obj_frame_idx = obj_frame_idx.astype(np.int32)
        obj_frames = frames[obj_frame_idx]
        obj['start_frame'] = np.min(obj_frames)
        obj['end_frame'] = np.max(obj_frames)
        
        objects_info[key] = obj

    # [num_frames, max_obj, track_id, x, y, z, qw, qx, qy, qz]
    objects_tracklets_world = np.concatenate(
        [visible_objects_ids[..., None], visible_objects_pose_world], axis=-1
    )
    
    objects_tracklets_vehicle = np.concatenate(
        [visible_objects_ids[..., None], visible_objects_pose_vehicle], axis=-1
    )
    
    
    return objects_tracklets_world, objects_tracklets_vehicle, objects_info


def generate_dataparser_outputs(
        datadir, 
        selected_frames=None, 
        build_pointcloud=True, 
        cameras=[0, 1, 2, 3, 4]
    ):

    start_frame, end_frame = selected_frames[0], selected_frames[1]
    num_frames = end_frame - start_frame + 1

    # ---------------- 1. 目录结构 ----------------
    anno_root = os.path.join(datadir, 'autolabel')       

    # ---------------- 2. 收集所有 json 与图像 ----------------
    all_jsons = []  
    for cam in cameras:      
        for jname in sorted(os.listdir(anno_root)):
            if not jname.endswith('.json'):
                continue
            frame_idx = int(jname.replace('.json', ''))
            if frame_idx < start_frame or frame_idx > end_frame:
                continue
            all_jsons.append((cam, frame_idx, os.path.join(anno_root, jname)))

    # 按帧号排序，保证时序
    all_jsons.sort(key=lambda x: x[1])

     # load camera, frame, path
    frames = []
    frames_idx = []
    cams = []
    image_filenames = []
    
    ixts = []
    exts = []
    poses = []
    c2ws = []
    # 添加
    ego_poses = []

    cams_timestamps = []
    frames_timestamps = []

    ego_frame_poses = []

    # load calibration and ego pose
    intrinsics, extrinsics, ego_frame_poses, ego_cam_poses = load_camera_info_cosmos(datadir, cameras)


    for uid, (cam, frame_id, json_path) in enumerate(tqdm(all_jsons, desc='parse cameras')):
        with open(json_path, 'r') as f:
            js = json.load(f)

        # if uid == 0:
        #     timestamp_offset = js['timestamp'] / 1e6
        # if cam == 0:
        # 修改
        if cam == cameras[0]:
            frames_timestamps.append(js['global_properties'][f'{_label2camera[cam]}_timestamp'] / 1e9)

        frames.append(frame_id)
        frames_idx.append(frame_id - start_frame)
        cams.append(cam)

        # ---- 内外参 ----
        ixts.append(intrinsics[cam])
        exts.append(extrinsics[cam])
        # poses.append(ego_cam_poses[cam,frame_id])
        # c2w = ego_cam_poses[cam,frame_id] @ extrinsics[cam]
        # 修改
        poses.append(ego_cam_poses[cam][frame_id - start_frame])
        c2w = ego_cam_poses[cam][frame_id - start_frame] @ extrinsics[cam]
        c2ws.append(c2w)
        # 添加
        ego_poses.append(ego_frame_poses[frame_id - start_frame])
        cams_timestamps.append(js['global_properties'][f'{_label2camera[cam]}_timestamp'] / 1e9)
        image_name = js['global_properties'][f'{_label2camera[cam]}_timestamp']
        image_filename = os.path.join(datadir, _label2camera[cam], f'{image_name}.jpg')
        image_filenames.append(image_filename)
        
    exts = np.stack(exts, axis=0)
    ixts = np.stack(ixts, axis=0)
    poses = np.stack(poses, axis=0)
    c2ws = np.stack(c2ws, axis=0)

    _, object_tracklets_vehicle, object_info = get_obj_pose_tracking(datadir, ego_frame_poses, cameras, selected_frames)
    
    timestamp_offset = min(cams_timestamps + frames_timestamps)
    cams_timestamps = np.array(cams_timestamps) - timestamp_offset
    frames_timestamps = np.array(frames_timestamps) - timestamp_offset
    min_timestamp, max_timestamp = np.array(frames_timestamps).min(), np.array(frames_timestamps).max()
    for track_id in object_info.keys():
        object_start_frame = object_info[track_id]['start_frame']
        object_end_frame = object_info[track_id]['end_frame']
        object_start_timestamp = frames_timestamps[object_start_frame - start_frame] - 0.1
        object_end_timestamp = frames_timestamps[object_end_frame - start_frame] + 0.1
        object_info[track_id]['start_timestamp'] = max(object_start_timestamp, min_timestamp)
        object_info[track_id]['end_timestamp'] = min(object_end_timestamp, max_timestamp)
        
    # for frame_id in range(start_frame, end_frame + 1):
    #     for cam in cameras:
    #         image_filename = os.path.join(datadir, _label2camera[cam], f'{frame_id}_{cam}.png')
    #         image_filenames.append(image_filename)

    result = dict()
    result['num_frames'] = num_frames
    result['exts'] = exts
    result['ixts'] = ixts
    result['poses'] = poses
    result['c2ws'] = c2ws
    result['obj_tracklets'] = object_tracklets_vehicle
    result['obj_info'] = object_info
    result['frames'] = frames
    result['cams'] = cams
    result['frames_idx'] = frames_idx
    result['image_filenames'] = image_filenames
    result['cams_timestamps'] = cams_timestamps
    result['tracklet_timestamps'] = frames_timestamps
    # 添加
    result['ego_poses'] = np.stack(ego_poses, axis=0)


    obj_bounds = []
    for i, image_filename in tqdm(enumerate(image_filenames)):
        cam = cams[i]
        h, w = image_heights[cam], image_widths[cam]
        obj_bound = np.zeros((h, w)).astype(np.uint8)
        obj_tracklets = object_tracklets_vehicle[frames_idx[i]]
        ixt, ext = ixts[i], exts[i]
        for obj_tracklet in obj_tracklets:
            track_id = int(obj_tracklet[0])
            if track_id >= 0:
                obj_pose_vehicle = np.eye(4)    
                obj_pose_vehicle[:3, :3] = quaternion_to_matrix_numpy(obj_tracklet[4:8])
                obj_pose_vehicle[:3, 3] = obj_tracklet[1:4]
                obj_length = object_info[track_id]['length']
                obj_width = object_info[track_id]['width']
                obj_height = object_info[track_id]['height']
                bbox = np.array([[-obj_length, -obj_width, -obj_height], 
                                 [obj_length, obj_width, obj_height]]) * 0.5
                corners_local = bbox_to_corner3d(bbox)
                corners_local = np.concatenate([corners_local, np.ones_like(corners_local[..., :1])], axis=-1)
                corners_vehicle = corners_local @ obj_pose_vehicle.T # 3D bounding box in vehicle frame
                mask = get_bound_2d_mask(
                    corners_3d=corners_vehicle[..., :3],
                    K=ixt,
                    pose=np.linalg.inv(ext), 
                    H=h, W=w
                )
                obj_bound = np.logical_or(obj_bound, mask)
        obj_bounds.append(obj_bound)
    result['obj_bounds'] = obj_bounds         

    # out_dir = './overmask'                       # 当前路径，想换就改这里
    # os.makedirs(out_dir, exist_ok=True)
    # alpha = 0.3
    # for idx, mask in enumerate(result['obj_bounds']): 
    #     img_bgr = cv2.imread(str(image_filenames[idx]))
    #     # 0/1 → 0/255 灰度图
    #     gray = (mask * 255).astype('uint8')
    #     mask_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    #     blended = cv2.addWeighted(img_bgr, 1 - alpha, mask_bgr, alpha, 0)
    #     save_path = os.path.join(out_dir, f'obj_bound_{idx:02d}.png')
    #     cv2.imwrite(save_path, blended)      # opencv 默认写单通道 PNG
    #     print('saved', save_path)
    
    
    colmap_basedir = os.path.join(f'{cfg.model_path}/colmap')
    # if not os.path.exists(os.path.join(colmap_basedir, 'triangulated/sparse/model')):
    #     from script.waymo.colmap_cosmos_full import run_colmap_cosmos
    #     run_colmap_cosmos(result)
    
    if build_pointcloud:
        print('build point cloud')
        pointcloud_dir = os.path.join(cfg.model_path, 'input_ply')
        os.makedirs(pointcloud_dir, exist_ok=True)
        
        points_xyz_dict = dict()
        points_rgb_dict = dict()
        points_xyz_dict['bkgd'] = []
        points_rgb_dict['bkgd'] = []
        for track_id in object_info.keys():
            points_xyz_dict[f'obj_{track_id:03d}'] = []
            points_rgb_dict[f'obj_{track_id:03d}'] = []

        # print('initialize from sfm pointcloud')
        # points_colmap_path = os.path.join(colmap_basedir, 'triangulated/sparse/model/points3D.bin')
        # points_colmap_xyz, points_colmap_rgb, points_colmap_error = read_points3D_binary(points_colmap_path)
        # points_colmap_rgb = points_colmap_rgb / 255.
                     
        for i, frame in tqdm(enumerate(range(start_frame, end_frame+1))):

            pcd_path = os.path.join(datadir,'pcd', f'{frame}.pcd')
            # 读取
            pcd = o3d.io.read_point_cloud(pcd_path)  
            points_xyz_vehicle = np.asarray(pcd.points)   
            points_xyz_vehicle = np.concatenate([points_xyz_vehicle, np.ones_like(points_xyz_vehicle[..., :1])], axis=-1)           
            points_rgb = np.ones((points_xyz_vehicle.shape[0], 3), dtype=np.float32)

            ego_pose = ego_frame_poses[frame - start_frame]
            points_xyz_world = points_xyz_vehicle @ ego_pose.T

            # points_xyz_vehicle = np.empty((0, 4), dtype=np.float32)
            # points_rgb = np.empty((0, 3), dtype=np.float32)
            points_xyz_obj_mask = np.zeros(points_xyz_vehicle.shape[0], dtype=np.bool_)

            for tracklet in object_tracklets_vehicle[i]:
                track_id = int(tracklet[0])
                if track_id >= 0:
                    obj_pose_vehicle = np.eye(4)                    
                    obj_pose_vehicle[:3, :3] = quaternion_to_matrix_numpy(tracklet[4:8])
                    obj_pose_vehicle[:3, 3] = tracklet[1:4]
                    vehicle2local = np.linalg.inv(obj_pose_vehicle)
                    
                    points_xyz_obj = points_xyz_vehicle @ vehicle2local.T
                    points_xyz_obj = points_xyz_obj[..., :3]
                    
                    length = object_info[track_id]['length']
                    width = object_info[track_id]['width']
                    height = object_info[track_id]['height']
                    bbox = [[-length/2, -width/2, -height/2], [length/2, width/2, height/2]]
                    obj_corners_3d_local = bbox_to_corner3d(bbox)
                    
                    points_xyz_inbbox = inbbox_points(points_xyz_obj, obj_corners_3d_local)
                    points_xyz_obj_mask = np.logical_or(points_xyz_obj_mask, points_xyz_inbbox)
                    points_xyz_dict[f'obj_{track_id:03d}'].append(points_xyz_obj[points_xyz_inbbox])
                    points_rgb_dict[f'obj_{track_id:03d}'].append(points_rgb[points_xyz_inbbox])
                    
          
            points_lidar_xyz = points_xyz_world[~points_xyz_obj_mask][..., :3]
            points_lidar_rgb = points_rgb[~points_xyz_obj_mask]
            
            points_xyz_dict['bkgd'].append(points_lidar_xyz)
            points_rgb_dict['bkgd'].append(points_lidar_rgb)

        initial_num_obj = 40000

        for k, v in points_xyz_dict.items():
            if len(v) == 0:
                continue
            else:
                points_xyz = np.concatenate(v, axis=0)
                points_rgb = np.concatenate(points_rgb_dict[k], axis=0)

                if k == 'bkgd':
                    # downsample lidar pointcloud with voxels
                    points_lidar = o3d.geometry.PointCloud()
                    points_lidar.points = o3d.utility.Vector3dVector(points_xyz)
                    points_lidar.colors = o3d.utility.Vector3dVector(points_rgb)
                    downsample_points_lidar = points_lidar.voxel_down_sample(voxel_size=0.15)
                    downsample_points_lidar, _ = downsample_points_lidar.remove_radius_outlier(nb_points=10, radius=0.5)
                    points_lidar_xyz = np.asarray(downsample_points_lidar.points).astype(np.float32)
                    points_lidar_rgb = np.asarray(downsample_points_lidar.colors).astype(np.float32)
                                       
                elif k.startswith('obj'):
                    if len(points_xyz) > initial_num_obj:
                        random_indices = np.random.choice(len(points_xyz), initial_num_obj, replace=False)
                        points_xyz = points_xyz[random_indices]
                        points_rgb = points_rgb[random_indices]
                        
                    points_xyz_dict[k] = points_xyz
                    points_rgb_dict[k] = points_rgb

                else:
                    raise NotImplementedError()

        # Get sphere center and radius
        # lidar_sphere_normalization = get_Sphere_Norm(points_lidar_xyz)
        # lidar_sphere_normalization = get_Sphere_Norm(points_lidar_xyz)
        # sphere_center = lidar_sphere_normalization['center']
        # sphere_radius = lidar_sphere_normalization['radius']

        try:
            if True:
                points_lidar_mask = np.ones(points_lidar_xyz.shape[0], dtype=np.bool_)
                for i, ext in enumerate(exts):
                    # if frames_idx[i] not in train_frames:
                    #     continue
                    camera_position = c2ws[i][:3, 3]
                    radius = np.linalg.norm(points_lidar_xyz - camera_position, axis=-1)
                    mask = np.logical_or(radius < cfg.data.get('extent', 10),points_lidar_xyz[:, 2] < camera_position[2])
                    points_lidar_mask = np.logical_and(points_lidar_mask, ~mask)        
                points_lidar_xyz = points_lidar_xyz[points_lidar_mask]
                points_lidar_rgb = points_lidar_rgb[points_lidar_mask]
            
            # points_colmap_dist = np.linalg.norm(points_colmap_xyz - sphere_center, axis=-1)
            # mask = points_colmap_dist < 2 * sphere_radius
            # points_colmap_xyz = points_colmap_xyz[mask]
            # points_colmap_rgb = points_colmap_rgb[mask]
            # points_colmap_xyz, points_colmap_rgb, _ = generate_colmap_pointcloud()
        
            points_bkgd_xyz = points_lidar_xyz.astype(np.float32)
            points_bkgd_rgb = points_lidar_rgb.astype(np.float32)
        except:
            print('No colmap pointcloud')
            points_bkgd_xyz = np.empty((0, 3), dtype=np.float32)
            points_bkgd_rgb = np.empty((0, 3), dtype=np.float32)
        
        points_xyz_dict['lidar'] = points_lidar_xyz
        points_rgb_dict['lidar'] = points_lidar_rgb
        # points_xyz_dict['colmap'] = points_colmap_xyz
        # points_rgb_dict['colmap'] = points_colmap_rgb
        points_xyz_dict['bkgd'] = points_bkgd_xyz
        points_rgb_dict['bkgd'] = points_bkgd_rgb
        
        result['points_xyz_dict'] = points_xyz_dict
        result['points_rgb_dict'] = points_rgb_dict

    

        for k in points_xyz_dict.keys():
            points_xyz = points_xyz_dict[k]
            points_rgb = points_rgb_dict[k]
            ply_path = os.path.join(pointcloud_dir, f'points3D_{k}.ply')
            try:
                storePly(ply_path, points_xyz, points_rgb)
                print(f'saving pointcloud for {k}, number of initial points is {points_xyz.shape}')
            except:
                print(f'failed to save pointcloud for {k}')
                continue
    
    return result


