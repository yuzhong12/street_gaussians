
from lib.utils.idc_utils import generate_dataparser_outputs
from lib.utils.graphics_utils import focal2fov, BasicPointCloud
from lib.utils.data_utils import get_val_frames
from lib.datasets.base_readers import CameraInfo, SceneInfo, getNerfppNorm, fetchPly, get_PCA_Norm, get_Sphere_Norm
import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from lib.config import cfg



################################ 添加视角偏移 ################################
from collections import defaultdict
from typing import Dict, List, Literal, Tuple, Type
import copy
import math
LANE_SHIFT_SIGN: Dict[str, Literal[-1, 1]] = defaultdict(lambda: -1)

def affine_inverse(A: np.ndarray):
    R = A[..., :3, :3]  # ..., 3, 3
    T = A[..., :3, 3:]  # ..., 3, 1
    P = A[..., 3:, :]  # ..., 1, 4
    return np.concatenate([np.concatenate([R.T, -R.T @ T], axis=-1), P], axis=-2)

def get_lane_shift_direction(ego_frame_poses, frame):
    assert frame >= 0 and frame < len(ego_frame_poses)
    if len(cfg.data.cameras)==1:
        if frame == 0:
            ego_pose_delta = ego_frame_poses[1][:3, 3] - ego_frame_poses[0][:3, 3]
        else:
            ego_pose_delta = ego_frame_poses[frame][:3, 3] - ego_frame_poses[frame - 1][:3, 3]

    elif len(cfg.data.cameras)>1:
        cam_n = len(cfg.data.cameras)
        if frame == 0 or frame == 1:
            ego_pose_delta = ego_frame_poses[2][:3, 3] - ego_frame_poses[0][:3, 3]
        else:
            ego_pose_delta = ego_frame_poses[frame * cam_n][:3, 3] - ego_frame_poses[(frame - 1)*cam_n][:3, 3]

    ego_pose_delta = ego_pose_delta[:2]  # x, y
    ego_pose_delta /= np.linalg.norm(ego_pose_delta)
    direction = np.array([-ego_pose_delta[1], ego_pose_delta[0], 0])  # y, x 左偏移为正方向
    # direction = np.array([ego_pose_delta[1], -ego_pose_delta[0], 0])  # y, x 右偏移为正方向
    direction_front_rear = np.array([ego_pose_delta[0], ego_pose_delta[1], 0])  # x,y 前偏移为正方向

    return direction, direction_front_rear

def novel_view_cameras(cameras: List[CameraInfo], ego_frame_poses, obj_info, camera_tracklets):
    from lib.config import cfg
    modes = []
    
    # shifts = cfg.render.novel_view.shift if isinstance(cfg.render.novel_view.shift, list) else [cfg.render.novel_view.shift]
    shifts = [1]

    if cfg.mode == 'train':
        shifts = [x for x in shifts if x != 0]
    for shift in shifts:
        modes.append({'shift': shift, 'rotate': 0.0})
    # rotates = cfg.render.novel_view.rotate if isinstance(cfg.render.novel_view.rotate, list) else [cfg.render.novel_view.rotate]
    # rotates = [x for x in rotates if x != 0]
    # for rotate in rotates:
    #     modes.append({'shift': 0, 'rotate': rotate})

    novel_view_cameras = []
    skip_count = 0
    
    # cameras = [camera for camera in cameras if camera.metadata['cam'] == 0]  # only consider the FRONT camera (whose cam_idx is marked as 0)
    cameras = [camera for camera in cameras]  # only consider the FRONT camera (whose cam_idx is marked as 0)
    
    pbar = tqdm(total=len(cameras) * len(modes), desc='Making novel view cameras')
    for mode in modes:
        for i in range(len(cameras)):
            novel_view_camera = copy.copy(cameras[i])
            novel_view_camera = novel_view_camera._replace(metadata=copy.copy(cameras[i].metadata))

            image_name = novel_view_camera.image_name

            # make novel view path
            shift, rotate = mode['shift'], mode['rotate']
            tag = ''
            if shift != 0: tag += f'_shift_{shift:.2f}'
            if rotate != 0: tag += f'_rotate_{rotate:.2f}'
            
            novel_view_dir = os.path.join(cfg.source_path, 'lidar', f'color_render{tag}')
            novel_view_image_name = f'{image_name}{tag}.png'
            metadata = novel_view_camera.metadata
            metadata['is_novel_view'] = True
            metadata['novel_view_id'] = shift
            cam, frame = metadata['cam'], metadata['frame']
            novel_view_rgb_path = os.path.join(novel_view_dir, f'{str(frame).zfill(6)}_{cam}.png')
            novel_view_mask_path = os.path.join(novel_view_dir, f'{str(frame).zfill(6)}_{cam}_mask.png')
            metadata['guidance_rgb_path'] = novel_view_rgb_path
            metadata['guidance_mask_path'] = novel_view_mask_path

            # make novel view camera
            ego_pose = metadata['ego_pose'].copy()
            ext = metadata['extrinsic'].copy()


            shift_direction, direction_front_rear = get_lane_shift_direction(ego_frame_poses, frame)
            scene_idx = os.path.split(cfg.source_path)[-1]
            x, y, z = cfg.render.novel_view.shift
            # ego_pose[:3, 3] += [x*direction_front_rear[0], y*shift_direction[1], z]
            ego_pose[:3, 3] += x*direction_front_rear
            ego_pose[:3, 3] += y*shift_direction
            ego_pose[:3, 3] += [0,0,z]
            # print("-------------------------------------")
            ######################### 偏移结束 ############################

            # rotate
            c, s = math.cos(rotate), math.sin(rotate)
            rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            ego_pose[:3, :3] = rot @ ego_pose[:3, :3]

            c2w = ego_pose @ ext
            RT = affine_inverse(c2w)
            R = RT[:3, :3].T
            T = RT[:3, 3]

            novel_view_camera = novel_view_camera._replace(
                image_name=novel_view_image_name,  R=R, T=T, guidance=dict(), metadata=metadata)
            novel_view_cameras.append(novel_view_camera)

            # # TODO: fix obj_pose and sky and lidar
            # cam = novel_view_camera.metadata['cam']
            # frame_idx = novel_view_camera.metadata['frame_idx']

            # skip_camera = False
            # for obj_id in obj_info.keys():
            #     id = obj_info[obj_id]['id']
            #     if camera_tracklets[cam, frame_idx, id, -1] < 0.:
            #         continue
            #     trans = camera_tracklets[cam, frame_idx, id, :3]
            #     view = (novel_view_camera.R).T @ trans + novel_view_camera.T
            #     depth = view[2]
            #     if depth < cfg.render.novel_view.train_actor_distance_thresh and \
            #         depth > -cfg.render.novel_view.train_actor_distance_thresh:
            #         skip_camera = True
            #     break

            # skip_count += skip_camera
            # novel_view_camera.metadata['skip_camera'] = skip_camera  # will skip camera for training if this is present

            pbar.update()

    novel_view_cameras = sorted(novel_view_cameras, key=lambda x: x.uid)
    return novel_view_cameras

############################视角偏移################################



# ------------- 主入口 -------------
def readIDCInfo(path,images='images',split_train=-1,split_test=-1,**kwargs):
    selected_frames = cfg.data.get('selected_frames', None)
    if cfg.debug:
        selected_frames = [0, 0]
        
    bkgd_ply_path = os.path.join(cfg.model_path, 'input_ply/points3D_bkgd.ply')
    build_pointcloud = (cfg.mode == 'train') and (not os.path.exists(bkgd_ply_path) or cfg.data.get('regenerate_pcd', False))
    
    # dynamic mask
    dynamic_mask_dir = os.path.join(path, 'dynamic_mask')
    load_dynamic_mask = True

    # sky mask
    sky_mask_dir = os.path.join(path, 'sky_mask')
    load_sky_mask = (cfg.mode == 'train') and os.path.exists(sky_mask_dir)
 
    
    # lidar depth
    lidar_depth_dir = os.path.join(path, 'lidar_depth')
    load_lidar_depth = (cfg.mode == 'train') and os.path.exists(lidar_depth_dir)

    output = generate_dataparser_outputs(
        datadir=path, 
        selected_frames=selected_frames,
        build_pointcloud=build_pointcloud,
        cameras=cfg.data.get('cameras', [0, 1, 2]),
    )

    exts = output['exts']
    ixts = output['ixts']
    poses = output['poses']
    c2ws = output['c2ws']
    image_filenames = output['image_filenames']
    obj_tracklets = output['obj_tracklets']
    obj_info = output['obj_info']
    frames, cams = output['frames'], output['cams']
    frames_idx = output['frames_idx']
    num_frames = output['num_frames']
    cams_timestamps = output['cams_timestamps']
    tracklet_timestamps = output['tracklet_timestamps']
    obj_bounds = output['obj_bounds']
    train_frames, test_frames = get_val_frames(
        num_frames, 
        test_every=split_test if split_test > 0 else None,
        train_every=split_train if split_train > 0 else None,
    )

    scene_metadata = dict()
    scene_metadata['obj_tracklets'] = obj_tracklets
    scene_metadata['tracklet_timestamps'] = tracklet_timestamps
    scene_metadata['obj_meta'] = obj_info
    scene_metadata['num_images'] = len(exts)
    scene_metadata['num_cams'] = len(cfg.data.cameras)
    scene_metadata['num_frames'] = num_frames
    
    camera_timestamps = dict()
    for cam in cfg.data.get('cameras', [0, 1, 2]):
        camera_timestamps[cam] = dict()
        camera_timestamps[cam]['train_timestamps'] = []
        camera_timestamps[cam]['test_timestamps'] = []      

    ########################################################################################################################
    cam_infos = []
    for i in tqdm(range(len(exts))):
        # generate pose and image
        ext = exts[i]
        ixt = ixts[i]
        c2w = c2ws[i]
        pose = poses[i]
        image_path = image_filenames[i]
        image_name = os.path.basename(image_path).split('.')[0]
        image = Image.open(image_path)

        width, height = image.size
        fx, fy = ixt[0, 0], ixt[1, 1]
        FovY = focal2fov(fx, height)
        FovX = focal2fov(fy, width)    
        
        RT = np.linalg.inv(c2w)
        R = RT[:3, :3].T
        T = RT[:3, 3]
        K = ixt.copy()
        
        metadata = dict()
        metadata['frame'] = frames[i]
        metadata['cam'] = cams[i]
        metadata['frame_idx'] = frames_idx[i]
        metadata['ego_pose'] = pose
        metadata['extrinsic'] = ext
        metadata['timestamp'] = cams_timestamps[i]

        if frames_idx[i] in train_frames:
            metadata['is_val'] = False
            camera_timestamps[cams[i]]['train_timestamps'].append(cams_timestamps[i])
        else:
            metadata['is_val'] = True
            camera_timestamps[cams[i]]['test_timestamps'].append(cams_timestamps[i])
        
        guidance = dict()

        # load dynamic mask
        if load_dynamic_mask:
            guidance['obj_bound'] = Image.fromarray(obj_bounds[i])

        # load lidar depth
        if load_lidar_depth:
            depth_path = os.path.join(lidar_depth_dir, f'{image_name}.npy')
            depth = np.load(depth_path, allow_pickle=True)
            depth = dict(depth.item())
            mask = depth['mask']
            value = depth['value']
            depth = np.zeros_like(mask).astype(np.float32)
            depth[mask] = value
            guidance['lidar_depth'] = depth
            
        # load sky mask
        if load_sky_mask:
            sky_mask_path = os.path.join(sky_mask_dir, f'{image_name}.png')
            sky_mask = (cv2.imread(sky_mask_path)[..., 0]) > 0.
            guidance['sky_mask'] = Image.fromarray(sky_mask)
        
        mask = None        
        cam_info = CameraInfo(
            uid=i, R=R, T=T, FovY=FovY, FovX=FovX, K=K,
            image=image, image_path=image_path, image_name=image_name,
            width=width, height=height,
            metadata=metadata,
            guidance=guidance,
        )
        cam_infos.append(cam_info)
        
        # sys.stdout.write('\n')
    train_cam_infos = [cam_info for cam_info in cam_infos if not cam_info.metadata['is_val']]
    test_cam_infos = [cam_info for cam_info in cam_infos if cam_info.metadata['is_val']]
    
    for cam in cfg.data.get('cameras', [0, 1, 2]):
        camera_timestamps[cam]['train_timestamps'] = sorted(camera_timestamps[cam]['train_timestamps'])
        camera_timestamps[cam]['test_timestamps'] = sorted(camera_timestamps[cam]['test_timestamps'])
    scene_metadata['camera_timestamps'] = camera_timestamps
        
    novel_view_cam_infos = []
    
    #######################################################################################################################3
    # Get scene extent
    # 1. Default nerf++ setting
    if cfg.mode == 'novel_view':
        # 添加
        novel_view_cam_infos = novel_view_cameras(cam_infos, output['ego_poses'], None, None)
        nerf_normalization = getNerfppNorm(novel_view_cam_infos)
    else:
        nerf_normalization = getNerfppNorm(train_cam_infos)

    # 2. The radius we obtain should not be too small (larger than 10 here)
    nerf_normalization['radius'] = max(nerf_normalization['radius'], 10)
    
    # 3. If we have extent set in config, we ignore previous setting
    if cfg.data.get('extent', False):
        nerf_normalization['radius'] = cfg.data.extent
    
    # 4. We write scene radius back to config
    cfg.data.extent = float(nerf_normalization['radius'])

    # 5. We write scene center and radius to scene metadata    
    scene_metadata['scene_center'] = nerf_normalization['center']
    scene_metadata['scene_radius'] = nerf_normalization['radius']
    print(f'Scene extent: {nerf_normalization["radius"]}')

    # Get sphere center
    lidar_ply_path = os.path.join(cfg.model_path, 'input_ply/points3D_lidar.ply')
    if os.path.exists(lidar_ply_path):
        sphere_pcd: BasicPointCloud = fetchPly(lidar_ply_path)
    else:
        sphere_pcd: BasicPointCloud = fetchPly(bkgd_ply_path)
    
    sphere_normalization = get_Sphere_Norm(sphere_pcd.points)
    scene_metadata['sphere_center'] = sphere_normalization['center']
    scene_metadata['sphere_radius'] = sphere_normalization['radius']
    print(f'Sphere extent: {sphere_normalization["radius"]}')

    pcd: BasicPointCloud = fetchPly(bkgd_ply_path)
    if cfg.mode == 'train':
        point_cloud = pcd
    else:
        point_cloud = None
        bkgd_ply_path = None

    scene_info = SceneInfo(
        point_cloud=point_cloud,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=bkgd_ply_path,
        metadata=scene_metadata,
        novel_view_cameras=novel_view_cam_infos,
    )
    
    return scene_info
        
