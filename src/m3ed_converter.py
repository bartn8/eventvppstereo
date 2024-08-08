import numpy as np
import h5py
import cv2
import glob
import os
import tqdm
from ouster.client import LidarPacket, SensorInfo, Scans, Packets, ChanField, XYZLut
import open3d as o3d
import klampt.math.se3 as se3
import yaml
import matplotlib.pyplot as plt
import json
import numba

import argparse
import errno
import re

def range_for_field(f):
    if f in (ChanField.RANGE2, ChanField.SIGNAL2,
             ChanField.REFLECTIVITY2):
        return ChanField.RANGE2
    else:
        return ChanField.RANGE

def get_cloud(h5f, idx, start_pose=None, stop_pose=None):    
    info = json.loads(h5f['/ouster/metadata'][()])
    #Remove warnings
    info["build_date"] = ""
    info["image_rev"] = ""
    info["prod_pn"] = ""
    info["status"] = ""
    
    info = SensorInfo(json.dumps(info))
    packet_buf = h5f['/ouster/data'][idx,...]

    packets = [LidarPacket(packet_buf[i], info) for i in range(packet_buf.shape[0])]
    scans = Scans(Packets( packets, info ))
    scan = next(iter(scans))
    
    metadata = scans.metadata
    xyzlut = XYZLut(metadata, use_extrinsics=True)

    fields = list(scan.fields)

    xyz_field = scan.field(range_for_field(fields[-1]))
    xyz = xyzlut(xyz_field) # xyz_destaggered)
    
    #Rolling shutter correction; assume xyz in staggered form
    if start_pose is not None and stop_pose is not None:
        scanlines = xyz.shape[0]
        n_poses = xyz.shape[1]

        se3_start_pose = se3.from_ndarray(start_pose)
        se3_stop_pose = se3.from_ndarray(stop_pose)
        
        for i in range(n_poses):
            t = (scan.timestamp[i] - scan.timestamp[0]) / (scan.timestamp[-1]-scan.timestamp[0])
            interpolated_pose = se3.ndarray(se3.interpolate(se3_start_pose, se3_stop_pose, t))
            xyz[:,i,:] = (interpolated_pose[:3] @ np.vstack([xyz[:,i,:].T, np.ones(scanlines)])).T

    return xyz

def get_view(L0_points, rotomtx, K, D, height, width):

    L0_cloud = o3d.geometry.PointCloud()
    L0_cloud.points = o3d.utility.Vector3dVector(L0_points)

    Cn_cloud = L0_cloud.transform(rotomtx)

    z_limit = 250
    camera_fov = 64/180*np.pi # The ec FoV is 62, but we are taking a bit more
    xy_limit = z_limit * np.tan(camera_fov/2)

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-xy_limit, -xy_limit, 0.05),
                                               max_bound=(xy_limit, xy_limit, z_limit))
    Cn_cloud_crop = Cn_cloud.crop(bbox)

    diameter = np.linalg.norm(np.array(Cn_cloud_crop.get_max_bound()) -
                              np.array(Cn_cloud_crop.get_min_bound()))
    
    # https://github.com/isl-org/Open3D/blob/ff22900c958d0216c85305ee8b3841e5699f9d57/examples/python/geometry/point_cloud_hidden_point_removal.py#L24C4-L24C4
    hpr_radius = diameter * 100
    if hpr_radius > 0 and len(Cn_cloud_crop.points) > 4:
        _, pt_map = Cn_cloud_crop.hidden_point_removal(np.zeros((3,)), hpr_radius)
        Cn_cloud_HPR = Cn_cloud_crop.select_by_index(pt_map)

        camera_points = np.array(Cn_cloud_HPR.points)

        rvecs = np.zeros((3,1))
        tvecs = np.zeros((3,1))

        imgpts, _ = cv2.projectPoints(camera_points, rvecs, tvecs, K, D)

        imgpts = imgpts[:,0,:]
        valid_points = (imgpts[:, 1] >= 0) & (imgpts[:, 1] < height) & \
                    (imgpts[:, 0] >= 0) & (imgpts[:, 0] < width)
        imgpts = imgpts[valid_points,:]
        depth = camera_points[valid_points,2]


        image = np.zeros([height, width]) + np.inf
        image[imgpts[:,1].astype(int), imgpts[:,0].astype(int)] = depth

        image[ image==0.0 ] = np.inf

        return image

    return np.zeros([height, width]) + np.inf

def get_ts_mapping(ts_a, ts_b):
    mymap = {}

    for i,ts in enumerate(ts_a):
        t_b = 0
        prev_t_b = 0
        prev_k = 0
        current_k = 0
        for k,other_ts in enumerate(ts_b):
            prev_t_b = t_b
            t_b = other_ts
            prev_k = current_k
            current_k = k

            if t_b >= ts:
                if abs(t_b-ts) <= abs(prev_t_b-ts):
                    mymap[i] = current_k
                else:
                    mymap[i] = prev_k
                break

    if ts_b[-1] < ts_a[-1]:
        mymap[i] = len(ts_b)-1

    return mymap

def get_ts_pose(ts, poses_gt, ts_gt):
    if ts_gt[0] > ts:
        start_pose_a = se3.from_ndarray(poses_gt[0])
        start_pose_b = se3.from_ndarray(poses_gt[1])
        alpha = (ts-ts_gt[0]) / (ts_gt[1]-ts_gt[0])
        return se3.ndarray(se3.interpolate(start_pose_a, start_pose_b, alpha))
    
    if ts_gt[-1] < ts:
        start_pose_a = se3.from_ndarray(poses_gt[-2])
        start_pose_b = se3.from_ndarray(poses_gt[-1])
        alpha = (ts-ts_gt[-2]) / (ts_gt[-1]-ts_gt[-2])
        return se3.ndarray(se3.interpolate(start_pose_a, start_pose_b, alpha))

    start_pose_a = None
    prev_t = ts_gt[0]
    current_t = ts_gt[0]
    prev_pose = poses_gt[0]
    current_pose = poses_gt[0]

    for i in range(len(ts_gt)):
        t, pose = ts_gt[i], poses_gt[i]
        prev_t = current_t
        prev_pose = current_pose
        current_t = t
        current_pose = pose

        if current_t > ts:
            start_pose_a = prev_pose
            t_start_a = prev_t
            start_pose_b = current_pose
            t_start_b = current_t
            break

    if start_pose_a is None:
        start_pose_a = prev_pose
        t_start_a = prev_t
        start_pose_b = current_pose
        t_start_b = current_t

    start_pose_a = se3.from_ndarray(start_pose_a)
    start_pose_b = se3.from_ndarray(start_pose_b)
    alpha = (ts-t_start_a) / (t_start_b-t_start_a)

    return se3.ndarray(se3.interpolate(start_pose_a, start_pose_b, alpha))

def get_start_stop_poses(t_start, t_stop, poses_gt, ts_gt):
    if ts_gt[-1] <= t_start:
        t_a = ts_gt[-2]
        t_b = ts_gt[-1]
        pose_a = se3.from_ndarray(poses_gt[-2])
        pose_b = se3.from_ndarray(poses_gt[-1])
        alpha_start = (t_start-t_a) / (t_b-t_a)
        alpha_stop = (t_stop-t_a) / (t_b-t_a)

        start_pose = se3.ndarray(se3.interpolate(pose_a, pose_b, alpha_start))
        stop_pose = se3.ndarray(se3.interpolate(pose_a, pose_b, alpha_stop))

        return start_pose, stop_pose

    
    if ts_gt[0] >= t_stop:
        t_a = ts_gt[0]
        t_b = ts_gt[1]
        pose_a = se3.from_ndarray(poses_gt[0])
        pose_b = se3.from_ndarray(poses_gt[1]) 

        alpha_start = (t_start-t_a) / (t_b-t_a)
        alpha_stop = (t_stop-t_a) / (t_b-t_a)

        start_pose = se3.ndarray(se3.interpolate(pose_a, pose_b, alpha_start))
        stop_pose = se3.ndarray(se3.interpolate(pose_a, pose_b, alpha_stop))

        return start_pose, stop_pose

    if ts_gt[0] >= t_start:
        t_start_a = ts_gt[0]
        t_start_b = ts_gt[1]
        start_pose_a = poses_gt[0]
        start_pose_b = poses_gt[1]    
        i = 1
    else:
        prev_t = ts_gt[0]
        current_t = ts_gt[0]
        prev_pose = poses_gt[0]
        current_pose = poses_gt[0]
        
        for i in range(len(ts_gt)):
            t, pose = ts_gt[i], poses_gt[i]
            prev_t = current_t
            prev_pose = current_pose
            current_t = t
            current_pose = pose

            if current_t > t_start:
                start_pose_a = prev_pose
                t_start_a = prev_t
                start_pose_b = current_pose
                t_start_b = current_t
                break

    start_pose_a = se3.from_ndarray(start_pose_a)
    start_pose_b = se3.from_ndarray(start_pose_b)
    alpha = (t_start-t_start_a) / (t_start_b-t_start_a)

    start_pose = se3.ndarray(se3.interpolate(start_pose_a, start_pose_b, alpha))

    prev_t = ts_gt[i-1]
    current_t = ts_gt[i-1]
    prev_pose = poses_gt[i-1]
    current_pose = poses_gt[i-1]
    stop_pose_a = None

    for i in range(i,len(ts_gt)):
        t, pose = ts_gt[i], poses_gt[i]
        prev_t = current_t
        prev_pose = current_pose
        current_t = t
        current_pose = pose

        if current_t > t_stop:
            stop_pose_a = prev_pose
            t_stop_a = prev_t
            stop_pose_b = current_pose
            t_stop_b = current_t
            break

    if stop_pose_a is None:
        stop_pose_a = prev_pose
        t_stop_a = prev_t
        stop_pose_b = current_pose
        t_stop_b = current_t
    
    stop_pose_a = se3.from_ndarray(stop_pose_a)
    stop_pose_b = se3.from_ndarray(stop_pose_b)
    alpha = (t_stop-t_stop_a) / (t_stop_b-t_stop_a)

    stop_pose = se3.ndarray(se3.interpolate(stop_pose_a, stop_pose_b, alpha))

    return start_pose, stop_pose
    

def transform_inv(T):
    T_inv = np.eye(4)
    T_inv[:3,:3] = T[:3,:3].T
    T_inv[:3,3] = -1.0 * ( T_inv[:3,:3] @ T[:3,3] )
    return T_inv

def rectify(stereo_left_cam_matrix, stereo_left_dist, stereo_right_cam_matrix, stereo_right_dist, stereo_left_to_stereo_right, h, w):
    R1,R2,P1,P2,Q,_,_ = cv2.stereoRectify(stereo_left_cam_matrix, stereo_left_dist, stereo_right_cam_matrix, stereo_right_dist, (w,h), stereo_left_to_stereo_right[:3,:3], stereo_left_to_stereo_right[:3,3], flags=cv2.CALIB_ZERO_DISPARITY, alpha=1)
    leftmapX, leftmapY = cv2.initUndistortRectifyMap(stereo_left_cam_matrix, stereo_left_dist, R1, P1, (w,h), cv2.CV_32FC1)
    rightmapX, rightmapY = cv2.initUndistortRectifyMap(stereo_right_cam_matrix, stereo_right_dist, R2, P2, (w,h), cv2.CV_32FC1)

    leftmap = np.concatenate([np.expand_dims(leftmapX, -1), np.expand_dims(leftmapY, -1)], -1)
    rightmap = np.concatenate([np.expand_dims(rightmapX, -1), np.expand_dims(rightmapY, -1)], -1)

    return R1,R2,P1,P2, Q, leftmap, rightmap

#https://github.com/daniilidis-group/m3ed/blob/main/build_system/semantics/internimage.py
#https://stackoverflow.com/questions/41703210/inverting-a-real-valued-index-grid/68706787#68706787
#https://stackoverflow.com/questions/72635492/what-are-the-inaccuracies-of-this-inverse-map-function-in-opencv/72649764#72649764
def invert_map(F):
    # shape is (h, w, 2), an "xymap"
    (h, w) = F.shape[:2]
    I = np.zeros_like(F)
    I[:,:,1], I[:,:,0] = np.indices((h, w)) # identity map
    P = np.copy(I)
    for i in range(10):
        correction = I - cv2.remap(F, P, None, interpolation=cv2.INTER_LINEAR)
        P += correction * 0.5
    return P

def reproject_depth(W_start, H_start, K, depth, P_end, R_end, RT, W_end, H_end):
    R_end_4x4 = np.eye(4)
    R_end_4x4[:3,:3] = R_end

    xx, yy = np.meshgrid(np.arange(W_start), np.arange(H_start))
    points_grid = np.stack(((xx-K[0,2])/K[0,0], (yy-K[1,2])/K[1,1], np.ones_like(xx)), axis=0) * depth
    mask = np.ones((H_start, W_start), dtype=bool)
    mask[depth<=0] = False
    depth_pts = points_grid.transpose(1,2,0)[mask]

    camera_points = (R_end_4x4 @ RT @ np.vstack([depth_pts.T, np.ones(depth_pts.shape[0])])).T

    if camera_points.shape[0] == 0:
        return np.zeros([H_end, W_end])

    rvecs = np.zeros((3,1)) 
    tvecs = np.zeros((3,1))
    D_end = np.zeros((4,1))
    K_end = P_end[:3,:3]

    _camera_points = camera_points[:,:3]

    imgpts, _ = cv2.projectPoints(_camera_points, rvecs, tvecs, K_end, D_end)
    
    imgpts = imgpts[:,0,:]
    valid_points = (imgpts[:, 1] >= 0) & (imgpts[:, 1] < H_end) & \
                (imgpts[:, 0] >= 0) & (imgpts[:, 0] < W_end)
    imgpts = imgpts[valid_points,:]

    _end_depth = camera_points[valid_points,2]

    end_depth = np.zeros([H_end, W_end])
    end_depth[imgpts[:,1].astype(int), imgpts[:,0].astype(int)] = _end_depth

    return end_depth

@numba.njit
def forward_remap(disp: np.ndarray, map: np.ndarray):
    """
    disp: np.ndarray HxW
    map: np.ndarray HxWx2
    """
    re_disp = np.zeros_like(disp)
    H,W = disp.shape[:2]

    for y in range(H):
        for x in range(W):
            if disp[y,x] > 0:
                rx,ry = map[y,x]
                rx,ry = round(rx), round(ry)
                if 0 <= rx < W and 0 <= ry < H:
                    re_disp[ry,rx] = disp[y,x]
    
    return re_disp


def main(args):
    input_paths = args.input_folder
    output_path = args.output_folder
    delete = args.delete
    ignore_glob = args.ignore_glob
    ignore_h5 = args.ignore_h5
    threshold = args.threshold
    max_offset_us = args.max_offset_us
    offset_bins = args.offset_bins
    offset_gamma = args.offset_gamma

    offsets_us = set()
    offsets_us.add(0)

    for i in range(offset_bins):
        offsets_us.add(round((((i+1)/offset_bins)**offset_gamma)*max_offset_us))

    offsets_us = sorted(list(offsets_us))

    #Search for valid pair of data+depth_gt
    if ignore_glob:
        data_glob = input_paths
    else:
        data_glob = []
        for input_path in input_paths:
            data_glob += sorted(glob.glob(os.path.join(input_path, "*/*_data.h5")))

    data_list = []
    gt_list = []

    print("Checking for valid pairs...")

    for data_path in tqdm.tqdm(data_glob):
        gt_path = data_path.replace('_data.h5', '_depth_gt.h5')
        if os.path.exists(gt_path):
            data_list.append(data_path)
            gt_list.append(gt_path)

    print("Converting to DSEC style...")

    pbar = tqdm.tqdm(zip(data_list, gt_list), total=len(data_list))

    for data_path, gt_path in pbar:

        pbar.set_description(os.path.basename(data_path).replace('_data.h5', ''))

        data_file = h5py.File(data_path, 'r+')
        depth_file = h5py.File(gt_path, 'r+')

        basename = os.path.basename(data_path).replace("_data.h5", "")

        for dir in ["calibration", "disparity/event", "events/left", "events/right"]:
            os.makedirs(os.path.join(output_path, basename, dir), exist_ok=True)

        #Create left and right event data h5
        for cam in ['left', 'right']:
            pbar.set_description(f"Creating {cam} event history...")

            try:
                event_file = h5py.File(os.path.join(output_path, basename, f"events/{cam}/events.h5"), 'x')
                events_group = event_file.create_group("events")

                events_group.create_dataset("p", data=data_file[f'/prophesee/{cam}/p'], compression="lzf")
                pbar.set_description(f"Creating {cam} p event history... (20%)")

                events_group.create_dataset("t", data=data_file[f'/prophesee/{cam}/t'], compression="lzf")
                pbar.set_description(f"Creating {cam} t event history... (40%)")

                events_group.create_dataset("x", data=data_file[f'/prophesee/{cam}/x'], compression="lzf")
                pbar.set_description(f"Creating {cam} x event history... (60%)")

                events_group.create_dataset("y", data=data_file[f'/prophesee/{cam}/y'], compression="lzf")
                pbar.set_description(f"Creating {cam} y event history... (80%)")

                event_file.create_dataset("ms_to_idx", data=data_file[f'/prophesee/{cam}/ms_map_idx'], compression="lzf")
                event_file.create_dataset("t_offset", data=0)
                pbar.set_description(f"Creating {cam} event history... (100%)")

                event_file.close()
            except OSError as e:
                if not ignore_h5:
                    raise e
                if e.errno is not None:
                    myerrno = e.errno
                else:
                    mymsgs = str(e).split(",")
                    myerrno = -1
                    for mymsg in mymsgs:
                        mymsg = mymsg.replace(" ", "")
                        if re.match("^errno=[0-9]+$", mymsg):
                            myerrno = int(mymsg.split("=")[1])
                            break
                if myerrno != errno.EEXIST:
                    raise e

        #Get calibration files
        pbar.set_description(f"Creating calibration files...")

        """
        Intrinsics
        cam0: Event camera left
        cam1: Frame camera left
        cam2: Frame camera right
        cam3: Event camera right
        camRectX: Rectified version of camX. E.g. camRect0 is the rectified version of cam0.
        Extrinsics
        T_XY: Rigid transformation that transforms a point from the camY coordinate frame into the camX coordinate frame.
        R_rectX: Rotation that transforms a point from the camX coordinate frame into the camRectX coordinate frame.
        """

        K_event_left = np.eye(3)
        K_event_left[0, 0] = data_file["/prophesee/left/calib/intrinsics"][0]
        K_event_left[1, 1] = data_file["/prophesee/left/calib/intrinsics"][1]
        K_event_left[0, 2] = data_file["/prophesee/left/calib/intrinsics"][2]
        K_event_left[1, 2] = data_file["/prophesee/left/calib/intrinsics"][3]
        D_event_left = np.array(data_file['/prophesee/left/calib/distortion_coeffs'])

        K_event_right = np.eye(3)
        K_event_right[0, 0] = data_file["/prophesee/right/calib/intrinsics"][0]
        K_event_right[1, 1] = data_file["/prophesee/right/calib/intrinsics"][1]
        K_event_right[0, 2] = data_file["/prophesee/right/calib/intrinsics"][2]
        K_event_right[1, 2] = data_file["/prophesee/right/calib/intrinsics"][3]
        D_event_right = np.array(data_file['/prophesee/right/calib/distortion_coeffs'])

        event_right_to_event_left = np.array(data_file["/prophesee/right/calib/T_to_prophesee_left"])
        RT_event = transform_inv(event_right_to_event_left)

        R1_event, R2_event, P1_event, P2_event, Q_event, leftmap_event, rightmap_event = rectify(K_event_left, D_event_left, K_event_right, D_event_right, RT_event, 720, 1280)

        inv_leftmap_event = invert_map(leftmap_event)
        inv_rightmap_event = invert_map(rightmap_event)
        
        R1_event_4x4 = np.eye(4)
        R1_event_4x4[:3,:3] = R1_event

        R2_event_4x4 = np.eye(4)
        R2_event_4x4[:3,:3] = R2_event

        focal_event = abs(Q_event[2,3])
        baseline_event = abs(1/Q_event[3,2])

        K_gray_left = np.eye(3)
        K_gray_left[0, 0] = data_file["/ovc/left/calib/intrinsics"][0]
        K_gray_left[1, 1] = data_file["/ovc/left/calib/intrinsics"][1]
        K_gray_left[0, 2] = data_file["/ovc/left/calib/intrinsics"][2]
        K_gray_left[1, 2] = data_file["/ovc/left/calib/intrinsics"][3]
        D_gray_left = np.array(data_file['/ovc/left/calib/distortion_coeffs'])

        K_gray_right = np.eye(3)
        K_gray_right[0, 0] = data_file["/ovc/right/calib/intrinsics"][0]
        K_gray_right[1, 1] = data_file["/ovc/right/calib/intrinsics"][1]
        K_gray_right[0, 2] = data_file["/ovc/right/calib/intrinsics"][2]
        K_gray_right[1, 2] = data_file["/ovc/right/calib/intrinsics"][3]
        D_gray_right = np.array(data_file['/ovc/right/calib/distortion_coeffs'])

        gray_left_to_event_left = np.array(data_file["/ovc/left/calib/T_to_prophesee_left"])
        gray_right_to_event_left = np.array(data_file["/ovc/right/calib/T_to_prophesee_left"])
        RT_gray = transform_inv(gray_right_to_event_left) @ gray_left_to_event_left

        R1_gray, R2_gray, P1_gray, P2_gray, Q_gray, _, _ = rectify(K_gray_left, D_gray_left, K_gray_right, D_gray_right, RT_gray, 800, 1280)

        R1_gray_4x4 = np.eye(4)
        R1_gray_4x4[:3,:3] = R1_gray

        R2_gray_4x4 = np.eye(4)
        R2_gray_4x4[:3,:3] = R2_gray

        cam_to_cam = {}
        cam_to_cam['intrinsics'] = {}
        
        cam_to_cam['intrinsics']['cam0'] = {}
        cam_to_cam['intrinsics']['cam0']['camera_type'] = 'event'
        cam_to_cam['intrinsics']['cam0']['canera_location'] = 'left'
        cam_to_cam['intrinsics']['cam0']['is_rectified'] = False
        cam_to_cam['intrinsics']['cam0']['camera_model'] = 'pinhole'
        cam_to_cam['intrinsics']['cam0']['distortion_coeffs'] = D_event_left.tolist()
        cam_to_cam['intrinsics']['cam0']['distortion_model'] = 'radtan'
        cam_to_cam['intrinsics']['cam0']['resolution'] = [1280, 720]
        cam_to_cam['intrinsics']['cam0']['camera_matrix'] = np.array(data_file['/prophesee/left/calib/intrinsics']).tolist()

        cam_to_cam['intrinsics']['camRect0'] = {}
        cam_to_cam['intrinsics']['camRect0']['camera_type'] = 'event'
        cam_to_cam['intrinsics']['camRect0']['canera_location'] = 'left'
        cam_to_cam['intrinsics']['camRect0']['is_rectified'] = True
        cam_to_cam['intrinsics']['camRect0']['camera_model'] = 'pinhole'
        cam_to_cam['intrinsics']['camRect0']['resolution'] = [1280, 720]
        cam_to_cam['intrinsics']['camRect0']['camera_matrix'] = np.array([P1_event[0,0], P1_event[1,1], P1_event[0,2], P1_event[1,2]]).tolist()

        cam_to_cam['intrinsics']['cam1'] = {}
        cam_to_cam['intrinsics']['cam1']['camera_type'] = 'frame'
        cam_to_cam['intrinsics']['cam1']['canera_location'] = 'left'
        cam_to_cam['intrinsics']['cam1']['is_rectified'] = False
        cam_to_cam['intrinsics']['cam1']['T_cn_cnm1'] = transform_inv(gray_left_to_event_left).tolist()#cam0->cam1
        cam_to_cam['intrinsics']['cam1']['camera_model'] = 'pinhole'
        cam_to_cam['intrinsics']['cam1']['distortion_coeffs'] = D_gray_left.tolist()
        cam_to_cam['intrinsics']['cam1']['distortion_model'] = 'radtan'
        cam_to_cam['intrinsics']['cam1']['resolution'] = [1280, 800]
        cam_to_cam['intrinsics']['cam1']['camera_matrix'] = np.array(data_file['/ovc/left/calib/intrinsics']).tolist()

        cam_to_cam['intrinsics']['camRect1'] = {}
        cam_to_cam['intrinsics']['camRect1']['camera_type'] = 'frame'
        cam_to_cam['intrinsics']['camRect1']['canera_location'] = 'left'
        cam_to_cam['intrinsics']['camRect1']['is_rectified'] = True
        cam_to_cam['intrinsics']['camRect1']['camera_model'] = 'pinhole'
        cam_to_cam['intrinsics']['camRect1']['resolution'] = [1280, 800]
        cam_to_cam['intrinsics']['camRect1']['camera_matrix'] = np.array([P1_gray[0,0], P1_gray[1,1], P1_gray[0,2], P1_gray[1,2]]).tolist()

        cam_to_cam['intrinsics']['cam2'] = {}
        cam_to_cam['intrinsics']['cam2']['camera_type'] = 'frame'
        cam_to_cam['intrinsics']['cam2']['canera_location'] = 'right'
        cam_to_cam['intrinsics']['cam2']['is_rectified'] = False
        cam_to_cam['intrinsics']['cam2']['T_cn_cnm1'] = (transform_inv(gray_right_to_event_left) @ gray_left_to_event_left).tolist()#cam1 -> cam2
        cam_to_cam['intrinsics']['cam2']['camera_model'] = 'pinhole'
        cam_to_cam['intrinsics']['cam2']['distortion_coeffs'] = D_gray_right.tolist()
        cam_to_cam['intrinsics']['cam2']['distortion_model'] = 'radtan'
        cam_to_cam['intrinsics']['cam2']['resolution'] = [1280, 800]
        cam_to_cam['intrinsics']['cam2']['camera_matrix'] = np.array(data_file['/ovc/right/calib/intrinsics']).tolist()

        cam_to_cam['intrinsics']['camRect2'] = {}
        cam_to_cam['intrinsics']['camRect2']['camera_type'] = 'frame'
        cam_to_cam['intrinsics']['camRect2']['canera_location'] = 'right'
        cam_to_cam['intrinsics']['camRect2']['is_rectified'] = True
        cam_to_cam['intrinsics']['camRect2']['camera_model'] = 'pinhole'
        cam_to_cam['intrinsics']['camRect2']['resolution'] = [1280, 800]
        cam_to_cam['intrinsics']['camRect2']['camera_matrix'] = np.array([P2_gray[0,0], P2_gray[1,1], P2_gray[0,2], P2_gray[1,2]]).tolist()


        cam_to_cam['intrinsics']['cam3'] = {}
        cam_to_cam['intrinsics']['cam3']['camera_type'] = 'event'
        cam_to_cam['intrinsics']['cam3']['canera_location'] = 'right'
        cam_to_cam['intrinsics']['cam3']['is_rectified'] = False
        cam_to_cam['intrinsics']['cam3']['T_cn_cnm1'] = (transform_inv(event_right_to_event_left) @ gray_right_to_event_left).tolist()#cam2 -> cam3
        cam_to_cam['intrinsics']['cam3']['camera_model'] = 'pinhole'
        cam_to_cam['intrinsics']['cam3']['distortion_coeffs'] = D_event_right.tolist()
        cam_to_cam['intrinsics']['cam3']['distortion_model'] = 'radtan'
        cam_to_cam['intrinsics']['cam3']['resolution'] = [1280, 720]
        cam_to_cam['intrinsics']['cam3']['camera_matrix'] = np.array(data_file['/prophesee/right/calib/intrinsics']).tolist()

        cam_to_cam['intrinsics']['camRect3'] = {}
        cam_to_cam['intrinsics']['camRect3']['camera_type'] = 'event'
        cam_to_cam['intrinsics']['camRect3']['canera_location'] = 'right'
        cam_to_cam['intrinsics']['camRect3']['is_rectified'] = True
        cam_to_cam['intrinsics']['camRect3']['camera_model'] = 'pinhole'
        cam_to_cam['intrinsics']['camRect3']['resolution'] = [1280, 720]
        cam_to_cam['intrinsics']['camRect3']['camera_matrix'] = np.array([P2_event[0,0], P2_event[1,1], P2_event[0,2], P2_event[1,2]]).tolist()


        cam_to_cam['extrinsics'] = {}
        cam_to_cam['extrinsics']['T_10'] = transform_inv(gray_left_to_event_left).tolist()
        cam_to_cam['extrinsics']['T_21'] = (transform_inv(gray_right_to_event_left) @ gray_left_to_event_left).tolist()
        cam_to_cam['extrinsics']['T_32'] = (transform_inv(event_right_to_event_left) @ gray_right_to_event_left).tolist()

        cam_to_cam['extrinsics']['R_rect0'] = R1_event.tolist()
        cam_to_cam['extrinsics']['R_rect1'] = R1_gray.tolist()
        cam_to_cam['extrinsics']['R_rect2'] = R2_gray.tolist()
        cam_to_cam['extrinsics']['R_rect3'] = R2_event.tolist()

        cam_to_cam['disparity_to_depth'] = {}
        cam_to_cam['disparity_to_depth']['cams_03'] = Q_event.tolist()
        cam_to_cam['disparity_to_depth']['cams_12'] = Q_gray.tolist()

        #Save cam_to_cam.yaml
        with open(os.path.join(output_path, basename, "calibration/cam_to_cam.yaml"), "w") as f:
            yaml.dump(cam_to_cam, f)

        cam_to_lidar = {}

        cam_to_lidar['T_lidar_cam0'] = transform_inv(np.array(data_file["/ouster/calib/T_to_prophesee_left"])).tolist()
        cam_to_lidar['T_lidar_camRect0'] = transform_inv(R1_event_4x4 @ np.array(data_file["/ouster/calib/T_to_prophesee_left"])).tolist()
        
        cam_to_lidar['T_lidar_cam1'] = transform_inv(np.array(transform_inv(gray_left_to_event_left) @ data_file["/ouster/calib/T_to_prophesee_left"])).tolist()
        cam_to_lidar['T_lidar_camRect1'] = transform_inv(R1_gray_4x4 @ transform_inv(gray_left_to_event_left) @ np.array(data_file["/ouster/calib/T_to_prophesee_left"])).tolist()
        
        cam_to_lidar['T_lidar_cam2'] = transform_inv(transform_inv(gray_right_to_event_left) @ np.array(data_file["/ouster/calib/T_to_prophesee_left"])).tolist()
        cam_to_lidar['T_lidar_camRect2'] = transform_inv(R2_gray_4x4 @ transform_inv(gray_right_to_event_left) @ np.array(data_file["/ouster/calib/T_to_prophesee_left"])).tolist()
        
        cam_to_lidar['T_lidar_cam3'] = transform_inv(np.array(transform_inv(event_right_to_event_left) @ data_file["/ouster/calib/T_to_prophesee_left"])).tolist()
        cam_to_lidar['T_lidar_camRect3'] = transform_inv(R2_event_4x4 @ transform_inv(event_right_to_event_left) @ np.array(data_file["/ouster/calib/T_to_prophesee_left"])).tolist()

        #Fake data
        cam_to_lidar['calibration_details'] = {}
        cam_to_lidar['calibration_details']['image_indices'] = [0,0]
        cam_to_lidar['calibration_details']['n_after'] = 0
        cam_to_lidar['calibration_details']['n_before'] = 0
        cam_to_lidar['calibration_details']['sequence'] = '1970-01-01-00-00-00'
        cam_to_lidar['calibration_details']['temporal_offset_us'] = 0

        #Save cam_to_lidar.yaml
        with open(os.path.join(output_path, basename, "calibration/cam_to_lidar.yaml"), "w") as f:
            yaml.dump(cam_to_lidar, f)

        pbar.set_description(f"Creating event rectification maps files...")

        #Create left and right rectification maps
        for cam, map in zip(['left', 'right'], [inv_leftmap_event, inv_rightmap_event]):
            rect_file = h5py.File(os.path.join(output_path, basename, f"events/{cam}/rectify_map.h5"), 'w')
            rect_file.create_dataset("rectify_map", data=map.astype(np.float32), compression='lzf')
            rect_file.close()

        #Save groundtruth disparity values.

        ts_gt = np.array(depth_file['ts'])
        ts_start = np.array(data_file['ouster']['ts_start'])
        ts_stop = np.array(data_file['ouster']['ts_end'])
        mymap_lidar = get_ts_mapping(ts_gt, ts_start)
        
        event_poses_gt = np.array(depth_file['Cn_T_C0'])      
        lidar_poses_gt = np.array(depth_file['Ln_T_L0'])

        RT_lidar = np.array(data_file["/ouster/calib/T_to_prophesee_left"])
        n_imgs = len(depth_file['depth']['prophesee']['left'])

        timestamps_event = []
        
        maes = []
        filtered_maes = []
        mae = -1

        disp_factor_dict = {}

        for offset_us in offsets_us:
            k = 0
            
            for i in range(n_imgs):
                pbar.set_description(f"Creating event groundtruth disparity files (offset: {offset_us} us) mae: {mae} ({i+1}/{n_imgs}) ...")

                depth_i = depth_file['depth']['prophesee']['left'][i]
                depth_i[np.isinf(depth_i)] = 0
                depth_i = forward_remap(depth_i, inv_leftmap_event)

                t_start = int(ts_start[mymap_lidar[i]])
                t_stop = int(ts_stop[mymap_lidar[i]])

                #Interpolate start and stop poses relative to lidar start scan and stop scan timestamps
                start_pose, stop_pose = get_start_stop_poses(t_start, t_stop, lidar_poses_gt, ts_gt)
                                
                #Velocity-aware raw lidar:
                #Ouster lidar gives a HxW raw data where each column correspond to a specific timestamp.
                #Intermediate poses are interpolated using column timestamp and start and stop poses.
                raw_pcd = get_cloud(data_file, mymap_lidar[i], transform_inv(start_pose), transform_inv(stop_pose)).reshape(-1, 3)

                #After correction, move pcd from origin to correleted gt pose
                pose_gt = transform_inv(get_ts_pose(ts_gt[i]-offset_us, lidar_poses_gt, ts_gt))

                raw_depth_i = get_view(raw_pcd, RT_lidar @ transform_inv(pose_gt), K_event_left, D_event_left, 720, 1280)# Ln -> L0 -> Ln
                raw_depth_i[np.isinf(raw_depth_i)] = 0
                raw_depth_i = forward_remap(raw_depth_i, inv_leftmap_event)

                #Save also groundtruth with offset 
                offset_depth_i = depth_i.copy()
                event_pose_gt = transform_inv(get_ts_pose(ts_gt[i]-offset_us, event_poses_gt, ts_gt))
                offset_depth_i = reproject_depth(1280, 720, P1_event[:3,:3], offset_depth_i, P1_event, np.eye(3), transform_inv(event_pose_gt) @ transform_inv(event_poses_gt[i]), 1280, 720)

                if offset_us == 0:
                    mask = np.logical_and(raw_depth_i>0, depth_i>0)
                    errormap = np.abs(raw_depth_i-depth_i)
                    errormap[np.logical_not(mask)] = 0
                    errormap[np.isnan(errormap)] = 0
                    mae = np.nanmean(errormap[mask])
                    maes.append(mae)
                else:
                    mae = maes[i]

                #check if MAE is not so big
                if threshold <= 0 or maes[i] <= threshold:
                    if offset_us == 0:
                        timestamps_event.append(int(ts_gt[i]))
                        filtered_maes.append(mae)
                        #Convert depth to disparity and save with 6 zeros digits
                        disp_i = depth_i.copy()
                        disp_i[disp_i>0] = (focal_event*baseline_event) / disp_i[disp_i>0]

                        quantization = 65535 / np.max(disp_i)
                        if 'event' not in disp_factor_dict:
                            disp_factor_dict['event'] = {}
                        disp_factor_dict['event'][f'{k:06d}.png'] = quantization.item()

                        cv2.imwrite(os.path.join(output_path, basename, f"disparity/event/{k:06d}.png"), np.clip(disp_i*quantization,0,65535).astype(np.uint16))

                    raw_disp_i = raw_depth_i.copy()
                    raw_disp_i[raw_disp_i>0] = (focal_event*baseline_event) / raw_disp_i[raw_disp_i>0]

                    quantization = 65535 / np.max(raw_disp_i)
                    if f'event_raw_{offset_us}' not in disp_factor_dict:
                        disp_factor_dict[f'event_raw_{offset_us}'] = {}
                    disp_factor_dict[f'event_raw_{offset_us}'][f'{k:06d}.png'] = quantization.item()

                    os.makedirs(os.path.join(output_path, basename, f"disparity/event_raw_{offset_us}"), exist_ok=True)
                    cv2.imwrite(os.path.join(output_path, basename, f"disparity/event_raw_{offset_us}/{k:06d}.png"), np.clip(raw_disp_i*quantization,0,65535).astype(np.uint16))

                    offset_disp_i = offset_depth_i.copy()
                    offset_disp_i[offset_disp_i>0] = (focal_event*baseline_event) / offset_disp_i[offset_disp_i>0]

                    quantization = 65535 / np.max(offset_disp_i)
                    if f'event_{offset_us}' not in disp_factor_dict:
                        disp_factor_dict[f'event_{offset_us}'] = {}
                    disp_factor_dict[f'event_{offset_us}'][f'{k:06d}.png'] = quantization.item()

                    os.makedirs(os.path.join(output_path, basename, f"disparity/event_{offset_us}"), exist_ok=True)
                    cv2.imwrite(os.path.join(output_path, basename, f"disparity/event_{offset_us}/{k:06d}.png"), np.clip(offset_disp_i*quantization,0,65535).astype(np.uint16))

                    k+=1
  
        with open(os.path.join(output_path, basename, "disparity/disp_factor.yaml"), "w") as f:
            yaml.dump(disp_factor_dict, f)

        np.savetxt(os.path.join(output_path, basename, "disparity/raw_event_mae.txt"), np.array(maes))
        np.savetxt(os.path.join(output_path, basename, "disparity/raw_event_mae_filtered.txt"), np.array(filtered_maes))

        plt.figure()
        plt.bar([i for i in range(len(maes))], maes)
        plt.savefig(os.path.join(output_path, basename, "disparity/raw_event_mae.png"))
        plt.close()

        plt.figure()
        plt.bar([i for i in range(len(filtered_maes))], filtered_maes)
        plt.savefig(os.path.join(output_path, basename, "disparity/raw_event_mae_filtered.png"))
        plt.close()

        np.savetxt(os.path.join(output_path, basename, "disparity/timestamps.txt"), np.array(timestamps_event).astype(np.uint64), fmt='%i')

        data_file.close()
        depth_file.close()

        if delete:
            os.remove(data_path)
            os.remove(gt_path)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="M3ED Converter into DSEC format"
    )

    parser.add_argument(
        "-i",
        "--input_folder",
        required=True,
        nargs='+',
        help="""Input folder containinbg a list of folder and for each folder a p\
                air of *_data.h5 and *_depth_gt.h5"""
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        required=True,
        help="""Output folder where modified files will be moved from original source folder"""
    )
    parser.add_argument(
        "-d",
        "--delete",
        action='store_true',
        help="""Delete original files"""
    )
    parser.add_argument(
        "--offset_gamma",
        type=float,
        default=2.2,
        help="""Choose offset gamma of disparities with lidar captured before groundtruth."""
    )
    parser.add_argument(
        "--max_offset_us",
        type=int,
        default=0,
        help="""Choose max offset of disparities with lidar captured before groundtruth."""
    )
    parser.add_argument(
        "--offset_bins",
        type=int,
        default=10,
        help="""Choose length of the series of disparities with lidar captured before groundtruth."""
    )
    parser.add_argument(
        "--ignore_glob",
        action='store_true',
        help="""Ignore glob"""
    )
    parser.add_argument(
        "--ignore_h5",
        action='store_true',
        help="""Ignore h5 exceptions"""
    )
    parser.add_argument(
        '--threshold',
        type=float,
        help="""Use a threshold to ignore frames with high difference \
                between GT and raw (meters)""",
        default=0.0
    )


    args = parser.parse_args()

    main(args)