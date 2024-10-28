import os
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET

import torch
import numpy as np
import json
import open3d

from typing import Dict, List, Tuple

from typing import List, Tuple
import numpy as np
from transforms3d.quaternions import quat2mat, qinverse, rotate_vector, qmult


def project(pts: np.ndarray, K: np.ndarray, img_size: List[int] or Tuple[int] = None) -> np.ndarray:
    """Projects 3D points to image plane.

    Args:
        - pts [N, 3/4]: points in camera coordinates (homogeneous or non-homogeneous)
        - K [3, 3]: intrinsic matrix
        - img_size (width, height): optional, clamp projection to image borders
        Outputs:
        - uv [N, 2]: coordinates of projected points
    """

    assert len(pts.shape) == 2, 'incorrect number of dimensions'
    assert pts.shape[1] in [3, 4], 'invalid dimension size'
    assert K.shape == (3, 3), 'incorrect intrinsic shape'

    uv_h = (K @ pts[:, :3].T).T
    uv = uv_h[:, :2] / uv_h[:, -1:]

    if img_size is not None:
        uv[:, 0] = np.clip(uv[:, 0], 0, img_size[0])
        uv[:, 1] = np.clip(uv[:, 1], 0, img_size[1])

    return uv


def get_grid_multipleheight() -> np.ndarray:
    # create grid of points
    ar_grid_step = 0.3
    ar_grid_num_x = 7
    ar_grid_num_y = 4
    ar_grid_num_z = 7
    ar_grid_z_offset = 1.8
    ar_grid_y_offset = 0

    ar_grid_x_pos = np.arange(0, ar_grid_num_x)-(ar_grid_num_x-1)/2
    ar_grid_x_pos *= ar_grid_step

    ar_grid_y_pos = np.arange(0, ar_grid_num_y)-(ar_grid_num_y-1)/2
    ar_grid_y_pos *= ar_grid_step
    ar_grid_y_pos += ar_grid_y_offset

    ar_grid_z_pos = np.arange(0, ar_grid_num_z).astype(float)
    ar_grid_z_pos *= ar_grid_step
    ar_grid_z_pos += ar_grid_z_offset

    xx, yy, zz = np.meshgrid(ar_grid_x_pos, ar_grid_y_pos, ar_grid_z_pos)
    ones = np.ones(xx.shape[0]*xx.shape[1]*xx.shape[2])
    eye_coords = np.concatenate([c.reshape(-1, 1)
                                for c in (xx, yy, zz, ones)], axis=-1)
    return eye_coords


# global variable, avoids creating it again
eye_coords_glob = get_grid_multipleheight()


def reprojection_error(
        q_est: np.ndarray, t_est: np.ndarray, q_gt: np.ndarray, t_gt: np.ndarray, K: np.ndarray,
        W: int, H: int) -> float:
    eye_coords = eye_coords_glob

    # obtain ground-truth position of projected points
    uv_gt = project(eye_coords, K, (W, H))

    # residual transformation
    cam2w_est = np.eye(4)
    cam2w_est[:3, :3] = quat2mat(q_est)
    cam2w_est[:3, -1] = t_est
    cam2w_gt = np.eye(4)
    cam2w_gt[:3, :3] = quat2mat(q_gt)
    cam2w_gt[:3, -1] = t_gt

    # residual reprojection
    eyes_residual = (np.linalg.inv(cam2w_est) @ cam2w_gt @ eye_coords.T).T
    uv_pred = project(eyes_residual, K, (W, H))

    # get reprojection error
    repr_err = np.linalg.norm(uv_gt - uv_pred, ord=2, axis=1)
    mean_repr_err = float(repr_err.mean().item())
    return mean_repr_err


def convert_world2cam_to_cam2world(q, t):
    qinv = qinverse(q)
    tinv = -rotate_vector(t, qinv)
    return qinv, tinv


def KRK_inv(K, R, t=None):
    """Assuming there is only rotation and no translation between the cameras
    """
    K_inv = np.linalg.inv(K)
    if isinstance(t, np.ndarray):
        n = np.array([0, 0, 1])
        return K.dot(R.dot(K_inv) - np.outer(t, n))
    else:
        return K @ R @ K_inv


def backproject_3d(uv, depth, K):
    '''
    Backprojects 2d points given by uv coordinates into 3D using their depth values and intrinsic K
    :param uv: array [B, N, 2]
    :param depth: array [B, N, 1]
    :param K: array [B, 3, 3]
    :return: xyz: array [B, N, 3]
    '''

    B, num_corr, _ = uv.shape

    ones_vector = torch.ones((B, num_corr, 1)).to(uv.device)
    uv1 = torch.cat([uv, ones_vector], dim=-1)
    xyz = depth * (torch.linalg.inv(K) @ uv1.transpose(2, 1)).transpose(2, 1)

    return xyz


def project_2d(XYZ, K):
    '''
    Backprojects 3d points given by XYZ coordinates into 2D using their depth values and intrinsic K
    XYZ - Size: B, n, 3
    '''

    B, num_corr, _ = XYZ.shape

    xyz_cam = (K @ XYZ.transpose(2, 1)).transpose(2, 1)
    xy = xyz_cam / (xyz_cam[:, :, 2].view(B, num_corr, 1)+1e-16)

    return xy[:, :, :2]


def world2cam():
    pass

def cam2world():
    pass

def cam2pixel():
    pass

def pixel2cam():
    pass

def pixel2world():
    pass

def world2pixel():
    pass


def xml2dict(element):
    # Convert an XML element and its children to a dictionary
    result = {}
    
    # Add element's attributes to the dictionary
    for key, value in element.attrib.items():
        result[f'@{key}'] = value
    
    # Add element's children to the dictionary
    for child in element:
        child_result = xml2dict(child)
        if child.tag in result:
            # If there's already an entry for this tag, append to the list
            if type(result[child.tag]) is list:
                result[child.tag].append(child_result)
            else:
                result[child.tag] = [result[child.tag], child_result]
        else:
            result[child.tag] = child_result
    
    # Add element's text content to the dictionary (if any)
    text = element.text.strip() if element.text else ''
    if text:
        if result:
            result['#text'] = text
        else:
            result = text
    
    return result

def parse_xml(file_path) -> Dict:
    # Parse XML file and convert it to a dictionary
    tree = ET.parse(file_path)
    root = tree.getroot()
    return {root.tag: xml2dict(root)}

def get_bboxes(detection):
    bboxes = []
    if 'object' in detection['annotation']:
        for obj in detection['annotation']['object']:
            xmin = int(obj['bndbox']['xmin'])
            xmax = int(obj['bndbox']['xmax'])
            ymin = int(obj['bndbox']['ymin'])
            ymax = int(obj['bndbox']['ymax'])
            bboxes.append((xmin,ymin,xmax,ymax))
    return bboxes


def draw_bounding_boxes(
    image_path,
    output_path,
    bboxes,
    color='red',
    width=3,
    fill_color=None,
):
    """
    Draw bounding boxes on an image and save the output.

    Parameters:
    - image_path (str): Path to the input image.
    - output_path (str): Path to save the output image with bounding boxes.
    - bboxes (list of tuples): List of bounding boxes, each represented as a tuple (xmin, ymin, xmax, ymax).
    """
    # Open an image file
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
        
        # Draw each bounding box
        for bbox in bboxes:
            draw.rectangle(bbox, outline=color, width=width, fill=fill_color)
        
        # Save the output image
        img.save(output_path)
        print(f"Saved image with bounding boxes to {output_path}")


def draw_camera(K, R, t, w, h, scale=1, color=[0.8, 0.2, 0.8]):
    """Create axis, plane and pyramed geometries in Open3D format.
    :param K: calibration matrix (camera intrinsics)
    :param R: rotation matrix
    :param t: translation
    :param w: image width
    :param h: image height
    :param scale: camera model scale
    :param color: color of the image plane and pyramid lines
    :return: camera model geometries (axis, plane and pyramid)
    """

    # intrinsics
    K = K.copy() / scale
    Kinv = np.linalg.inv(K)

    # 4x4 transformation
    T = np.column_stack((R, t))
    T = np.vstack((T, (0, 0, 0, 1)))

    # axis
    axis = open3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5 * scale
    )
    axis.transform(T)

    # points in pixel
    points_pixel = [
        [0, 0, 0],
        [0, 0, 1],
        [w, 0, 1],
        [0, h, 1],
        [w, h, 1],
    ]

    # pixel to camera coordinate system
    points = [Kinv @ p for p in points_pixel]

    # image plane
    width = abs(points[1][0]) + abs(points[3][0])
    height = abs(points[1][1]) + abs(points[3][1])
    plane = open3d.geometry.TriangleMesh.create_box(width, height, depth=1e-6)
    plane.paint_uniform_color(color)
    plane.translate([points[1][0], points[1][1], scale])
    plane.transform(T)

    # pyramid
    points_in_world = [(R @ p + t) for p in points]
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
    ]
    colors = [color for i in range(len(lines))]
    line_set = open3d.geometry.LineSet(
        points=open3d.utility.Vector3dVector(points_in_world),
        lines=open3d.utility.Vector2iVector(lines),
    )
    line_set.colors = open3d.utility.Vector3dVector(colors)

    # return as list in Open3D format
    return [axis, plane, line_set]

if __name__ == '__main__':
    img1 = 'debug/cam_pose_transform/frame_00040.png'
    xml1 = 'debug/cam_pose_transform/frame_00040.xml'
    img2 = 'debug/cam_pose_transform/frame_00060.png'
    xml2 = 'debug/cam_pose_transform/frame_00060.xml'
    # img1 = 'debug/cam_pose_transform/frame_00001.png'
    # xml1 = 'debug/cam_pose_transform/frame_00001.xml'
    # img2 = 'debug/cam_pose_transform/frame_00020.png'
    # xml2 = 'debug/cam_pose_transform/frame_00020.xml'
    alg_res1 = parse_xml(xml1)
    alg_res2 = parse_xml(xml2)
    boxes1 = get_bboxes(alg_res1)
    boxes2 = get_bboxes(alg_res2)

    rendered_img = f"debug/rendered/{os.path.basename(img2)}"
    draw_bounding_boxes(img2, rendered_img, boxes1, color='red')
    draw_bounding_boxes(rendered_img, rendered_img, boxes2, color='blue')

    transform_file_path = '/home/qizhinas01/yixu.cui/data/stitching/for_nerf/cooked/scene_heap/transforms.json'
    with open(transform_file_path, 'r') as fin:
        all_infos = json.load(fin)
        K = [
            [all_infos['fl_x'], 0, all_infos['cx']],
            [0, all_infos['fl_y'], all_infos['cy']],
            [0, 0, 1],
        ]
        K = np.array(K)
        all_frames = all_infos['frames']
        R1, R2 = None, None
        for frame in all_frames:
            if frame['colmap_im_id'] == 40:
                R1 = [
                    frame['transform_matrix'][0][:3],
                    frame['transform_matrix'][1][:3],
                    frame['transform_matrix'][2][:3],
                ]
                t1 = [
                    frame['transform_matrix'][0][3],
                    frame['transform_matrix'][1][3],
                    frame['transform_matrix'][2][3],
                ]
            elif frame['colmap_im_id'] == 60:
                R2 = [
                    frame['transform_matrix'][0][:3],
                    frame['transform_matrix'][1][:3],
                    frame['transform_matrix'][2][:3],
                ]
                t2 = [
                    frame['transform_matrix'][0][3],
                    frame['transform_matrix'][1][3],
                    frame['transform_matrix'][2][3],
                ]
            if isinstance(R1, list) and isinstance(R2, list):
                R1 = np.array(R1)
                R2 = np.array(R2)
                t1 = np.array(t1)
                t2 = np.array(t2)
                R_1to2 = np.dot(R2, np.linalg.inv(R1))
                t_1to2 = t2 - np.dot(R_1to2, t1)
                H = KRK_inv(K, R_1to2)
                print(H)
                boxes1 = np.array(boxes1)
                z = np.ones((boxes1.shape[0], 1))
                boxes1 = np.insert(boxes1,(2, 4), z, axis=1)
                boxes1_lt = boxes1[:, :3] @ H
                boxes1_rd = boxes1[:, 3:] @ H
                # boxes1 = np.c_[boxes1_lt[:, :2], boxes1_rd[:, :2]].tolist()
                boxes1 = np.append(boxes1_lt[:, :2], boxes1_rd[:, :2], axis=1).tolist()
                draw_bounding_boxes(
                    rendered_img, rendered_img, boxes1, color='yellow'
                )

                H = KRK_inv(K, R_1to2, t_1to2)
                boxes1 = get_bboxes(alg_res1)
                boxes1 = np.array(boxes1)
                z = np.ones((boxes1.shape[0], 1))
                boxes1 = np.insert(boxes1,(2, 4), z, axis=1)
                boxes1_lt = boxes1[:, :3] @ H - t_1to2
                boxes1_rd = boxes1[:, 3:] @ H - t_1to2
                # boxes1_lt = boxes1[:, :3] @ H
                # boxes1_rd = boxes1[:, 3:] @ H
                # boxes1 = np.c_[boxes1_lt[:, :2], boxes1_rd[:, :2]].tolist()
                boxes1 = np.append(
                    # np.maximum(boxes1_lt[:, :2] / np.abs(boxes1_lt[:, [2]]), 0),
                    # np.maximum(boxes1_rd[:, :2] / np.abs(boxes1_rd[:, [2]]), 0),
                    boxes1_lt[:, :2],
                    boxes1_rd[:, :2],
                    axis=1,
                ).tolist()
                draw_bounding_boxes(
                    rendered_img, rendered_img, boxes1, color='green', width=2,
                )

