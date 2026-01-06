import os
import json
import math
import glob
import numpy as np
from magnum import Vector3, Quaternion, Rad
from scipy.spatial.transform import Rotation


def get_glb_path(group, scene):
    """根据group和scene获取glb路径"""
    # 处理 group 中的数字部分 (如 hm3d_14)
    group_parts = group.split('_')
    dataset_type = group_parts[0]

    if dataset_type == 'mp3d':
        return f"/data1/liuy/scene_datasets/mp3d/{scene}/{scene}.glb"
    elif dataset_type == 'gibson':
        return f"/data1/liuy/scene_datasets/gibson/{scene}.glb"
    elif dataset_type == 'hm3d':
        base_path = f"/data1/liuy/scene_datasets/hm3d"

        if not os.path.exists(base_path):
            raise FileNotFoundError(f"HM3D数据集路径不存在: {base_path}")

        matching_folders = glob.glob(os.path.join(base_path, f"*{scene}*"))

        if not matching_folders:
            raise FileNotFoundError(f"在 {base_path} 中找不到包含 '{scene}' 的文件夹")

        target_folder = None
        for folder in matching_folders:
            folder_name = os.path.basename(folder)
            if scene in folder_name:
                target_folder = folder
                break

        if not target_folder:
            raise FileNotFoundError(f"在 {base_path} 中找不到包含 '{scene}' 的文件夹")

        glb_file = os.path.join(target_folder, f"{scene}.basis.glb")

        if not os.path.exists(glb_file):
            possible_files = glob.glob(os.path.join(target_folder, f"*{scene}*.glb"))
            if possible_files:
                glb_file = possible_files[0]
            else:
                raise FileNotFoundError(f"在 {target_folder} 中找不到 {scene}.basis.glb 或类似文件")

        return glb_file
    elif dataset_type == 'replica':
        return f"/data1/liuy/scene_datasets/replica/{scene}/habitat/mesh_preseg_semantic.ply"
    else:
        raise ValueError(f"未知的group类型: {group}")


def normalize_angle(angle):
    """将角度标准化到0-360度范围"""
    angle = angle % (2*np.pi)
    if angle < 0:
        angle += 2*np.pi
    return angle


def is_success(current_coords, target_coords, pos_threshold=0.1, angle_threshold=np.pi/4):
    """
    检查是否成功到达目标位置
    current_coords: [x, y, z, angle]
    target_coords: [x, y, z, angle]
    """
    if len(current_coords) < 4 or len(target_coords) < 4:
        return False

    # 计算位置距离
    pos_distance = math.sqrt(
        (current_coords[0] - target_coords[0]) ** 2 +
        (current_coords[1] - target_coords[1]) ** 2 +
        (current_coords[2] - target_coords[2]) ** 2
    )

    # 计算角度差异
    current_angle = normalize_angle(current_coords[3])
    target_angle = normalize_angle(target_coords[3])
    angle_diff = min(
        abs(current_angle - target_angle),
        2* np.pi - abs(current_angle - target_angle)
    )

    return pos_distance <= pos_threshold and angle_diff <= angle_threshold


def load_posture(posture_path, start_idx, end_idx):
    """从posture.json加载起始帧和结束帧的坐标"""
    with open(posture_path, 'r') as f:
        posture_data = json.load(f)

    # 确保索引在有效范围内
    start_idx = max(0, min(start_idx, len(posture_data) - 1))
    end_idx = max(0, min(end_idx, len(posture_data) - 1))

    start_frame = posture_data[start_idx]
    end_frame = posture_data[end_idx]
    start_frame[3] = start_frame[3]/180*np.pi
    end_frame[3] = end_frame[3]/180*np.pi
    # 确保坐标格式为 [x, y, z, angle]
    if len(start_frame) == 4:
        start_coords = start_frame
    else:
        start_coords = [start_frame[0], start_frame[1], start_frame[2], 0.0]

    if len(end_frame) == 4:
        end_coords = end_frame
    else:
        end_coords = [end_frame[0], end_frame[1], end_frame[2], 0.0]

    return start_coords, end_coords


def parse_transform_matrix(numbers):
    """将数字列表转换为位置和旋转"""
    matrix = np.array(numbers[:16]).reshape(4, 4)
    position = Vector3(numbers[12], numbers[-3] + 1.5, numbers[14])
    rotation_matrix = matrix[:3, :3]
    quat = Quaternion.from_matrix(rotation_matrix)
    return {
        "position": position,
        "rotation": [quat.scalar, quat.vector.x, quat.vector.y, quat.vector.z]
    }


def load_position(file_path):
    """从文件加载位置信息并转换为变换矩阵"""
    with open(file_path, 'r') as f:
        data = json.load(f)

    action = data['action']
    a = action[3]
    x, y, z = action[:3]

    # 精度处理
    x = round(x, 6)
    y = round(y, 6)
    z = round(z, 2)
    sin_a = round(np.sin(a), 6)
    cos_a = round(np.cos(a), 6)

    # 构建变换矩阵数据
    data_str = [
        cos_a, 0, sin_a, 0,
        0, 1, 0, 0,
        -sin_a, 0, cos_a, 0,
        x, 0, y, 1,
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, z, 0, 1
    ]

    return parse_transform_matrix(data_str)