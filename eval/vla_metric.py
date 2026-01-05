import os
import json
import math
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw  # 需要安装fastdtw库：pip install fastdtw

# 常量定义
THRESHOLD_STOP_DIST = 0.15
THRESHOLD_STOP_ANGLE = math.pi / 12
THRESHOLD_SUCCESS_DIST = 0.5
THRESHOLD_SUCCESS_ANGLE = math.pi / 4
ALPHA = 1  # NDTW参数


def angle_difference(a, b):
    """计算两个角度之间的最小差异（考虑圆周性）"""
    diff = abs(a - b)
    return min(diff, 2 * math.pi - diff)


def calculate_ndtw(seq_a, seq_b, is_angle=False):
    """
    计算两个序列之间的NDTW
    :param seq_a: 序列A
    :param seq_b: 序列B
    :param is_angle: 是否为角度序列
    :return: (NDTW值, 参考路径长度L)
    """
    if len(seq_a) == 0 or len(seq_b) == 0:
        return 0.0, 0.0

    # 计算累积距离
    distance, _ = fastdtw(
        np.array(seq_a).reshape(-1, 1) if is_angle else np.array(seq_a),
        np.array(seq_b).reshape(-1, 1) if is_angle else np.array(seq_b),
        dist=angle_difference if is_angle else euclidean
    )

    # 计算参考路径长度L
    if is_angle:
        L = sum(angle_difference(seq_b[i], seq_b[i - 1]) for i in range(1, len(seq_b)))
    else:
        L = sum(euclidean(seq_b[i], seq_b[i - 1]) for i in range(1, len(seq_b)))

    # 避免除以零
    L = max(L, 1e-5)

    # 计算NDTW
    ndtw_value = math.exp(-distance / (ALPHA * L))

    return ndtw_value, L


def process_episode(trajectory_file):
    """处理单个episode文件"""
    try:
        # 加载轨迹文件
        with open(trajectory_file, 'r', encoding='gbk') as f:
            traj_data = json.load(f)

        # 获取episode_key和轨迹
        episode_key = traj_data["episode_key"].lstrip('/')
        trajectory = traj_data["trajectory"]

        # 解析路径
        parts = episode_key.split('/')
        scene_name = parts[0]
        env_name = parts[1]
        traj_folder = parts[2]
        vla_file = parts[3]

        # 构建vla_ins文件路径
        vla_ins_path = f"/data1/liuy/test_pi0/vla_ins/{scene_name}/{env_name}/{traj_folder}/{vla_file}"
        with open(vla_ins_path, 'r', encoding='gbk') as f:
            vla_data = json.load(f)
        source = vla_data["source"]

        # 构建posture文件路径
        posture_path = f"/data1/liuy/test_pi0/without_screenshot/{scene_name}/{env_name}/{traj_folder}/posture.json"
        with open(posture_path, 'r') as f:
            posture_data = json.load(f)

        # 提取gt序列 (0-based索引)
        start_idx = source[0] - 1
        end_idx = source[1] - 1
        gt_full_seq = posture_data[start_idx:end_idx + 1]

        # 转换角度为弧度
        gt_seq = []
        for point in gt_full_seq:
            x, y, z, yaw_deg = point
            yaw_rad = yaw_deg * math.pi / 180.0
            gt_seq.append([x, y, z, yaw_rad])

        # 处理预测轨迹 (跳过第0个点)
        pred_full_seq = trajectory[2:16]  # 取第1到15个点

        # 检查是否满足停止条件
        stop_index = None
        for i in range(1, len(pred_full_seq)):
            prev = pred_full_seq[i - 1]
            curr = pred_full_seq[i]

            # 计算距离和角度差
            dist = euclidean(prev[:3], curr[:3])
            angle_diff = angle_difference(prev[3], curr[3])

            if dist < THRESHOLD_STOP_DIST and angle_diff < THRESHOLD_STOP_ANGLE:
                stop_index = i - 1  # 使用前一个点作为终点
                break

        # 确定最终使用的预测序列
        if stop_index is not None:
            pred_seq = pred_full_seq[:stop_index + 1]  # 包含停止点
        else:
            pred_seq = pred_full_seq  # 使用完整序列

        # 获取最后一个预测点和gt点
        last_pred = pred_seq[-1]
        last_gt = gt_seq[-1]

        # 计算最终距离和角度差
        final_dist = euclidean(last_pred[:3], last_gt[:3])
        final_angle_diff = angle_difference(last_pred[3], last_gt[3])

        # 判断是否成功
        success = (final_dist < THRESHOLD_SUCCESS_DIST and
                   final_angle_diff < THRESHOLD_SUCCESS_ANGLE)

        # 准备NDTW计算
        pred_positions = [p[:3] for p in pred_seq]
        pred_angles = [p[3] for p in pred_seq]

        gt_positions = [p[:3] for p in gt_seq]
        gt_angles = [p[3] for p in gt_seq]

        # 计算位置NDTW和参考长度
        nDTW_pos, L_pos = calculate_ndtw(pred_positions, gt_positions)

        # 计算角度NDTW和参考长度
        nDTW_ang, L_ang = calculate_ndtw(pred_angles, gt_angles, is_angle=True)

        L_pos_adjusted = L_pos / 2.2

        # 计算权重 (避免除零错误)
        total_adjusted = L_pos_adjusted + L_ang
        if total_adjusted > 0:
            weight_pos = L_pos_adjusted / total_adjusted
            weight_ang = L_ang / total_adjusted
        else:
            weight_pos = 0.5
            weight_ang = 0.5

        # 计算综合NDTW
        nDTW_total = weight_pos * nDTW_pos + weight_ang * nDTW_ang
        if stop_index is not None:
            return {
                "episode": episode_key,
                "success": success,
                "nDTW": nDTW_total,
                "final_dist": final_dist,
                "final_angle_diff": final_angle_diff
            }
        else:
            return {
                "episode": episode_key,
                "success": success,
                "nDTW": None,
                "final_dist": final_dist,
                "final_angle_diff": final_angle_diff
            }
    except Exception as e:
        print(f"Error processing {trajectory_file}: {str(e)}")
        return None


def main():
    # 配置路径
    trajectories_dir = "/data1/liuy/test_pi0/shared_folder/trajectories"
    output_file = "evaluation_results_openvla.json"

    # 收集所有结果
    all_results = []
    success_count = 0
    total_count = 0
    ndtw_count = 0
    total_nDTW = 0.0

    # 遍历所有轨迹文件
    for filename in os.listdir(trajectories_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(trajectories_dir, filename)
            result = process_episode(filepath)

            if result:
                all_results.append(result)
                total_count += 1
                if result["nDTW"] is not None and result["nDTW"]>1:
                    result["nDTW"] = None
                if result["nDTW"] is not None and result["nDTW"] < 1:
                    ndtw_count += 1
                    total_nDTW += result["nDTW"]
                    # if result["nDTW"] >1:
                    #     print(filepath)
                # total_nDTW += result["nDTW"]

                if result["success"]:
                    success_count += 1

    # 计算总体指标
    success_rate = success_count / total_count if total_count > 0 else 0.0
    # avg_nDTW = total_nDTW / total_count if total_count > 0 else 0.0
    avg_nDTW = total_nDTW / ndtw_count if ndtw_count > 0 else 0.0
    print(ndtw_count)
    # 保存结果
    with open(output_file, 'w') as f:
        json.dump({
            "per_episode_results": all_results,
            "overall_metrics": {
                "success_rate": success_rate,
                "average_nDTW": avg_nDTW,
                "total_episodes": total_count
            }
        }, f, indent=2)

    # 打印结果
    print(f"Total episodes processed: {total_count}")
    print(f"Success rate: {success_rate:.4f} ({success_count}/{total_count})")
    print(f"Average nDTW: {avg_nDTW:.4f}")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()