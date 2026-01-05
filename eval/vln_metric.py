import os
import json
import math
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

# 常量定义
THRESHOLD_SUCCESS_DIST = 2  # 成功距离阈值 (米)
THRESHOLD_SUCCESS_ANGLE = math.pi*2   # 成功角度阈值 (弧度)
ALPHA = 10  # nDTW参数


def angle_difference(a, b):
    """计算两个角度之间的最小差异（考虑圆周性）"""
    diff = abs(a - b)
    return min(diff, 2 * math.pi - diff)


def calculate_ndtw(seq_a, seq_b):
    """
    计算两个序列之间的nDTW（仅使用三维坐标）
    :param seq_a: 序列A（三维坐标）
    :param seq_b: 序列B（三维坐标）
    :return: (nDTW值, 参考路径长度L)
    """
    if len(seq_a) == 0 or len(seq_b) == 0:
        return 0.0, 0.0

    # 计算累积距离
    distance, _ = fastdtw(
        np.array(seq_a),
        np.array(seq_b),
        dist=euclidean
    )

    # 计算参考路径长度L
    L = sum(euclidean(seq_b[i], seq_b[i - 1]) for i in range(1, len(seq_b)))
    L_a = sum(euclidean(seq_a[i], seq_a[i - 1]) for i in range(1, len(seq_a)))
    # 避免除以零
    L = max(L, 1e-5)
    # 计算nDTW
    ndtw_value = math.exp(-distance / (ALPHA * L))

    return ndtw_value, L


def process_episode(json_path):
    """处理单个episode文件"""
    try:
        # 加载轨迹文件
        with open(json_path, 'r') as f:
            data = json.load(f)

        # 检查终止原因
        if data.get("termination_reason") != "no_more_instructions":
            return None

        # 获取episode_key和预测轨迹
        episode_key = data["episode_key"].lstrip('/')
        pred_trajectory = data["trajectory"]

        # 构建posture文件路径
        base_path = "/data1/liuy/vln_pi0/without_screenshot"
        posture_path = os.path.join(base_path, os.path.dirname(episode_key), "posture.json")

        # 加载目标轨迹
        with open(posture_path, 'r') as f:
            gt_full_seq = json.load(f)

        # 转换角度为弧度并提取三维坐标
        gt_seq = []
        gt_positions = []
        for point in gt_full_seq:
            x, y, z, yaw_deg = point
            yaw_rad = yaw_deg * math.pi / 180.0
            gt_seq.append([x, y, z, yaw_rad])
            gt_positions.append([x, y, z])  # 仅位置用于nDTW

        # 获取目标终点
        gt_end = gt_seq[-1]

        # 提取预测轨迹的三维坐标
        pred_positions = []
        for point in pred_trajectory:
            # 预测轨迹已经是弧度制，我们只需要位置
            pred_positions.append(point[:3])

        # 获取预测终点
        pred_end = pred_trajectory[-1]

        # 计算NE（导航误差）
        ne = euclidean(pred_end[:3], gt_end[:3])

        # 计算SR（成功率）
        final_dist = euclidean(pred_end[:3], gt_end[:3])
        final_angle_diff = angle_difference(pred_end[3], gt_end[3])
        sr = 1 if (final_dist <= THRESHOLD_SUCCESS_DIST and
                   final_angle_diff <= THRESHOLD_SUCCESS_ANGLE) else 0

        # 计算OSR（在线成功率）
        osr = 0
        for point in pred_trajectory:
            dist = euclidean(point[:3], gt_end[:3])
            angle_diff = angle_difference(point[3], gt_end[3])
            if dist <= THRESHOLD_SUCCESS_DIST and angle_diff <= THRESHOLD_SUCCESS_ANGLE:
                osr = 1
                break

        # 计算nDTW（归一化动态时间规整）
        nDTW, _ = calculate_ndtw(pred_positions, gt_positions)

        return {
            "episode_key": episode_key,
            "NE": ne,
            "SR": sr,
            "OSR": osr,
            "nDTW": nDTW,
            "final_dist": final_dist,
            "final_angle_diff": final_angle_diff
        }

    except Exception as e:
        print(f"处理文件 {json_path} 时出错: {str(e)}")
        return None


def main():
    # 配置路径
    trajectories_dir = "/data1/liuy/vln_pi0/shared_folder/trajectories"
    output_file = "evaluation_results_unseen.json"

    # 收集所有结果
    all_results = []
    total_count = 0
    valid_count = 0
    sr_count = 0
    osr_count = 0
    total_ne = 0.0
    total_ndtw = 0.0

    # 遍历所有轨迹文件
    for filename in os.listdir(trajectories_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(trajectories_dir, filename)
            result = process_episode(filepath)

            if result:
                all_results.append(result)
                total_count += 1
                valid_count += 1

                # 更新统计数据
                total_ne += result["NE"]
                total_ndtw += result["nDTW"]
                sr_count += result["SR"]
                osr_count += result["OSR"]

    # 计算总体指标
    avg_ne = total_ne / valid_count if valid_count > 0 else 0.0
    avg_ndtw = total_ndtw / valid_count if valid_count > 0 else 0.0
    sr_rate = sr_count / valid_count if valid_count > 0 else 0.0
    osr_rate = osr_count / valid_count if valid_count > 0 else 0.0

    # 保存结果
    with open(output_file, 'w') as f:
        json.dump({
            "per_episode_results": all_results,
            "overall_metrics": {
                "average_NE": avg_ne,
                "success_rate": sr_rate,
                "online_success_rate": osr_rate,
                "average_nDTW": avg_ndtw,
                "total_episodes": total_count,
                "valid_episodes": valid_count
            }
        }, f, indent=2)

    # 打印结果
    print(f"处理完成! 共处理 {total_count} 个文件，其中 {valid_count} 个有效")
    print(f"平均导航误差 (NE): {avg_ne:.4f} 米")
    print(f"成功率 (SR): {sr_rate:.4f} ({sr_count}/{valid_count})")
    print(f"在线成功率 (OSR): {osr_rate:.4f} ({osr_count}/{valid_count})")
    print(f"平均 nDTW: {avg_ndtw:.4f}")
    print(f"结果已保存至 {output_file}")


if __name__ == "__main__":
    main()
