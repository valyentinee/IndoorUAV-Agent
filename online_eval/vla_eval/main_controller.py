import os
import json
import time
import shutil
from utils import get_glb_path, is_success, load_posture

# 配置参数
MAX_INFERENCE_STEPS = 12
SHARED_FOLDER = "shared_folder"
TEST_VLA_FILE = "test_vla.json"
VLA_INS_BASE = "./vla_ins"
POSTURE_BASE = "./without_screenshot"
TRAJECTORY_OUTPUT = os.path.join(SHARED_FOLDER, "trajectories")
CONTROLLER_INPUT = os.path.join(SHARED_FOLDER, "controller_input")
SIM_INPUT_DIR = os.path.join(SHARED_FOLDER, "sim_input")
SIM_OUTPUT_DIR = os.path.join(SHARED_FOLDER, "sim_output")
MODEL_INPUT_DIR = os.path.join(SHARED_FOLDER, "model_input")
MODEL_OUTPUT_DIR = os.path.join(SHARED_FOLDER, "model_output")
INSTRUCTIONS_DIR = os.path.join(SHARED_FOLDER, "instructions")
IMAGE_STORAGE = os.path.join(SHARED_FOLDER, "images")
INDOOR_UAV_BASE = "/data1/liux/Indoor_UAV"  # 新增基础路径

# 确保目录存在
for dir_path in [CONTROLLER_INPUT, SIM_INPUT_DIR, SIM_OUTPUT_DIR,
                 MODEL_INPUT_DIR, MODEL_OUTPUT_DIR, TRAJECTORY_OUTPUT,
                 INSTRUCTIONS_DIR, IMAGE_STORAGE]:
    os.makedirs(dir_path, exist_ok=True)


class EpisodeController:
    def __init__(self, episode_key):
        self.episode_key = episode_key
        self.trajectory = []
        self.step_count = 0
        self.success = False
        self.start_coords = None
        self.end_coords = None
        self.glb_path = None
        self.instruction = None
        self.start_image_path = None

        # 解析路径
        path_parts = episode_key.rsplit('/', 1)[0]
        parts = path_parts.strip('/').split('/')
        self.group = parts[0]
        self.scene = parts[1]
        self.traj = parts[2]

    def setup_episode(self):
        """设置新episode的初始状态"""
        print(f"\n=== 开始测试: {self.episode_key} ===")

        # 加载指令和source
        vla_ins_path = os.path.join(VLA_INS_BASE, self.group, self.scene, self.traj,
                                    os.path.basename(self.episode_key))
        with open(vla_ins_path, 'r', encoding='gbk') as f:
            vla_data = json.load(f)

        self.instruction = vla_data["instruction"]
        source = vla_data["source"]
        start_idx = source[0] - 1
        end_idx = source[1] - 1

        # 获取glb路径
        self.glb_path = get_glb_path(self.group, self.scene)

        # 加载posture数据
        posture_path = os.path.join(POSTURE_BASE, self.group, self.scene, self.traj, "posture.json")
        self.start_coords, self.end_coords = load_posture(posture_path, start_idx, end_idx)

        # 添加起始坐标到轨迹
        self.trajectory.append(self.start_coords)

        # 获取起始帧图像 (新增)
        start_frame_number = source[0]  # 起始帧号
        screenshots_dir = os.path.join(INDOOR_UAV_BASE, self.group, self.scene, self.traj, "screenshots")
        self.start_image_path = os.path.join(screenshots_dir, f"{start_frame_number}.png")

        # 确保图像存在
        if not os.path.exists(self.start_image_path):
            print(f"警告: 起始图像不存在: {self.start_image_path}")
            self.start_image_path = None

        # 保存当前指令 (包含起始图像路径)
        instruction_file = os.path.join(INSTRUCTIONS_DIR, "current_instruction.json")
        with open(instruction_file, 'w') as f:
            json.dump({
                "episode_key": self.episode_key,
                "instruction": self.instruction,
                "end_coords": self.end_coords,
                "glb_path": self.glb_path,
                "start_coords": self.start_coords,
                "start_image_path": self.start_image_path  # 新增起始图像路径
            }, f)

        # 发送起始坐标到模拟器
        self.send_to_simulator(self.start_coords, True)

    def send_to_simulator(self, coords, is_new_scene=False):
        """发送坐标到模拟器"""
        timestamp = time.time()
        sim_input_file = os.path.join(SIM_INPUT_DIR, f"sim_input_{timestamp}.json")
        with open(sim_input_file, 'w') as f:
            json.dump({
                "episode_key": self.episode_key,
                "coordinates": coords,
                "glb_path": self.glb_path if is_new_scene else None,
                "is_new_scene": is_new_scene
            }, f)
        print(f"发送坐标到模拟器: {coords}")

    def send_to_model(self, image_path, coords):
        """发送图像和坐标到模型"""
        timestamp = time.time()
        model_input_file = os.path.join(MODEL_INPUT_DIR, f"model_input_{timestamp}.json")
        with open(model_input_file, 'w') as f:
            json.dump({
                "episode_key": self.episode_key,
                "image_path": image_path,
                "coordinates": coords
            }, f)
        print(f"发送图像到模型: {os.path.basename(image_path)}")

    def process_sim_output(self, sim_data):
        """处理模拟器输出"""
        if sim_data.get("episode_key") != self.episode_key:
            return False

        # 第一次输出是起始图像
        if self.start_image_path is None:
            # 使用预定义的起始图像路径
            instruction_file = os.path.join(INSTRUCTIONS_DIR, "current_instruction.json")
            if os.path.exists(instruction_file):
                with open(instruction_file, 'r') as f:
                    data = json.load(f)
                self.start_image_path = data.get("start_image_path", None)

            if not self.start_image_path:
                print("警告: 无法获取起始图像路径")
                return False

        # 后续输出是推理步骤
        current_coords = sim_data["coordinates"]
        self.trajectory.append(current_coords)

        # 检查是否成功
        self.success = is_success(current_coords, self.end_coords)

        if self.success:
            print(f"成功! 在步骤 {self.step_count} 达到目标位置")
            self.terminate_episode()
            return False

        if self.step_count >= MAX_INFERENCE_STEPS:
            print(f"达到最大推理步数 ({MAX_INFERENCE_STEPS})")
            self.terminate_episode()
            return False

        # 发送新图像到模型
        self.send_to_model(sim_data["image_path"], current_coords)
        return True

    def process_model_output(self, model_data):
        """处理模型输出"""
        if model_data.get("episode_key") != self.episode_key:
            return False

        new_coords = model_data["coordinates"]
        self.step_count += 1
        print(f"推理步骤 {self.step_count}/{MAX_INFERENCE_STEPS} - 新坐标: {new_coords}")

        # 发送新坐标到模拟器
        self.send_to_simulator(new_coords)
        return True

    def terminate_episode(self):
        """终止当前episode并保存结果"""
        # 保存轨迹
        safe_episode_key = self.episode_key.replace('/', '_').replace(':', '_').replace(' ', '_')
        trajectory_file = os.path.join(TRAJECTORY_OUTPUT, f"{safe_episode_key}.json")

        with open(trajectory_file, 'w') as f:
            json.dump({
                "episode_key": self.episode_key,
                "success": self.success,
                "steps": self.step_count,
                "trajectory": self.trajectory
            }, f, indent=2)

        print(f"测试完成. 成功: {self.success}, 步数: {self.step_count}")

        # 发送终止信号
        terminate_file = os.path.join(SIM_INPUT_DIR, "terminate.json")
        with open(terminate_file, 'w') as f:
            json.dump({
                "episode_key": self.episode_key,
                "action": "terminate"
            }, f)


class FileMover:
    """负责在目录间移动文件的组件"""

    def __init__(self):
        self.active = True

    def move_sim_output_to_model_input(self):
        """将模拟器输出移动到模型输入"""
        for file_name in os.listdir(SIM_OUTPUT_DIR):
            if not file_name.endswith('.json'):
                continue

            src_path = os.path.join(SIM_OUTPUT_DIR, file_name)
            dst_path = os.path.join(CONTROLLER_INPUT, f"sim_{file_name}")

            try:
                shutil.move(src_path, dst_path)
                print(f"移动模拟器输出: {file_name} -> {os.path.basename(dst_path)}")
            except Exception as e:
                print(f"移动文件失败: {str(e)}")

    def move_model_output_to_sim_input(self):
        """将模型输出移动到模拟器输入"""
        for file_name in os.listdir(MODEL_OUTPUT_DIR):
            if not file_name.endswith('.json'):
                continue

            src_path = os.path.join(MODEL_OUTPUT_DIR, file_name)
            dst_path = os.path.join(CONTROLLER_INPUT, f"model_{file_name}")

            try:
                shutil.move(src_path, dst_path)
                print(f"移动模型输出: {file_name} -> {os.path.basename(dst_path)}")
            except Exception as e:
                print(f"移动文件失败: {str(e)}")


def main():
    # 加载测试配置
    with open(TEST_VLA_FILE, 'r') as f:
        test_vla = json.load(f)

    # 初始化文件搬运工
    file_mover = FileMover()

    # 运行所有episode
    results = {}
    episode_keys = list(test_vla.keys())

    for i, episode_key in enumerate(episode_keys):
        print(f"\n{'=' * 40}")
        print(f"开始测试 {i + 1}/{len(episode_keys)}: {episode_key}")

        # 初始化当前episode
        controller = EpisodeController(episode_key)
        controller.setup_episode()

        # 清空目录
        for dir_path in [CONTROLLER_INPUT, SIM_OUTPUT_DIR, MODEL_OUTPUT_DIR]:
            for item in os.listdir(dir_path):
                item_path = os.path.join(dir_path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)

        # 运行episode
        start_time = time.time()
        while True:
            # 移动文件
            file_mover.move_sim_output_to_model_input()
            file_mover.move_model_output_to_sim_input()

            # 处理控制器输入
            processed = False
            for file_name in os.listdir(CONTROLLER_INPUT):
                if not file_name.endswith('.json'):
                    continue

                file_path = os.path.join(CONTROLLER_INPUT, file_name)

                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)

                    # 处理模拟器输出
                    if file_name.startswith('sim_'):
                        if controller.process_sim_output(data):
                            processed = True

                    # 处理模型输出
                    elif file_name.startswith('model_'):
                        if controller.process_model_output(data):
                            processed = True

                    # 删除处理过的文件
                    os.remove(file_path)

                except Exception as e:
                    print(f"处理文件 {file_name} 出错: {str(e)}")
                    continue

            # 检查终止条件
            if controller.success or controller.step_count >= MAX_INFERENCE_STEPS:
                controller.terminate_episode()
                time.sleep(0.2)
                break

            # 超时检查 (5分钟)
            if time.time() - start_time > 240:
                print("episode超时，终止测试")
                controller.terminate_episode()
                time.sleep(0.2)
                break

            # 如果没有处理任何文件，等待一会儿
            if not processed:
                time.sleep(0.1)

        # 记录结果
        results[episode_key] = {
            "success": controller.success,
            "steps": controller.step_count,
            "difficulty": test_vla[episode_key].get("difficulty", "unknown"),
            "action_type": test_vla[episode_key].get("action_type", [])
        }

    # 保存最终结果
    results_file = os.path.join(TRAJECTORY_OUTPUT, "final_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n所有测试完成!")


if __name__ == "__main__":
    main()