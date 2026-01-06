import os
import json
import time
import numpy as np
from PIL import Image


# 初始化模型
def init_model():
    from openpi.training import config
    from openpi.policies import policy_config

    config = config.get_config("pi0_uav_low_mem_finetune")
    checkpoint_dir = "/data1/liuy/pi0_ck/pi0_uav_low_mem_finetune/15k_2/29999"
    return policy_config.create_trained_policy(config, checkpoint_dir)


def infer(policy, inputs):
    return policy.infer(inputs)["actions"]


policy = init_model()

# 配置
SHARED_FOLDER = "shared_folder"
MODEL_INPUT_DIR = os.path.join(SHARED_FOLDER, "model_input")
MODEL_OUTPUT_DIR = os.path.join(SHARED_FOLDER, "model_output")
INSTRUCTIONS_DIR = os.path.join(SHARED_FOLDER, "instructions")
os.makedirs(MODEL_INPUT_DIR, exist_ok=True)
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(INSTRUCTIONS_DIR, exist_ok=True)


class ModelService:
    def __init__(self):
        self.current_episode = None
        self.instruction = None
        self.end_coords = None
        self.ref_image_array = None
        self.last_ref_image_path = None
        self.last_instruction_mtime = 0  # 跟踪上次加载的时间戳

    def load_instruction(self):
        """加载当前指令并更新参考图像"""
        instruction_file = os.path.join(INSTRUCTIONS_DIR, "current_instruction.json")
        if not os.path.exists(instruction_file):
            return

        # 获取文件修改时间
        current_mtime = os.path.getmtime(instruction_file)

        # 检查文件是否已更新
        if current_mtime <= self.last_instruction_mtime:
            return

        self.last_instruction_mtime = current_mtime

        try:
            with open(instruction_file, 'r') as f:
                data = json.load(f)
                print(f"加载指令文件: {data.get('episode_key', 'unknown')}")

            # 检查是否是新episode
            if self.current_episode != data.get("episode_key"):
                self.current_episode = data.get("episode_key")
                self.instruction = data.get("instruction")
                self.end_coords = data.get("end_coords")
                self.last_ref_image_path = None
                print(f"新episode: {self.current_episode}")
            else:
                # 检查指令是否更新
                new_instruction = data.get("instruction")
                if new_instruction and new_instruction != self.instruction:
                    self.instruction = new_instruction
                    print(f"指令更新: {self.instruction}")

            # 获取参考图像路径
            ref_image_path = data.get("ref_image_path", data.get("start_image_path"))

            # 检查是否有新的参考图像路径
            if ref_image_path and ref_image_path != self.last_ref_image_path:
                self.last_ref_image_path = ref_image_path

                # 加载参考图像
                if os.path.exists(ref_image_path):
                    try:
                        ref_img = Image.open(ref_image_path).convert('RGB')
                        self.ref_image_array = np.asarray(ref_img, dtype=np.uint8)
                        print(f"更新参考图像: {os.path.basename(ref_image_path)}")
                    except Exception as e:
                        print(f"加载参考图像失败: {str(e)}")
                        self.ref_image_array = None
                else:
                    print(f"警告: 参考图像不存在: {ref_image_path}")
                    self.ref_image_array = None
        except Exception as e:
            print(f"加载指令文件失败: {str(e)}")

    def process_file(self, file_path):
        try:
            # 在每次处理前加载最新指令
            self.load_instruction()

            # 如果没有有效指令，跳过处理
            if not self.instruction:
                print("警告: 没有有效指令，跳过处理")
                return False

            with open(file_path, 'r') as f:
                data = json.load(f)

            episode_key = data.get("episode_key", "")
            image_path = data.get("image_path", "")
            coordinates = data.get("coordinates", [])

            # 检查是否匹配当前episode
            if episode_key != self.current_episode:
                print(f"忽略文件: 不属于当前episode ({episode_key} vs {self.current_episode})")
                return False

            # 获取输入图像
            if not os.path.exists(image_path):
                print(f"图像文件不存在: {image_path}")
                return False

            img = Image.open(image_path).convert('RGB')
            img_array = np.asarray(img, dtype=np.uint8)

            # 确保coordinates有4个元素
            if len(coordinates) < 4:
                coordinates = coordinates + [0.0] * (4 - len(coordinates))

            state = np.array(coordinates[:4], dtype=np.float32)

            # 准备模型输入
            example = {
                "observation/image": img_array,
                "observation/ref_image": self.ref_image_array,
                "observation/state": state,
                "task": self.instruction
            }

            # 执行推理
            output_all = infer(policy, example)
            output = output_all[9]
            new_coords = output[:4].tolist()

            # 保存模型输出
            timestamp = time.time()
            output_file = os.path.join(MODEL_OUTPUT_DIR, f"model_output_{timestamp}.json")
            with open(output_file, 'w') as f:
                json.dump({
                    "episode_key": self.current_episode,
                    "coordinates": new_coords
                }, f)

            print(f"推理完成 - 新坐标: {new_coords}")
            return True

        except Exception as e:
            print(f"处理文件 {file_path} 出错: {str(e)}")
            return False
        finally:
            # 清理输入文件
            if os.path.exists(file_path):
                os.remove(file_path)


def main():
    print("模型推理服务启动...")
    model_service = ModelService()

    try:
        while True:
            # 定期检查指令更新
            model_service.load_instruction()

            # 检查输入目录
            processed = False
            for file_name in os.listdir(MODEL_INPUT_DIR):
                if not file_name.endswith('.json'):
                    continue

                file_path = os.path.join(MODEL_INPUT_DIR, file_name)
                if model_service.process_file(file_path):
                    processed = True

            # 如果没有处理任何文件，等待一会儿
            if not processed:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("模型推理服务停止")


if __name__ == "__main__":
    main()