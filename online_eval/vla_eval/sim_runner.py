import os
import json
import time
from test_sim import setup_simulator, get_img
import cv2

# 配置
SHARED_FOLDER = "shared_folder"
SIM_INPUT_DIR = os.path.join(SHARED_FOLDER, "sim_input")
SIM_OUTPUT_DIR = os.path.join(SHARED_FOLDER, "sim_output")
IMAGE_STORAGE = os.path.join(SHARED_FOLDER, "images")
os.makedirs(SIM_INPUT_DIR, exist_ok=True)
os.makedirs(SIM_OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_STORAGE, exist_ok=True)


class SimulatorService:
    def __init__(self):
        self.sim = None
        self.agent = None
        self.current_glb_path = None

    def process_file(self, file_path):
        try:
            # 检查终止信号
            if "terminate.json" in file_path:
                print("收到终止信号，关闭模拟器")
                if self.sim:
                    self.sim.close()
                    self.sim = None
                # 移除终止信号文件
                os.remove(file_path)
                return True
            time.sleep(0.16)
            with open(file_path, 'r') as f:
                data = json.load(f)

            episode_key = data.get("episode_key", "")
            coords = data.get("coordinates", [])
            glb_path = data.get("glb_path", None)
            is_new_scene = data.get("is_new_scene", False)

            # 处理场景设置
            if is_new_scene and glb_path:
                if self.sim:
                    self.sim.close()
                print(f"初始化模拟器: {glb_path}")
                self.sim = setup_simulator(glb_path)
                self.agent = self.sim.initialize_agent(0)
                self.current_glb_path = glb_path
            elif not self.sim:
                print("错误: 模拟器未初始化")
                return False

            # 渲染图像
            timestamp = time.time()
            safe_episode_key = episode_key.replace('/', '_').replace(':', '_').replace(' ', '_')
            image_filename = f"image_{safe_episode_key}_{timestamp}.png"
            image_path = os.path.join(IMAGE_STORAGE, image_filename)

            # 创建临时文件存储坐标
            temp_coords_file = f"temp_coords_{timestamp}.json"
            with open(temp_coords_file, 'w') as f:
                json.dump({"action": coords}, f)

            # 获取图像
            frame = get_img(temp_coords_file, self.sim, self.agent)

            # 保存图像
            cv2.imwrite(image_path, frame)

            # 删除临时文件
            os.remove(temp_coords_file)

            # 保存输出
            output_file = os.path.join(SIM_OUTPUT_DIR, f"sim_output_{timestamp}.json")
            with open(output_file, 'w') as f:
                json.dump({
                    "episode_key": episode_key,
                    "coordinates": coords,
                    "image_path": image_path
                }, f)

            print(f"已生成图像: {image_path}")
            return True

        except Exception as e:
            print(f"处理文件 {file_path} 出错: {str(e)}")
            return False
        finally:
            # 清理输入文件 (不包括终止信号文件)
            if os.path.exists(file_path) and "terminate.json" not in file_path:
                os.remove(file_path)


def main():
    print("模拟器服务启动...")
    simulator = SimulatorService()

    try:
        while True:
            # 检查输入目录
            processed = False
            for file_name in os.listdir(SIM_INPUT_DIR):
                if not file_name.endswith('.json'):
                    continue

                file_path = os.path.join(SIM_INPUT_DIR, file_name)
                if simulator.process_file(file_path):
                    processed = True

            # 如果没有处理任何文件，等待一会儿
            if not processed:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("模拟器服务停止")

    finally:
        if simulator.sim:
            simulator.sim.close()


if __name__ == "__main__":
    main()
