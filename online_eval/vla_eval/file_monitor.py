import os
import time
import shutil

# 配置
SHARED_FOLDER = "shared_folder"
SIM_INPUT_DIR = os.path.join(SHARED_FOLDER, "sim_input")
SIM_OUTPUT_DIR = os.path.join(SHARED_FOLDER, "sim_output")
MODEL_INPUT_DIR = os.path.join(SHARED_FOLDER, "model_input")
MODEL_OUTPUT_DIR = os.path.join(SHARED_FOLDER, "model_output")
CONTROLLER_INPUT = os.path.join(SHARED_FOLDER, "controller_input")


def move_files(src_dir, dst_dir, prefix=""):
    """移动文件并添加前缀"""
    moved = False
    for file_name in os.listdir(src_dir):
        if not file_name.endswith('.json'):
            continue

        src_path = os.path.join(src_dir, file_name)
        dst_path = os.path.join(dst_dir, f"{prefix}{file_name}")

        try:
            shutil.move(src_path, dst_path)
            print(f"移动文件: {file_name} -> {os.path.basename(dst_path)}")
            moved = True
        except Exception as e:
            print(f"移动文件失败: {str(e)}")

    return moved


def main():
    print("文件监控服务启动...")

    try:
        while True:
            # 监控并移动文件
            moved = False

            # 移动模拟器输出到控制器输入 (添加'sim_'前缀)
            if move_files(SIM_OUTPUT_DIR, CONTROLLER_INPUT, "sim_"):
                moved = True

            # 移动模型输出到控制器输入 (添加'model_'前缀)
            if move_files(MODEL_OUTPUT_DIR, CONTROLLER_INPUT, "model_"):
                moved = True

            # 如果没有移动任何文件，等待一会儿
            if not moved:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("文件监控服务停止")


if __name__ == "__main__":
    main()