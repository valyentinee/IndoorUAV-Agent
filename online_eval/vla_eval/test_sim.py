import os
import numpy as np
import habitat_sim
from habitat_sim.utils import viz_utils as vut
from habitat_sim import Simulator, AgentConfiguration, SimulatorConfiguration
from magnum import Vector3
import cv2
from utils import load_position  # 从utils导入


def setup_simulator(glb_path):
    """设置模拟器环境"""
    sim_config = SimulatorConfiguration()
    sim_config.scene_id = glb_path
    sim_config.enable_physics = False

    camera_sensor_spec = habitat_sim.CameraSensorSpec()
    camera_sensor_spec.uuid = "rgb"
    camera_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    camera_sensor_spec.resolution = [720, 1280]
    camera_sensor_spec.position = Vector3(0, 1.5, 0)
    camera_sensor_spec.orientation = Vector3(0, np.pi, 0)

    agent_config = AgentConfiguration()
    agent_config.sensor_specifications = [camera_sensor_spec]
    agent_config.height = 1.5
    agent_config.radius = 0.1

    config = habitat_sim.Configuration(sim_config, [agent_config])
    return Simulator(config)


def get_img(coords_file, sim, agent):
    """根据坐标文件获取图像"""
    transform = load_position(file_path=coords_file)

    new_state = habitat_sim.AgentState()
    new_state.position = transform["position"]
    new_state.rotation = transform["rotation"]

    agent.set_state(new_state)
    obs = sim.get_sensor_observations()

    # 图像处理
    frame = obs["rgb"]
    frame = np.flipud(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    frame = frame[:, ::-1, :]

    return frame