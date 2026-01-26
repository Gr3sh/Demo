import sys
import random
import numpy as np
import os
from PIL import Image
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import time
import mujoco_viewer
import glfw
import mujoco
from mujoco_env.env import XLeRobotController


# 随机生成种子，如果是 None，随机生成场景
SEED = None

REPO_NAME = 'XleRobot-demo'  # 数据集名称
ROOT = "./demo_data"  # 数据存放根目录
NUM_DEMO = 1  # 收集次数

TASK_NAME = 'test dataset'
xml_path = './xml/scene.xml'  # 场景

# 如果已经存在数据，询问删除并新建
create_new = True

if os.path.exists(ROOT):
    print(f"Directory {ROOT} already exists.")
    ans = input("已经存在数据，是否删除并重新创建? (y/n) ")
    if ans == 'y':
        import shutil
        shutil.rmtree(ROOT)
    else:
        create_new = False

if create_new:
    # 创建 LeRobot 的 Dataset 格式
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,  # 数据集名称
        root=ROOT,  # 数据集保存的根目录
        robot_type="XleRobot",  # 机器人型号 自定义
        fps=20,  # 帧率
        features={
            "observation.state": {
                "dtype": "float64",
                "shape": (18,),
                "names": ["base_xy_yaw_and_joints"],
            },
            "action": {
                "dtype": "float64",
                "shape": (18,),
                "names": [
                    "slider_x_vel", "slider_y_vel", "yaw_vel",
                    "R_rot", "R_pitch", "R_elbow", "R_wrist_p", "R_wrist_r", "R_jaw",
                    "L_rot", "L_pitch", "L_elbow", "L_wrist_p", "L_wrist_r", "L_jaw",
                    "wheel1_vel", "wheel2_vel", "wheel3_vel",
                ],
            },
        },
    )
else:
    print("从先前的数据集加载")
    dataset = LeRobotDataset(REPO_NAME, root=ROOT)

try:
    controller = XLeRobotController(xml_path)
    controller.init_dataset(dataset)
    controller.init_mode("collect")
    controller.run()
except KeyboardInterrupt:
    print("\nReceived keyboard interrupt, shutting down...")

    