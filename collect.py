import sys
import random
import numpy as np
import os
from PIL import Image
from mujoco_env.y_env import SimpleEnv
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# 如果是 None，随机生成场景
SEED = None

REPO_NAME = 'XleRobot-demo'         # 数据集名称
ROOT = "./demo_data"                # 数据存放根目录
NUM_DEMO = 2 # 收集次数

TASK_NAME = 'Put mug cup on the plate'
xml_path = './asset/example_scene_y.xml' # 场景

PnPEnv = SimpleEnv(xml_path, seed = SEED, state_type = 'joint_angle')


# 如果已经存在数据，询问删除并新建
create_new = True
if os.path.exists(ROOT):
    print(f"Directory {ROOT} already exists.")
    ans = input("Do you want to delete it? (y/n) ")
    if ans == 'y':
        import shutil
        shutil.rmtree(ROOT)
    else:
        create_new = False

if create_new:
    # 创建 LeRobot 的 Dataset 格式
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,                  # 数据集名称
        root = ROOT,                        # 数据集保存的根目录
        robot_type="omy",                   # 机器人型号 自定义
        fps=20,                             # 帧率
        features={
            "observation.image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["state"], # x, y, z, roll, pitch, yaw
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["action"], # 6 joint angles and 1 gripper
            },
            "obj_init": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["obj_init"], # just the initial position of the object. Not used in training.
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
else:
    print("Load from previous dataset")
    dataset = LeRobotDataset(REPO_NAME, root=ROOT)

action = np.zeros(7)
episode_id = 0
record_flag = False # 只有开始运动了才开始记录数据

while PnPEnv.env.is_viewer_alive() and episode_id < NUM_DEMO:
    PnPEnv.step_env()
    if PnPEnv.env.loop_every(HZ=20):
        # 检查是否完成目标
        done = PnPEnv.check_success()
        if done:
            dataset.save_episode()
            PnPEnv.reset(seed = SEED)
            episode_id += 1
        # 获取键盘输入，并转换成动作
        action, reset  = PnPEnv.teleop_robot()
        if not record_flag and sum(action) != 0:
            record_flag = True
            print("Start recording")
        if reset:
            # 按 z 重置环境
            PnPEnv.reset(seed=SEED)
            dataset.clear_episode_buffer()
            record_flag = False
        # 采集图像和机械臂状态
        ee_pose = PnPEnv.get_ee_pose()
        agent_image,wrist_image = PnPEnv.grab_image()
        # 图像处理
        agent_image = Image.fromarray(agent_image)
        wrist_image = Image.fromarray(wrist_image)
        agent_image = agent_image.resize((256, 256))
        wrist_image = wrist_image.resize((256, 256))
        agent_image = np.array(agent_image)
        wrist_image = np.array(wrist_image)
        # 执行动作
        joint_q = PnPEnv.step(action)

        if record_flag:
            # 添加到数据集
            dataset.add_frame( {
                    "observation.image": agent_image,
                    "observation.wrist_image": wrist_image,
                    "observation.state": ee_pose,
                    "action": joint_q,
                    "obj_init": PnPEnv.obj_init_pose,
                    # "task": TASK_NAME,
                }, task = TASK_NAME
            )
        PnPEnv.render(teleop=True)

PnPEnv.env.close_viewer()

# Clean up the images folder
import shutil
shutil.rmtree(dataset.root / 'images')