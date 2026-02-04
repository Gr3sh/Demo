from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import numpy as np
from lerobot.common.datasets.utils import write_json, serialize_dict
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.utils import dataset_to_policy_features
import torch
from PIL import Image
import torchvision
import mujoco
import time

from mujoco_env.env import XLeRobotController

device = 'cuda'

dataset_metadata = LeRobotDatasetMetadata("XleRobot-demo", root='./demo_data')
features = dataset_to_policy_features(dataset_metadata.features)

output_features = {
    key: ft for key,
    ft in features.items() if ft.type is FeatureType.ACTION
}
input_features = {
    key: ft for key,
    ft in features.items() if key not in output_features
}

# Policies are initialized with a configuration class, in this case `DiffusionConfig`. For this example,
# we'll just use the defaults and so no arguments other than input/output features need to be passed.
# Temporal ensemble to make smoother trajectory predictions
cfg = ACTConfig(input_features=input_features, output_features=output_features, chunk_size= 10, n_action_steps=1, temporal_ensemble_coeff = 0.9)
delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)
# We can now instantiate our policy with this config and the dataset stats.
policy = ACTPolicy.from_pretrained('./ckpt/act_y', config = cfg, dataset_stats=dataset_metadata.stats)
policy.to(device)

xml_path = './xml/scene.xml'
controller = XLeRobotController(xml_path)

step = 0
policy.reset()
policy.eval()

control_hz = 20
control_dt = 1.0 / control_hz


while True:
    loop_start = time.time()
    if controller.check_success():
        print("Success")
        policy.reset()
        controller.reset()
        step = 0
        continue

    state = controller.qFb[:18].copy()

    data = {
        "observation.environment_state": torch.tensor(
            state, dtype=torch.float32
        ).unsqueeze(0).to(device),
        "observation.state": torch.tensor(
            state, dtype=torch.float32
        ).unsqueeze(0).to(device),
        'task': ['XleRobot-demo'],
    }

    with torch.no_grad():
        action = policy.select_action(data)

    action = action[0].cpu().numpy()

    controller.data.ctrl[:18] = action

    mujoco.mj_step(controller.model, controller.data)

    controller.render_ui()
    step += 1

    # time.sleep(0.002)