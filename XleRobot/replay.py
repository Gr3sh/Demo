import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import numpy as np
from lerobot.common.datasets.utils import write_json, serialize_dict
from mujoco_env.env import XLeRobotController

class EpisodeSampler(torch.utils.data.Sampler):
    """
    取一个单独的 Episode 出来
    """
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)

dataset = LeRobotDataset('XleRobot-demo', root='./demo_data')

# 选择哪个记录播放
episode_index = 0

episode_sampler = EpisodeSampler(dataset, episode_index)
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=1,
    batch_size=1,
    sampler=episode_sampler,
)

xml_path = './xml/scene.xml'
controller = XLeRobotController(xml_path)
controller.init_dataset(dataset)
controller.init_mode("replay")

step = 0
iter_dataloader = iter(dataloader)

controller.start_replay(dataloader)
controller.run()

stats = dataset.meta.stats
PATH = dataset.root / 'meta' / 'stats.json'
stats = serialize_dict(stats)

write_json(stats, PATH)