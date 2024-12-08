import json
import os
import random

import imgaug
import numpy as np
import torch

seed = 42
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import torch.nn.functional as F
from tqdm import tqdm

import models
from utils.datasets.fer2013dataset import fer2013
from utils.generals import make_batch

model_dict = [
    ("cbam_resnet50", "cbam_resnet50_test_2024Dec03_02.22"),
    ("resnet18", "resnet18_test_2024Dec03_00.35"),
    ("resnet34", "resnet34_test_2024Dec03_00.09"),
    ('resmasking', 'resmasking_rot30_2024Dec04_04.38'),
    ("resmasking_dropout1", "resmasking_dropout1_test_2024Dec03_00.49"),
]


def main():
    with open("./configs/fer2013_config.json") as f:
        configs = json.load(f)

    test_set = fer2013("test", configs, tta=True, tta_size=8)

    for model_name, checkpoint_path in model_dict:
        prediction_list = []  # each item is 7-ele array

        print("Processing", checkpoint_path)
        if os.path.exists("./saved/results/{}.npy".format(checkpoint_path)):
            continue

        model = getattr(models, model_name)
        model = model(in_channels=3, num_classes=7)

        state = torch.load(os.path.join("saved/checkpoints", checkpoint_path))
        model.load_state_dict(state["net"])

        model.cuda()
        model.eval()

        with torch.no_grad():
            for idx in tqdm(range(len(test_set)), total=len(test_set), leave=False):
                images, targets = test_set[idx]
                images = make_batch(images)
                images = images.cuda(non_blocking=True)

                outputs = model(images).cpu()
                outputs = F.softmax(outputs, 1)
                outputs = torch.sum(outputs, 0)  # outputs.shape [tta_size, 7]

                outputs = [round(o, 4) for o in outputs.numpy()]
                prediction_list.append(outputs)

        np.save("./saved/results/{}.npy".format(checkpoint_path), prediction_list)


if __name__ == "__main__":
    main()
