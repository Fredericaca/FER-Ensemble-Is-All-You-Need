import itertools
import json
import random

import imgaug
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# for consistent latex font
from matplotlib import rc
from sklearn.metrics import confusion_matrix

rc("font", **{"family": "serif", "serif": ["Computer Modern Roman"]})
rc("text", usetex=True)

seed = 41
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


from tqdm import tqdm

from utils.datasets.fer2013dataset import fer2013
from utils.generals import make_batch
import EmoNeXt
import EmoViT

class_names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
    print(cm)

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title, fontsize=12)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label", fontsize=12)
    plt.xlabel("Predicted label", fontsize=12)
    plt.tight_layout()

from models import cbam_resnet50, resnet18, resnet34, resmasking_dropout1,resmasking
from EmoViT import EmoVit

EmoNeXt_model = EmoNeXt.get_model(
    num_classes=len(class_names),
    model_size="base",
).to(device)

EmoViT_model = EmoVit(len(class_names)).to(device)

model_dict = [
    (resnet18, "resnet18_test_2024Dec03_00.35"),
    (resnet34, "resnet34_test_2024Dec03_00.09"),
    (cbam_resnet50, "cbam_resnet50_test_2024Dec03_02.22"),
    (resmasking, "resmasking_rot30_2024Dec04_04.38"),
    (resmasking_dropout1, "resmasking_dropout1_test_2024Dec03_00.49"),
    (EmoNeXt_model, "EmoNeXt_base.pt"),
    (EmoViT_model, "EmoViT.pth")
]


# Trained Model Accuracy
model_weights = [
                 0.67261, 
                 0.73279, 
                 0.73558, 
                 0.72917,
                 0.67986,
                 0.68475,
                 0.69605,
                 ]


def majorityVotingEnsemble():
    
    with open("./configs/fer2013_config.json") as f:
        configs = json.load(f)
    models = []

    for model, checkpoint in model_dict:
        state = torch.load("./checkpoint/{}".format(checkpoint))
        if checkpoint.endswith("pt"):
            if "model" in state:
                model.load_state_dict(state["model"])
            else:
                model.load_state_dict(state)
        elif checkpoint.endswith("pth"):
            if "model" in state:
                model.load_state_dict(state["model"])
            else:
                model.load_state_dict(state)
        else:
            model = model(num_classes=7, in_channels=3).to(device)
            model.load_state_dict(state["net"])
        model.eval()
        models.append(model)

    correct = 0
    total = 0
    all_target = []
    all_output = []

    test_set = fer2013("test", configs, tta=True, tta_size=8)
    # test_set = fer2013('test', configs, tta=False, tta_size=0)

    with torch.no_grad():
        for idx in tqdm(range(len(test_set)), total=len(test_set), leave=False):
            images, targets = test_set[idx]

            images = make_batch(images)
            images = images.cuda(non_blocking=True)
            weighted_sums = torch.zeros(7)
            for i,some_model in enumerate(models):
                if i == 5:
                    outputs = some_model(images)[1]
                elif i == 6:
                    outputs = some_model(images)
                    outputs = outputs.logits
                else: 
                    outputs = some_model(images)
                outputs = outputs.cpu()
                outputs = F.softmax(outputs, 1)
                outputs = torch.sum(outputs, 0)
                vote1 = torch.argmax(outputs, 0)
                weighted_sums[vote1] += 1
                outputs = torch.cat([outputs[0:vote1], outputs[vote1+1:]])
                vote2 = torch.argmax(outputs)
                weighted_sums[vote2] += 1
            predicted = []
            first = torch.argmax(weighted_sums, 0).item()
            predicted.append(first)
            weighted_sums = torch.cat([weighted_sums[0:first], weighted_sums[first+1:]])
            second = torch.argmax(weighted_sums, 0).item()
            predicted.append(second)
            targets = targets.item()

            if (targets in predicted):
                correct += 1
                index = predicted.index(targets)
            else:
                index = 0
            total += 1

            all_target.append(targets)
            all_output.append(predicted[index])

            if idx % 500 == 1:
                acc = 100. * correct / total
                print("Accuracy: {:.03f}".format(acc))
    print(correct, total)
    acc = 100. * correct / total
    print("Accuracy {:.03f}".format(acc))

    all_target = np.array(all_target)
    all_output = np.array(all_output)

    matrix = confusion_matrix(all_target, all_output)
    np.set_printoptions(precision=2)

    # plt.figure(figsize=(5, 5))
    plot_confusion_matrix(
        matrix,
        classes=class_names,
        normalize=True,
        title='{} \n Accuracc: {:.03f}'.format("Majority_Voting_Ensemble with 2 possible expressions", acc),
    )

    from datetime import datetime

    plt.savefig("./ConfusionMatrix/cm_{}.png".format("majority_voting_ensemble_with_2possibleExpression"))
    plt.show()
    plt.close()


if __name__ == "__main__":
    majorityVotingEnsemble()