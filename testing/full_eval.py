import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import jaccard_score, accuracy_score


from mirrornet import MirrorNet
from misc import crf_refine
from config import msd_testing_root
from dataset import ImageFolder
import lovasz_losses as L


args = {"snapshot": "160", "scale": 384, "crf": True}
pretrain_directory = "/home/research/Datasets/NormalNet/dataset"
data_root = "/home/research/Datasets/NormalNet/MSD/test"
ckpt_path = "./final_results"
exp_name = "MirrorNet"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 30
learning_rate = 0.001
momentum = 0.9
power = 0.9
DIM = 384
pretrained_path = os.path.join(
    ckpt_path, f"{exp_name}_frozen_backbone_pretrain_e100.pth"
)
f1p = os.path.join(
    ckpt_path, f"{exp_name}_finetuned_e50_frozen_backbone_pretrain_e100.pth"
)
f2p = os.path.join(
    ckpt_path, f"{exp_name}_finetuned_e100_frozen_backbone_pretrain_e100.pth"
)
f3p = os.path.join(
    ckpt_path, f"{exp_name}_finetuned_e200_frozen_backbone_pretrain_e100.pth"
)

base_y = os.path.join(ckpt_path, f"{exp_name}_base.pth")
base = os.path.join(ckpt_path, f"{exp_name}_BASE.pth")
basev2 = os.path.join(ckpt_path, f"{exp_name}_BASE_V2.pth")
low = os.path.join(ckpt_path, f"{exp_name}_BASE_bs10_frozen.pth")
# model_paths = [pretrained_path, f1p, f2p, f3p, base, basev2, low]
model_paths = [base_y, low, pretrained_path, f1p, f2p, f3p]
labels = ["Yang et al. Mirrornet", "Reproduced base", "pretrain only", "f1", "f2", "f3"]

img_transform = transforms.Compose(
    [
        transforms.Resize((args["scale"], args["scale"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

target_transform = transforms.Compose(
    [transforms.Resize((384, 384), interpolation=Image.NEAREST), transforms.ToTensor()]
)

# Dataset and DataLoader for finetuning
train_dataset = ImageFolder(
    root=data_root, img_transform=img_transform, target_transform=target_transform
)
train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)

test_dataset = ImageFolder(
    root=data_root, img_transform=img_transform, target_transform=target_transform
)
test_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)


results = {}

for i, model_path in enumerate(model_paths):
    net = MirrorNet(
        backbone_path="/home/research/Datasets/NormalNet/ICCV2019_MirrorNet/backbone/resnext/resnext_101_32x4d.pth"
    ).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    model_name = labels[i]

    iou_scores, accuracies = [], []

    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            f_4, f_3, f_2, f_1 = net(inputs)

        outputs_flat1 = (torch.sigmoid(f_1) > 0.4).view(-1).cpu().numpy()
        outputs_flat2 = (torch.sigmoid(f_1) > 0.8).view(-1).cpu().numpy()
        # outputs_flat = f_1.view(-1).cpu().numpy()
        targets_flat = targets.view(-1).cpu().numpy()

        iou = jaccard_score(
            targets_flat, outputs_flat1, average="binary", zero_division=0
        )
        acc = accuracy_score(targets_flat, outputs_flat2)

        iou_scores.append(iou)
        accuracies.append(acc)

        # Store results
        results[model_name] = {
            "mean_iou": np.mean(iou_scores) * 100,
            "mean_accuracy": np.mean(accuracies) * 100,
        }

# TODO iterate over all of the created dirs, and compute metrics with authors' functions
