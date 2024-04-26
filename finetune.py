import torch
from torch import optim, nn
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import wandb

from mirrornet import MirrorNet
from dataset import ImageFolder
import lovasz_losses as L


# Configuration
data_root = "MSD_sample/train"  # Change this path to point to dataset
ckpt_path = "./ckpt"
exp_name = "Experiment_1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
finetuned_path = os.path.join(ckpt_path, f"{exp_name}_e200_frozen.pth")
args = {"snapshot": "160", "scale": 384, "crf": True}

# training settings
epochs = 200
batch_size = 30
learning_rate = 0.001
momentum = 0.9
DIM = 384

# Logging
wandb.init(
    project="MSNet",
    config={
        "Architecture": "BASE",
        "Dataset": "Custom finetuning MirrorNet after pretraining with backbone weights frozen. Pretrain was done on 100 eopochs",
        "learning-rate": learning_rate,
        "momemtum": momentum,
        "finetuned_path": finetuned_path,
        "epochs": epochs,
    },
    name="Finetune with frozen backbone 200 epochs",
)

# Transforms
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

# Model initialization and loading pretrained weights
backbone_path = "./backbone/resnext/resnext_101_32x4d.pth"

net = MirrorNet(backbone_path=backbone_path).to(device)

# comment or uncomment the block below to freeze weights or not
for name, param in net.named_parameters():
    if any(  # add in a not to freeze backbone or top layers
        layer_name in name
        for layer_name in ["layer0", "layer1", "layer2", "layer3", "layer4"]
    ):
        param.requires_grad = False

optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

scheduler = torch.optim.lr_scheduler.PolynomialLR(
    optimizer, total_iters=160, power=0.9, last_epoch=-1
)


# Finetuning
epoch_losses = []
for epoch in range(epochs):
    net.train()
    total_loss = 0.0

    with tqdm(train_loader, unit="batch") as tepoch:
        for data in tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}")

            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            f_4, f_3, f_2, f_1 = net(inputs)
            loss = L.lovasz_hinge(f_1, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            tepoch.set_postfix(loss=total_loss / len(tepoch))

            wandb.log({"Finetune_LH_Loss": loss})

        scheduler.step()

    avg_loss = total_loss / len(train_loader)
    epoch_losses.append(avg_loss)
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")


# Save the finetuned model
torch.save(net.state_dict(), finetuned_path)

print("Finetuning complete.")
