import torch
from torch import optim, nn
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import os
from PIL import Image
from tqdm import tqdm
import wandb

from mirrornet import MirrorNet
from dataset import ImageFolder
from image_dataset import ImageDataset
import lovasz_losses as L


def tqdm_print(*args, **kwargs):
    """Utility function to print messages above the tqdm progress bar."""
    tqdm.write(*args, **kwargs)


# Configuration
TEST = False
HIGH_LR = False
pretrain_directory = "/home/research/Datasets/NormalNet/dataset"
data_root = "/home/research/Datasets/NormalNet/MSD/train"
ckpt_path = "./final_results"
exp_name = "MirrorNet"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pre_epochs = 100 if not TEST else 1
f1_epochs = 60 if not TEST else 1
f2_epochs = 160 if not TEST else 2
f3_epochs = 400 if not TEST else 3
# f4_epochs = 300
batch_size = 30
learning_rate = 0.001 if not HIGH_LR else 0.01
momentum = 0.9
power = 0.9
DIM = 384
pretrained_path = os.path.join(
    ckpt_path, f"{exp_name}_frozen_backbone_pretrain_e100.pth"
)
f1p = os.path.join(
    ckpt_path,
    f"{exp_name}_finetuned_e50_frozen_backbone_pretrain_e100_lr_{HIGH_LR}.pth",
)
f2p = os.path.join(
    ckpt_path,
    f"{exp_name}_finetuned_e100_frozen_backbone_pretrain_e100_lr_{HIGH_LR}.pth",
)
f3p = os.path.join(
    ckpt_path,
    f"{exp_name}_finetuned_e400_frozen_backbone_pretrain_e100.pth",
)

if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)

# # INIT CODE
# pretrain = wandb.init(
#     project="NormalNet-Final",
#     config={
#         "Architecture": "Resnext backbone with attention net",
#         "Dataset": "Custom pretraining MirrorNormalNet",
#         "infos": "Frozen backbone layers, pretraining on classifier only",
#         "epochs": pre_epochs,
#         "learning-rate": learning_rate,
#         "bs": batch_size,
#     },
#     name="Pretrain",
# )


img_transform = transforms.Compose(
    [
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # from imagenet
    ]
)

target_transform = transforms.Compose(
    [transforms.Resize((384, 384), interpolation=Image.NEAREST), transforms.ToTensor()]
)

pretrain_dataset = ImageDataset(
    directory=pretrain_directory,
    mod="binary",
    transform=img_transform,
    target_transform=target_transform,
    shape=["cube", "plane"],
)

train_dataset = ImageFolder(
    root=data_root, img_transform=img_transform, target_transform=target_transform
)

if TEST:
    pretrain_dataset = Subset(pretrain_dataset, list(range(5)))
    train_dataset = Subset(train_dataset, list(range(5)))

pretrain_loader = DataLoader(
    dataset=pretrain_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)

train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)

######### PRETRAIN LOOP #########
# Model initialization
net = MirrorNet(
    backbone_path="/home/research/Datasets/NormalNet/ICCV2019_MirrorNet/backbone/resnext/resnext_101_32x4d.pth"
).to(device)

# for name, param in net.named_parameters():
#     if any(  # add in a not to freeze backbone or top layers
#         layer_name in name
#         for layer_name in ["layer0", "layer1", "layer2", "layer3", "layer4"]
#     ):
#         param.requires_grad = False

# optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
# total_batches = len(pretrain_loader) * pre_epochs
# with tqdm(total=total_batches, unit="batch") as pbar:
#     for epoch in range(pre_epochs):
#         net.train()
#         total_loss = 0.0

#         # with tqdm(pretrain_loader, unit="batch") as tepoch:
#         for i, data in enumerate(pretrain_loader):
#             # tepoch.set_description(f"Epoch {epoch + 1}")

#             inputs, targets = data
#             inputs, targets = inputs.to(device), targets.to(device)

#             optimizer.zero_grad()

#             # Forward pass
#             f_4, f_3, f_2, f_1 = net(inputs)
#             loss = L.lovasz_hinge(f_1, targets, ignore=255)

#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#             avg_loss = total_loss / (i + 1)

#             pbar.set_description(
#                 f"Epoch {epoch + 1}/{pre_epochs}, Avg Loss: {avg_loss:.4f}"
#             )
#             pbar.update(1)

#             pretrain.log({"Pretrain_LH_Loss": loss})

#     avg_loss = total_loss / len(pretrain_loader)
#     tqdm_print(f"Epoch [{epoch + 1}/{pre_epochs}], Loss: {avg_loss:.8f}")

# pbar.close()
# torch.save(
#     net.state_dict(),
#     pretrained_path,
# )

# wandb.finish()

# FINETUNE
finetune = wandb.init(
    project="NormalNet-Final",
    config={
        "Architecture": "Resnext backbone with attention net",
        "Dataset": "Custom f1 MirrorNormalNet",
        "epochs": [f1_epochs, f2_epochs, f3_epochs],
        "learning-rate": learning_rate,
        "bs": batch_size,
    },
    name="Finetune 400 epochs",
)

for param in net.parameters():  # unfreeze all layers for finetuning
    param.requires_grad = True

# save_epochs = [f1_epochs, f2_epochs, f3_epochs]
save_epochs = [f3_epochs]

# poly decay NOT an exact representation of the original decay lr but whatver
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

# scheduler = torch.optim.lr_scheduler.PolynomialLR(
#     optimizer, total_iters=f3_epochs, power=0.9, lr_end=1e-6, last_epoch=-1
# )

total_batches = len(train_loader) * f3_epochs
with tqdm(total=total_batches, unit="batch") as pbar:
    for epoch in range(f3_epochs):
        net.train()
        total_loss = 0.0

        for i, data in enumerate(train_loader):

            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # Forward pass
            f_4, f_3, f_2, f_1 = net(inputs)
            loss = L.lovasz_hinge(f_1, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (i + 1)

            pbar.set_description(
                f"Epoch {epoch + 1}/{f3_epochs}, Avg Loss: {avg_loss:.4f}"
            )
            pbar.update(1)

            finetune.log({"Finetune_LH_Loss": loss})
            # finetune.log({"LR": optimizer.param_groups[0]["lr"]})

        # scheduler.step()

        # if epoch == f1_epochs - 1:
        #     torch.save(net.state_dict(), f1p)
        #     print(f"Saved model checkpoint at epoch {epoch+1} to {f1p}")
        # if epoch == f2_epochs - 1:
        #     torch.save(net.state_dict(), f2p)
        #     print(f"Saved model checkpoint at epoch {epoch+1} to {f2p}")
        if epoch == f3_epochs - 1:
            torch.save(net.state_dict(), f3p)
            print(f"Saved model checkpoint at epoch {epoch+1} to {f3p}")

    avg_loss = total_loss / len(pretrain_loader)
    tqdm_print(f"Epoch [{epoch + 1}/{f3_epochs}], Loss: {avg_loss:.8f}")

pbar.close()

wandb.finish()
