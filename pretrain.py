import torch
from torch import optim, nn
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import os
from PIL import Image
from tqdm import tqdm
import wandb
import time

from mirrornet import MirrorNet
from dataset import ImageFolder
from image_dataset import ImageDataset
import lovasz_losses as L


def tqdm_print(*args, **kwargs):
    """Utility function to print messages above the tqdm progress bar."""
    tqdm.write(*args, **kwargs)


# Configuration
pretrain_directory = "MSNet_sample"
ckpt_path = "./ckpt"
exp_name = "MirrorNet_Pretrain"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 100
batch_size = 30
learning_rate = 0.001
save_path = os.path.join(ckpt_path, f"{exp_name}_frozen_backbone_pretrain.pth")

wandb.init(
    project="MSNet",
    config={
        "Architecture": "MirrorNet",
        "epochs": epochs,
        "learning-rate": learning_rate,
        "save_path": save_path,
    },
    name="Yang et al. MirrorNet",
)

# Transforms
img_transform = transforms.Compose(
    [
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # transforms.Normalize([0.485], [0.229]),
    ]
)

target_transform = transforms.Compose(
    [transforms.Resize((384, 384), interpolation=Image.NEAREST), transforms.ToTensor()]
)

pretrain_dataset = ImageDataset(
    directory=pretrain_directory,
    mod="binary",
    transform=img_transform,
    shape=["cube", "plane"],
)


# for testing
# pretrain_dataset = Subset(pretrain_dataset, [0, 1])

pretrain_loader = DataLoader(
    dataset=pretrain_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)

######### PRETRAIN LOOP #########
# Model initialization
net = MirrorNet(backbone_path="./backbone/resnext/resnext_101_32x4d.pth").to(device)

# freeze non backbone layers
for name, param in net.named_parameters():
    if any(  # add in a not to freeze backbone or top layers
        layer_name in name
        for layer_name in ["layer0", "layer1", "layer2", "layer3", "layer4"]
    ):
        param.requires_grad = False

# Loss function and optimizer
# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(net.parameters(), lr=0.00001)
# TODO Setting momentum stabilizes gradients significantly but still not quite
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

total_batches = len(pretrain_loader) * epochs

start_time = time.time()

# Pretraining loop
with tqdm(total=total_batches, unit="batch") as pbar:
    for epoch in range(epochs):
        net.train()
        total_loss = 0.0

        # with tqdm(pretrain_loader, unit="batch") as tepoch:
        for i, data in enumerate(pretrain_loader):
            # tepoch.set_description(f"Epoch {epoch + 1}")

            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # Forward pass
            f_4, f_3, f_2, f_1 = net(inputs)
            loss = L.lovasz_hinge(f_1, targets, ignore=255)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (i + 1)

            # tepoch.set_postfix(loss=total_loss / len(tepoch))

            pbar.set_description(
                f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}"
            )
            pbar.update(1)

            wandb.log({"lovasz_hinge_loss": loss})

    avg_loss = total_loss / len(pretrain_loader)
    tqdm_print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.8f}")

pbar.close()
# Save the pretrained model
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)


# Save the pretrained model where we only tuned the backbone weights
torch.save(
    net.state_dict(),
    save_path,
)

print("Pretraining complete.")
