import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

from mirrornet import MirrorNet
from dataset import ImageFolder

# Configuration
data_root = "/home/research/Datasets/NormalNet/MSD/test"  # Update this path
ckpt_path = "./ckpt"
exp_name = "MirrorNet"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
batch_size = 10

args = {"snapshot": "160", "scale": 384, "crf": True}

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

# Test Dataset and DataLoader
test_dataset = ImageFolder(
    root=data_root, img_transform=img_transform, target_transform=target_transform
)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)

# Model initialization and load the finetuned weights
net = MirrorNet(backbone_path='/home/research/Datasets/NormalNet/ICCV2019_MirrorNet/backbone/resnext/resnext_101_32x4d.pth').to(device)
finetuned_path = os.path.join(ckpt_path, f"{exp_name}_base.pth")
# finetuned_path = os.path.join(ckpt_path, f"{exp_name}_finetuned.pth")

net.load_state_dict(torch.load(finetuned_path))
net.eval()


# Function to convert tensor to numpy image
# def tensor_to_numpy(tensor):
#     return tensor.cpu().detach().numpy().transpose(1, 2, 0)
# Function to convert tensor to numpy image and denormalize
def tensor_to_numpy(tensor):
    # Reverse Normalize for displaying purposes
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    tensor = tensor.cpu().detach()
    tensor = tensor * torch.Tensor(std).view(3, 1, 1) + torch.Tensor(mean).view(3, 1, 1)
    tensor = tensor.clamp(0, 1)  # Clamp values to maintain them within [0, 1]
    return tensor.numpy().transpose(1, 2, 0)


# Visualization
fig, axs = plt.subplots(batch_size, 3, figsize=(15, 5 * batch_size))
with torch.no_grad():
    for i, (images, masks) in enumerate(test_loader):
        images, masks = images.to(device), masks.to(device)
        outputs = net(images)[-1
        ]  
        # Assuming the model's forward returns a tuple and the last element is the output
        outputs = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities
        
        # print("unique")

        for j in range(batch_size):
            if i * batch_size + j >= len(
                axs
            ):  # If we have more images than the subplot size
                break

            # print(np.unique(outputs.cpu()))
            mean_val = outputs[j].mean().item()
            thresholded_output = (outputs[j] > mean_val).float()
            
            axs[j, 0].imshow(tensor_to_numpy(images[j]))
            axs[j, 0].set_title("Input Image")
            axs[j, 1].imshow(tensor_to_numpy(thresholded_output), cmap="gray")
            axs[j, 1].set_title("Predicted Mask")
            axs[j, 2].imshow(tensor_to_numpy(masks[j]), cmap="gray")
            axs[j, 2].set_title("Ground Truth Mask")

        plt.tight_layout()
        plt.savefig(f"visualization_batch_{i + 1}.png")
        plt.show()

        if i >= 0:  # Stop after saving images for the first batch
            break
