import os

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(
        self, directory, mod=None, transform=None, target_transform=None, shape=None
    ):
        self.directory = directory
        self.mod = mod
        self.transform = transform
        self.target_transform = target_transform
        self.shape = shape

        # Assuming the naming convention is consistent and all images are in the same directory
        if self.shape:
            self.image_pairs = [
                filename
                for filename in os.listdir(directory)
                if "normal" in filename and any(s in filename for s in self.shape)
            ]
        else:
            self.image_pairs = [
                filename for filename in os.listdir(directory) if "normal" in filename
            ]

    def binary_mask(self, image, threshold=10):
        image = np.array(image)
        # if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # _, binary_mask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        # binary_mask =
        binary_mask = (image > threshold).astype(np.uint8) * 255

        return Image.fromarray(binary_mask)
        # return binary_mask

    def bw(self, image):
        gray = image.convert("L")
        three_channel_bw = gray.convert("RGB")
        return three_channel_bw

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        normal_image_name = self.image_pairs[idx]
        # Assuming the naming convention to find the corresponding reflective image
        reflective_image_name = normal_image_name.replace("normal", "reflective")

        normal_image_path = os.path.join(self.directory, normal_image_name)
        reflective_image_path = os.path.join(self.directory, reflective_image_name)

        normal_image = Image.open(normal_image_path).convert("RGB")
        reflective_image = Image.open(reflective_image_path).convert("RGB")

        if self.mod == "binary":
            normal_image = self.binary_mask(normal_image)
        elif self.mod == "bw":
            normal_image = self.bw(normal_image)

        if self.transform:
            normal_image = self.target_transform(normal_image)
            reflective_image = self.transform(reflective_image)

        return reflective_image, normal_image
