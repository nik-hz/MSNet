"""
# SLIGHTLY MODIFIED FROM THE ORIGINAL

 @Time    : 9/29/19 17:14
 @Author  : TaylorMei
 @Email   : mhy666@mail.dlut.edu.cn
 
 @Project : ICCV2019_MirrorNet
 @File    : infer.py
 @Function: predict mirror map.
 
"""

import numpy as np
import os
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from config import msd_testing_root
from misc import check_mkdir, crf_refine
from mirrornet import MirrorNet

device_ids = [0]
torch.cuda.set_device(device_ids[0])

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
low2 = os.path.join(ckpt_path, f"{exp_name}_e200_frozen.pth")
# model_paths = [pretrained_path, f1p, f2p, f3p, base, basev2, low]
model_paths = [base_y, base, low, pretrained_path, f1p, f2p, f3p, low2]
labels = [
    "Yang et al. Mirrornet",
    "Repr.",
    "Repr. freeze",
    "Pretrain",
    "FT 50E",
    "FT 100E",
    "FT 200E",
    "test",
]


img_transform = transforms.Compose(
    [
        transforms.Resize((args["scale"], args["scale"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

to_test = {"MSD": msd_testing_root}

to_pil = transforms.ToPILImage()


def main():
    for i, model_path in enumerate(model_paths):

        net = MirrorNet().cuda(device)
        net.load_state_dict(torch.load(model_path, map_location=device))
        net.eval()

        model_name = os.path.split(model_path)[1].split(".")[0]
        print(model_name)

        with torch.no_grad():
            for name, root in to_test.items():
                img_list = [
                    img_name
                    for img_name in os.listdir(
                        "/home/research/Datasets/NormalNet/MSD/test/image"
                    )
                ]
                start = time.time()
                for idx, img_name in enumerate(img_list):
                    print(
                        "predicting for {}: {:>4d} / {}".format(
                            name, idx + 1, len(img_list)
                        )
                    )
                    check_mkdir(os.path.join(ckpt_path, model_name))
                    img = Image.open(
                        os.path.join(
                            "/home/research/Datasets/NormalNet/MSD/test/",
                            "image",
                            img_name,
                        )
                    )
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                        print("{} is a gray image.".format(name))
                    w, h = img.size
                    img_var = Variable(img_transform(img).unsqueeze(0)).cuda(
                        device_ids[0]
                    )
                    f_4, f_3, f_2, f_1 = net(img_var)
                    f_4 = f_4.data.squeeze(0).cpu()
                    f_3 = f_3.data.squeeze(0).cpu()
                    f_2 = f_2.data.squeeze(0).cpu()
                    f_1 = f_1.data.squeeze(0).cpu()
                    f_4 = np.array(transforms.Resize((h, w))(to_pil(f_4)))
                    f_3 = np.array(transforms.Resize((h, w))(to_pil(f_3)))
                    f_2 = np.array(transforms.Resize((h, w))(to_pil(f_2)))
                    f_1 = np.array(transforms.Resize((h, w))(to_pil(f_1)))
                    if args["crf"]:
                        f_1 = crf_refine(np.array(img.convert("RGB")), f_1)

                    Image.fromarray(f_1).save(
                        os.path.join(ckpt_path, model_name, img_name[:-4] + ".png")
                    )

                end = time.time()
                print("Average Time Is : {:.2f}".format((end - start) / len(img_list)))


if __name__ == "__main__":
    main()
