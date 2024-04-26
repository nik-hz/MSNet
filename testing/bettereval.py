import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from mirrornet import MirrorNet
from misc import crf_refine
from config import msd_testing_root

data_root = "/home/research/Datasets/NormalNet/MSD/test"
ckpt_path = "./ckpt"
exp_name = "MirrorNet"
args = {"snapshot": "160", "scale": 384, "crf": True}

img_transform = transforms.Compose(
    [
        transforms.Resize((args["scale"], args["scale"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

to_test = {"MSD": msd_testing_root}

to_pil = transforms.ToPILImage()

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def visualize_results():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    net = MirrorNet(
        backbone_path="/home/research/Datasets/NormalNet/ICCV2019_MirrorNet/backbone/resnext/resnext_101_32x4d.pth"
    ).to(device)

    # m_path = os.path.join(ckpt_path, f"{exp_name}_finetuned_no_pretrain.pth")
    m_path = os.path.join(
        ckpt_path, f"{exp_name}_finetuned_e100_frozen_backbone_pretrain_e100.pth"
    )

    # # m_path = os.path.join(ckpt_path, f"{exp_name}_pretrained.pth")

    net.load_state_dict(torch.load(m_path, map_location=device))
    net.eval()

    img_transform = transforms.Compose(
        [
            transforms.Resize((args["scale"], args["scale"])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    to_pil = transforms.ToPILImage()
    test_image_dir = os.path.join(data_root, "image")
    test_mask_dir = os.path.join(
        data_root, "mask"
    )  # Adjust if your ground truth directory has a different structure

    num_images = 5
    fig, axs = plt.subplots(
        num_images, 3, figsize=(10, num_images * 3)
    )  # Adjust figsize as needed

    with torch.no_grad():
        for i, img_name in enumerate(sorted(os.listdir(test_image_dir))[:num_images]):
            base_name = os.path.splitext(img_name)[0]
            img_path = os.path.join(test_image_dir, img_name)
            mask_path = os.path.join(test_mask_dir, f"{base_name}.png")

            img = Image.open(img_path)
            if img.mode != "RGB":
                img = img.convert("RGB")

            w, h = img.size
            img_var = Variable(img_transform(img).unsqueeze(0)).to(device)
            f_4, f_3, f_2, f_1 = net(img_var)

            f_1 = f_1.data.squeeze(0).cpu()
            f_1 = np.array(transforms.Resize((h, w))(to_pil(f_1)))

            # Optionally refine f_1 with CRF
            # if args['crf']:
            #     f_1 = crf_refine(np.array(img.convert('RGB')), np.array(transforms.Resize((h, w))(to_pil(f_1.data.squeeze(0).cpu()))))

            # pred_resized = pred.resize((w, h))

            # Load and prepare ground truth image for visualization
            true_mask = Image.open(mask_path)

            # Display images
            # fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[i, 0].imshow(img)
            axs[i, 0].set_title("Input Image")
            axs[i, 0].axis("off")

            axs[i, 1].imshow(f_1, cmap="gray")
            axs[i, 1].set_title("Predicted Mask")
            axs[i, 1].axis("off")

            axs[i, 2].imshow(true_mask, cmap="gray")
            axs[i, 2].set_title("Ground Truth")
            axs[i, 2].axis("off")

            plt.savefig(
                f"visualization__frozen_backbone_pretrain_e100_finetune_e100_frl.png"
            )


if __name__ == "__main__":
    visualize_results()
