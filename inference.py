import cv2
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from main import Base
from src.dataset import img_transform


def main(args):
    model = Base(n_channels=1, n_classes=2, bilinear=True)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to("cuda")
    model.eval()

    img = cv2.imread(args.image_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = np.expand_dims(img, -1)

    transform = img_transform()
    img = transform(image=img)["image"]
    img = torch.from_numpy(img).permute(2, 0, 1)
    mask = model(img.float().unsqueeze(0).to("cuda"))
    mask = (mask.softmax(dim=1)[:, 1] > 0.5).cpu()[0]
    mask = mask.to(torch.uint8)
    cv2.imwrite("mask.png", mask.numpy() * 255)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--image_path", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
