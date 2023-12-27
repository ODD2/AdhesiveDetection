# %%
import os
import cv2
import torch
import random
import numpy as np
import torchvision as tv
import albumentations as alb
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
from functools import partial
from src.synthesizer import process_refraction
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode


def img_transform(n_px=224):
    return alb.Compose([
        alb.Resize(n_px, n_px, always_apply=True),
        alb.Normalize(mean=0.48145466, std=0.26862954)
    ])


class SynthesisGlueDataset():
    def __init__(
        self,
        root_dir,
        file_match_pairs,
        split="train",
        n_px=224,
        duplicate=30,
        no_transform=False
    ):
        suffix = "_{}.{}"
        self.split = split
        self.duplicate = 1 if self.split == "test" else duplicate
        self.n_px = n_px
        self.pair_files = []

        split_folder = "train" if split == "valid" else split
        for glob_exp, indicators, ext in file_match_pairs:
            pure_files = sorted(set(
                glob(
                    os.path.join(
                        root_dir,
                        split_folder,
                        glob_exp + suffix.format(indicators[0], ext)
                    )
                )
            ))
            glue_files = sorted(set(
                glob(
                    os.path.join(
                        root_dir,
                        split_folder,
                        glob_exp + suffix.format(indicators[1], ext)
                    )
                )
            ))

            mask_files = sorted(set(
                glob(
                    os.path.join(
                        root_dir,
                        split_folder,
                        glob_exp + suffix.format("mask", "png")
                    )
                )
            ))
            for pure_file in pure_files:
                glue_file = pure_file.replace(
                    suffix.format(indicators[0], ext),
                    suffix.format(indicators[1], ext)
                )
                mask_file = pure_file.replace(
                    suffix.format(indicators[0], ext),
                    suffix.format("mask", "png")
                )
                if glue_file in glue_files and mask_file in mask_files:
                    self.pair_files.append((pure_file, glue_file, mask_file))
        rng = random.Random()
        rng.seed(1019)
        rng.shuffle(self.pair_files)

        if self.split == "train":
            self.pair_files = self.pair_files[:int(len(self.pair_files) * 0.7)]
        elif self.split == "valid":
            self.pair_files = self.pair_files[int(len(self.pair_files) * 0.7):]

        if (no_transform):
            self.img_transform = lambda **x: x
        else:
            self.img_transform = img_transform(n_px)

        if (self.split == "test"):
            pass
        else:
            self.ssl_transform = alb.Compose([
                # alb.Affine(),
                alb.RandomResizedCrop(
                    n_px,
                    n_px,
                    scale=[0.5, 1.0],
                    always_apply=True
                ),
                alb.Flip(),
                alb.ImageCompression(
                    always_apply=True,
                    quality_lower=80,
                    quality_upper=100
                ),
                # alb.Blur(blur_limit=3, always_apply=True),
            ])

            self.light_angle = [i for i in range(60, 90, 10)]
            self.light_intensity = [150, 200, 250]
            self.light_focusness = [1.5, 1.8, 2, 3]
            # self.light_intensity = [0]
            # self.light_focusness = [1]

            # self.amb_intensity = [0, 2, 4, 6]
            # self.amb_focusness = [1.1, 1.15, 1.2]
            self.amb_intensity = [0]
            self.amb_focusness = [3, 4, 5]

            # self.light_intensity = [0]
            self.glue_la_coef = [1.03, 1.09, 1.15]
            self.glue_n = [2, 3, 4]
            self.mask_a = [i for i in range(50, 100, 10)]
            self.mask_b = [i for i in range(50, 100, 10)]
            self.mask_c = [i for i in range(10, 40, 5)]
            if (self.split == "train"):
                self.granulity = [10, 15, 20]
                self.color_mask_thresh = 20
            elif (self.split == "valid"):
                self.granulity = [9, 13, 17]
                self.color_mask_thresh = 5

    def __len__(self):
        if self.split == "test":
            return len(self.pair_files)
        else:
            return len(self.pair_files) * self.duplicate

    def __getitem__(self, idx, records=False):
        idx = int(idx // self.duplicate)
        pipe_records = {}
        if (self.split == "test"):
            glue_image = cv2.imread(self.pair_files[idx][1])
            mask = cv2.imread(self.pair_files[idx][2], cv2.IMREAD_GRAYSCALE)
            mask = (mask / mask.max()).astype(np.uint8)
        else:
            ssl_cond = random.random()

            if (ssl_cond < 0.2):
                pipe_records["ssl"] = False
                glue_image = cv2.imread(self.pair_files[idx][1])
                mask = cv2.imread(self.pair_files[idx][2], cv2.IMREAD_GRAYSCALE)
                mask = (mask / mask.max()).astype(np.uint8)
                result = self.ssl_transform(image=glue_image, mask=mask)
                glue_image = result["image"]
                mask = result["mask"]
            else:
                pipe_records["ssl"] = True
                pure_image = cv2.imread(self.pair_files[idx][0])
                pure_image = self.ssl_transform(image=pure_image)["image"]
                cond = random.random()
                if (cond < 0.8 or self.split == "valid"):
                    pipe_records["blank"] = False
                    while (True):
                        try:
                            pipe_params = dict(
                                mask_type="ell",
                                mask_params=dict(
                                    a=random.choice(self.mask_a),
                                    b=random.choice(self.mask_b),
                                    c=random.choice(self.mask_c),
                                    base_height=1,
                                ),
                                n=random.choice(self.glue_n),
                                light_angle=random.choice(self.light_angle),
                                color_mask=[
                                    random.randint(0, self.color_mask_thresh)
                                    for _ in range(3)
                                ],
                                glue_la_coef=random.choice(self.glue_la_coef),
                                granulity=random.choice(self.granulity),
                                light_intensity=random.choice(self.light_intensity),
                                light_focusness=random.choice(self.light_focusness),
                                amb_intensity=random.choice(self.amb_intensity),
                                amb_focusness=random.choice(self.amb_focusness),
                                rand_count=3
                            )
                            glue_image, mask = process_refraction(
                                pure_image,
                                **pipe_params
                            )
                            pipe_records["params"] = pipe_params
                        except Exception as e:
                            raise e
                            continue
                        else:
                            break
                    mask = mask.astype(np.uint8)
                else:
                    pipe_records["blank"] = True
                    glue_image = pure_image
                    mask = np.zeros(
                        shape=glue_image.shape[:2],
                        dtype=np.uint8
                    )

            # result = self.ssl_transform(image=glue_image, mask=mask)
            # glue_image = result["image"]
            # mask = result["mask"]

        glue_image = cv2.cvtColor(glue_image, cv2.COLOR_RGB2GRAY)
        glue_image = np.expand_dims(glue_image, -1)
        result = self.img_transform(image=glue_image, mask=mask)
        glue_image = torch.from_numpy(result["image"]).permute(2, 0, 1)
        mask = torch.from_numpy(result["mask"])
        ret = partial(
            dict, img=glue_image,
            mask=mask,
            file=self.pair_files[idx][1],
        )
        if (records):
            return ret(pipe_records=pipe_records)
        else:
            return ret()


# %%
if __name__ == "__main__":
    import json
    # dataset = SynthesisGlueDataset(
    #     root_dir="./dataset",
    #     split="train",
    #     file_match_pairs=[
    #         ("*/*", (1, 2), "bmp"),
    #         ("*/*", (0, 1), "jpg")
    #     ]
    # )

    # # data = dataset[random.randrange(0, len(dataset))]
    # # plt.imshow(data["img"].squeeze(0), cmap="gray")
    # # plt.show()
    # # plt.imshow(data["mask"], cmap="gray")
    # # plt.show()

    # for _ in range(10):
    #     data = dataset[random.randrange(0, len(dataset))]
    #     print(data["img"].shape)
    #     plt.imshow(data["img"].squeeze(0), cmap="gray")
    #     plt.show()
    #     plt.imshow(data["mask"], cmap="gray")
    #     plt.show()
    for split in ["train", "valid", "test"]:
        dataset = SynthesisGlueDataset(
            root_dir="./dataset",
            split=split,
            file_match_pairs=[
                ("*/*", (1, 2), "bmp"),
                ("*/*", (0, 1), "jpg")
            ]
        )

        os.makedirs(f"./misc/{split}", exist_ok=True)
        for i in range(10):
            data = dataset.__getitem__(random.randrange(0, len(dataset)), records=True)
            plt.figure(figsize=(20, 10), layout="constrained")
            plt.suptitle(json.dumps(data["pipe_records"]), wrap=True)
            plt.subplot(1, 2, 1)
            plt.imshow(data["img"].squeeze(0), cmap="gray")
            plt.subplot(1, 2, 2)
            plt.imshow(data["mask"], cmap="gray")
            plt.savefig(f"./misc/{split}/{i}.png")
            plt.close()


# %%
