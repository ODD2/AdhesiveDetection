# %%
import os
import cv2
import random
import numpy as np
import albumentations as alb
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
from src.synthesizer import process_refraction


class SynthesisGlueDataset():
    def __init__(self, root_dir, file_match_pairs, split="train", n_px=224, duplicate=30):
        if (split == "test"):
            raise NotImplementedError()

        suffix = "_{}.{}"
        self.split = split
        self.duplicate = duplicate
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
            for pure_file in pure_files:
                glue_file = pure_file.replace(
                    suffix.format(indicators[0], ext),
                    suffix.format(indicators[1], ext)
                )
                if glue_file in glue_files:
                    self.pair_files.append((pure_file, glue_file))

        random.seed(1019)
        random.shuffle(self.pair_files)
        if self.split == "train":
            self.pair_files = self.pair_files[:int(len(self.pair_files) * 0.7)]
        elif self.split == "valid":
            self.pair_files = self.pair_files[int(len(self.pair_files) * 0.7):]

        self.img_transform = alb.Compose([
            # alb.Affine(),
            alb.RandomResizedCrop(
                n_px,
                n_px,
                scale=[0.8, 1.0],
                always_apply=True
            )
        ])

        self.light_angle = [i for i in range(60, 90, 5)]
        self.mask_radius = [i for i in range(15, 30, 5)]
        # self.light_intensity = [i for i in range(0, 120, 20)]
        self.light_intensity = [0]
        self.glue_la_coef = [0.01, 0.005, 0.001, 0.0005, 0.0001]
        self.glue_n = [1.2, 1.25, 1.3, 1.35]
        if (self.split == "train"):
            self.granulity = [i for i in range(5, 15)]
            self.color_mask_thresh = 20
        elif (self.split == "valid"):
            self.granulity = [i for i in range(8, 15)]
            self.color_mask_thresh = 5

    def __len__(self):
        return len(self.pair_files) * self.duplicate

    def __getitem__(self, idx):
        idx = int(idx // self.duplicate)
        pure_image = cv2.imread(self.pair_files[idx][0])
        pure_image = self.img_transform(image=pure_image)["image"]
        while (True):
            try:
                glue_image, mask = process_refraction(
                    pure_image,
                    mask_radius=random.choice(self.mask_radius),
                    glue_height=10,
                    n=random.choice(self.glue_n),
                    light_angle=random.choice(self.light_angle),
                    color_mask=[
                        random.randint(0, self.color_mask_thresh)
                        for _ in range(3)
                    ],
                    glue_la_coef=random.choice(self.glue_la_coef),
                    granulity=random.choice(self.granulity),
                    light_intensity=random.choice(self.light_intensity),
                )
            except:
                continue
            else:
                break

        glue_image = cv2.cvtColor(glue_image, cv2.COLOR_RGB2GRAY)
        glue_image = np.expand_dims(glue_image, 0)
        return dict(
            img=glue_image,
            mask=mask,
            file=self.pair_files[idx][1]
        )


# %%
if __name__ == "__main__":
    dataset = SynthesisGlueDataset(
        root_dir="./dataset",
        split="train",
        file_match_pairs=[
            ("*/*", (1, 2), "bmp"),
            ("*/*", (0, 1), "jpg")
        ]
    )
    # count = 0
    # for data in tqdm(dataset):
    #     count += 1
    #     plt.imshow(data["img"].squeeze(0), cmap="gray")
    #     plt.show()
   #     if (count > 10):
    #         break

    for _ in range(10):
        plt.imshow(dataset[random.randrange(0, len(dataset))]["img"].squeeze(0), cmap="gray")
        plt.show()

    # dataset = SynthesisGlueDataset(
    #     root_dir="./dataset",
    #     split="train",
    #     file_match_pairs=[
    #         ("*/*", (1, 2), "bmp"),
    #         ("*/*", (0, 1), "jpg")
    #     ]
    # )
    # count = 0
    # for data in tqdm(dataset):
    #     count += 1
    #     plt.imshow(data["img"].squeeze(0), cmap="gray")
    #     plt.show()
    #     if (count > 10):
    #         break

# %%
