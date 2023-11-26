# %%
import os
import cv2
import random
import numpy as np
import albumentations as alb
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
from test import process_refraction


class GlueDataset():
    def __init__(self, root_dir, file_match_pairs, split="train", n_px=224):
        suffix = "_{}.{}"
        self.split = split
        self.pair_files = []
        for glob_exp, indicators, ext in file_match_pairs:
            pure_files = sorted(set(glob(os.path.join(root_dir, split, glob_exp + suffix.format(indicators[0], ext)))))
            glue_files = sorted(set(glob(os.path.join(root_dir, split, glob_exp + suffix.format(indicators[1], ext)))))
            for pure_file in pure_files:
                glue_file = pure_file.replace(
                    suffix.format(indicators[0], ext),
                    suffix.format(indicators[1], ext)
                )
                if glue_file in glue_files:
                    self.pair_files.append((pure_file, glue_file))

        self.img_transform = alb.Compose([
            alb.Affine(),
            alb.RandomResizedCrop(n_px, n_px)
        ])
        self.granulity = [i for i in range(5, 11)]
        self.mask_radius = [i for i in range(20, 45, 5)]
        self.light_intensity = [i for i in range(70, 140)]
        self.light_angle = [i for i in range(60, 90, 5)]

    def __len__(self):
        return len(self.pair_files)

    def __getitem__(self, idx):
        pure_image = cv2.imread(self.pair_files[idx][0])
        pure_image = self.img_transform(image=pure_image)["image"]
        # glue_image = cv2.imread(self.pair_files[idx][1], cv2.IMREAD_GRAYSCALE)
        # mask = np.abs((glue_image - pure_image))
        glue_image, mask = process_refraction(
            pure_image,
            mask_radius=random.choice(self.mask_radius),
            glue_height=10,
            n=1.3,
            light_angle=random.choice(self.light_angle),
            color_mask=[10, 10, 10],
            granulity=random.choice(self.granulity),
            light_intensity=random.choice(self.light_intensity),
        )
        glue_image = cv2.cvtColor(glue_image, cv2.COLOR_RGB2GRAY)
        glue_image = np.expand_dims(glue_image, 0)
        return dict(
            img=glue_image,
            mask=mask,
            file=self.pair_files[idx][1]
        )


# dataset = GlueDataset(
#     root_dir="./dataset",
#     split="test",
#     file_match_pairs=[
#         ("*/*", (1, 2), "bmp"),
#         ("*/*", (0, 1), "jpg")
#     ]
# )

# glue, mask, file = dataset[0]
# print(mask.shape)
# print(mask.dtype)
# print(mask.sum())
# plt.imshow(glue, cmap="gray")
# plt.show()
# plt.imshow(mask)
# plt.show()


# if __name__ == "__main__":
#     dataset = GlueDataset(
#         root_dir="./dataset",
#         split="test",
#         file_match_pairs=[
#             ("*/*", (1, 2), "bmp"),
#             ("*/*", (0, 1), "jpg")
#         ]
#     )
#     for i in tqdm(dataset):
#         pass
# %%

# %%
