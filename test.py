# %%
import cv2
import matplotlib.pyplot as plt
from src.synthesizer import process_refraction

# %%
img = cv2.imread("/home/od/Desktop/repos/GlueFinder/dataset/train/type1/53_1.bmp")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cmap="gray")
plt.show()

# %%
import random
import numpy as np
# random.seed(1019)
# np.random.seed(1019)
a = process_refraction(
    img,
    mask_type="ell",
    mask_params=dict(
        a=250, b=200, c=40,
        base_height=1,
    ),
    n=4,
    visualize=True,
    light_angle=80,
    color_mask=[10, 10, 10],
    granulity=6,
    glue_la_coef=1.05,
    # light_intensity=128,
    light_intensity=100,
    light_focusness=5,
    amb_intensity=1.3,
    amb_focusness=3,
    # disable_mask_transform=True,
    rand_count=1
)

plt.savefig("test.png")

# %%
