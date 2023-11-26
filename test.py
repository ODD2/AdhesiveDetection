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


# %%


process_refraction(
    img,
    mask_radius=40,
    glue_height=10,
    n=1.3,
    visualize=True,
    light_angle=60,
    color_mask=[10, 10, 10],
    granulity=10,
    light_intensity=30,

)


# %%
if __name__ == "__main__":
    x = process_refraction(
        img,
        mask_radius=40,
        glue_height=10,
        n=1.3,
        visualize=True,
        light_angle=60,
        color_mask=[10, 10, 10],
        granulity=5,
        light_intensity=70,

    )

    # %%
    for i in range(90, 30, -10):
        x = process_refraction(
            img,
            mask_radius=50,
            glue_height=0,
            n=1.3,
            visualize=True,
            light_angle=i,
        )

    # %%
    for i in range(50, 0, -5):
        x = process_refraction(
            img,
            mask_radius=50,
            glue_height=0,
            n=1.3,
            visualize=True,
            glue_la_coef=float(i) / 1000,
        )
    # %%
    for i in range(1, 50, 5):
        x = process_refraction(
            img,
            mask_radius=50,
            glue_height=0,
            n=1.3,
            visualize=True,
            light_focusness=i
        )
