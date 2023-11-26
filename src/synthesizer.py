import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import albumentations as alb

COEF = (180 / math.pi)


def coord_rotate(x, y, z, ry, rz):
    ay, az = coord_angles(x, y, z)
    l2 = math.sqrt(y**2 + x**2 + z**2)
    _l1 = abs(l2 * math.cos((az + rz) / COEF))
    return (
        _l1 * math.cos((ay + ry) / COEF),
        l2 * math.sin((az + rz) / COEF),
        _l1 * math.sin((ay + ry) / COEF)
    )

# x horizontal, y vertical, z depth -> y rotate, z rotate


def coord_angles(x, y, z):
    return (
        math.atan2(z, x) * COEF,
        math.atan2(y, math.sqrt(z**2 + x**2)) * COEF
    )


def refraction(theta1, n=1.333):
    n_air = 1.003
    theta2 = math.asin((n_air / n) * math.sin(theta1 / COEF))
    return theta2 * COEF


def driver(surf_loc, norm_a, n=1.333):
    # print("norm", norm_a)
    assert norm_a[1] <= 90
    theta1 = (90 - norm_a[1])
    assert theta1 <= 90
    theta2 = refraction(theta1=theta1, n=n)
    assert theta2 < 90
    theta3 = 90 - theta2 - norm_a[1]
    pro_vec = [surf_loc[1] * math.tan(theta3 / COEF), surf_loc[1], 0]
    pro_vec = np.array(
        coord_rotate(
            *pro_vec,
            180 + norm_a[0],
            180
        )
    )
    imp_loc = surf_loc + pro_vec
    return imp_loc


# print(coord_angles(1, 2*math.sqrt(3), math.sqrt(3)))
# coord_angles(*coord_rotate(1, 2, math.sqrt(3), 120, -5))
# driver(
#     np.array([0, math.sqrt(3), 0]),
#     np.array([0, 46.7]),
#     n=3
# )


def random_create_mask(img, radius=10, base_height=0):
    y_axis = int(img.shape[0] // 2)
    from_loc = int(img.shape[1] / 5)
    dest_loc = int(img.shape[1] * (1 - 1 / 5))
    mask = cv2.rectangle(
        np.zeros(img.shape[:2]),
        (from_loc, y_axis - radius),
        (dest_loc, y_axis + radius),
        255,
        -1
    )
    for _loc in [from_loc, dest_loc]:
        mask = cv2.circle(
            mask,
            (_loc, y_axis),
            radius,
            255,
            -1
        )

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if (mask[y, x] > 0):
                if (from_loc <= x <= dest_loc):
                    mask[y, x] = math.sqrt(radius**2 - (y - y_axis)**2)
                else:
                    if (x < from_loc):
                        anchor = from_loc
                    else:
                        anchor = dest_loc
                    mask[y, x] = math.sqrt(
                        radius**2 - (y - y_axis)**2 - (x - anchor) ** 2
                    )
                mask[y, x] += base_height
    return mask


def process_refraction(
    img,
    mask_radius,
    glue_height,
    n=1,
    visualize=False,
    granulity=10,
    color_mask=[50, 50, 50],
    glue_la_coef=0.01,
    light_angle=60,
    light_intensity=256,
    light_focusness=50,
):
    # create mask for the glue
    mask = random_create_mask(
        img,
        radius=mask_radius,
        base_height=glue_height
    )

    # synthesis the glue
    mask_transforms = alb.Compose(
        [
            alb.ElasticTransform(
                alpha=80,
                sigma=granulity,
                alpha_affine=0,
                p=1
            ),
            alb.GaussianBlur(
                blur_limit=[7, 19],
                p=1
            ),
            alb.Blur(
                blur_limit=[3, 7],
                p=1
            ),
            alb.Affine(
                scale=1.0,
                shear=0,
                rotate=[-360, 360],
                translate_percent=[0.05, 0.2]
            )
        ]
    )

    # get the depth of the glue
    depth = mask_transforms(image=mask)["image"]
    # depth = mask

    # get the norm of the glue from the depth
    # !!! the coordinate system of the normal is slightly different with
    # !!! the environment, where x is horizontal, y is vertical and z is depth.
    # !!! the normal coordinate differs with the origin system for the last two axis,
    # !!! where the second(y) axis is depth and the third axis(z) is vertical.
    zy, zx = np.gradient(depth)
    normal = np.dstack((-zx, -zy, np.ones_like(depth)))
    _norm = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= _norm
    normal[:, :, 1] /= _norm
    normal[:, :, 2] /= _norm

    # process refraction
    _img = img.copy()
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if (depth[y, x] > 1):
                surf_loc = (x, depth[y, x], y)
                norm_coord = normal[y, x].copy().flatten()
                norm_coord[1], norm_coord[2] = (
                    norm_coord[2], norm_coord[1]
                )

                ref_loc = driver(
                    n=n,
                    norm_a=coord_angles(*norm_coord),
                    surf_loc=surf_loc,
                )

                norm_angle = coord_angles(*norm_coord)[1]
                ref_coef = math.cos((light_angle - norm_angle) * 2 / COEF)

                # Phase 1:
                color = img[int(ref_loc[2]), int(ref_loc[0])].astype(float)
                # glue tint color
                color -= color_mask
                # glue property to obsorb light
                color *= max((1 - glue_la_coef * depth[y, x]), 0)
                color[color > 255] = 255
                color[color < 0] = 0
                # Phase 2:
                # reflection
                ref_coef = max(min(ref_coef, 1), 0)
                color += int((ref_coef**light_focusness) * light_intensity)
                color[color > 255] = 255
                color[color < 0] = 0
                _img[y, x] = color.astype(np.uint8)

    if (visualize):
        plt.imshow(img)
        plt.show()
        plt.imshow(mask)
        plt.show()
        plt.imshow(depth)
        plt.show()
        plt.imshow(((normal + 1) / 2)[:, :, ::-1])
        plt.show()
        plt.imshow(_img)
        plt.show()
        plt.imshow(cv2.cvtColor(_img, cv2.COLOR_RGB2GRAY), cmap="gray")
        plt.show()

    return _img, depth > 1
