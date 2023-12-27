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
    pro_angle = coord_angles(*pro_vec)
    pro_vec = np.array(
        coord_rotate(
            *pro_vec,
            180 + norm_a[0],
            -pro_angle[1] * 2
        )
    )
    pro_vec[2] = -pro_vec[2]
    imp_loc = surf_loc + pro_vec
    return imp_loc


# print(coord_angles(1, 2*math.sqrt(3), math.sqrt(3)))
# coord_angles(*coord_rotate(1, 2, math.sqrt(3), 120, -5))
# driver(
#     np.array([0, math.sqrt(3), 0]),
#     np.array([0, 46.7]),
#     n=3
# )

def create_ellipsoid(img, a=10, b=5, c=5, base_height=0, *args, **kargs):
    y_axis = int(img.shape[0] // 2)
    x_axis = int(img.shape[1] // 2)
    mask = cv2.ellipse(
        np.zeros(img.shape[:2]),
        (x_axis, y_axis),
        (a, b),
        0,
        0,
        360,
        255,
        -1,
    )

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if (mask[y, x] > 0):
                try:
                    mask[y, x] = math.sqrt((
                        (a * b * c)**2 -
                        ((y - y_axis) * a * c)**2 -
                        ((x - x_axis) * b * c)**2
                    ) / (a * b)**2)
                    mask[y, x] += base_height
                except:
                    mask[y, x] = 0
    return mask


def create_bar(img, radius=10, base_height=0, *args, **kargs):
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


def create_dot(img, radius=10, base_height=0, *args, **kargs):
    y_axis = int(img.shape[0] // 2)
    x_axis = int(img.shape[1] // 2)

    mask = cv2.circle(
        np.zeros(img.shape[:2]),
        (x_axis, y_axis),
        radius,
        255,
        -1
    )

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if (mask[y, x] > 0):
                mask[y, x] = math.sqrt(
                    radius**2 - (y - y_axis)**2 - (x - x_axis) ** 2
                )
                mask[y, x] += base_height
    return mask


def process_refraction(
    img,
    mask_params,
    n=1,
    visualize=False,
    granulity=10,
    color_mask=[50, 50, 50],
    glue_la_coef=0.01,
    light_angle=60,
    light_intensity=256,
    light_focusness=50,
    amb_intensity=0.3,
    amb_focusness=5,
    mask_type="bar",
    disable_mask_transform=False,
    rand_count=3
):
    mask_type_list = {
        "bar": create_bar,
        "dot": create_dot,
        "ell": create_ellipsoid,
    }

    depth = None

    # synthesis the glue
    if disable_mask_transform:
        mask_transforms = alb.Compose([])
    else:
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
                    rotate=[0, 180],
                    translate_percent=[-0.5, 0.5],
                    p=1
                )
            ]
        )

    for _ in range(rand_count):
        # create mask for the glue
        raw_mask = mask_type_list[mask_type](
            img,
            **mask_params
        )
        # get the depth of the glue
        if (depth is None):
            depth = mask_transforms(image=raw_mask)["image"]
        else:
            depth = np.maximum(depth, mask_transforms(image=raw_mask)["image"])

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

    mask = (depth >= 0.1)

    max_y = img.shape[0]
    max_x = img.shape[1]

    def check_in_bound(ref_loc):
        if (ref_loc[0] < 0 or ref_loc[1] < 0 or ref_loc[0] >= max_y or ref_loc[1] >= max_x):
            return False
        return True

    # process refraction
    _img = img.copy()
    light_infos = []
    amb_light_matrix = np.zeros(_img.shape)
    ref_light_matrix = np.ones(_img.shape, dtype=float) * 90
    for y in range(img.shape[0]):
        _light_infos = []
        for x in range(img.shape[1]):
            light_data = None

            if (mask[y, x]):
                surf_loc = (x, depth[y, x], y)
                norm_coord = normal[y, x].copy().flatten()
                norm_coord[1], norm_coord[2] = (
                    norm_coord[2], -norm_coord[1]
                )
                # vector to angles
                norm_angles = coord_angles(*norm_coord)

                # derive refract location
                ref_loc = driver(
                    n=n,
                    norm_a=norm_angles,
                    surf_loc=surf_loc
                )
                ref_loc = [round(ref_loc[2]), round(ref_loc[0])]

                # save for further usage
                light_data = dict(
                    ref_loc=ref_loc
                )

                # account for ambient light
                if (check_in_bound(ref_loc)):
                    amb_light_matrix[ref_loc[0], ref_loc[1]] += 1

                # account for light reflection
                pitch_angle = norm_angles[1]
                if (light_angle < pitch_angle):
                    ref_light_matrix[y, x] = 90 - (2 * pitch_angle - light_angle)

            _light_infos.append(light_data)
        light_infos.append(_light_infos)

    # Phase 0:
    img = img.astype(float)

    # glue property to obsorb light
    img_la = img * np.expand_dims(np.clip(1 / np.power(glue_la_coef, depth), 0, 1), -1)
    img_la[img_la > 255] = 255
    img_la[img_la < 0] = 0
    img_la = img_la.astype(np.uint8)

    # reflection
    ref_light_matrix = np.abs(ref_light_matrix)
    ref_light_matrix = cv2.GaussianBlur(ref_light_matrix, (3, 3), 0, 0)
    ref_light_matrix = 1 / np.power(light_focusness, ref_light_matrix)
    img_ref = ref_light_matrix * light_intensity
    img_ref[img_ref > 255] = 255
    img_ref[img_ref < 0] = 0
    img_ref = img_ref.astype(np.uint8)

    # refraction
    amb_light_matrix = cv2.GaussianBlur(amb_light_matrix, (11, 11), 0, 0)
    amb_light_matrix = np.power(amb_focusness, amb_light_matrix) - 1
    img_la_amb = img_la * (amb_light_matrix * amb_intensity + 1)
    img_la_amb[img_la_amb > 255] = 255
    img_la_amb[img_la_amb < 0] = 0
    img_la_amb = img_la_amb.astype(np.uint8)

    for y in range(img_la_amb.shape[0]):
        for x in range(img_la_amb.shape[1]):
            if (not light_infos[y][x] is None):
                ref_loc = light_infos[y][x]["ref_loc"]

                if (not check_in_bound(ref_loc)):
                    color = np.zeros(3, dtype=float)
                else:
                    color = img_la_amb[ref_loc[0], ref_loc[1]].astype(float)

                # Phase 1:
                # glue tint color
                color -= color_mask
                # Phase 2:
                # reflection
                color += img_ref[y, x]
                # clip
                color[color > 255] = 255
                color[color < 0] = 0
                _img[y, x] = color.astype(np.uint8)

    if (visualize):
        # plt.imshow(img)
        # plt.show()
        # plt.imshow(raw_mask)
        # plt.show()
        # plt.imshow(depth)
        # plt.show()
        # plt.imshow(img_la)
        # plt.show()
        # plt.imshow(((normal + 1) / 2)[:, :, ::-1])
        # plt.show()
        plt.imshow(_img)
        # plt.show()
        # plt.imshow(cv2.cvtColor(_img, cv2.COLOR_RGB2GRAY), cmap="gray")
        # plt.show()
        # plt.imshow(amb_light_matrix / amb_light_matrix.max(), vmax=1, vmin=0)
        # plt.show()
        # plt.imshow(mask)
        # plt.show()

    return _img, mask
