from typing import Literal

import cv2
import matplotlib.pyplot as plt
import numpy as np

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)


def stack_frames_horizontally(frames: list[np.ndarray], vspace: int = 10, hspace: int = 10):
    img_w = sum([img.shape[1] for img in frames]) + (len(frames) + 1) * vspace
    img_h = max([img.shape[0] for img in frames]) + 2 * hspace
    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    ymin = hspace
    xmin = vspace

    for frame in frames:
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        h, w = frame.shape[:2]

        if h < img_h - 2 * hspace:
            ymin = img_h // 2 - h // 2
        else:
            ymin = hspace
        img[ymin : ymin + h, xmin : xmin + w, :] = frame
        xmin = xmin + w + vspace
    return img


def make_grid(
    images: list[np.ndarray], nrows: int = 1, pad: int = 5, match_size: bool = False
) -> np.ndarray:
    _images = [img.copy() for img in images]
    if match_size:
        max_h = max([img.shape[0] for img in _images])
        max_w = max([img.shape[1] for img in _images])
        for i in range(len(_images)):
            _images[i] = cv2.resize(_images[i], (max_w, max_h))

    h, w = _images[0].shape[:2]
    ncols = int(np.ceil(len(_images) / nrows).item())
    grid_h = (h + pad) * nrows + pad
    grid_w = (w + pad) * ncols + pad
    grid = np.zeros((grid_h, grid_w, 3), dtype=_images[0].dtype)
    for idx, image in enumerate(_images):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        row = idx // ncols
        col = idx % ncols
        grid_y = pad + row * (h + pad)
        grid_x = pad + col * (w + pad)
        grid[grid_y : grid_y + h, grid_x : grid_x + w] = image
    return grid


# def resize_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:


def stack_horizontally(images: list[np.ndarray], pad: int = 5):
    max_h = max([img.shape[0] for img in images])
    new_h = max_h
    resized_images = []
    for img in images:
        h, w = img.shape[:2]
        if h != new_h:
            new_w = int(w / h * new_h)
            resized_images.append(cv2.resize(img, (new_w, new_h)))
        else:
            resized_images.append(img)
    grid_h = pad + new_h + pad
    grid_w = sum([img.shape[1] + pad for img in resized_images]) + pad
    grid = np.zeros((grid_h, grid_w, 3), dtype=images[0].dtype)

    x = pad
    for idx, image in enumerate(resized_images):
        img_w = image.shape[1]
        grid[pad:-pad, x : x + img_w] = image
        x = x + img_w + pad
    return grid


def put_txt(
    image: np.ndarray,
    labels: list[str],
    vspace: int = 10,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.5,
    thickness=1,
    loc: Literal["tl", "tc", "tr", "bl", "bc", "br"] = "tl",
    bg_color: tuple[int, int, int] = GRAY,
    txt_color: tuple[int, int, int] = WHITE,
    alpha: float = 1.0,
) -> np.ndarray:
    img_h, img_w = image.shape[:2]
    txt_h = vspace
    txt_w = 0
    for label in labels:
        (width, height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        txt_h += height + vspace
        txt_w = max(txt_w, width)
    txt_w += 2 * vspace

    if loc == "tl":
        y = vspace
        x = vspace
    elif loc == "tr":
        y = vspace
        x = img_w - txt_w - vspace
    elif loc == "bl":
        y = img_h - txt_h - vspace
        x = vspace
    elif loc == "br":
        y = img_h - txt_h - vspace
        x = img_w - txt_w - vspace
    elif loc == "tc":
        y = vspace
        x = img_w // 2 - txt_w // 2 - vspace
    elif loc == "bc":
        y = img_h - txt_h - vspace
        x = img_w // 2 - txt_w // 2 - vspace

    _x = x - vspace
    _y = y - vspace
    if alpha < 1:
        sub_img = image[_y : _y + txt_h, _x : _x + txt_w]
        white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
        cv2.rectangle(white_rect, (0, 0), (txt_w, txt_h), bg_color, -1)
        res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
        image[_y : _y + txt_h, _x : _x + txt_w] = res
    else:
        cv2.rectangle(image, (_x, _y), (_x + txt_w, _y + txt_h), bg_color, -1)

    for label in labels:
        (width, height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.putText(image, label, (x, y + height), font, font_scale, txt_color)
        y += height + vspace
    return image


def add_labels_to_frames(
    frames: list[np.ndarray],
    labels: list[str],
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.5,
    thickness=1,
):
    labeled_frames = []
    for i in range(len(frames)):
        label = labels[i]
        image = frames[i]
        if isinstance(label, str):
            label = [label]

        labeled_image = put_txt(
            image.copy(),
            label,
            vspace=5,
            font=font,
            font_scale=font_scale,
            thickness=thickness,
        )
        labeled_frames.append(labeled_image)
    return labeled_frames


# cmap = plt.cm.get_cmap("tab10")
# get_color = lambda i: (np.array(cmap(i)[:3][::-1]) * 255).astype(np.uint8)

colors = [
    (144, 238, 144),  # Light Green
    (255, 105, 180),  # Hot Pink
    (135, 206, 250),  # Sky Blue
    (255, 215, 0),  # Gold
    (255, 69, 0),  # Red-Orange
    (255, 182, 193),  # Light Pink
    (0, 128, 128),  # Teal
    (255, 160, 122),  # Light Salmon
    (0, 191, 255),  # Deep Sky Blue
    (70, 130, 180),  # Steel Blue
    (255, 99, 71),  # Tomato
    (0, 255, 255),  # Cyan
    (0, 255, 127),  # Spring Green
    (255, 0, 255),  # Magenta
    (255, 215, 0),  # Gold
    (255, 140, 0),  # Dark Orange
    (30, 144, 255),  # Dodger Blue
    (255, 20, 147),  # Deep Pink
    (255, 165, 0),  # Orange
    (218, 112, 214),  # Orchid
]
colors = colors + colors + colors + colors + colors
get_color = lambda i: np.array(colors[i]).astype(np.uint8)
