import numpy as np
import cv2


def xyxy_to_mask(xyxy: list[int] | list[tuple[int, int]], h: int, w: int) -> np.ndarray:
    if not isinstance(xyxy[0], int):
        xyxy = [xyxy[0][0], xyxy[0][1], xyxy[1][0], xyxy[1][1]]

    xmin, ymin, xmax, ymax = xyxy
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[ymin:ymax, xmin:xmax] = 255
    return mask


def mask_to_polygons(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return list of polygons. Each polygon is a list of corner coords"""
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt.squeeze().tolist() for cnt in contours]
    return contours


def mask_to_bounding_xyxy_coords(mask: np.ndarray) -> list[list[int]]:
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0].squeeze()
    xmin, ymin, width, height = cv2.boundingRect(contour)
    xmax, ymax = xmin + width, ymin + height
    return [[xmin, ymin], [xmax, ymax]]


def coco_poly_seg_to_mask(
    segmentation: list[list[float]], h: int, w: int
) -> np.ndarray:
    contours = [
        np.array(polygon).reshape(-1, 2).astype(int) for polygon in segmentation
    ]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask = cv2.drawContours(mask, contours, -1, 255, -1)
    return mask


def coco_rle_seg_to_mask(segmentation: dict[str, list[int]]) -> np.ndarray:
    h, w = segmentation["size"]
    counts = segmentation["counts"]
    values = []
    value = 0
    for count in counts:
        values.extend([value] * count)
        value = (not value) * 255
    mask = np.array(values, dtype=np.uint8).reshape(w, h)
    return mask
