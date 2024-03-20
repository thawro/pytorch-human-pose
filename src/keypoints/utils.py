import cv2
import numpy as np

_xy_coords = tuple[int, int]
_polygon = list[_xy_coords]
_polygons = list[_polygon]

# list of smaller polygons [[x11, y11, x12, y12, ..., x1n, y1n], [x21, y21, ...]]
_coco_polygon = list[float]
_coco_polygons = list[_coco_polygon]
_mask = np.ndarray


def mask_to_polygons(mask: _mask) -> _polygons:
    """Return list of polygons. Each polygon is a list of corner coords"""
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt.squeeze().flatten().tolist() for cnt in contours]
    return contours


def coco_polygons_to_mask(polygons: _coco_polygons, h: int, w: int) -> _mask:
    if isinstance(polygons, dict):
        return coco_rle_seg_to_mask(polygons)
    seg_polygons = [np.array(polygon).reshape(-1, 2).astype(np.int32) for polygon in polygons]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask = cv2.drawContours(mask, seg_polygons, -1, 255, -1)
    return mask


def coco_rle_seg_to_mask(segmentation: dict[str, list[int]]) -> _mask:
    h, w = segmentation["size"]
    counts = segmentation["counts"]
    values = []
    value = 0
    for count in counts:
        values.extend([value] * count)
        value = (not value) * 255
    mask = np.array(values, dtype=np.uint8).reshape(w, h).T
    return mask


def coco_rle_to_seg(segmentation: dict[str, list[int]]) -> _coco_polygons | _polygons:
    mask = coco_rle_seg_to_mask(segmentation)
    return mask_to_polygons(mask)
