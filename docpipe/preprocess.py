"""Photo preprocessing for image inputs.

Three jobs, all gated by the ``preprocess`` block of config/scenario.yaml:

``rectify``
    Geometry: deskew by projection-profile search, then perspective rectify when
    a near-rectangular document outline is found. Runs once while normalising the
    image into the pipeline-ready copy, so OCR, the review renderer and the
    report all see the same page and the same coordinates. The file in the inbox
    is never rewritten.

``ocr_input``
    Appearance: max-of-channels suppresses coloured seals and stamps (red 公章,
    blue 转章) without erasing dark ink; a local-paper delta pulls a glare band
    back down so the strokes hiding inside it stay recoverable; CLAHE / gamma
    lift rescues backlit or washed-out photos. Only the pixels handed to the OCR
    engine change, so the reviewer still sees the photo as it was taken.

``quality_flags``
    Judgement: blurry, too dark, low contrast, skew beyond the correction cap,
    page partly outside the frame, or a finger-shaped blob over the corner.
    Flags surface in the review UI and the report as "建议重拍" hints instead of
    being silently swallowed; occlusion is only marked, never guessed at.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

GEOMETRY_DEFAULTS = {
    "analyze_max_side": 900,
    "deskew": True,
    "search_range_degrees": 24.0,
    "max_deskew_degrees": 18.0,
    "search_step_degrees": 0.5,
    "min_deskew_degrees": 0.6,
    "min_projection_contrast": 1.35,
    # 旋转前的行文投影峰值至少要是不旋转时的这么多倍，否则「矫正」没有明确收益。
    # 实测：真正拍歪 7° 的页面增益 ~17 倍，强透视页的伪峰值只有 ~1.3 倍。
    "min_skew_gain": 1.5,
    "min_foreground_pixels": 400,
    "rectify": True,
    "min_quad_area_ratio": 0.45,
    "polygon_epsilon_ratio": 0.02,
    "min_corner_deviation_degrees": 9.0,
    "min_border_margin_ratio": 0.012,
    # 只碰到一条边通常只是页面贴边扫描（发票页右侧贴框很常见），至少两条边出框才说明
    # 拍漏了一角或一截。
    "partial_min_border_sides": 2,
    # 拉正后的输出边长相对文档轮廓的倍数；1.0 会缩小重采样、丢小字细节。
    "warp_output_scale": 1.6,
    "min_aspect": 0.5,
    "max_aspect": 2.6,
    "fill": 255,
}

ENHANCE_DEFAULTS = {
    "seal_suppression": True,
    "seal_chroma_threshold": 45,
    "min_colored_ratio": 0.004,
    "max_colored_ratio": 0.30,
    # 骑缝章残片专用页级下限：跨页处的章只有半边或一角入画，整页彩色占比落到主门之下
    # （实测四分之一章 ≈ 0.0028、主门 0.004），主门会静默跳过。低于主门但高于本门时，
    # 只认**贴图片边缘**的色块；内部彩色小块照旧被拒，免得把小图标、红字当章。
    "edge_seal_min_colored_ratio": 0.0015,
    # 上一条「贴边」的容差（按图片长宽比例，两轴各算）：实测残片都严丝合缝贴边，
    # 容差只用来兜住形态学开运算对边角的一两像素磨损。
    "edge_seal_margin_ratio": 0.005,
    # 彩色规则线（发票红框、表格红线）又长又细，形态学开运算先把它们从印章候选里去掉，
    # 只留下块状色斑；再按长宽比与包围盒上限筛掉整页偏色。
    "seal_open_kernel": 3,
    "seal_min_aspect": 0.25,
    "seal_max_aspect": 4.0,
    "max_seal_bbox_ratio": 0.25,
    # 抑制掩码的膨胀量：只覆盖章体与压章笔画，不做整页通道取最大（那会擦掉章外的红字）。
    "seal_mask_dilate": 9,
    "seal_merge_radius": 15,
    "min_seal_area_ratio": 0.002,
    "max_seal_area_ratio": 0.5,
    "max_seal_regions": 12,
    # 反光带压制：只动「够亮、不带颜色、而且带内的笔画已经被冲淡到接近纸面」的成片高光。
    # 判据是灰度跨度之比：同一页带外还有 165 级的纸墨跨度，带内只剩 30 级，说明字还在、
    # 只是被光冲掉了——把这个比值当作拉伸倍数用回去，既证明值得救，也说清楚该救多少。
    # 整页均匀过曝落在面积上限之外，不在这条路里：那该重拍，硬拉伸只会把纸面噪点变成假墨。
    "highlight_suppression": True,
    "highlight_min_luma": 232,
    "highlight_max_chroma": 14,
    # 种子：一个行高窗口里纸墨跨度塌到该值以下、够亮、不带颜色的地方。
    "highlight_local_window": 25,
    "highlight_max_local_range": 70,
    # 区域内至少要还剩这么多级跨度，才说明里面确有被冲淡的字（纯空白页边跨度接近 0）。
    "highlight_min_band_range": 8,
    # 单个塌掉区域占整页的比例区间：低于下限忽略碎片，高于上限说明整页都过曝，该重拍而不是拉伸。
    "min_highlight_area_ratio": 0.01,
    "max_highlight_area_ratio": 0.55,
    # 跨度比低于该值认为「没什么可救」（干净页的字缝、纸边都在 1 附近），跳过不动。
    "highlight_min_gain": 1.8,
    # 拉伸倍数上限：再大就是把纸面噪点放大成假笔画了。
    "highlight_max_gain": 6.0,
    # 目标纸面亮度 = 带外纸面亮度 + 该提升量，留一点余量，避免压过头看起来像一片浅灰底。
    "highlight_local_lift": 16,
    # 估计局部纸面亮度用的高斯半径（同样按 900px 分析图计）。
    "highlight_blur_sigma": 12,
    "contrast_lift": True,
    "min_luma_range": 100.0,
    "clahe_clip": 2.2,
    "clahe_tiles": 8,
    "brightness_lift": True,
    "min_median_brightness": 140.0,
    "gamma_floor": 0.55,
}

# Thresholds are measured on the 900px analysis copy of real page renders:
# a clean text-layer render sits at lap_var 950-2700, p1 98-128, luma range 127-157,
# median 254-255; "usable but soft" is still ~400, visibly blurred ~70, unusable ~2.
QUALITY_DEFAULTS = {
    "min_sharpness": 220.0,
    "min_median_brightness": 140.0,
    "max_ink_luma": 165.0,
    "min_luma_range": 100.0,
    "min_ocr_mean_confidence": 0.72,
    # 遮挡只标记、不硬修：手指压住的那块像素本来就没有信息，猜出来比留白更危险。
    # 判定条件全部同时成立才报：有颜色、比纸面暗、够大、够长条、而且贴到画面边缘。
    "occlusion_check": True,
    "occlusion_min_chroma": 26,
    "occlusion_min_darker": 24,
    "occlusion_min_area_ratio": 0.03,
    "occlusion_min_aspect": 2.4,
    "occlusion_max_area_ratio": 0.55,
    # 贴边判定容差：手指一定伸出画面，但拉正后的白边会让它差几个像素。
    "occlusion_border_ratio": 0.03,
}

FLAG_LABELS = {
    "LOW_RESOLUTION": "页面分辨率偏低",
    "PHOTO_BLUR": "照片偏糊",
    "PHOTO_DARK": "光线偏暗",
    "PHOTO_WASHED": "反光或字迹太浅",
    "PHOTO_SKEW_UNCORRECTED": "倾斜过大，未能拉正",
    "PHOTO_PARTIAL": "只拍到部分页面",
    "PHOTO_LOW_OCR": "文字识别置信度偏低",
    "PHOTO_OCCLUSION": "疑似手指或物体遮挡",
}


def _block(config: Dict, name: str, defaults: Dict) -> Dict:
    options = dict(defaults)
    options.update(((config.get("preprocess") or {}).get(name)) or {})
    return options


def enabled(config: Dict) -> bool:
    return bool((config.get("preprocess") or {}).get("enabled", True))


def geometry_options(config: Dict) -> Dict:
    return _block(config, "geometry", GEOMETRY_DEFAULTS)


def enhance_options(config: Dict) -> Dict:
    return _block(config, "enhance", ENHANCE_DEFAULTS)


def quality_options(config: Dict) -> Dict:
    return _block(config, "quality", QUALITY_DEFAULTS)


def flag_label(flag: str) -> str:
    return FLAG_LABELS.get(flag, flag)


def _gray(array: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(array, cv2.COLOR_RGB2GRAY) if array.ndim == 3 else array


def _shrunk(array: np.ndarray, max_side: int) -> np.ndarray:
    longest = max(array.shape[0], array.shape[1])
    if max_side <= 0 or longest <= max_side:
        return array
    ratio = max_side / float(longest)
    size = (max(1, round(array.shape[1] * ratio)), max(1, round(array.shape[0] * ratio)))
    return cv2.resize(array, size, interpolation=cv2.INTER_AREA)


def _rotation(angle: float, array: np.ndarray) -> np.ndarray:
    return cv2.getRotationMatrix2D((array.shape[1] / 2.0, array.shape[0] / 2.0), angle, 1.0)


def _rotate_small(binary: np.ndarray, angle: float) -> np.ndarray:
    return cv2.warpAffine(binary, _rotation(angle, binary),
                          (binary.shape[1], binary.shape[0]), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _profile_variance(binary: np.ndarray) -> float:
    height = binary.shape[0]
    top = int(height * 0.06)
    bottom = max(top + 1, int(height * 0.94))
    profile = binary[top:bottom].sum(axis=1, dtype=np.float64) / 255.0
    return float(np.var(profile))


def projection_skew(binary: np.ndarray, options: Dict) -> Tuple[Optional[float], float, float]:
    """Return the rotation that straightens text rows, plus peak contrast and gain.

    ``angle`` is what must be *applied* to fix the page, so callers never reason
    about opposite sign conventions. ``None`` means the row profile carries no
    usable signal (blank page, or a photo dominated by pictures). ``gain`` compares
    the best angle against the page as shot: a perspective photo has no single
    correct angle, so its peak is only marginally better than doing nothing.
    """
    if int(binary.sum() // 255) < int(options["min_foreground_pixels"]):
        return None, 0.0, 0.0
    sweep = float(options["search_range_degrees"])
    step = max(0.1, float(options["search_step_degrees"]))
    if sweep <= 0:
        return None, 0.0, 0.0
    angles = np.arange(-sweep, sweep + step * 0.5, step)
    scores = np.array([_profile_variance(_rotate_small(binary, float(angle))) for angle in angles])
    if scores.size == 0 or scores.max() <= 0:
        return None, 0.0, 0.0
    best = int(np.argmax(scores))
    contrast = float(scores[best] / (np.median(scores) + 1e-9))
    gain = float(scores[best] / (float(_profile_variance(binary)) + 1e-9))
    if contrast < float(options["min_projection_contrast"]):
        return None, contrast, gain
    return float(angles[best]), contrast, gain


def _corner_order(points: np.ndarray) -> np.ndarray:
    ordered = np.array(points, dtype=np.float32).reshape(-1, 2)
    sums = ordered.sum(axis=1)
    diffs = ordered[:, 0] - ordered[:, 1]
    return np.array([ordered[int(np.argmin(sums))], ordered[int(np.argmax(diffs))],
                     ordered[int(np.argmax(sums))], ordered[int(np.argmin(diffs))]], dtype=np.float32)


def _angle_at(previous: np.ndarray, corner: np.ndarray, following: np.ndarray) -> float:
    a, b = previous - corner, following - corner
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cosine)))))


def _quad_deviation(corners: np.ndarray) -> float:
    return float(max(abs(_angle_at(corners[(index - 1) % 4], corners[index],
                                   corners[(index + 1) % 4]) - 90.0) for index in range(4)))


def document_quad(small_gray: np.ndarray, options: Dict) -> Tuple[Optional[np.ndarray], float, bool]:
    """Find the page outline: (ordered corners, area ratio, touches border)."""
    height, width = small_gray.shape
    edges = cv2.Canny(cv2.GaussianBlur(small_gray, (5, 5), 0), 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0, False
    contour = max(contours, key=cv2.contourArea)
    area_ratio = float(cv2.contourArea(contour)) / float(max(1, width * height))
    epsilon = float(options["polygon_epsilon_ratio"]) * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(approx) != 4 or area_ratio < float(options["min_quad_area_ratio"]):
        return None, area_ratio, False
    corners = _corner_order(approx)
    margin = float(options["min_border_margin_ratio"])
    sides = [bool(np.any(corners[:, 0] <= margin * width)),
             bool(np.any(corners[:, 1] <= margin * height)),
             bool(np.any(corners[:, 0] >= width * (1 - margin))),
             bool(np.any(corners[:, 1] >= height * (1 - margin)))]
    on_border = sum(sides) >= int(options.get("partial_min_border_sides", 2))
    return corners, area_ratio, on_border


def _target_size(corners: np.ndarray, options: Dict) -> Optional[Tuple[int, int]]:
    width = max(2, round(float(np.mean([np.linalg.norm(corners[1] - corners[0]),
                                        np.linalg.norm(corners[2] - corners[3])]))))
    height = max(2, round(float(np.mean([np.linalg.norm(corners[3] - corners[0]),
                                         np.linalg.norm(corners[2] - corners[1])]))))
    if not float(options["min_aspect"]) <= width / float(height) <= float(options["max_aspect"]):
        return None
    scale = float(options.get("warp_output_scale", 1.0))
    return max(2, round(width * scale)), max(2, round(height * scale))


def _flatten(array: np.ndarray, corners: np.ndarray, target: Tuple[int, int], fill: int) -> np.ndarray:
    matrix = cv2.getPerspectiveTransform(
        corners.astype(np.float32),
        np.array([[0, 0], [target[0], 0], [target[0], target[1]], [0, target[1]]], dtype=np.float32))
    return cv2.warpPerspective(array, matrix, (target[0], target[1]), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(fill, fill, fill) if array.ndim == 3 else fill)


def rectify(array_rgb: np.ndarray, config: Dict) -> Tuple[np.ndarray, Dict]:
    """Deskew and perspective-correct one page, returning the array and a record."""
    options = geometry_options(config)
    record: Dict = {"deskew_degrees": 0.0, "warped": False, "skew_detected": None,
                    "quad_area_ratio": None, "partial": False, "uncorrected_skew": False}
    if not enabled(config) or array_rgb.size == 0:
        return array_rgb, record

    def analysis(array: np.ndarray) -> np.ndarray:
        return cv2.threshold(_shrunk(_gray(array), int(options["analyze_max_side"])), 0, 255,
                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    gray = _shrunk(_gray(array_rgb), int(options["analyze_max_side"]))
    corners, area_ratio, on_border = document_quad(gray, options)
    record["quad_area_ratio"] = round(area_ratio, 3)
    if corners is not None:
        record["partial"] = on_border
    if (corners is not None and not on_border and options["rectify"]
            and _quad_deviation(corners) >= float(options["min_corner_deviation_degrees"])):
        target = _target_size(corners, options)
        if target:
            cap = int((config.get("ingest") or {}).get("image_max_side", 2600) or 0)
            if cap > 0 and max(target) > cap:
                factor = cap / float(max(target))
                target = (max(2, round(target[0] * factor)), max(2, round(target[1] * factor)))
            scale = np.array([array_rgb.shape[1] / float(gray.shape[1]),
                              array_rgb.shape[0] / float(gray.shape[0])], dtype=np.float32)
            array_rgb = _flatten(array_rgb, corners * scale, target, int(options["fill"]))
            record["warped"] = True

    binary = analysis(array_rgb)
    angle, _contrast, gain = projection_skew(binary, options)
    record["skew_detected"] = None if angle is None else round(angle, 2)
    record["skew_gain"] = round(gain, 3)
    if angle is not None and options["deskew"]:
        if abs(angle) < float(options["min_deskew_degrees"]):
            pass
        # Two reasons not to rotate: past the cap the page was shot at an angle
        # rather than crooked, and a weak profile gain means no single angle really
        # straightens it (that is perspective, and rotating makes it read worse).
        elif abs(angle) > float(options["max_deskew_degrees"]):
            record["uncorrected_skew"] = True
        elif gain < float(options["min_skew_gain"]):
            record["uncertain_skew"] = True
        else:
            array_rgb = cv2.warpAffine(
                array_rgb, _rotation(angle, array_rgb),
                (array_rgb.shape[1], array_rgb.shape[0]), flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(int(options["fill"]),) * 3)
            record["deskew_degrees"] = round(angle, 2)
    if record["uncorrected_skew"]:
        record["partial_skew"] = True
    return array_rgb, record


def _luma_stats(small_gray: np.ndarray) -> Dict:
    return {
        "median": float(np.median(small_gray)),
        "ink": float(np.percentile(small_gray, 1)),
        "paper": float(np.percentile(small_gray, 99)),
    }


def _sharpness(small_gray: np.ndarray) -> float:
    """Edge energy after stretching the histogram, so dim or flat pages are judged
    on focus alone rather than on illumination."""
    low, high = np.percentile(small_gray, 1), np.percentile(small_gray, 99)
    if float(high) - float(low) < 2.0:
        return 0.0
    stretched = np.clip((small_gray.astype(np.float32) - low) * (255.0 / (float(high) - low)),
                        0, 255).astype(np.uint8)
    return float(cv2.Laplacian(stretched, cv2.CV_64F).var())


def quality_flags(array_rgb: np.ndarray, record: Dict, config: Dict) -> List[str]:
    """Judge whether the corrected photo is good enough to trust without a re-shoot."""
    options = quality_options(config)
    flags: List[str] = []
    if array_rgb.size == 0:
        return flags
    analyze_side = int(geometry_options(config)["analyze_max_side"])
    small = _shrunk(_gray(array_rgb), analyze_side)
    if _sharpness(small) < float(options["min_sharpness"]):
        flags.append("PHOTO_BLUR")
    stats = _luma_stats(small)
    if stats["median"] < float(options["min_median_brightness"]):
        flags.append("PHOTO_DARK")
    elif stats["ink"] > float(options["max_ink_luma"]) or (
            stats["paper"] - stats["ink"]) < float(options["min_luma_range"]):
        # A dark page already fails on brightness; calling it "washed out" too
        # would only repeat the same advice twice.
        flags.append("PHOTO_WASHED")
    if record.get("partial"):
        flags.append("PHOTO_PARTIAL")
    if record.get("partial_skew"):
        flags.append("PHOTO_SKEW_UNCORRECTED")
    if bool(options["occlusion_check"]) and _occlusion_regions(_shrunk(array_rgb, analyze_side), options):
        flags.append("PHOTO_OCCLUSION")
    return flags


def flag_from_ocr(mean_confidence: Optional[float], config: Dict) -> Optional[str]:
    floor = float(quality_options(config)["min_ocr_mean_confidence"])
    if mean_confidence is None or floor <= 0:
        return None
    return "PHOTO_LOW_OCR" if float(mean_confidence) < floor else None


def _blobs(mask: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
    """(x, y, width, height, area_ratio) for every connected component of a mask."""
    count, _labels, stats, _centroid = cv2.connectedComponentsWithStats(mask, connectivity=8)
    total = float(max(1, mask.size))
    blobs = []
    for index in range(1, count):
        blobs.append((int(stats[index, cv2.CC_STAT_LEFT]), int(stats[index, cv2.CC_STAT_TOP]),
                      int(stats[index, cv2.CC_STAT_WIDTH]), int(stats[index, cv2.CC_STAT_HEIGHT]),
                      int(stats[index, cv2.CC_STAT_AREA]) / total))
    return blobs


def _colored_regions(small_bgr: np.ndarray, options: Dict,
                     edge_only: bool = False) -> Tuple[np.ndarray, List[List[float]]]:
    """Seal-like coloured blobs and the pixel mask to suppress them.

    Red invoice rules are long thin coloured lines, not stamps; treating every
    coloured pixel as a seal used to mark a whole coloured invoice as one giant
    seal and erase text outside it. Morphological opening removes those thin
    lines first, then components are kept only when their shape and size look
    like a stamp. Nearby fragments (a ring broken by black text) are merged for
    the review box, while suppression stays inside the actual stamp pixels.

    ``edge_only`` keeps only components touching an image border (within
    ``edge_seal_margin_ratio``): the caller uses it for page-wide coloured
    ratios below the main gate, where the only explainable seal is a 骑缝章
    fragment running off the edge of the frame.
    """
    channels = small_bgr.astype(np.int16)
    spread = channels.max(axis=2) - channels.min(axis=2)
    mask = np.where(spread > int(options["seal_chroma_threshold"]), 255, 0).astype(np.uint8)
    kernel_size = max(1, int(options.get("seal_open_kernel", 3)))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                              np.ones((kernel_size, kernel_size), np.uint8))
    height, width = opened.shape[:2]
    total = float(max(1, opened.size))
    count, labels, stats, _centroid = cv2.connectedComponentsWithStats(opened, connectivity=8)
    min_area = float(options.get("min_seal_area_ratio", 0.002))
    max_area = float(options.get("max_seal_area_ratio", 0.5))
    min_aspect = float(options.get("seal_min_aspect", 0.25))
    max_aspect = float(options.get("seal_max_aspect", 4.0))
    max_bbox = float(options.get("max_seal_bbox_ratio", 0.25))
    margin_x = float(options.get("edge_seal_margin_ratio", 0.005)) * width
    margin_y = float(options.get("edge_seal_margin_ratio", 0.005)) * height
    keep = np.zeros_like(opened)
    components = []
    for index in range(1, count):
        x, y, w, h, area = (int(stats[index, cv2.CC_STAT_LEFT]),
                            int(stats[index, cv2.CC_STAT_TOP]),
                            int(stats[index, cv2.CC_STAT_WIDTH]),
                            int(stats[index, cv2.CC_STAT_HEIGHT]),
                            int(stats[index, cv2.CC_STAT_AREA]))
        ratio = area / total
        if not min_area <= ratio <= max_area:
            continue
        aspect = w / float(max(1, h))
        if not min_aspect <= aspect <= max_aspect:
            continue
        if (w * h) / total > max_bbox:
            continue
        if edge_only and not (x <= margin_x or y <= margin_y
                              or x + w >= width - margin_x
                              or y + h >= height - margin_y):
            continue
        keep[labels == index] = 255
        components.append((x, y, w, h, ratio))
    if not components:
        return np.zeros_like(opened), []
    radius = max(1, int(options.get("seal_merge_radius", 15)))
    merged = cv2.dilate(keep, np.ones((radius, radius), np.uint8))
    count, labels, stats, _centroid = cv2.connectedComponentsWithStats(merged, connectivity=8)
    regions: List[List[float]] = []
    for index in range(1, count):
        x, y, w, h = (int(stats[index, cv2.CC_STAT_LEFT]), int(stats[index, cv2.CC_STAT_TOP]),
                      int(stats[index, cv2.CC_STAT_WIDTH]), int(stats[index, cv2.CC_STAT_HEIGHT]))
        if (w * h) / total > max_bbox:
            # A merge that grew into a page-sized box is the old failure mode;
            # keep its individual components instead.
            members = [item for item in components
                       if x <= item[0] <= x + w and y <= item[1] <= y + h]
            for mx, my, mw, mh, _ratio in members:
                regions.append([round(mx / width, 4), round(my / height, 4),
                                round(min(1.0, (mx + mw) / width), 4),
                                round(min(1.0, (my + mh) / height), 4)])
            continue
        regions.append([round(x / width, 4), round(y / height, 4),
                        round(min(1.0, (x + w) / width), 4),
                        round(min(1.0, (y + h) / height), 4)])
    regions = regions[:int(options.get("max_seal_regions", 12))]
    if not regions:
        return np.zeros_like(opened), []
    dilate = max(1, int(options.get("seal_mask_dilate", 9)))
    suppression = cv2.dilate(keep, np.ones((dilate, dilate), np.uint8))
    return suppression, regions


def _occlusion_regions(small_bgr: np.ndarray, options: Dict) -> List[List[float]]:
    """Long coloured blobs that are darker than this page's paper and touch a frame edge.

    That combination is a finger or the heel of the shooter's hand rather than a stamp:
    a seal is round and printed colour is not darker than its own paper. Elongation is
    measured on the minimum-area rectangle, not the bounding box — a finger comes in
    diagonally, and its bounding box looks almost square.
    """
    gray = _gray(small_bgr).astype(np.int16)
    channels = small_bgr.astype(np.int16)
    spread = channels.max(axis=2) - channels.min(axis=2)
    paper = float(np.percentile(gray, 90))
    mask = np.where((spread >= int(options["occlusion_min_chroma"]))
                    & (paper - gray >= int(options["occlusion_min_darker"])), 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    height, width = mask.shape[:2]
    border = max(2, int(round(min(height, width) * float(options["occlusion_border_ratio"]))))
    count, labels, stats, _centroid = cv2.connectedComponentsWithStats(mask, connectivity=8)
    total = float(max(1, mask.size))
    regions = []
    for index in range(1, count):
        ratio = int(stats[index, cv2.CC_STAT_AREA]) / total
        if not float(options["occlusion_min_area_ratio"]) <= ratio <= float(options["occlusion_max_area_ratio"]):
            continue
        x, y = int(stats[index, cv2.CC_STAT_LEFT]), int(stats[index, cv2.CC_STAT_TOP])
        w, h = int(stats[index, cv2.CC_STAT_WIDTH]), int(stats[index, cv2.CC_STAT_HEIGHT])
        if min(x, y, width - x - w, height - y - h) > border:
            continue
        contours, _hierarchy = cv2.findContours((labels == index).astype(np.uint8),
                                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        (_cx, _cy), (rect_w, rect_h), _angle = cv2.minAreaRect(max(contours, key=cv2.contourArea))
        if min(rect_w, rect_h) < 1.0 or max(rect_w, rect_h) < (
                float(options["occlusion_min_aspect"]) * min(rect_w, rect_h)):
            continue
        regions.append([round(x / width, 4), round(y / height, 4),
                        round(min(1.0, (x + w) / width), 4), round(min(1.0, (y + h) / height), 4),
                        round(ratio, 4)])
    regions.sort(key=lambda item: item[4], reverse=True)
    return [item[:4] for item in regions]


def _glare_band(small_bgr: np.ndarray, options: Dict) -> Optional[Tuple[np.ndarray, float, float]]:
    """(region to stretch, gain, area share) for a washed-out band, or None when there is nothing to rescue.

    Brightness alone cannot find a reflection: an evenly lit page is white everywhere,
    and stretching that turns paper noise into fake strokes. What says "specular" is
    that the ink inside has collapsed towards the paper, so the local grey span inside
    the band is a fraction of the span the rest of the page shows. The gain is exactly
    that ratio, which both proves the rescue is worth doing and says how far to go — and
    it is measured per blob, so a partly covered page stays partly untouched. The span
    is local on purpose: a page-wide percentile would measure the lamp gradient instead.
    """
    gray = _gray(small_bgr)
    channels = small_bgr.astype(np.int16)
    spread = channels.max(axis=2) - channels.min(axis=2)
    window = max(3, int(options["highlight_local_window"]) | 1)
    kernel = np.ones((window, window), np.uint8)
    # 种子 = 一个行高窗口里纸墨跨度已经塌掉、而且发亮不带颜色的地方。
    # 空白页也满足「跨度塌掉」，但它的笔画本来就没有，救完只剩噪声——所以要求这里还有字。
    paper_local = cv2.dilate(gray, kernel)
    local_span = paper_local.astype(np.int16) - cv2.erode(gray, kernel).astype(np.int16)
    seed = np.where((local_span <= int(options["highlight_max_local_range"]))
                    & (paper_local >= int(options["highlight_min_luma"]))
                    & (spread <= int(options["highlight_max_chroma"])), 255, 0).astype(np.uint8)
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    low, high = float(options["min_highlight_area_ratio"]), float(options["max_highlight_area_ratio"])
    page_range = float(np.percentile(gray, 90)) - float(np.percentile(gray, 1))
    count, labels, stats, _centroid = cv2.connectedComponentsWithStats(seed, connectivity=8)
    region = np.zeros_like(seed)
    area, strongest = 0.0, 1.0
    for index in range(1, count):
        if not low <= int(stats[index, cv2.CC_STAT_AREA]) / float(max(1, seed.size)) <= high:
            continue
        component = labels == index
        washed = float(np.median(local_span[component]))
        # 跨度太小 = 这里根本没有字（纯空白页边），救它只会把纸面噪声放大成假墨。
        if washed < float(options["highlight_min_band_range"]):
            continue
        gain = float(np.clip(page_range / washed, 1.0, float(options["highlight_max_gain"])))
        if gain < float(options["highlight_min_gain"]):
            continue
        region[component] = 255
        area += float(component.mean())
        strongest = max(strongest, gain)
    if area < low:
        return None
    # 种子只在塌掉的笔画周围；膨胀一行字的高度，把带内被冲淡的字整个圈进待救区域。
    return cv2.dilate(region, kernel), round(strongest, 3), round(area, 4)


def _suppress_glare(bgr: np.ndarray, region: np.ndarray, gain: float,
                    analysis: np.ndarray, options: Dict) -> np.ndarray:
    """Stretch the band's contrast around the surrounding paper level.

    The paper level comes from a mask-aware blur, so only pixels outside the band vote;
    in the middle of a wide band, where that leaves no coverage, the page-wide paper
    median takes over. The estimate is low-resolution on purpose: the correction is
    applied against the full-resolution pixels, so real strokes only get deeper while
    their edges stay where they were.
    """
    gray = _gray(analysis).astype(np.float32)
    outside = (region == 0).astype(np.float32)
    sigma = float(options["highlight_blur_sigma"])
    kernel = int(sigma * 6) | 1
    coverage = cv2.GaussianBlur(outside, (kernel, kernel), sigma)
    local = cv2.GaussianBlur(gray * outside, (kernel, kernel), sigma) / np.maximum(coverage, 1e-3)
    blend = np.clip(coverage / 0.15, 0.0, 1.0)
    fallback = float(np.median(gray[region == 0])) if (region == 0).any() else 255.0
    # 留一点余量：压到与纸面齐平会让带子边缘看起来像一圈浅灰底。
    lift = float(options["highlight_local_lift"])
    paper = np.clip(blend * local + (1.0 - blend) * fallback + lift, lift, 255.0)
    size = (bgr.shape[1], bgr.shape[0])
    paper_full = cv2.resize(paper, size, interpolation=cv2.INTER_LINEAR)[..., None]
    soft = cv2.resize(region, size, interpolation=cv2.INTER_LINEAR)[..., None] / 255.0
    source = bgr.astype(np.float32)
    # 每个通道按同一倍数加深与纸面的差值；比纸面还亮的（被冲淡的白）直接落回纸面。
    corrected = paper_full - np.maximum(paper_full - source, 0.0) * gain
    return np.clip(source + (corrected - source) * soft, 0, 255).astype(np.uint8)


def ocr_input(bgr: np.ndarray, config: Dict) -> Tuple[np.ndarray, Dict]:
    """Rewrite the pixels handed to OCR: seal suppression plus contrast rescue."""
    options = enhance_options(config)
    info: Dict = {"seal_suppressed": False, "highlight_suppressed": False,
                  "contrast_lifted": False, "brightness_lifted": False,
                  "seal_edge_fragment": False}
    if not enabled(config) or bgr.size == 0 or bgr.ndim != 3 or bgr.shape[2] != 3:
        return bgr, info
    analysis = _shrunk(bgr, int(geometry_options(config)["analyze_max_side"]))
    channels = analysis.astype(np.int16)
    # Coloured seals (red 公章, blue 转章) have a wide spread between channels while
    # ink stays neutral, so chroma rather than hue is the trigger.
    spread = channels.max(axis=2) - channels.min(axis=2)
    colored_ratio = float((spread > int(options["seal_chroma_threshold"])).mean())
    info["colored_ratio"] = round(colored_ratio, 4)
    if bool(options["seal_suppression"]):
        main_min = float(options["min_colored_ratio"])
        main_max = float(options["max_colored_ratio"])
        # 主门之外再开一档「贴边残片门」：只有半边/一角入画的骑缝章整页占比低于主门，
        # 但它的色块真贴图片边缘——满足这一点才认，内部彩色小块（图标、红字）仍被拒。
        edge_only = False
        active = main_min <= colored_ratio <= main_max
        if not active and colored_ratio < main_min:
            edge_only = colored_ratio >= float(options.get("edge_seal_min_colored_ratio", 0.0015))
            active = edge_only
        if active:
            mask, regions = _colored_regions(analysis, options, edge_only=edge_only)
            if regions:
                # Max of channels keeps dark ink black while the seal reads as paper.
                # Applied only inside the stamp mask: a page-wide rewrite also erases
                # red pre-printed rules and red text away from any seal.
                top = np.maximum(np.maximum(bgr[:, :, 0], bgr[:, :, 1]), bgr[:, :, 2])
                neutral = cv2.merge([top, top, top])
                full = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
                rewritten = bgr.copy()
                rewritten[full > 0] = neutral[full > 0]
                bgr = rewritten
                info["seal_suppressed"] = True
                info["seal_regions"] = regions
                if edge_only:
                    info["seal_edge_fragment"] = True
    if bool(options["highlight_suppression"]):
        band = _glare_band(analysis, options)
        if band is not None:
            region, gain, area = band
            bgr = _suppress_glare(bgr, region, gain, analysis, options)
            info["highlight_suppressed"] = True
            info["highlight_area"] = area
            info["highlight_gain"] = gain
    single = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    stats = _luma_stats(single)
    if bool(options["brightness_lift"]) and stats["median"] < float(options["min_median_brightness"]):
        gamma = max(float(options["gamma_floor"]),
                    float(np.log(0.5) / np.log(max(0.02, stats["median"] / 255.0))))
        lut = (np.clip(np.power(np.arange(256) / 255.0, gamma), 0, 1) * 255).astype(np.uint8)
        bgr = cv2.LUT(bgr, lut)
        single = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        stats = _luma_stats(single)
        info["brightness_lifted"] = True
    if bool(options["contrast_lift"]) and (stats["paper"] - stats["ink"]) < float(options["min_luma_range"]):
        clahe = cv2.createCLAHE(clipLimit=float(options["clahe_clip"]),
                                tileGridSize=(int(options["clahe_tiles"]),) * 2)
        lifted = clahe.apply(single)
        bgr = cv2.merge([lifted, lifted, lifted])
        info["contrast_lifted"] = True
    return bgr, info
