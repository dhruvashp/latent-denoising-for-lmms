"""
Self-contained image corruption module for VLM robustness evaluation.

Reimplements the 15 corruption types from Hendrycks & Dietterich (ICLR 2019)
using only numpy, PIL, scipy, and cv2 — no wand/ImageMagick dependency.
All corruptions work at arbitrary image sizes (not hardcoded to 224x224).

4 categories, each with corruption pools:
  Noise:   gaussian_noise, shot_noise, impulse_noise
  Blur:    defocus_blur, glass_blur, motion_blur, zoom_blur
  Weather: snow, frost, fog, brightness
  Digital: contrast, elastic_transform, pixelate, jpeg_compression
"""

import numpy as np
from PIL import Image as PILImage
from io import BytesIO
import cv2
from scipy.ndimage import zoom as scizoom
from scipy.ndimage import map_coordinates, gaussian_filter
import os
import warnings

warnings.simplefilter("ignore", UserWarning)

# Path to frost overlay images (from the robustness repo)
FROST_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'robustness',
    'ImageNet-C', 'imagenet_c', 'imagenet_c', 'frost'
)

# Pre-load frost overlays into memory at import time (avoids cv2.imread failures at runtime)
_FROST_IMAGES = []
for _fname in ['frost1.png', 'frost2.png', 'frost3.png',
               'frost4.jpg', 'frost5.jpg', 'frost6.jpg']:
    _fpath = os.path.join(FROST_DIR, _fname)
    if os.path.exists(_fpath):
        _img = cv2.imread(_fpath)
        if _img is not None:
            _FROST_IMAGES.append(_img[..., [2, 1, 0]])  # BGR -> RGB

# ============================================================
# Corruption categories
# ============================================================

CORRUPTION_CATEGORIES = {
    'noise':   ['gaussian_noise', 'shot_noise', 'impulse_noise'],
    'blur':    ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'],
    'weather': ['snow', 'frost', 'fog', 'brightness'],
    'digital': ['contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'],
}

# ============================================================
# Helpers
# ============================================================

def _disk(radius, alias_blur=0.1, dtype=np.float32):
    """Create a disk-shaped convolution kernel."""
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


def _plasma_fractal(mapsize=256, wibbledecay=3):
    """Generate a heightmap using diamond-square algorithm."""
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
                 stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def _clipped_zoom(img, zoom_factor):
    """Zoom into the center of the image by zoom_factor, keeping original size.
    Uses cv2.resize instead of scipy.ndimage.zoom for ~10x speedup."""
    h = img.shape[0]
    w = img.shape[1] if img.ndim >= 2 else h
    ch = int(np.ceil(h / float(zoom_factor)))
    cw = int(np.ceil(w / float(zoom_factor)))
    top = (h - ch) // 2
    left = (w - cw) // 2
    cropped = img[top:top + ch, left:left + cw]
    # cv2.resize is much faster than scipy.ndimage.zoom
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    if img.ndim == 2 and zoomed.ndim == 3:
        zoomed = zoomed[:, :, 0]
    return zoomed


def _cv2_motion_blur_kernel(size, angle):
    """Create a motion blur kernel using cv2 (replaces wand MotionImage)."""
    k = np.zeros((size, size), dtype=np.float32)
    k[size // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle, 1.0)
    k = cv2.warpAffine(k, M, (size, size))
    k = k / k.sum()
    return k


def _to_array(x):
    """Convert PIL Image to numpy array."""
    return np.array(x)


# ============================================================
# Noise corruptions
# ============================================================

def gaussian_noise(x, severity=1):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
    x = _to_array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, severity=1):
    c = [60, 25, 12, 5, 3][severity - 1]
    x = _to_array(x) / 255.
    return np.clip(np.random.poisson(x * c) / float(c), 0, 1) * 255


def impulse_noise(x, severity=1):
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]
    x = _to_array(x) / 255.
    # Salt-and-pepper noise (reimplemented without skimage)
    out = x.copy()
    # Salt
    salt = np.random.random(x.shape[:2]) < c / 2
    out[salt] = 1.0
    # Pepper
    pepper = np.random.random(x.shape[:2]) < c / 2
    out[pepper] = 0.0
    return np.clip(out, 0, 1) * 255


# ============================================================
# Blur corruptions
# ============================================================

def defocus_blur(x, severity=1):
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]
    x = _to_array(x) / 255.
    kernel = _disk(radius=c[0], alias_blur=c[1])
    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))
    return np.clip(channels, 0, 1) * 255


def glass_blur(x, severity=1):
    # sigma, max_delta, iterations
    c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]
    x = _to_array(x)
    H, W = x.shape[:2]
    x = np.uint8(
        gaussian_filter(x / 255., sigma=(c[0], c[0], 0)) * 255
    )
    # Vectorized local pixel shuffling
    d = c[1]
    for i in range(c[2]):
        # Generate random source coords for every pixel
        dy = np.random.randint(-d, d + 1, size=(H, W))
        dx = np.random.randint(-d, d + 1, size=(H, W))
        h_coords = np.arange(H)[:, None] + dy
        w_coords = np.arange(W)[None, :] + dx
        h_coords = np.clip(h_coords, 0, H - 1)
        w_coords = np.clip(w_coords, 0, W - 1)
        x = x[h_coords, w_coords]
    return np.clip(gaussian_filter(x / 255., sigma=(c[0], c[0], 0)), 0, 1) * 255


def motion_blur(x, severity=1):
    c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]
    x = _to_array(x).astype(np.float32) / 255.
    angle = np.random.uniform(-45, 45)
    # Kernel size must be odd
    ksize = max(3, int(c[0]) | 1)
    kernel = _cv2_motion_blur_kernel(ksize, angle)
    # Apply kernel per channel
    out = cv2.filter2D(x, -1, kernel)
    return np.clip(out, 0, 1) * 255


def zoom_blur(x, severity=1):
    c = [np.arange(1, 1.11, 0.01),
         np.arange(1, 1.16, 0.01),
         np.arange(1, 1.21, 0.02),
         np.arange(1, 1.26, 0.02),
         np.arange(1, 1.31, 0.03)][severity - 1]
    x = (_to_array(x) / 255.).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += _clipped_zoom(x, zoom_factor)
    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255


# ============================================================
# Weather corruptions
# ============================================================

def snow(x, severity=1):
    c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
         (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
         (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
         (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
         (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]

    x = _to_array(x).astype(np.float32) / 255.
    H, W = x.shape[:2]
    snow_layer = np.random.normal(size=(H, W), loc=c[0], scale=c[1])
    snow_layer = _clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0

    # Motion blur the snow layer using cv2 instead of wand
    snow_gray = (np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8)
    angle = np.random.uniform(-135, -45)
    ksize = max(3, int(c[4]) | 1)
    kernel = _cv2_motion_blur_kernel(ksize, angle)
    snow_layer = cv2.filter2D(snow_gray.astype(np.float32), -1, kernel) / 255.
    snow_layer = snow_layer[..., np.newaxis]

    gray = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(H, W, 1)
    x = c[6] * x + (1 - c[6]) * np.maximum(x, gray * 1.5 + 0.5)
    return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255


def frost(x, severity=1):
    c = [(1, 0.4), (0.8, 0.6), (0.7, 0.7), (0.65, 0.7), (0.6, 0.75)][severity - 1]
    x = _to_array(x)
    H, W = x.shape[:2]

    # Load a random frost overlay using PIL (avoids cv2 libpng issues under multiprocessing)
    frost_files = ['frost1.png', 'frost2.png', 'frost3.png',
                   'frost4.jpg', 'frost5.jpg', 'frost6.jpg']
    idx = np.random.randint(len(frost_files))
    frost_path = os.path.join(FROST_DIR, frost_files[idx])

    if os.path.exists(frost_path):
        frost_pil = PILImage.open(frost_path).convert('RGB')
        frost_pil = frost_pil.resize((W, H), PILImage.BILINEAR)
        frost_img = np.array(frost_pil)
    else:
        frost_img = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
        frost_img = cv2.GaussianBlur(frost_img, (0, 0), sigmaX=10)

    return np.clip(c[0] * x + c[1] * frost_img, 0, 255)


def fog(x, severity=1):
    c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][severity - 1]
    x = _to_array(x) / 255.
    H, W = x.shape[:2]
    max_val = x.max()
    # Generate fractal at power-of-2 size, then crop/resize
    mapsize = max(256, 1 << (max(H, W) - 1).bit_length())
    fractal = _plasma_fractal(mapsize=mapsize, wibbledecay=c[1])
    fractal = fractal[:H, :W]
    x += c[0] * fractal[..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def brightness(x, severity=1):
    c = [.1, .2, .3, .4, .5][severity - 1]
    x = _to_array(x) / 255.
    # RGB -> HSV, boost V, HSV -> RGB (using cv2 instead of skimage)
    x_uint8 = np.uint8(np.clip(x, 0, 1) * 255)
    hsv = cv2.cvtColor(x_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] / 255. + c, 0, 1) * 255.
    rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return np.clip(rgb, 0, 255).astype(np.float64)


# ============================================================
# Digital corruptions
# ============================================================

def contrast(x, severity=1):
    c = [0.4, .3, .2, .1, .05][severity - 1]
    x = _to_array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def elastic_transform(image, severity=1):
    c = [(244 * 2, 244 * 0.7, 244 * 0.1),
         (244 * 2, 244 * 0.08, 244 * 0.2),
         (244 * 0.05, 244 * 0.01, 244 * 0.02),
         (244 * 0.07, 244 * 0.01, 244 * 0.02),
         (244 * 0.12, 244 * 0.01, 244 * 0.02)][severity - 1]

    image = _to_array(image).astype(np.float32) / 255.
    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = (gaussian_filter(np.random.uniform(-1, 1, size=shape[:2]),
                          c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dy = (gaussian_filter(np.random.uniform(-1, 1, size=shape[:2]),
                          c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    return np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255


def pixelate(x, severity=1):
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
    if not isinstance(x, PILImage.Image):
        x = PILImage.fromarray(np.uint8(x))
    W, H = x.size  # PIL uses (W, H)
    x = x.resize((int(W * c), int(H * c)), PILImage.BOX)
    x = x.resize((W, H), PILImage.BOX)
    return np.array(x).astype(np.float64)


def jpeg_compression(x, severity=1):
    c = [25, 18, 15, 10, 7][severity - 1]
    if not isinstance(x, PILImage.Image):
        x = PILImage.fromarray(np.uint8(x))
    output = BytesIO()
    x.save(output, 'JPEG', quality=c)
    x = PILImage.open(output)
    return np.array(x).astype(np.float64)


# ============================================================
# Registry
# ============================================================

CORRUPTION_FUNCTIONS = {
    'gaussian_noise': gaussian_noise,
    'shot_noise': shot_noise,
    'impulse_noise': impulse_noise,
    'defocus_blur': defocus_blur,
    'glass_blur': glass_blur,
    'motion_blur': motion_blur,
    'zoom_blur': zoom_blur,
    'snow': snow,
    'frost': frost,
    'fog': fog,
    'brightness': brightness,
    'contrast': contrast,
    'elastic_transform': elastic_transform,
    'pixelate': pixelate,
    'jpeg_compression': jpeg_compression,
}


def apply_corruption(pil_image, corruption_category, severity=3, image_index=None):
    """
    Apply a random corruption from the given category to a PIL image.

    Args:
        pil_image: PIL.Image in RGB mode
        corruption_category: one of 'noise', 'blur', 'weather', 'digital'
        severity: int in [1, 5]
        image_index: int, deterministic seed so baseline and SGLD see
                     identical corrupted inputs for fair comparison.
                     Seed = hash(category, severity, image_index).

    Returns:
        PIL.Image with corruption applied
    """
    if corruption_category not in CORRUPTION_CATEGORIES:
        raise ValueError(f"Unknown category '{corruption_category}'. "
                         f"Choose from: {list(CORRUPTION_CATEGORIES.keys())}")

    pool = CORRUPTION_CATEGORIES[corruption_category]

    # Deterministic seeding: same (category, severity, index) → same corruption
    if image_index is not None:
        import hashlib
        seed = int(hashlib.sha256(f"{corruption_category}_{severity}_{image_index}".encode()).hexdigest(), 16) % (2**31)
        rng_state = np.random.get_state()
        np.random.seed(seed)

    corruption_name = pool[np.random.randint(len(pool))]
    fn = CORRUPTION_FUNCTIONS[corruption_name]

    # Some functions expect PIL input, others numpy — pass PIL, they convert internally
    result = fn(pil_image, severity=severity)

    # Restore global RNG state so we don't disturb anything else
    if image_index is not None:
        np.random.set_state(rng_state)

    # Ensure output is a PIL Image
    if isinstance(result, np.ndarray):
        result = PILImage.fromarray(np.uint8(np.clip(result, 0, 255)))
    elif not isinstance(result, PILImage.Image):
        result = PILImage.fromarray(np.uint8(np.clip(np.array(result), 0, 255)))

    return result.convert('RGB')
