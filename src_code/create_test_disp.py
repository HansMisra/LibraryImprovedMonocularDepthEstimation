import os

import cv2
import numpy as np
from tqdm import tqdm


def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    return image


def create_matchers(num_disparities=128, block_size=7):
    if num_disparities % 16 != 0:
        raise ValueError("num_disparities must be divisible by 16.")

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * block_size**2,
        P2=32 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    try:
        import cv2.ximgproc as ximgproc
    except ImportError as exc:
        raise ImportError(
            "opencv-contrib-python is required for WLS filtering."
        ) from exc

    right_matcher = ximgproc.createRightMatcher(left_matcher)
    return left_matcher, right_matcher, ximgproc


def generate_disparity_map(left_image, right_image, num_disparities=128, block_size=7):
    left_matcher, right_matcher, ximgproc = create_matchers(
        num_disparities=num_disparities,
        block_size=block_size
    )

    left_raw = left_matcher.compute(left_image, right_image)
    right_raw = right_matcher.compute(right_image, left_image)

    wls_filter = ximgproc.createDisparityWLSFilter(left_matcher)
    wls_filter.setLambda(8000.0)
    wls_filter.setSigmaColor(1.5)

    filtered_raw = wls_filter.filter(
        left_raw,
        left_image,
        disparity_map_right=right_raw
    )

    disparity = filtered_raw.astype(np.float32) / 16.0
    confidence = wls_filter.getConfidenceMap().astype(np.float32)

    valid = (
        np.isfinite(disparity)
        & (disparity > 0.0)
        & np.isfinite(confidence)
        & (confidence > 0.0)
    )

    return disparity, confidence, valid


def encode_kitti_disparity(disparity, valid):
    encoded = np.zeros(disparity.shape, dtype=np.uint16)

    values = np.rint(disparity[valid] * 256.0)
    values = np.clip(values, 0, np.iinfo(np.uint16).max)

    encoded[valid] = values.astype(np.uint16)
    return encoded


def make_valid_mask(valid):
    return valid.astype(np.uint8) * 255


def make_disparity_visualization(disparity, valid):
    normalized = np.zeros(disparity.shape, dtype=np.uint8)

    if np.any(valid):
        values = disparity[valid]
        low = float(np.percentile(values, 1))
        high = float(np.percentile(values, 99))

        scale = max(high - low, 1e-6)
        mapped = np.clip((disparity - low) / scale, 0.0, 1.0)

        normalized[valid] = np.rint(mapped[valid] * 255.0).astype(np.uint8)

    visualization = cv2.applyColorMap(normalized, cv2.COLORMAP_PLASMA)
    visualization[~valid] = 0
    return visualization


def save_disparity_maps(
    image_dir1,
    image_dir2,
    output_dir,
    mask_dir=None,
    vis_dir=None,
    confidence_dir=None,
    num_disparities=128,
    block_size=7
):
    if mask_dir is None:
        mask_dir = output_dir.rstrip("/\\") + "_mask"

    if vis_dir is None:
        vis_dir = output_dir.rstrip("/\\") + "_vis"

    if confidence_dir is None:
        confidence_dir = output_dir.rstrip("/\\") + "_confidence"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(confidence_dir, exist_ok=True)

    left_files = {
        f for f in os.listdir(image_dir1)
        if f.endswith("_10.png")
    }
    right_files = {
        f for f in os.listdir(image_dir2)
        if f.endswith("_10.png")
    }

    paired_files = sorted(left_files & right_files)

    if not paired_files:
        raise ValueError("No matching *_10.png stereo pairs were found.")

    for filename in tqdm(
        paired_files,
        desc="Generating disparity maps",
        unit="map"
    ):
        left_path = os.path.join(image_dir1, filename)
        right_path = os.path.join(image_dir2, filename)

        left_image = preprocess_image(left_path)
        right_image = preprocess_image(right_path)

        disparity, confidence, valid = generate_disparity_map(
            left_image,
            right_image,
            num_disparities=num_disparities,
            block_size=block_size
        )

        encoded = encode_kitti_disparity(disparity, valid)
        mask = make_valid_mask(valid)
        visualization = make_disparity_visualization(disparity, valid)

        cv2.imwrite(os.path.join(output_dir, filename), encoded)
        cv2.imwrite(os.path.join(mask_dir, filename), mask)
        cv2.imwrite(os.path.join(vis_dir, filename), visualization)

        confidence_name = os.path.splitext(filename)[0] + ".npy"
        np.save(os.path.join(confidence_dir, confidence_name), confidence)


if __name__ == "__main__":
    image_dir1 = os.path.join(
        "kitti_data", "data_scene_flow", "testing", "image_2"
    )
    image_dir2 = os.path.join(
        "kitti_data", "data_scene_flow", "testing", "image_3"
    )
    output_dir = os.path.join(
        "kitti_data", "data_scene_flow", "testing", "test_disp"
    )

    save_disparity_maps(image_dir1, image_dir2, output_dir)
