import os
import cv2
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    # Apply histogram equalization
    image = cv2.equalizeHist(image)
    return image

def generate_disparity_map(left_image, right_image):
    # Use Semi-Global Block Matching
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,  # Increased from 16 to 64
        blockSize=11,       # Recommended block size for SGBM
        P1=8 * 3 * 11**2,   # Controls disparity smoothness
        P2=32 * 3 * 11**2,  # Larger than P1
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disparity = stereo.compute(left_image, right_image).astype(np.float32) / 16.0
    return disparity

def apply_wls_filter(left_image, disparity):
    # Ensure you have opencv-contrib-python installed to access ximgproc
    try:
        import cv2.ximgproc as ximgproc
    except ImportError:
        raise ImportError("opencv-contrib-python is not installed. Install it with 'pip install opencv-contrib-python'.")

    # Create WLS filter and apply to disparity map
    wls_filter = ximgproc.createDisparityWLSFilterGeneric(False)
    wls_filter.setLambda(8000)
    wls_filter.setSigmaColor(1.5)
    filtered_disp = wls_filter.filter(disparity, left_image)
    disp_normalized = cv2.normalize(filtered_disp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return disp_normalized

def save_disparity_maps(image_dir1, image_dir2, output_dir):
    left_image_files = [f for f in os.listdir(image_dir1) if f.endswith('_10.png')]
    right_image_files = [f for f in os.listdir(image_dir2) if f.endswith('_10.png')]
    left_image_files.sort()  
    right_image_files.sort()  

    for left_image_file, right_image_file in zip(left_image_files, right_image_files):
        left_image_path = os.path.join(image_dir1, left_image_file)
        right_image_path = os.path.join(image_dir2, right_image_file)

        left_image = preprocess_image(left_image_path)
        right_image = preprocess_image(right_image_path)

        disparity = generate_disparity_map(left_image, right_image)
        disparity_filtered = apply_wls_filter(left_image, disparity)

        output_filename = os.path.splitext(left_image_file)[0] + '.png'
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, disparity_filtered)
        print(f'Saved disparity map to {output_path}')

# Example usage
image_dir1 = os.path.join('kitti_data', 'data_scene_flow', 'testing', 'image_2')
image_dir2 = os.path.join('kitti_data', 'data_scene_flow', 'testing', 'image_3')
output_dir = os.path.join('kitti_data', 'data_scene_flow', 'testing', 'test_disp')

save_disparity_maps(image_dir1, image_dir2, output_dir)
