import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(file_path):
    """Load an image from a file."""
    return cv2.cvtColor(cv2.imread(file_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

def load_disparity(file_path):
    """Load a disparity map from a file and convert to float type."""
    disparity = cv2.imread(file_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    return disparity / 256.0  # Disparity maps are saved as 16-bit PNGs

def calculate_depth_map(disparity, baseline, focal_length):
    """Convert disparity to depth map using the given baseline and focal length."""
    depth_map = np.where(disparity > 0, (focal_length * baseline) / disparity, 0)
    return depth_map

def main():
    data_dir = 'path/to/data_scene_flow'
    calib_dir = 'path/to/data_scene_flow_calib'
    
    # Example file names (adjust as needed)
    left_image_path = os.path.join(data_dir, 'image_2', '000000_10.png')
    right_image_path = os.path.join(data_dir, 'image_3', '000000_10.png')
    disparity_path = os.path.join(data_dir, 'disp_occ_0', '000000_10.png')
    
    # Load images and disparity
    left_image = load_image(left_image_path)
    right_image = load_image(right_image_path)
    disparity = load_disparity(disparity_path)
    
    # Assuming a fixed baseline and focal length (you need to get these values from calib files)
    baseline = 0.54  # meters
    focal_length = 721.5377  # pixels

    # Calculate depth map
    depth_map = calculate_depth_map(disparity, baseline, focal_length)

    # Display results
    plt.figure(figsize=(10, 7))
    plt.subplot(1, 3, 1)
    plt.imshow(left_image)
    plt.title('Left Image')
    plt.subplot(1, 3, 2)
    plt.imshow(disparity, cmap='viridis')
    plt.title('Disparity Map')
    plt.subplot(1, 3, 3)
    plt.imshow(depth_map, cmap='plasma')
    plt.title('Depth Map')
    plt.show()

if __name__ == '__main__':
    main()
