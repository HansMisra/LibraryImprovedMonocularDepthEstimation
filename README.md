# Library-Improved Monocular Depth Estimation

Course project prototype for monocular-style depth/disparity estimation using KITTI scene flow data.

The project trains a small PyTorch encoder-decoder model (`DepthNet`) to predict a single-channel disparity map from an RGB image. It also includes an OpenCV stereo-disparity pipeline using StereoSGBM and WLS filtering to generate test disparity maps from KITTI left/right image pairs.

## What it does

- Loads KITTI RGB images and disparity maps
- Applies preprocessing and image transforms
- Trains a PyTorch CNN encoder-decoder model for disparity prediction
- Generates stereo disparity maps with OpenCV StereoSGBM + WLS filtering
- Evaluates predictions with error metrics and visualization plots
- Displays original image, target disparity, and predicted disparity side by side

## Main files

- `src_code/model.py` — defines the `DepthNet` model
- `src_code/train.py` — trains the model on KITTI image/disparity pairs
- `src_code/evaluate.py` — evaluates prediction error and visualization output
- `src_code/run_evaluation.py` — evaluation runner
- `src_code/create_test_disp.py` — creates test disparity maps from stereo image pairs

## Tools

Python, PyTorch, OpenCV, NumPy, scikit-learn, matplotlib, KITTI data

## Status

Academic computer vision prototype. The goal was to explore the full pipeline for image-based disparity estimation, including data loading, model training, stereo-derived target generation, and prediction visualization. This is not a production depth-estimation system.