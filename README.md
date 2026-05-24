# Library-Improved Monocular Depth Estimation

Course project prototype for monocular-style depth/disparity estimation using KITTI scene flow data.

This project trains a small PyTorch encoder-decoder model (`DepthNet`) to predict a single-channel disparity map from an RGB image. It also includes an OpenCV stereo-disparity pipeline using StereoSGBM and WLS filtering to generate test disparity maps from KITTI left/right image pairs.

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
- `src_code/data_utils.py` — dataset loading and preprocessing utilities
- `src_code/main.py` — main project entry point, if using the full pipeline

## Basic usage

Place KITTI scene flow data under:

```bash
src_code/kitti_data/data_scene_flow/
```

Then run training from the `src_code` folder:

```bash
python train.py
```

To generate stereo-derived test disparity maps:

```bash
python create_test_disp.py
```

To run evaluation:

```bash
python run_evaluation.py
```

Depending on the local dataset location, paths inside the scripts may need to be updated before running.

## Tools

Python, PyTorch, OpenCV, NumPy, scikit-learn, matplotlib, KITTI data

## Notes

This repo does not include the KITTI dataset. The dataset should be downloaded separately and placed locally according to the expected folder structure.

Model weights and large generated outputs are intentionally excluded from version control.

## Status

Academic computer vision prototype. The goal was to explore the full pipeline for image-based disparity estimation, including data loading, model training, stereo-derived target generation, evaluation, and prediction visualization.

This is not a production depth-estimation system or a state-of-the-art monocular depth model.
