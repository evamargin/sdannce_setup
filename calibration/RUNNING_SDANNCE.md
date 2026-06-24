# Running sDANNCE — on the Linux GPU server

Training and inference need an **NVIDIA GPU on Linux** (tested RTX 3090 / A5000 /
A6000; CUDA 11.1; PyTorch 1.9). Labeling is already done on the laptop. This is all
headless — plain SSH to the server is fine.

## 1. Environment (once)
```bash
git clone https://github.com/tqxli/sdannce
cd sdannce
conda create --name sdannce python=3.8 -y
conda activate sdannce
conda install pytorch=1.9.1 torchvision=0.10.1 cudatoolkit=11.1 cudnn ffmpeg -c pytorch -c nvidia -y
pip install setuptools==59.5.0
pip install -e .
```
Check the GPU is visible: `python -c "import torch; print(torch.cuda.is_available())"` → `True`.

## 2. Stage your data
Lay out an experiment folder like the repo's `demo/` (see
`notebooks/1.visualize_mv_dataset_annotations.ipynb`). You need, per experiment:
- the **4 synced videos** (`videos/Camera1/0.mp4`, … — convert/rename as the demo expects),
- the **calibration + labels** `*_dannce.mat` (your Label3D output: `labelData` + `params` + `sync` + `camnames`),
- `io.yaml` + the COM/sDANNCE config files (copy from `demo/` and edit paths, `camnames`, `n_views: 4`, `n_channels`, skeleton/`n_keypoints`).

Copy from the laptop, e.g.:
```bash
scp -r labeling/ data/behavior_raw/ output/params/ user@server:/path/to/exp/
```

## 3. Train + predict the COM (centroid)
sDANNCE is top-down: first find the mouse's center each frame.
```bash
# edit demo/train_com.sh / predict_com.sh paths first
dannce train com   <com_config.yaml>      # or: sh demo/train_com.sh
dannce predict com <com_config.yaml>      # or: sh demo/predict_com.sh
```
Produces a `com3d` file used to crop a 3D cube around the animal.

## 4. Train / finetune sDANNCE
Recommended for ~300 labels: **finetune the pretrained model** rather than train
from scratch — see `notebooks/3.finetune_on_new_dataset.ipynb`.
```bash
dannce train sdannce <sdannce_config.yaml>   # or: sh demo/train_sdannce.sh
```
Guide's recipe: warm up the DANNCE backbone ≥100 epochs (COM augmentation on),
then finetune sDANNCE <70 epochs. Output: model weights `*.pth`.

## 5. Predict 3D pose + visualize
```bash
dannce predict sdannce <sdannce_config.yaml>   # or: sh demo/predict_sdannce.sh
dannce predict-multi-gpu ...                    # for long recordings
sh demo/vis_sdannce.sh                          # render overlays to sanity-check
```
Output: `save_data_AVG.mat` with per-frame 3D keypoint coordinates (in mm, because
calibration used mm squares).

## Notes
- The demo quickstart predicts the first 500 frames using a pretrained model
  (`demo/weights/SDANNCE_gcn_bsl_FM_ep100.pth`) — run it first to confirm the
  install works before training your own.
- If your keypoint set matches the pretrained model's, finetuning converges fast.
  If it differs a lot, you may train the pose head from scratch on your labels.
- Reference: `INSTALL.md`, `DEMO.md`, `GUIDE.md` in the sdannce repo.
