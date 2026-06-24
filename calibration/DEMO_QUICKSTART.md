# sDANNCE demo — one-night quickstart (predict → visualize → train)

Goal for tonight: learn the sDANNCE flow on the **ready-made demo data** (no cameras, no
labeling, **no MATLAB**). The demo ships with labels + calibration + COM + pretrained weights,
so you just run it. Do it on your **Linux GPU server** where sDANNCE is installed.

> **Is MATLAB needed?** Not tonight. MATLAB (Label3D) only *creates* labels on your own data.
> The demo is already labeled. (For your own data later, labeling = MATLAB Label3D — there's no
> official non-MATLAB labeler — or a converter we build then.)

Order matters: **get predict + visualize working first** (guaranteed win), then train.

---

## 0. Confirm the environment (server)
```bash
conda activate sdannce
python -c "import torch; print(torch.cuda.is_available())"   # must print: True
cd /path/to/sdannce        # the repo you installed
```

## 1. Get the demo data (~5 GB) via download + scp
The server is SSH-only for big files, and Box links don't `wget` cleanly, so:
```bash
# On your LAPTOP: open this in a browser and download demo_data.zip (~5 GB)
#   https://duke.box.com/s/2aw5r4hb3u57p1abt99n15f6hkl36x5k
# then copy it to the server's demo/ folder:
scp demo_data.zip USER@SERVER:/path/to/sdannce/demo/
```
On the **server**:
```bash
cd /path/to/sdannce/demo
sh prepare_demo.sh         # unzips into the experiment folders
ls                         # you should see 2021_07_05_M4_M7/, weights/, *.sh, ...
```

## 2. Predict + visualize FIRST (the guaranteed win)
```bash
cd /path/to/sdannce/demo
sh predict_sdannce.sh      # uses pretrained weights, ~500 frames
sh vis_sdannce.sh          # renders the 3D pose overlay
```
Output lands in `demo/2021_07_05_M4_M7/SDANNCE/predict02/` plus a visualization.
View it on your laptop:
```bash
# On your LAPTOP:
scp -r USER@SERVER:/path/to/sdannce/demo/2021_07_05_M4_M7/SDANNCE/predict02 ./
```
✅ If you see a 3D skeleton tracking the animals, your install works and you've done
**predict + visualize** end to end.

## 3. (Learn the COM step)
sDANNCE is top-down: find the animal centroid (COM), then pose in a cube around it. The predict
above used a provided `COM/predict01/com3d.mat`. To see COM prediction itself:
```bash
sh predict_com.sh          # uses configs/com_mouse_config.yaml
```

## 4. Training / finetune pass
```bash
# Optional: make it finish tonight — lower epochs first
nano ../configs/sdannce_rat_config.yaml     # set epochs: 2  (or 3-5)

sh train_sdannce.sh        # dannce train sdannce ... --train-mode finetune --use-npy True
watch -n 5 nvidia-smi      # (separate terminal) confirm the GPU is working
```
✅ Success = it runs ≥1 epoch and writes a checkpoint under `.../SDANNCE/train.../` with no
error. (`sh train_com.sh` similarly trains the COM network.)

## 5. Understand the layout (so YOUR rig slots in later)
Open these and match each piece to what we already build:
- `configs/sdannce_rat_config.yaml`, `configs/com_mouse_config.yaml` — the knobs (n_views,
  n_keypoints, paths, epochs, batch size).
- One experiment folder, e.g. `demo/2021_07_05_M4_M7/`: the **videos**, the **`*_dannce.mat`**
  (calibration `params` + `sync` + `labelData` labels), the **`COM/`** outputs, `io.yaml`.

Mapping to your rig:
- Our `output/params/hires_cam*_params.mat` = the **calibration** half of that `*_dannce.mat`.
- Label3D would add the **`labelData`** (your hand labels).
- So later: calibrate (done) → label in Label3D → drop into this exact demo layout → same
  commands.

---

## Troubleshooting
| Symptom | Fix |
|---|---|
| `train_sdannce.sh` errors about missing `.npy` | Edit it: `--use-npy True` → `--use-npy False` (computes volumes on the fly; slower but no preprocessing detour). |
| `torch.cuda.is_available()` is `False` | Wrong CUDA/PyTorch; re-check the env from `INSTALL.md` (CUDA 11.1 / torch 1.9.1+cu111). |
| Can't view visualization over SSH | It only writes files — `scp` the `predict02/` folder to your laptop and open there. |
| CUDA out of memory in training | Lower `batch_size` (and/or volume size) in the config yaml. |
| Box link won't download on server | Expected — download on the laptop, `scp` over (step 1). |

## References
- Repo: `DEMO.md`, `GUIDE.md`, `INSTALL.md`.
- Notebooks: `2.inference_*` (predict), `3.finetune_on_new_dataset.ipynb` (train on new data),
  `1.visualize_mv_dataset_annotations.ipynb`.
- Our `RUNNING_SDANNCE.md` (the same flow, for your own data later).
