# Training the first-party XESS5 model

XESS5 now has an original, trainable scale-2 reconstruction architecture plus a paired-frame training pipeline. **It has no trained weights yet.** Training only begins once the user supplies a valid dataset created from scenes and assets licensed for the intended use.

## Install the training runtime

For Intel Arc on a supported Windows 11 or Linux setup, install the official PyTorch XPU wheels and then XESS5. Check Intel's current driver/runtime prerequisites before training:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/xpu
python -m pip install -e ".[training,test]"
```

For CPU-only development, the regular PyPI package can be used:

```powershell
python -m pip install -e ".[training,test]"
```

CPU training is much slower. Verify the selected device before a long run; `--device xpu` deliberately fails instead of silently training on the CPU.

## Model contract

`XESS5TemporalSR` consumes ten low-resolution input channels:

| Channels | Content | Contract |
|---|---|---|
| 0–2 | Current RGB | Display-encoded RGB, [0, 1] |
| 3 | Depth | Real depth buffer, normalized [0, 1] with a documented encoding |
| 4–5 | Motion | HWC input array of pixel displacement per frame; divided by `--motion-scale-px` and bounded to [-1, 1] |
| 6–8 | Previous output | Previous XESS5/reference output already reprojected into the current view, then downsampled |
| 9 | History-validity mask | Real mask in [0, 1]; zero means the corresponding history must not be used |

The model outputs a four-channel head at exactly 2x input width and height: RGB residuals plus a temporal blend gate. The output compositor adds the learned residual to a bicubic base and blends in history only where the supplied validity mask allows it. This is an initial architecture to benchmark, not a claim of DLSS equivalence or production quality.

Count the parameters and estimate raw weight storage for any configuration:

```powershell
python -m xess5.training.model --channels 48 --blocks 10 --context-blocks 4
```

The report estimates tensor bytes at FP16 and FP32; it does not include checkpoint metadata or guarantee the final file size. A 512 MB weight file is **not** an acceptance criterion by itself: extra parameters can increase inference cost and latency without improving image quality. Compare image quality, latency, power, and VRAM on the real target hardware before scaling the network.

## Dataset manifest

The manifest is UTF-8 JSON Lines: one JSON object per paired frame. File paths are relative to the manifest directory or absolute. A tiny example for the first frame or a scene/camera reset is:

```json
{"low_res":"input/000001.png","target":"reference/000001.png","depth":"depth/000001.npy","motion":"motion/000001.npy","history":null,"history_valid":null}
```

Each entry must explicitly include all six keys. `low_res` is the rendered input image, and `target` is its ground-truth render at exactly 2x width and height. Both are RGB images. `depth` is a float32-compatible NumPy `.npy` array with shape H×W or H×W×1, normalized to [0, 1] using one consistent depth convention. `motion` is a float32-compatible `.npy` array with shape H×W×2 in HWC order, storing pixel displacement per frame. Confirm the actual renderer's axis and sign convention and keep it consistent with history reprojection.

For a non-reset frame, `history` must be an RGB image aligned to the high-resolution target and already reprojected into the current view. `history_valid` must be an image mask or `.npy` array at target resolution, with values in [0, 1]. It marks pixels where warped history is trustworthy, including disocclusion handling. If there is no valid previous frame, both fields must explicitly be `null`; the dataset uses a zero history tensor with a zero validity mask for that reset case only. Depth and motion are never fabricated by the loader.

Create separate training and validation manifests. Split by scene or capture sequence rather than randomly splitting neighboring frames, otherwise almost identical frames can leak across the two sets. Use original renders and asset packs whose licences permit dataset creation and model training; keep their sources and licences outside the checkpoint metadata or in an accompanying provenance file.

## Start training

```powershell
python -m xess5.training.train --train-manifest data/train.jsonl --validation-manifest data/validation.jsonl --output-dir models/xess5-quality --device xpu --channels 48 --blocks 10 --context-blocks 4 --patch-size 128 --batch-size 1 --epochs 100 --amp
```

On an Intel GPU, the script uses PyTorch's XPU backend. `--amp` requests BF16 autocast and fails if the selected GPU does not report BF16 support; omit it to train in FP32. Patches are cropped on the CPU before transfer to conserve VRAM. A batch should contain frames with the same dimensions, so keep `--batch-size 1` when resolutions vary. Begin with a small dataset and a few epochs to confirm that the pipeline runs, then scale up once you have a clean validation baseline.

Training writes `checkpoint-last.pt`, `checkpoint-best.pt`, and `training-report.json` in the output directory. Without a validation manifest, “best” is selected on training loss, so a separate validation set is strongly recommended. Generated weights and local data are not meant to be committed to the source repository.

## Run a trained checkpoint on an offline frame

After training, run the checkpoint with the same real auxiliary buffers:

```powershell
python -m xess5.training.infer --checkpoint models/xess5-quality/checkpoint-best.pt --color frames/low.png --depth frames/depth.npy --motion frames/motion.npy --output work/xess5-output.png --device xpu --warmup 2 --iterations 5 --report logs/inference.json
```

For a non-reset frame, also pass `--history` and `--history-valid`. The history image must already be reprojected to the current view and exactly 2x the color frame dimensions; the mask must have the same output dimensions. Omitting both history arguments is an explicit first-frame/reset case. The report includes checkpoint SHA-256, model settings, inference time, full offline frame-path time, and a labelled FPS-equivalent estimate. It is not a measurement of in-game FPS.

## What this does not do yet

- It does not create or download training frames.
- It does not supply pretrained model weights.
- It does not yet convert this model's checkpoint into the vendored native Vulkan/XMX graph. The existing `xmx-frame` runner still expects its own fixed research graph and compatible logical-weight format.
- It does not capture a running game's color, depth, motion, or previous output. A trustworthy graphics-capture/reprojection pipeline must produce those buffers first.
- It does not claim a frame-rate target. Measure output quality, end-to-end latency, VRAM, power, and temperature on the Arc B580 before using it alongside a game.

The useful next milestone is to assemble a small, legally usable paired dataset, run the pipeline end to end, record validation metrics and visual comparisons, and only then tune model size or begin an XMX kernel implementation for this architecture.

## Reference

PyTorch's official Intel GPU installation guidance documents the XPU wheel index and supported hardware: <https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html>. The available driver/runtime requirements depend on the exact operating system and PyTorch release.
