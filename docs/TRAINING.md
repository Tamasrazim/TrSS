# Training the first-party TrSS model

TrSS has an original, trainable scale-2 reconstruction baseline and a paired-frame training pipeline. **It has no trained weights yet.** Training begins only after you provide a valid dataset made from scenes and assets licensed for the intended use.

## Install the training runtime

For Intel Arc on a supported Windows 11 or Linux setup, install the official PyTorch XPU wheels and then TrSS. Check Intel's current driver/runtime prerequisites before training:

\`\`\`powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/xpu
python -m pip install -e ".[training,test]"
\`\`\`

For CPU-only development, the regular PyPI package can be used:

\`\`\`powershell
python -m pip install -e ".[training,test]"
\`\`\`

CPU training is much slower. Verify the selected device before a long run; \`--device xpu\` deliberately fails rather than silently training on the CPU.

## Model contract

\`TrSSTemporalSR\` consumes ten low-resolution input channels:

| Channels | Content | Contract |
|---|---|---|
| 0–2 | Current RGB | Display-encoded RGB, [0, 1] |
| 3 | Depth | Real depth buffer, normalized [0, 1] with a documented encoding |
| 4–5 | Motion | HWC input array of pixel displacement per frame; divided by \`--motion-scale-px\` and bounded to [-1, 1] |
| 6–8 | Previous output | Previous TrSS/reference output already reprojected into the current view, then downsampled |
| 9 | History-validity mask | Real mask in [0, 1]; zero means the corresponding history must not be used |

The model outputs a four-channel head at exactly 2x input width and height: RGB residuals plus a temporal blend gate. The output compositor adds the learned residual to a bicubic base and blends in history only where the supplied validity mask allows it. This is a trainable reconstruction baseline, not a claim of DLSS equivalence or production quality.

## Dataset manifest

The manifest is UTF-8 JSON Lines: one JSON object per paired frame. File paths are relative to the manifest directory or absolute. A tiny example for the first frame or a scene/camera reset is:

\`\`\`json
{"low_res":"input/000001.png","target":"reference/000001.png","depth":"depth/000001.npy","motion":"motion/000001.npy","history":null,"history_valid":null}
\`\`\`

Each entry must explicitly include all six keys. \`low_res\` is the rendered input image, and \`target\` is its ground-truth render at exactly 2x width and height. Both are RGB images. \`depth\` is a float32-compatible NumPy \`.npy\` array with shape H×W or H×W×1, normalized to [0, 1] using one consistent depth convention. \`motion\` is a float32-compatible \`.npy\` array with shape H×W×2 in HWC order, storing pixel displacement per frame. Confirm the actual renderer's axis and sign convention and keep it consistent with history reprojection.

For a non-reset frame, \`history\` must be an RGB image aligned to the high-resolution target and already reprojected into the current view. \`history_valid\` must be an image mask or \`.npy\` array at target resolution, with values in [0, 1]. It marks pixels where warped history is trustworthy, including disocclusion handling. If there is no valid previous frame, both fields must explicitly be \`null\`; the dataset uses a zero history tensor with a zero validity mask for that reset case only. Depth and motion are never fabricated by the loader.

Create separate training and validation manifests. Split by scene or capture sequence rather than randomly splitting neighboring frames, otherwise almost identical frames can leak across the two sets. Use original renders and asset packs whose licences permit dataset creation and model training; keep their sources and licences in an accompanying provenance file.

## Start training

\`\`\`powershell
python -m trss.training.train --train-manifest data/train.jsonl --validation-manifest data/validation.jsonl --output-dir models/trss-quality --device xpu --channels 48 --blocks 10 --context-blocks 4 --patch-size 128 --batch-size 1 --epochs 100 --amp
\`\`\`

Training uses PyTorch's XPU backend when selected. \`--amp\` requests BF16 autocast and fails if the selected device does not report BF16 support; omit it to train in FP32. Patches are cropped on the CPU before transfer to conserve VRAM. Training samples receive synchronized horizontal/vertical flips; the corresponding motion-vector components have their signs reflected as well. Validation does not use random crops: it scores deterministic patches covering the full frame so the metric is not just a center-crop score. A batch should contain frames with the same dimensions, so keep \`--batch-size 1\` when resolutions vary.

Begin with a small dataset and a few epochs to confirm the pipeline runs, then scale up after establishing a clean validation baseline. Use a separate validation manifest split by scene/capture sequence.

## Resume after an interruption

Every completed epoch atomically updates \`checkpoint-last.pt\`, so an interruption while writing cannot leave the previous checkpoint half-written. The checkpoint keeps model weights, AdamW state, progress history, random crop state, and the training sample-shuffle state.

Resume with the same manifests and model dimensions; the checkpoint verifies the training/validation manifest paths and motion scale to prevent accidental dataset/preprocessing changes:

\`\`\`powershell
python -m trss.training.train --train-manifest data/train.jsonl --validation-manifest data/validation.jsonl --output-dir models/trss-quality --device xpu --channels 48 --blocks 10 --context-blocks 4 --patch-size 128 --batch-size 1 --epochs 100 --amp --resume models/trss-quality/checkpoint-last.pt
\`\`\`

Here \`--epochs\` means the total target epoch, not the number of extra epochs. To continue a run that stopped on epoch 34, keep the target at 100. On resume, TrSS restores the optimizer state and the saved random states. The cosine learning-rate schedule is computed from the absolute epoch, so continuing with the same total target matches the intended schedule. Pass \`--learning-rate\` explicitly only when deliberately changing the base rate.

Training writes \`checkpoint-last.pt\`, \`checkpoint-best.pt\`, and \`training-report.json\`. When a validation manifest is supplied, “best” is chosen by validation loss. Without one, it uses augmented training loss and is not a trustworthy generalization score. Generated weights and local data are not meant to be committed to the source repository.

## Run a trained checkpoint on an offline frame

\`\`\`powershell
python -m trss.training.infer --checkpoint models/trss-quality/checkpoint-best.pt --color frames/low.png --depth frames/depth.npy --motion frames/motion.npy --output work/trss-output.png --device xpu --warmup 2 --iterations 5 --report logs/inference.json
\`\`\`

For a non-reset frame, also pass \`--history\` and \`--history-valid\`. The history image must already be reprojected to the current view and exactly 2x the color frame dimensions; the mask must have the same output dimensions. Omitting both history arguments is an explicit first-frame/reset case. The report includes checkpoint SHA-256, model settings, inference time, full offline frame-path time, and a labelled FPS-equivalent estimate. It is not a measurement of in-game FPS.

## What this does not do yet

- It does not create or download training frames.
- It does not supply pretrained model weights.
- It does not yet convert this model's checkpoint into the vendored native Vulkan/XMX graph. The existing \`xmx-frame\` runner still expects its own fixed research graph and compatible logical-weight format.
- It does not capture a running game's color, depth, motion, or previous output. A trustworthy graphics-capture/reprojection pipeline must produce those buffers first.
- It does not claim a frame-rate target. Measure output quality, end-to-end latency, VRAM, power, and temperature on the Arc B580 before using it alongside a game.

The next milestone is to assemble a small, legally usable paired dataset, run several training epochs, compare validation metrics and visual outputs, and only then tune model size or begin an XMX kernel implementation for this architecture.

## Reference

PyTorch's official Intel GPU installation guidance documents the XPU wheel index and supported hardware: https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html. Driver/runtime requirements depend on the exact operating system and PyTorch release.
