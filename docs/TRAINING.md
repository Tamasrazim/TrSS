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

Create separate training and validation manifests. Split by scene or capture sequence rather than randomly splitting neighboring frames, otherwise almost identical frames can leak across the two sets. Before training, TrSS also rejects any exact low-resolution input frame or target frame reused across the two manifests. This catches direct file reuse, but it cannot detect near-duplicate frames saved under different paths; scene-level separation remains necessary. Use original renders and asset packs whose licences permit dataset creation and model training; keep their sources and licences in an accompanying provenance file.

### Generate and inspect a controlled synthetic pilot dataset

Run the one-command pipeline to generate the controlled 3D scenes, split complete scenes between training and validation, run the structural audit, and write an offline gallery:

```powershell
python -m trss.training.prepare_pilot --output-dir data/trss-pilot --scenes 6 --frames-per-scene 6 --width 160 --height 90 --seed 20261014
```

Open `data/trss-pilot/review-gallery/index.html` and inspect the pairs manually before treating this as training data. The pipeline writes `train.jsonl`, `validation.jsonl`, `dataset-audit.json`, `generation-report.json`, and `pipeline-report.json`. Use a fresh folder unless you explicitly pass `--force`; the command stages the result and promotes it after generation, splitting, auditing, and gallery creation.

The CPU software rasterizer uses first-party box geometry and is explicitly synthetic. It renders the same camera/scene state at 1× and 2× resolution, computes normalized linear view-space depth and current-minus-previous motion from known geometry, and estimates previous-frame history using object-ID/depth visibility checks. It is **not** a live-game capture hook, production GPU buffer reader, photorealistic renderer, or Arc performance test. Passing the structural audit cannot prove semantic alignment or model quality.

To verify that the generated manifests reach the actual optimizer and checkpoint writer, run a small **CPU-only smoke-training epoch**. It is a software-pipeline test; the resulting weights are trained only on the synthetic pilot and are not useful evidence of game quality.

```powershell
python -m trss.training.train --train-manifest data/trss-pilot/train.jsonl --validation-manifest data/trss-pilot/validation.jsonl --output-dir logs/trss-pilot-training-smoke --device cpu --channels 8 --blocks 1 --context-blocks 0 --patch-size 8 --batch-size 1 --epochs 1 --seed 20261014
```

This produces `training-report.json`, `checkpoint-last.pt`, and `checkpoint-best.pt`. Once the smoke test works, actual model development still requires a legally usable dataset captured from a suitable renderer and subsequent validation on the intended hardware.

### Run the complete synthetic pilot in one command

For a single reproducible smoke experiment, TrSS can run all currently implemented Python-side stages end to end: generate paired synthetic scenes, split whole scenes, audit the dataset, train the first-party model, evaluate the saved checkpoint against bicubic interpolation, export the same checkpoint and RGB compositor to ONNX, then run one held-out frame through OpenVINO. It saves the reports, both visual galleries, the training log, ONNX model and sidecar, and an output PNG in one directory.

```powershell
python -m pip install -e ".[test,export,gpu]"
python -m trss.training.pilot_experiment --output-dir logs/trss-full-pilot --scenes 4 --frames-per-scene 3 --epochs 2 --train-device cpu --evaluation-device cpu --inference-device CPU
```

The installed command is `trss-pilot-experiment`. To use Intel Arc for the OpenVINO stage, change `--inference-device ARC`; the project will require OpenVINO to identify one unique Intel Arc adapter and will fail rather than silently falling back to CPU. Training can separately be set to `--train-device xpu` when the installed PyTorch XPU runtime is available.

Open `logs/trss-full-pilot/experiment-report.json` for a machine-readable summary. `evaluation/index.html` compares the trained checkpoint to bicubic interpolation on all held-out frames; `dataset/review-gallery/index.html` shows the source pairs. This pipeline deliberately uses synthetic box scenes, so it verifies plumbing and records measurements—not photorealism, game compatibility, or in-game FPS. The result is a completed pilot experiment, not a completed neural renderer.

### Evaluate a trained checkpoint against bicubic interpolation

After training, run a held-out validation manifest through the saved checkpoint. TrSS writes native-size prediction PNGs, a four-panel comparison for every evaluated frame, aggregate and per-scene MAE/RMSE/PSNR, the checkpoint SHA-256, and a local HTML gallery. By default it evaluates the entire manifest using overlapping tiles to limit accelerator memory:

```powershell
python -m trss.training.evaluate_checkpoint --checkpoint logs/trss-pilot-training-smoke/checkpoint-best.pt --manifest data/trss-pilot/validation.jsonl --output-dir logs/trss-pilot-evaluation --device cpu --tile-size 64 --tile-overlap 8
```

Open `logs/trss-pilot-evaluation/index.html` to review the nearest-neighbour input, bicubic baseline, TrSS prediction, and target reference. The companion `evaluation-report.json` records the exact checkpoint hash and model settings, per-frame and per-scene metrics, evaluation device, tile settings, and whether this checkpoint beat bicubic interpolation on aggregate MAE and PSNR. Use a distinct output folder unless `--force` is explicitly passed. For a quick development loop, `--max-frames 6` selects evenly spaced frames, but that report explicitly identifies itself as a subset rather than a complete validation score.

MAE, RMSE, and PSNR are computed over normalized display-encoded RGB values—not linear-light radiance, perceptual preference, or whole-game frame timings. A metric improvement only says that the checkpoint matches this target set better than bicubic interpolation under the recorded settings. Synthetic-pilot results remain synthetic and must not be presented as proof of real-game generalization or Arc B580 performance.

### Export a trained checkpoint to ONNX

The exporter creates one portable ONNX graph for both the trained network and TrSS's exact RGB output composition. The graph takes four NCHW tensors—`features`, `low_res_rgb`, `history_rgb`, and `history_valid`—and returns the composed RGB frame at 2× input resolution. Dynamic batch and spatial axes are enabled by default, and a neighbouring JSON file records the checkpoint hash, model configuration, tensor contracts, and known limitations.

Install the optional export runtime and export a checkpoint:

```powershell
python -m pip install -e ".[export]"
python -m trss.training.export_onnx --checkpoint logs/trss-pilot-training-smoke/checkpoint-best.pt --output models/trss-composed.onnx --opset 17
```

The equivalent installed command is `trss-export-onnx`. The ONNX file is checked with the ONNX graph validator before it is promoted into place. If either the model or sidecar already exists, the command refuses to replace it unless `--force` is supplied. Use `--static-shapes` only when the model should accept the 16×24 input resolution used for export tracing.

The graph is designed to be imported by a compatible ONNX consumer, including an OpenVINO workflow where the installed operator set supports the exported graph. It is not a complete live renderer or a game hook: the application must supply real, aligned depth and motion data and a history frame already reprojected into the current view. The `features` channels 0–2 must agree with `low_res_rgb`, and the history channels/mask within `features` must agree with the separate full-resolution tensors. Export success and matching CPU tests do not prove visual quality or Arc B580 performance.

### Run the exported graph through OpenVINO

The composed ONNX graph can now be exercised through TrSS's OpenVINO device path using one record from the same paired-frame manifest. This command rebuilds the ten-channel feature tensor with the documented depth, normalized motion, reprojected history, and validity-mask rules, compiles the graph on a selected device, and writes a PNG plus a timing report.

```powershell
python -m trss.training.infer_onnx --model models/trss-composed.onnx --manifest data/trss-pilot/validation.jsonl --row 0 --output work/trss-onnx-frame.png --device ARC --report logs/trss-onnx-frame.json
```

Use `--device CPU` only as an explicit offline-debug option. The default `ARC` alias requires OpenVINO to identify exactly one Intel Arc adapter; an ambiguous or missing Arc GPU is an error, not a silent CPU fallback. This tests the model artifact and preprocessing path on the selected OpenVINO device, but is still offline manifest inference and does not acquire frames from a running game. The report separates model compilation and inference timing from preprocessing and output writing, records an output PNG SHA-256, and computes MAE/RMSE/PSNR against the paired high-resolution target for that row. These are single-frame display-RGB metrics, not perceptual scores, and none of the timing values is in-game FPS.

### Build scene-separated train and validation manifests

Put a `scene_id` on every row in one source JSONL manifest. Use the same ID for every frame from a complete scene or capture sequence, and distinct IDs for sequences that must not cross the training/validation boundary. The splitter validates that the referenced files exist, refuses exact low-resolution input or target files labelled as different scenes, preserves additional metadata fields, and rewrites file paths so the generated manifests still resolve from their output folder.

```powershell
python -m trss.training.split_manifest --input-manifest data/all-frames.jsonl --train-output data/train.jsonl --validation-output data/validation.jsonl --scene-field scene_id --validation-fraction 0.2 --seed 20261014
```

The installed `trss-split-dataset` command runs the same tool. Its JSON report lists which scenes and frame counts went into each split. The requested fraction is approximate because complete scenes are indivisible; use more balanced capture-sequence sizes when you need closer proportions. The tool only organizes existing records: it does not generate paired renders, validate visual alignment, synthesize depth/motion, or train weights. Existing output manifests are protected unless `--force` is supplied.

### Audit the actual training inputs before training

Run the structural audit on the generated manifests. It decodes each input/reference image, checks that every reference is exactly 2× the input resolution, validates depth, motion, and optional temporal-history shapes/ranges, reports motion values that will be clipped by the configured scale, and compares decoded pixels across the two splits. Pixel hashing catches exact duplicates saved under different file names. A second, warning-only check compares 64-bit difference hashes and mean RGB values to flag likely near-duplicates that were resized, lightly altered, or re-encoded. For each valid 2× pair, it also compares the low-resolution input with the reference target downsampled using the BOX filter and reports MAE, RMSE, and PSNR as a basic pair-consistency signal. These values are diagnostic only: rendering methods, anti-aliasing, and exposure can cause real differences, so inspect examples manually and keep scene/capture-level separation as the primary defence against leakage.

```powershell
python -m trss.training.audit_dataset --train-manifest data/train.jsonl --validation-manifest data/validation.jsonl --provenance-json data/provenance.json --report logs/dataset-audit.json --motion-scale-px 64 --near-duplicate-hamming-distance 4 --near-duplicate-mean-rgb-distance 10
```

The provenance file must be a JSON object with non-empty `dataset_name`, `source`, and `license` fields, for example:

```json
{
  "dataset_name": "My capture set",
  "source": "Original renders from my own scene",
  "license": "Original work; assets and capture permitted for this use"
}
```

The installed `trss-audit-dataset` command runs the same audit. It returns a non-zero exit code for structural errors, scene overlap, or exact pixel leakage; perceptual near-duplicate matches are warnings only and do not change the pass/fail result. The JSON report includes counts by image field and the nearest detected training example for each flagged validation image, plus per-frame `pair_consistency` metrics and split summaries. The Hamming-distance threshold can be 0–16; the mean RGB distance is measured on 0–255 RGB channel averages. The pair-consistency metric is not a hard pass/fail threshold, and it cannot establish semantic alignment. If provenance is omitted, the audit warns that licensing and origin are unverified. Inspect example pairs and verify every asset's terms yourself.

### Visually review paired frames offline

Structural checks cannot tell whether a reference render actually matches its low-resolution input. Build a local HTML gallery with a deterministic, scene-balanced sample of frame pairs:

```powershell
python -m trss.training.review_dataset --manifest data/train.jsonl --output-dir logs/train-review --max-frames 36 --scene-field scene_id --seed 20261014
```

Open `logs/train-review/index.html` in a browser. The gallery includes side-by-side input/reference thumbnails, original dimensions, scene labels, history-reset state, searchable cards, and source paths. The adjacent `review-report.json` records the selected manifest lines and any image decoding failures. Thumbnails are local files; the page uses no network resources. A successful gallery build confirms that the selected images could be decoded, not that they are visually aligned or licensed correctly.

## Start training

\`\`\`powershell
python -m trss.training.train --train-manifest data/train.jsonl --validation-manifest data/validation.jsonl --output-dir models/trss-quality --device xpu --channels 48 --blocks 10 --context-blocks 4 --patch-size 128 --batch-size 1 --epochs 100 --amp
\`\`\`

Training uses PyTorch's XPU backend when selected. \`--amp\` requests BF16 autocast and fails if the selected device does not report BF16 support; omit it to train in FP32. Patches are cropped on the CPU before transfer to conserve VRAM. Training samples receive synchronized horizontal/vertical flips; the corresponding motion-vector components have their signs reflected as well. Validation does not use random crops: it scores deterministic patches covering the full frame so the metric is not just a center-crop score. A batch should contain frames with the same dimensions, so keep \`--batch-size 1\` when resolutions vary.

Begin with a small dataset and a few epochs to confirm the pipeline runs, then scale up after establishing a clean validation baseline. Use a separate validation manifest split by scene/capture sequence.

## Resume after an interruption

Every completed epoch atomically updates \`checkpoint-last.pt\`, so an interruption while writing cannot leave the previous checkpoint half-written. The checkpoint keeps model weights, AdamW state, progress history, random crop state, and the training sample-shuffle state.

Resume with the same manifests, model dimensions, batch size, patch size, BF16 setting, and motion scale. The checkpoint verifies those run settings (as well as the manifest paths) so a changed batch or crop configuration cannot silently continue an incompatible run. Increasing the target epoch count is expected when continuing training. The guard checks manifest paths, not the contents of files at those paths, so keep the dataset immutable during a run:

\`\`\`powershell
python -m trss.training.train --train-manifest data/train.jsonl --validation-manifest data/validation.jsonl --output-dir models/trss-quality --device xpu --channels 48 --blocks 10 --context-blocks 4 --patch-size 128 --batch-size 1 --epochs 100 --amp --resume models/trss-quality/checkpoint-last.pt
\`\`\`

Here \`--epochs\` means the total target epoch, not the number of extra epochs. To continue a run that stopped on epoch 34, keep the target at 100. On resume, TrSS restores the optimizer state and the saved random states. The cosine learning-rate schedule is computed from the absolute epoch, so continuing with the same total target matches the intended schedule. Pass \`--learning-rate\` explicitly only when deliberately changing the base rate.

Training writes \`checkpoint-last.pt\`, \`checkpoint-best.pt\`, and \`training-report.json\`. When a validation manifest is supplied, “best” is chosen by validation loss. Without one, it uses augmented training loss and is not a trustworthy generalization score. Generated weights and local data are not meant to be committed to the source repository.

## Run a trained checkpoint on an offline frame

\`\`\`powershell
python -m trss.training.infer --checkpoint models/trss-quality/checkpoint-best.pt --color frames/low.png --depth frames/depth.npy --motion frames/motion.npy --output work/trss-output.png --device xpu --warmup 2 --iterations 5 --report logs/inference.json
\`\`\`

For a non-reset frame, also pass \`--history\` and \`--history-valid\`. The history image must already be reprojected to the current view and exactly 2x the color frame dimensions; the mask must have the same output dimensions. Omitting both history arguments is an explicit first-frame/reset case. The report separates input setup, isolated model inference, model-plus-composition timing, output encoding/writing, and total benchmark-call elapsed time. The FPS-equivalent estimate uses repeated model-plus-composition timing only; it excludes live capture, file decoding, input setup, device transfer, and output encoding. It is not a full-pipeline or in-game FPS measurement.

## What this does not do yet

- The ten-channel temporal training pipeline still cannot construct genuine game-renderer depth/motion/history data by itself. The separate RGB-only importer can create spatial pairs from user-recorded game video.
- It does not supply pretrained model weights.
- It does not yet convert this model's checkpoint into the vendored native Vulkan/XMX graph. The existing \`xmx-frame\` runner still expects its own fixed research graph and compatible logical-weight format.
- The first-party Windows DXGI sampler can record visible desktop color to PNG, and the separate importer can decode ordinary local video recordings. Neither tool reads private game depth/motion buffers, performs temporal reprojection, nor supplies real renderer buffers to the ten-channel temporal model.
- It does not claim a frame-rate target. Measure output quality, end-to-end latency, VRAM, power, and temperature on the Arc B580 before using it alongside a game.

The next milestone is to inspect spatial-model predictions visually on separate capture scenes, then build renderer-specific auxiliary-buffer access and temporal reprojection. Only after that should TrSS claim anything about a usable real-time temporal renderer.

## Reference

PyTorch's official Intel GPU installation guidance documents the XPU wheel index and supported hardware: https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html. Driver/runtime requirements depend on the exact operating system and PyTorch release.


## Train an RGB-only spatial baseline from recorded gameplay

The separate spatial baseline lets you begin with footage from games already installed on your PC. It uses normal video files from a recorder you already have; it never attaches to the game, reads game memory, or injects a DLL. The importer builds an aligned pair from every selected video frame: the captured frame is the target and a half-resolution reduction of that same frame is the input.

Install the optional decoder and training dependencies:

```powershell
python -m pip install -e ".[training,video,test]"
```

Start with **two separate ETS2 recordings** at the same display resolution. Choose different stretches of road or different locations so that validation is not mostly the same scene as training. A native 1920x1080 recording creates 960x540 input frames and 1920x1080 target frames. With the default sampling interval, a 60 fps recording contributes about four samples per second, up to the configured frame limit.

```powershell
python -m trss.training.import_game_video "D:\Captures\ETS2-road-a.mkv" "D:\Captures\ETS2-road-b.mkv" --output-dir data\ets2-spatial --sample-every 15 --max-frames-per-video 120 --seed 20261014
```

The command writes `train.jsonl`, `validation.jsonl`, `capture-report.json`, and `provenance.json`, plus the extracted RGB pairs under `frames\`. The complete recording file—not individual nearby frames—is assigned to one split. If only one clip is supplied, or two supplied files have identical content, the importer refuses to proceed. Existing output is protected unless `--force` is explicitly passed.

Train the spatial-only model. The explicit XPU setting fails if the installed PyTorch build cannot see an Intel GPU; use `--device cpu` for a small plumbing test:

```powershell
python -m trss.training.train_spatial --train-manifest data\ets2-spatial\train.jsonl --validation-manifest data\ets2-spatial\validation.jsonl --output-dir logs\ets2-spatial --device xpu --epochs 5 --patch-size 96
```

The training report records the checkpoint format, hashes of both manifests, device, per-epoch losses, per-frame and aggregate MAE/RMSE/PSNR, and whether the model's validation score is better than the bicubic baseline. Check `logs\ets2-spatial\checkpoint-best.pt` and `training-report.json` after the run.

**Important limits:** this is a spatial 2x upscaling experiment, not the temporal reconstruction model. It does not receive true game-rendered low-resolution inputs and cannot learn from this dataset how to use game depth, motion vectors, disocclusion masks, or temporal history. The pairs are formed by downsampling the recorded targets, which is useful for a first real-content experiment but does not reproduce the game's original low-resolution rendering pipeline. Do not treat good validation metrics as proof of in-game quality or Arc B580 real-time performance. Keep recordings and extracted game frames local unless you have permission to redistribute them.

Use ETS2 first because it is an easy place to collect long, controlled driving footage. Alan Wake can be a second dataset with darker lighting and bright flashlight/highlight transitions. The Crew 2 can follow for faster motion. Keep PUBG out of the initial capture/integration tests: do not use process hooks or other unapproved tools with its anti-cheat system. Ordinary recordings are enough for this RGB-only first step.


### Inspect the trained spatial output

After the spatial training command finishes, evaluate its best checkpoint on the held-out clips and build a local HTML gallery:

```powershell
python -m trss.training.evaluate_spatial --checkpoint logs\ets2-spatial\checkpoint-best.pt --manifest data\ets2-spatial\validation.jsonl --output-dir logs\ets2-spatial-evaluation --device xpu --tile-size 256 --tile-overlap 16 --max-previews 12
```

Open `logs\ets2-spatial-evaluation\index.html`. Each sample shows the enlarged low-resolution input, bicubic baseline, TrSS spatial prediction, and captured target. The accompanying `evaluation-report.json` records the checkpoint and manifest SHA-256 hashes, evaluation device, per-frame MAE/RMSE/PSNR, the aggregate scores, and whether the model beat bicubic interpolation. Unless `--max-frames` is supplied, scores cover the entire validation manifest; `--max-previews` only limits how many examples are shown in the gallery.

Start with a few modest recordings and a short training run before trying long sessions. A result that does not beat bicubic is still useful feedback: inspect the comparison gallery and metrics instead of assuming the learned output is better. Keep the first experiment RGB-only until a valid, renderer-specific path for depth, motion, and temporal history is available.
