# Model-specific input adapters

XESS5 cannot infer the meaning of an arbitrary model tensor from its name. A port called "depth" may contain linear distance, device-space Z, reversed-Z, normalized depth, or something else. Motion vectors can use different units and sign conventions. Color tensors can be RGB/BGR, linear or display-encoded, and scaled differently. The adapter specification records the selected model's real contract.

## Template

Start by inspecting a local model:

    xess5 inspect --model "models\your-model.onnx"

Copy the example adapter-template.json and edit it to match that model. The template deliberately describes a hypothetical graph; its input names and ranges are NOT correct for any particular neural-rendering model until verified.

Each input binding requires:

- source: one of color, depth, motion_vectors, noise, history, or conditioning;
- layout: one of NCHW, NHWC, CHW, HWC, HW, VECTOR, or NC;
- source_range: the value range in the frame data before normalization;
- target_range: the numerical range expected by the model;
- optional resize: nearest or bilinear for spatial sources (defaults to bilinear);
- optional channel_order: RGB or BGR for image-like sources (defaults to RGB). For a four-channel RGBA source, BGR reorders only the first three color channels and preserves alpha, producing BGRA.

An optional root-level preview_output section may declare one direct RGB/RGBA output tensor, for example:

    "preview_output": {
      "port": "rgb_output",
      "layout": "NCHW",
      "source_range": [0, 1],
      "channel_order": "RGB"
    }

The port name must exactly match a model output. The output must be rank four with batch size 1 and three or four color channels. Feature planes and packed neural heads are rejected; they need a model-specific composition function. If the model has a direct image output, this declaration lets XESS5 save a preview while still preserving every raw output tensor.

The engine uses the model's static width and height when available. Dynamic spatial dimensions use the supplied buffer size. The adapter refuses channel-count or static-shape mismatches rather than inventing channels.

The range conversion is an affine remapping:

    target = (source - source_min) / (source_max - source_min)
           * (target_max - target_min) + target_min

It does not clamp values, compute per-channel means/standard deviations, convert sRGB to linear light, scale motion vectors, convert depth encodings, create deterministic noise, or guess model-specific output composition. Add those steps in a dedicated adapter if the model requires them.

## Run one captured frame offline

The current render-frame command is an offline validation path, not a game hook. It reads a screenshot and whichever real auxiliary buffers were saved by another trusted capture path:

    xess5 render-frame ^
      --model "models\your-model.onnx" ^
      --spec "models\your-adapter.json" ^
      --color "frames\frame-0001.png" ^
      --depth "frames\depth-0001.npy" ^
      --motion "frames\motion-0001.npy" ^
      --output-dir "work\output-tensors" ^
      --device ARC ^
      --warmup 2 ^
      --iterations 10 ^
      --report "logs\frame-0001.json"

Every argument is local. Preview/history export is optional and requires a direct RGB preview_output declaration in the spec. The preview is normalized to RGB and resized to the source frame dimensions. The history file is H×W×3 float32 in [0, 1]; configure the next frame's history binding source_range as [0, 1] unless a model-specific transform says otherwise. Omit a buffer option only when the model's spec does not require that source. If the model requires depth, motion, noise, history, or conditioning and it is not supplied, XESS5 stops with an error instead of making one up.

Array inputs are numeric NumPy .npy files loaded with object deserialization disabled. Typical source shapes are:

| Source | Shape | Notes |
|---|---|---|
| color | H×W×3 | Loaded from the screenshot as RGB uint8 |
| depth | H×W or H×W×1 | Caller must state the real depth encoding and units |
| motion_vectors | H×W×2 | The adapter preserves source units; its configured range only remaps values |
| noise | H×W×3 | Must be the exact noise representation expected by the model |
| history | H×W×3 | An actual previous model output; it is not created automatically |
| conditioning | vector | The caller supplies the model's documented conditioning vector |

## Output rules

The adapter writes every model output as a separate .npy tensor and reports its shape and datatype. It intentionally does not assume an output named rgb is display-ready RGB. Some networks output residuals, feature planes, exposure/gain values, alpha, or a packed neural head; a model-specific composition routine must turn those values into the final frame. An optional direct-RGB preview only works when the spec explicitly identifies a rank-four RGB/RGBA output; it is not a generic decoder for neural heads.

The command reports preprocessing, inference, end-to-end timing, compile timing, the chosen OpenVINO device, frame metadata, input tensors, and output shapes. These are offline per-frame measurements; they are not in-game FPS.

## DLSS-NR boundary

The generic adapter is useful plumbing, not a compatible implementation of NVIDIA's neural-rendering contract by itself. The Intel-targeted open-source research project describes a much more specialized graph, weight layout, auxiliary-feature assembly, temporal behavior, and GPU compute path. Before claiming compatibility, XESS5 still needs the exact legal model/weights and exact feature preprocessing and output composition, plus hardware validation. The project will not scrape proprietary model files or silently bundle extracted weights.
