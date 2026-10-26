# Luckfox Pico Mini A deployment

The recovered deployment runs D2 YOLO26 Nano at 320 px on an RV1103 board
with 64 MB RAM. Source is in `deployment/luckfox/`; images, compiled binaries
and model weights are excluded from Git.

## Pipeline and model contract

The camera path uses opencv-mobile to initialize the ISP through rkaiq,
captures a BGR frame, letterboxes it to 320×320 with value 114, converts it
to RGB, runs RKNN through the zero-copy API, and applies class-wise NMS.
Detection confidence is 0.25; NMS IoU is 0.45.

The recovered model has one INT8 output, `1×8×2100`. Channels 0–3 are
normalized `cx, cy, width, height`; channels 4–7 are sigmoid class scores
for plastic, paper, metal and glass. There is no separate objectness channel.

The recovered D2 decoder used fixed affine parameters: zero point −128 and
scale 0.00463843858. The current decoder reads those parameters from the
loaded RKNN model and rejects unsupported tensor layouts. Its host replay
on all 32 preserved D2 output tensors produced exactly the same 52 detections;
[regression evidence](../reports/board/decoder-regression.json) records the inputs
and checks. A new D3 export still needs board parity verification.
The D3 training run does not by itself validate a D3 board deployment.

## Build

External dependencies:

- [Luckfox Pico SDK](https://github.com/LuckfoxTECH/luckfox-pico), including
  the `arm-rockchip830-linux-uclibcgnueabihf` toolchain.
- [Luckfox RKNN example](https://github.com/LuckfoxTECH/luckfox_pico_rknn_example),
  providing RKNN headers/runtime and its opencv-mobile libraries.

```bash
LUCKFOX_SDK_ROOT=/path/to/luckfox-pico \
RKNN_EXAMPLE_ROOT=/path/to/luckfox_pico_rknn_example \
sh deployment/luckfox/build.sh
```

The script creates `deployment/luckfox/build/rknn_smoke` and
`deployment/luckfox/build/camera_pipeline`. No third-party library binaries
are vendored here.

## Camera and field modes

Run on the board with the required runtime libraries available. Stop the
stock camera application before manual capture:

```sh
sh /oem/usr/bin/RkLunch-stop.sh
./camera_pipeline model.rknn 640 480 30
./camera_pipeline model.rknn --shots /userdata/shots 5 120
./camera_pipeline model.rknn --button /userdata/photos
./camera_pipeline model.rknn --field /userdata/field
```

Field mode uses the board button and work LED: one click saves a photo;
a double click starts a 30-second scan with a frame every two seconds;
a long press stops the current activity. Results persist under `/userdata`.

To install the selected RKNN artifact and enable startup at the next boot:

```bash
sh deployment/luckfox/field-install.sh /path/to/model.rknn
```

This replaces the installed detector binary/model and adds `S99fieldapp`.
`field-install.sh disable` removes its autostart and restores the stock camera
application. Use `ANDROID_SERIAL` when multiple ADB devices are connected.

## Optional unknown-material display

`TVL_MATERIAL_MARGIN` enables an experimental naming policy when the top two
class scores are close. It defaults to zero (disabled), preserves all boxes,
and shows the best material as a suggestion. See the [fixed abstention
experiment and usage](material-abstention.md) for evidence and limitations.
The additional mode has host tests and cross-compiles; it has no fresh board
validation and is not used to claim an accuracy or speed improvement.

## Recovered measurements

These measurements belong to the existing D2 export:

| Measurement | Result |
| --- | ---: |
| Capture, 30-frame camera run | 19.99 ms/frame |
| Preprocessing | 14.80 ms/frame |
| NPU inference | 31.39 ms/frame |
| Postprocessing | 0.38 ms/frame |
| Complete pipeline | 66.56 ms/frame, about 15 FPS |
| Peak process RSS | about 12.2 MB |

The 32-image PyTorch/RKNN comparison reported 50 matches from 51 PyTorch
and 52 RKNN detections, mean matched IoU 0.812 and mean absolute confidence
difference 0.028. These are output-agreement measurements, not ground-truth AP.
The short camera run used an empty scene with no detections. It does not
establish sustained throughput or postprocessing cost in crowded scenes.

The original source hashes are retained in `deployment/luckfox/provenance.json`.
The current camera runtime adds model-provided quantization and contract checks;
the decoder has host tests and the complete runtime was cross-compiled.
No board is currently connected for a fresh hardware run.

## Reproducible Nano export

Use a separate environment: RKNN Toolkit2 2.3.2 requires an older PyTorch
than the training recipe. The exporter runs on the Linux host CPU.

```bash
python3.12 -m venv .venv-rknn
. .venv-rknn/bin/activate
python -m pip install torch==2.4.0+cpu torchvision==0.19.0+cpu \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r configs/requirements-rknn.lock
python -m pip check
python -m runners.export_rknn \
  --weights /path/to/completed-nano/weights/best.pt \
  --run-metadata /path/to/completed-nano/run.json \
  --data-root data/versions/d3 --output runs/export/d3-nano
```

The exporter verifies the completed training record and frozen dataset hashes.
It copies all 256 real training calibration images into independent, letterboxed
320×320 PNGs. The source JPEGs and checkpoint are checked again after export.
Validation/test images are excluded from calibration. The source dataset scanner
is never invoked, preventing implicit JPEG repair from changing frozen images.

The pinned Ultralytics exporter creates normalized `1×8×2100` ONNX output
before RKNN INT8 conversion. Eight calibration tensors are compared with the
same PyTorch inference branch before compiling RKNN. The output directory
retains the ONNX graph, compiled RKNN, calibration-copy hashes, environment
and `export.json`. These large/private artifacts stay outside Git.
This ONNX graph is the RKNN conversion intermediate: consumers must decode
normalized coordinates, as the supplied material-abstention runner does.

The portable path was exercised with the known D2 Nano checkpoint and the D3
calibration membership. Its normalized ONNX maximum absolute error was below
0.000007 across eight inputs. That check validates the floating-point graph;
it does not establish INT8 accuracy or board performance. The new letterboxed
calibration copies also differ from the old export preparation, so the old
board measurements do not describe this newly compiled artifact.

D3 Nano has also completed this export. Its [export receipt](../reports/board/d3-export.json)
records the checkpoint, ONNX and RKNN hashes and pinned environment. The maximum
absolute ONNX/PyTorch error over eight inputs was 0.00000530. RKNN compilation
succeeded with all 256 real training calibration copies. This provides a D3
artifact ready for board testing; it does not supersede the board-tested D2
artifact or establish D3 INT8 accuracy.

The [official Ultralytics RKNN integration](https://docs.ultralytics.com/integrations/rockchip-rknn)
and [Rockchip Toolkit2](https://github.com/airockchip/rknn-toolkit2) describe the
conversion/runtime boundary. Fresh D3 board validation remains a separate step.
