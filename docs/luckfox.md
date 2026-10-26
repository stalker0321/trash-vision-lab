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

The original decoder contains D2-specific affine quantization constants:
zero point −128 and scale 0.00463843858. Its supported model SHA256 is recorded
in `deployment/luckfox/provenance.json`. A new D3 export must have its tensor
layout, quantization and parity checked before using this runtime.
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
The short camera run also does not establish sustained throughput.

The restored C/C++ inference sources and startup script retain their original
hashes. The portable build and installation wrappers replace personal paths.
No board is currently connected for a fresh hardware run.
