#!/bin/sh
# Build the recovered D2 runtime with the official Luckfox uClibc toolchain.
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
: "${LUCKFOX_SDK_ROOT:?Set LUCKFOX_SDK_ROOT to the Luckfox Pico SDK checkout}"
: "${RKNN_EXAMPLE_ROOT:?Set RKNN_EXAMPLE_ROOT to the Luckfox RKNN example checkout}"
TVL_TOOLCHAIN="$LUCKFOX_SDK_ROOT/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin"
TVL_OUT="${TVL_BUILD_DIR:-"$HERE/build"}"
TVL_CC="$TVL_TOOLCHAIN/arm-rockchip830-linux-uclibcgnueabihf-gcc"
TVL_CXX="$TVL_TOOLCHAIN/arm-rockchip830-linux-uclibcgnueabihf-g++"
TVL_INC="$RKNN_EXAMPLE_ROOT/include/rknn"
TVL_LIB="$RKNN_EXAMPLE_ROOT/lib/uclibc"
TVL_OPENCV="$TVL_LIB/lib"
test -x "$TVL_CC"
test -x "$TVL_CXX"
mkdir -p "$TVL_OUT"

"$TVL_CC" -O2 -Wall -Wextra -I"$TVL_INC" "$HERE/rknn_smoke.c" \
    -L"$TVL_LIB" -Wl,-rpath,'$ORIGIN' -lrknnmrt -lm -o "$TVL_OUT/rknn_smoke"
"$TVL_CXX" -O2 -Wall -I"$RKNN_EXAMPLE_ROOT/include" -I"$TVL_INC" "$HERE/camera_pipeline.cpp" \
    "$TVL_OPENCV/libopencv_highgui.a" "$TVL_OPENCV/libopencv_features2d.a" \
    "$TVL_OPENCV/libopencv_video.a" "$TVL_OPENCV/libopencv_photo.a" \
    "$TVL_OPENCV/libopencv_imgproc.a" "$TVL_OPENCV/libopencv_core.a" \
    -L"$TVL_LIB" -lrknnmrt -lpthread -ldl -lm -lrt -o "$TVL_OUT/camera_pipeline"
printf 'Built runtime in %s\n' "$TVL_OUT"
