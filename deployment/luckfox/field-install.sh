#!/bin/sh
# Install the field runtime and enable boot startup, or disable its autostart.
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DEST=/userdata/trash-yolo26n-int8-320
if [ "${1:-}" = disable ]; then
    adb shell 'rm -f /etc/init.d/S99fieldapp; killall camera_pipeline 2>/dev/null || true; sh /oem/usr/bin/RkLunch.sh >/dev/null 2>&1'
    printf 'Field autostart disabled; stock camera application restored.\n'
    exit 0
fi
if [ "$#" -ne 1 ]; then
    printf 'Usage: %s MODEL.rknn | disable\n' "$0" >&2
    exit 2
fi
test -f "$1"
test -f "$HERE/build/camera_pipeline"
adb shell "mkdir -p $DEST"
adb push "$HERE/build/camera_pipeline" "$DEST/camera_pipeline"
adb push "$1" "$DEST/model.rknn"
adb push "$HERE/S99fieldapp" /etc/init.d/S99fieldapp
adb shell "chmod +x $DEST/camera_pipeline /etc/init.d/S99fieldapp"
printf 'Installed; field mode starts at the next boot.\n'
