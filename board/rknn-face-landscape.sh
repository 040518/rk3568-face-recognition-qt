#!/bin/sh

# Rockchip's patched Weston DRM backend accepts runtime output commands from
# this file. The recognition launcher owns rotation while it is running.
export WESTON_DRM_CONFIG=/tmp/.weston_drm.conf
if [ ! -e "$WESTON_DRM_CONFIG" ]; then
    printf '%s\n' 'output:all:rotate0' >"$WESTON_DRM_CONFIG"
fi
