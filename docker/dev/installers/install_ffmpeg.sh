#! /bin/bash
set -euo pipefail

CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1090,SC1091
. "${CURR_DIR}/installer_base.sh"

# Homepage: https://ffmpeg.org/
apt_get_update_and_install \
  libavcodec-dev \
  libavutil-dev \
  libswresample-dev \
  libavformat-dev \
  libswscale-dev
apt_get_cleanup

ok "Successfully installed system-provided ffmpeg"
