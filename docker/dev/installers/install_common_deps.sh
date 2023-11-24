#!/bin/bash
set -euo pipefail

CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1090,SC1091
source "${CURR_DIR}/installer_base.sh"

apt_get_update_and_install \
  zlib1g-dev \
  libssl-dev

# gflags and glog
bash "${CURR_DIR}/install_gflags.sh"
bash "${CURR_DIR}/install_glog.sh"

bash "${CURR_DIR}/install_bzip2.sh"
bash "${CURR_DIR}/install_lz4.sh"
bash "${CURR_DIR}/install_zstd.sh"
bash "${CURR_DIR}/install_libxml2.sh"
bash "${CURR_DIR}/install_libarchive.sh"

bash "${CURR_DIR}/install_double_conversion.sh"
bash "${CURR_DIR}/install_eigen.sh"
bash "${CURR_DIR}/install_yaml_cpp.sh"
bash "${CURR_DIR}/install_protobuf.sh"

apt_get_cleanup
