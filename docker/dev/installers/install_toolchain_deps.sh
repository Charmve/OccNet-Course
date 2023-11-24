#! /bin/bash
set -euo pipefail

CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1090,SC1091
source "${CURR_DIR}/installer_base.sh"

bash "${CURR_DIR}/install_cmake.sh"
bash "${CURR_DIR}/install_bazel.sh"
bash "${CURR_DIR}/install_llvm_clang.sh"
bash "${CURR_DIR}/install_buildifier.sh"
bash "${CURR_DIR}/install_buildozer.sh"

pip3_install \
  isort \
  black \
  flake8
