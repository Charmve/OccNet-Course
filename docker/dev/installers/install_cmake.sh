#! /bin/bash
###############################################################################
# Copyright 2020 The Apollo Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################
set -euo pipefail

CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1090,SC1091
. "${CURR_DIR}/installer_base.sh"

VERSION="3.21.2"
TARGET_ARCH="$(uname -m)"

CMAKE_SH=
CHECKSUM=
if [[ "${TARGET_ARCH}" == "x86_64" ]]; then
  CMAKE_SH="cmake-${VERSION}-linux-x86_64.sh"
  CHECKSUM="3310362c6fe4d4b2dc00823835f3d4a7171bbd73deb7d059738494761f1c908c"
elif [[ "${TARGET_ARCH}" == "aarch64" ]]; then
  CMAKE_SH="cmake-${VERSION}-linux-aarch64.sh"
  CHECKSUM="1d1c08d8d9b5a705ec6415dfe74b9f88f5a3fb66ad8a26ca6b7568f63d4670b6"
fi

# https://github.com/Kitware/CMake/releases/download/v3.21.1/cmake-3.21.1-linux-x86_64.sh
DOWNLOAD_LINK="https://github.com/Kitware/CMake/releases/download/v${VERSION}/${CMAKE_SH}"
download_if_not_cached "${CMAKE_SH}" "${CHECKSUM}" "${DOWNLOAD_LINK}"

chmod a+x ${CMAKE_SH}
./${CMAKE_SH} --skip-license --prefix="${SYSROOT_DIR}"
rm -fr ${CMAKE_SH}
