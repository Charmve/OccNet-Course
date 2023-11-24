#!/usr/bin/env bash

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

# Fail on first error.
set -e

CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1090,SC1091
. "${CURR_DIR}/installer_base.sh"

ARCH="$(uname -m)"
if [[ "${ARCH}" != "x86_64" && "${ARCH}" != "aarch64" ]]; then
  error "Architecture ${ARCH} not supported yet"
  exit 1
fi

VERSION="5.1.0"
PKG_NAME=
CHECKSUM=
DOWNLOAD_LINK=

if [[ "$ARCH" == "x86_64" ]]; then
  PKG_NAME="buildifier-${VERSION}-linux-amd64"
  CHECKSUM="52bf6b102cb4f88464e197caac06d69793fa2b05f5ad50a7e7bf6fbd656648a3"
  DOWNLOAD_LINK="https://github.com/bazelbuild/buildtools/releases/download/${VERSION}/buildifier-linux-amd64"
else
  PKG_NAME="buildifier-${VERSION}-linux-arm64"
  CHECKSUM="917d599dbb040e63ae7a7e1adb710d2057811902fdc9e35cce925ebfd966eeb8"
  DOWNLOAD_LINK="https://github.com/bazelbuild/buildtools/releases/download/${VERSION}/buildifier-linux-arm64"
fi

download_if_not_cached "${PKG_NAME}" "${CHECKSUM}" "${DOWNLOAD_LINK}"

cp -f "${PKG_NAME}" "${SYSROOT_DIR}/bin/buildifier"
chmod a+x "${SYSROOT_DIR}/bin/buildifier"
rm -rf "${PKG_NAME}"

info "Done installing buildifier ${VERSION}"
