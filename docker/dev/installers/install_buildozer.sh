#! /bin/bash
set -eu

CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1090,SC1091
. "${CURR_DIR}/installer_base.sh"

ARCH="$(uname -m)"
VERSION="5.1.0"

PKG_NAME=
CHECKSUM=
DOWNLOAD_LINK=

if [[ "${ARCH}" == "x86_64" ]]; then
  PKG_NAME="buildozer-${VERSION}-linux-amd64"
  CHECKSUM="7346ce1396dfa9344a5183c8e3e6329f067699d71c4391bd28317391228666bf"
  DOWNLOAD_LINK="https://github.com/bazelbuild/buildtools/releases/download/${VERSION}/buildozer-linux-amd64"
elif [[ "${ARCH}" == "aarch64" ]]; then
  PKG_NAME="buildozer-${VERSION}-linux-arm64"
  CHECKSUM="0b08e384709ec4d4f5320bf31510d2cefe8f9e425a6565b31db06b2398ff9dc4"
  DOWNLOAD_LINK="https://github.com/bazelbuild/buildtools/releases/download/${VERSION}/buildozer-linux-arm64"
else
  error "Target arch ${ARCH} not supported yet"
  exit 1
fi
download_if_not_cached "${PKG_NAME}" "${CHECKSUM}" "${DOWNLOAD_LINK}"

mv -f "${PKG_NAME}" "${SYSROOT_DIR}/bin/buildozer"
chmod a+x "${SYSROOT_DIR}/bin/buildozer"

ok "Successfully installed buildozer ${VERSION}"
