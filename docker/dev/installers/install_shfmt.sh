#! /bin/bash
set -euo pipefail

CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1090,SC1091
. "${CURR_DIR}/installer_base.sh"

ARCH="$(uname -m)"
VERSION="3.5.1"

if [ "$ARCH" == "x86_64" ]; then
  PKG_NAME="shfmt_v${VERSION}_linux_amd64"
  CHECKSUM="56099a689b68534f98e1f8f05d3df6750ab53e3db68f514ee45595bf5b79d158"
elif [ "$ARCH" == "aarch64" ]; then
  PKG_NAME="shfmt_v${VERSION}_linux_arm64"
  CHECKSUM="09d7902de04d52ebe0b332d84a9746d195f7e930806bdc2436f84d0de6a2d368"
else
  error "Target arch ${ARCH} not supported yet"
  exit 1
fi

DOWNLOAD_LINK="https://github.com/mvdan/sh/releases/download/v${VERSION}/${PKG_NAME}"
download_if_not_cached "${PKG_NAME}" "${CHECKSUM}" "${DOWNLOAD_LINK}"

mv -f "${PKG_NAME}" "${SYSROOT_DIR}/bin/shfmt"
chmod a+x "${SYSROOT_DIR}/bin/shfmt"

ok "Successfully installed shfmt ${VERSION}."
