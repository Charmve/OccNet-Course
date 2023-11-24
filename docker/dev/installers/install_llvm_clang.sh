#! /bin/bash
set -euo pipefail

CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1090,SC1091
. "${CURR_DIR}/installer_base.sh"

ARCH="$(uname -m)"

VERSION="13.0.1"

PKG_NAME=
CHECKSUM=

if [[ "${ARCH}" == "aarch64" ]]; then
  PKG_NAME="clang+llvm-${VERSION}-aarch64-linux-gnu.tar.xz"
  CHECKSUM="15ff2db12683e69e552b6668f7ca49edaa01ce32cb1cbc8f8ed2e887ab291069"
else
  PKG_NAME="clang+llvm-${VERSION}-x86_64-linux-gnu-ubuntu-18.04.tar.xz"
  CHECKSUM="84a54c69781ad90615d1b0276a83ff87daaeded99fbc64457c350679df7b4ff0"
fi

DOWNLOAD_LINK="https://github.com/llvm/llvm-project/releases/download/llvmorg-${VERSION}/${PKG_NAME}"
download_if_not_cached "${PKG_NAME}" "${CHECKSUM}" "${DOWNLOAD_LINK}"

DEST_DIR=/opt/llvm
mkdir "${DEST_DIR}"
tar xJf "${PKG_NAME}" --strip-components=1 -C "${DEST_DIR}"
# chown -R root:root ${DEST_DIR}

if [[ ! -e /usr/bin/clang-format ]]; then
  ln -s "${DEST_DIR}/bin/clang-format" /usr/bin/clang-format
fi

rm -rf "${PKG_NAME}"
ok "Successfully installed clang+llvm ${VERSION}"
