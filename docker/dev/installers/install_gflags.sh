#! /bin/bash
set -euo pipefail

CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1090,SC1091
. "${CURR_DIR}/installer_base.sh"

VERSION="2.2.2"

PKG_NAME="gflags-${VERSION}.tar.gz"
DOWNLOAD_LINK="https://github.com/gflags/gflags/archive/v2.2.2.tar.gz"
CHECKSUM="34af2f15cf7367513b352bdcd2493ab14ce43692d2dcd9dfc499492966c64dcf"

download_if_not_cached "${PKG_NAME}" "${CHECKSUM}" "${DOWNLOAD_LINK}"

tar xzf "${PKG_NAME}"
pushd "gflags-${VERSION}" > /dev/null

mkdir build
pushd build > /dev/null

cmake .. -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release

make -j "$(nproc)"
make install
popd > /dev/null

popd > /dev/null

# Cleanup
rm -rf "${PKG_NAME}" "gflags-${VERSION}"
ldconfig

info "OK. gflags ${VERSION} successfully installed"
