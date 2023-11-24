#! /bin/bash
set -euo pipefail

CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1090,SC1091
. "${CURR_DIR}/installer_base.sh"

VERSION="0.5.0"

PKG_NAME="glog-${VERSION}.tar.gz"
DOWNLOAD_LINK="https://github.com/google/glog/archive/v0.5.0.tar.gz"
CHECKSUM="eede71f28371bf39aa69b45de23b329d37214016e2055269b3b5e7cfd40b59f5"

download_if_not_cached "${PKG_NAME}" "${CHECKSUM}" "${DOWNLOAD_LINK}"

tar xzf "${PKG_NAME}"
pushd "glog-${VERSION}" > /dev/null

mkdir build
pushd build > /dev/null

cmake .. -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release

make -j "$(nproc)"
make install
popd > /dev/null

popd > /dev/null

# Cleanup
rm -rf "${PKG_NAME}" "glog-${VERSION}"
ldconfig

info "OK. glog ${VERSION} successfully installed"
