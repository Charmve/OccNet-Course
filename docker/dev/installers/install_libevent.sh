#! /bin/bash
set -euo pipefail

CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1090,SC1091
. "${CURR_DIR}/installer_base.sh"

VERSION="2.1.8"
PKG_NAME="libevent-release-${VERSION}-stable.tar.gz"
DOWNLOAD_LINK="https://github.com/libevent/libevent/archive/release-${VERSION}-stable.tar.gz"
CHECKSUM="316ddb401745ac5d222d7c529ef1eada12f58f6376a66c1118eee803cb70f83d"

download_if_not_cached "${PKG_NAME}" "${CHECKSUM}" "${DOWNLOAD_LINK}"

tar xzf "${PKG_NAME}"

pushd "libevent-release-${VERSION}-stable" > /dev/null

mkdir build
pushd build > /dev/null

# Ref: https://github.com/libevent/libevent/blob/release-2.1.8-stable/CMakeLists.txt
cmake .. \
  -DEVENT__BUILD_SHARED_LIBRARIES=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DEVENT__DISABLE_BENCHMARK=OFF \
  -DEVENT__DISABLE_TESTS=OFF \
  -DEVENT__DISABLE_REGRESS=OFF \
  -DEVENT__DISABLE_SAMPLES=OFF

make -j "$(nproc)"
make install
popd > /dev/null

popd > /dev/null

# Cleanup
rm -rf "${PKG_NAME}" "libevent-release-${VERSION}-stable"
ldconfig

info "OK. libevent ${VERSION} successfully installed"
