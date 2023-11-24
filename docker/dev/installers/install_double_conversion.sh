#! /bin/bash
set -euo pipefail

CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1090,SC1091
. "${CURR_DIR}/installer_base.sh"

VERSION="3.1.6"

PKG_NAME="double-conversion-${VERSION}.tar.gz"
DOWNLOAD_LINK="https://github.com/google/double-conversion/archive/v3.1.6.tar.gz"
CHECKSUM="8a79e87d02ce1333c9d6c5e47f452596442a343d8c3e9b234e8a62fce1b1d49c"

download_if_not_cached "${PKG_NAME}" "${CHECKSUM}" "${DOWNLOAD_LINK}"

tar xzf "${PKG_NAME}"
pushd "double-conversion-${VERSION}" > /dev/null

# Ref: https://github.com/google/double-conversion#cmake
cmake . \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF

make -j "$(nproc)"
make install

popd > /dev/null

# Cleanup
rm -rf "${PKG_NAME}" "double-conversion-${VERSION}"
ldconfig

info "OK. double-conversion ${VERSION} successfully installed"
