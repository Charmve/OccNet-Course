#! /bin/bash
set -euo pipefail

CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1090,SC1091
. "${CURR_DIR}/installer_base.sh"

apt_get_update_and_install \
  zlib1g-dev \
  liblzma-dev

VERSION="2.9.12"

PKG_NAME="libxml2-${VERSION}.tar.gz"
DOWNLOAD_LINK="https://github.com/GNOME/libxml2/archive/refs/tags/v2.9.12.tar.gz"
CHECKSUM="8a4ddd706419c210b30b8978a51388937fd9362c34fc9a3d69e4fcc6f8055ee0"

download_if_not_cached "${PKG_NAME}" "${CHECKSUM}" "${DOWNLOAD_LINK}"

tar xzf "${PKG_NAME}"
pushd "libxml2-${VERSION}" > /dev/null

mkdir build
pushd build > /dev/null

cmake .. \
  -D CMAKE_INSTALL_PREFIX="${SYSROOT_DIR}" \
  -DLIBXML2_WITH_PYTHON=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release

make -j "$(nproc)"
make install
popd > /dev/null

popd > /dev/null

# Cleanup
rm -rf "${SYSROOT_DIR}/share/doc/libxml2"
rm -rf "${PKG_NAME}" "libxml2-${VERSION}"

apt_get_cleanup
ldconfig

info "OK. libxml2 ${VERSION} successfully installed"
