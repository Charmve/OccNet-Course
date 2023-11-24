#! /bin/bash
set -euo pipefail

CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1090,SC1091
. "${CURR_DIR}/installer_base.sh"

apt_get_update_and_install libacl1-dev
apt_get_cleanup

VERSION="3.5.2"
PKG_NAME="libarchive-${VERSION}.tar.gz"
DOWNLOAD_LINK="https://github.com/libarchive/libarchive/archive/v3.5.2.tar.gz"
CHECKSUM="126058cb4cf50e36bcf83307f5d987bde2ecebabcae985b6a153116362d25b7b"

download_if_not_cached "${PKG_NAME}" "${CHECKSUM}" "${DOWNLOAD_LINK}"

tar xzf "${PKG_NAME}"
pushd "libarchive-${VERSION}" > /dev/null

pushd build > /dev/null

cmake .. -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release

make -j "$(nproc)"
make install
popd > /dev/null

popd > /dev/null

# Cleanup
rm -rf "${PKG_NAME}" "libarchive-${VERSION}"
ldconfig

info "OK. libarchive ${VERSION} successfully installed"
