#! /bin/bash
set -euo pipefail

CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1090,SC1091
. "${CURR_DIR}/installer_base.sh"

apt_get_update_and_install \
  liblzma-dev

VERSION="1.5.2"

PKG_NAME="zstd-${VERSION}.tar.gz"
DOWNLOAD_LINK="https://github.com/facebook/zstd/archive/v1.5.2/zstd-1.5.2.tar.gz"
CHECKSUM="f7de13462f7a82c29ab865820149e778cbfe01087b3a55b5332707abf9db4a6e"

download_if_not_cached "${PKG_NAME}" "${CHECKSUM}" "${DOWNLOAD_LINK}"

tar xzf "${PKG_NAME}"
pushd "zstd-${VERSION}" > /dev/null

make -j "$(nproc)"
make install

popd > /dev/null

# Cleanup
rm -rf "${PKG_NAME}" "zstd-${VERSION}"
apt_get_cleanup
ldconfig

info "OK. zstd ${VERSION} successfully installed"
