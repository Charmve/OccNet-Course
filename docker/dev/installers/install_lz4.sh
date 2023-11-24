#! /bin/bash
set -euo pipefail

CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1090,SC1091
. "${CURR_DIR}/installer_base.sh"

VERSION="1.9.3"

PKG_NAME="lz4-${VERSION}.tar.gz"
DOWNLOAD_LINK="https://github.com/lz4/lz4/archive/refs/tags/v${VERSION}.tar.gz"
CHECKSUM="030644df4611007ff7dc962d981f390361e6c97a34e5cbc393ddfbe019ffe2c1"

download_if_not_cached "${PKG_NAME}" "${CHECKSUM}" "${DOWNLOAD_LINK}"

tar xzf "${PKG_NAME}"
pushd "lz4-${VERSION}" > /dev/null

make -j "$(nproc)"
make install

popd > /dev/null

# Cleanup
rm -rf "${PKG_NAME}" "lz4-${VERSION}"
ldconfig

info "OK. lz4 ${VERSION} successfully installed"
