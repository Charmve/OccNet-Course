#! /bin/bash
set -euo pipefail

CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1090,SC1091
. "${CURR_DIR}/installer_base.sh"

VERSION="3.3.9"

PKG_NAME="eigen-${VERSION}.tar.gz"
DOWNLOAD_LINK="https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.tar.gz"
CHECKSUM="7985975b787340124786f092b3a07d594b2e9cd53bbfe5f3d9b1daee7d55f56f"

download_if_not_cached "${PKG_NAME}" "${CHECKSUM}" "${DOWNLOAD_LINK}"

tar xzf "${PKG_NAME}"
pushd "eigen-${VERSION}" > /dev/null
mkdir build
pushd build > /dev/null

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DCMAKE_INSTALL_PREFIX=/usr \
  -DEIGEN_TEST_CXX11=ON

make install
popd > /dev/null

popd > /dev/null

# Cleanup
rm -rf "${PKG_NAME}" "eigen-${VERSION}"

info "OK. Eigen ${VERSION} successfully installed"
