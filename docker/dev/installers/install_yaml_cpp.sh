#! /bin/bash
set -euo pipefail

CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1090,SC1091
. "${CURR_DIR}/installer_base.sh"

VERSION="0.7.0"
PKG_NAME="yaml-cpp-${VERSION}.tar.gz"
DOWNLOAD_LINK="https://github.com/jbeder/yaml-cpp/archive/${PKG_NAME}"
CHECKSUM="43e6a9fcb146ad871515f0d0873947e5d497a1c9c60c58cb102a97b47208b7c3"

download_if_not_cached "${PKG_NAME}" "${CHECKSUM}" "${DOWNLOAD_LINK}"

tar xzf "${PKG_NAME}"
pushd "yaml-cpp-yaml-cpp-${VERSION}" > /dev/null

mkdir build
pushd build > /dev/null

cmake .. \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_TESTING=OFF \
  -DCMAKE_BUILD_TYPE=Release

make -j "$(nproc)"
make install
popd > /dev/null

popd > /dev/null

# Cleanup
rm -rf "${PKG_NAME}" "yaml-cpp-yaml-cpp-${VERSION}"
ldconfig

info "OK. yaml-cpp ${VERSION} successfully installed"
