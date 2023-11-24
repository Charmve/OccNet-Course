#! /bin/bash
set -euo pipefail

CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1090,SC1091
. "${CURR_DIR}/installer_base.sh"

apt_get_update_and_install \
  libglew-dev \
  libjpeg-dev \
  libpng-dev \
  libfreetype6-dev \
  libsnappy-dev \
  libflann-dev

pip3_install numpy

VERSION="4.5.5"
PKG_NAME="opencv-${VERSION}.tar.gz"
DOWNLOAD_LINK="https://github.com/opencv/opencv/archive/4.5.5/opencv-4.5.5.tar.gz"
CHECKSUM="a1cfdcf6619387ca9e232687504da996aaa9f7b5689986b8331ec02cb61d28ad"

download_if_not_cached "${PKG_NAME}" "${CHECKSUM}" "${DOWNLOAD_LINK}"

tar xzf "${PKG_NAME}"
pushd "opencv-${VERSION}" > /dev/null

mkdir build
pushd build > /dev/null

cmake .. \
  -DCMAKE_INSTALL_PREFIX=/usr \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_SKIP_RPATH=ON \
  -DENABLE_PRECOMPILED_HEADERS=OFF \
  -DBUILD_WITH_DEBUG_INFO=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_opencv_python2=OFF \
  -DBUILD_opencv_python3=ON \
  -DBUILD_DOCS=OFF \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DWITH_QT=OFF \
  -DWITH_IPP=OFF \
  -DWITH_ITT=OFF \
  -DWITH_TBB=OFF \
  -DWITH_OPENEXR=OFF \
  -DWITH_1394=OFF \
  -DWITH_OPENCL=OFF \
  -DWITH_WEBP=OFF \
  -DWITH_VTK=OFF \
  -DWITH_OPENNI=OFF \
  -DWITH_GSTREAMER=OFF \
  -DWITH_GSTREAMER_0_10=OFF \
  -Wno-dev

make -j "$(nproc)"
make install
popd > /dev/null

popd > /dev/null

# Cleanup
rm -rf "${PKG_NAME}" "opencv-${VERSION}"
apt_get_cleanup
ldconfig

info "OK. Opencv ${VERSION} successfully installed"
