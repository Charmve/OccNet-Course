#! /bin/bash
set -euo pipefail
CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1090,SC1091
. "${CURR_DIR}/installer_base.sh"

VERSION="0.6.0"
PKG_NAME_OSQP="osqp-${VERSION}.tar.gz"
CHECKSUM="6e00d11d1f88c1e32a4419324b7539b89e8f9cbb1c50afe69f375347c989ba2b"

DOWNLOAD_LINK="https://github.com/oxfordcontrol/osqp/archive/v${VERSION}.tar.gz"
download_if_not_cached "${PKG_NAME_OSQP}" "${CHECKSUM}" "${DOWNLOAD_LINK}"

tar xzf "${PKG_NAME_OSQP}"

pushd "osqp-${VERSION}" > /dev/null

PKG_NAME_QDLDL="qdldl-0.1.5.tar.gz"
CHECKSUM="2868b0e61b7424174e9adef3cb87478329f8ab2075211ef28fe477f29e0e5c99"
DOWNLOAD_LINK="https://github.com/oxfordcontrol/qdldl/archive/v0.1.5.tar.gz"
download_if_not_cached "${PKG_NAME_QDLDL}" "${CHECKSUM}" "${DOWNLOAD_LINK}"
tar xzf ${PKG_NAME_QDLDL} --strip-components=1 \
  -C ./lin_sys/direct/qdldl/qdldl_sources

mkdir build && cd build

cmake .. \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_INSTALL_PREFIX="${SYSROOT_DIR}" \
  -DCMAKE_BUILD_TYPE=Release
make "-j$(nproc)"
make install

rm -f "${SYSROOT_DIR}/lib/libqdldl.a" \
  "${SYSROOT_DIR}/lib/libosqp.a"
chmod 0755 \
  "${SYSROOT_DIR}/lib/libqdldl.so" \
  "${SYSROOT_DIR}/lib/libosqp.so"

popd > /dev/null
ldconfig

rm -rf "osqp-${VERSION}" "${PKG_NAME_OSQP}"
ok "Successfully installed OSQP ${VERSION}"
