#! /bin/bash
set -euo pipefail

CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1090,SC1091
. "${CURR_DIR}/installer_base.sh"

# Ref: https://www.boost.org/
VERSION="1.76.0"
UNDERLINED_VERSION="${VERSION//./_}"

PKG_NAME="boost_${UNDERLINED_VERSION}.tar.bz2"
DOWNLOAD_LINK="https://boostorg.jfrog.io/artifactory/main/release/${VERSION}/source/${PKG_NAME}"
CHECKSUM="f0397ba6e982c4450f27bf32a2a83292aba035b827a5623a14636ea583318c41"

download_if_not_cached "${PKG_NAME}" "${CHECKSUM}" "${DOWNLOAD_LINK}"

tar xjf "${PKG_NAME}"

# Ref: https://www.boost.org/doc/libs/1_73_0/doc/html/mpi/getting_started.html
pushd "boost_${UNDERLINED_VERSION}"
./bootstrap.sh \
  --prefix="${SYSROOT_DIR}" \
  --without-icu

./b2 -d+2 -q -j"$(nproc)" \
  --without-graph_parallel \
  --without-mpi \
  variant=release \
  link=shared \
  threading=multi \
  install
popd
ldconfig

# Clean up
rm -rf "boost_${UNDERLINED_VERSION}" "${PKG_NAME}"
