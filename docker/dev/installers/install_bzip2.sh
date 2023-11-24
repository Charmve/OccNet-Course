#! /bin/bash
set -euo pipefail

CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1090,SC1091
. "${CURR_DIR}/installer_base.sh"

ARCH="$(uname -m)"

VERSION="1.0.8-2"

if [[ "${ARCH}" == "x86_64" ]]; then
  PKG_NAME="bzip2-${VERSION}-amd64.tar.gz"
  DOWNLOAD_LINK="https://maiwei-web.oss-cn-beijing.aliyuncs.com/cache/packages/${PKG_NAME}"
  CHECKSUM="c50e818106ad8a3d96d4bedfda7598621f9ed77686d5d64752063bab8e949373"
else
  error "bzip2 for ${ARCH} not ready. Exiting..."
  exit 1
fi

download_if_not_cached "${PKG_NAME}" "${CHECKSUM}" "${DOWNLOAD_LINK}"

tar xzf "${PKG_NAME}"
pushd "${PKG_NAME%.tar.gz}" > /dev/null

run dpkg -i "bzip2_${VERSION}_amd64.deb" \
  "libbz2-1.0_${VERSION}_amd64.deb" \
  "libbz2-dev_${VERSION}_amd64.deb"

popd > /dev/null

# Cleanup
rm -rf "${PKG_NAME}" "${PKG_NAME%.tar.gz}"

info "OK. bzip2 ${VERSION} successfully installed"
