#! /bin/bash
set -euo pipefail

CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1090,SC1091
. "${CURR_DIR}/installer_base.sh"

apt_get_update_and_install libunwind-dev

VERSION="2.9.1"
PKG_NAME="gperftools-${VERSION}.tar.gz"
CHECKSUM="484a88279d2fa5753d7e9dea5f86954b64975f20e796a6ffaf2f3426a674a06a"
DOWNLOAD_LINK="https://github.com/gperftools/gperftools/archive/${PKG_NAME}"
download_if_not_cached "${PKG_NAME}" "${CHECKSUM}" "${DOWNLOAD_LINK}"
tar xzf "${PKG_NAME}"

pushd "gperftools-gperftools-${VERSION}" > /dev/null
./autogen.sh

./configure --disable-static --enable-frame-pointers
# --disable-deprecated-pprof --with-pic=yes

make -j "$(nproc)"
make install

popd > /dev/null

apt_get_cleanup
ldconfig
rm -rf "${PKG_NAME}" "gperftools-gperftools-${VERSION}"

ok "Successfully installed gperftools ${VERSION}"
