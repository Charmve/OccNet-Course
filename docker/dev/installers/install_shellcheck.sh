#! /bin/bash
set -euo pipefail

CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1090,SC1091
. "${CURR_DIR}/installer_base.sh"

VERSION="0.8.0"

TARGET_ARCH="$(uname -m)"

# As always, x86_64 and aarch64 only
PKG_NAME="shellcheck-v${VERSION}.linux.${TARGET_ARCH}.tar.xz"
DOWNLOAD_LINK="https://github.com/koalaman/shellcheck/releases/download/v${VERSION}/${PKG_NAME}"
CHECKSUM=
if [[ "${TARGET_ARCH}" == "x86_64" ]]; then
  CHECKSUM="ab6ee1b178f014d1b86d1e24da20d1139656c8b0ed34d2867fbb834dad02bf0a"
elif [[ "${TARGET_ARCH}" == "aarch64" ]]; then
  CHECKSUM="9f47bbff5624babfa712eb9d64ece14c6c46327122d0c54983f627ae3a30a4ac"
else
  warning "${TARGET_ARCH} architecture is currently not supported."
  exit 1
fi

download_if_not_cached "${PKG_NAME}" "${CHECKSUM}" "${DOWNLOAD_LINK}"
tar xJf "${PKG_NAME}"

pushd "shellcheck-v${VERSION}" &> /dev/null
mv shellcheck "${SYSROOT_DIR}/bin"
chmod a+x "${SYSROOT_DIR}/bin/shellcheck"
popd &> /dev/null

rm -rf "${PKG_NAME}" "shellcheck-v${VERSION}"
ok "shellcheck ${VERSION} succefully installed"
