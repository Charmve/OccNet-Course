#! /bin/bash
set -euo pipefail

CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1090,SC1091
. "${CURR_DIR}/installer_base.sh"

ARCH="$(uname -m)"
if [[ "${ARCH}" != "x86_64" && "${ARCH}" != "aarch64" ]]; then
  error "Architecture ${ARCH} not supported yet"
  exit 1
fi

BAZEL_VERSION="5.3.2"

# See ref: https://docs.bazel.build/versions/master/install-ubuntu.html
apt_get_update_and_install zlib1g-dev # openjdk-11-jdk

if [[ "$ARCH" == "x86_64" ]]; then
  PKG_NAME="bazel_${BAZEL_VERSION}-linux-x86_64.deb"
  DOWNLOAD_LINK="https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/${PKG_NAME}"
  CHECKSUM="898848a688aa05a906c14467a50cbef2daaf7b5fc649bd2d22490aced8c1702c"
  download_if_not_cached "${PKG_NAME}" "${CHECKSUM}" "${DOWNLOAD_LINK}"

  dpkg -i "${PKG_NAME}"
  rm -rf "${PKG_NAME}"
else
  PKG_NAME="bazel-${BAZEL_VERSION}-linux-arm64"
  CHECKSUM="dcad413da286ac1d3f88e384ff05c2ed796f903be85b253591d170ce258db721"
  DOWNLOAD_LINK="https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/${PKG_NAME}"

  download_if_not_cached "${PKG_NAME}" "${CHECKSUM}" "${DOWNLOAD_LINK}"

  cp -f ${PKG_NAME} /usr/bin/bazel
  chmod a+x /usr/bin/bazel

  # NOTE:
  #   `bazel_completion.bash` was copied from `/etc/bash_completion.d/bazel`
  # on x86_64 with bazel deb-installed.
  # cp -f "${RCFILES_DIR}/bazel_completion.bash" /etc/bash_completion.d/bazel
  rm -rf "${PKG_NAME}"
fi

info "Done installing bazel ${BAZEL_VERSION}"

# Clean up cache to reduce layer size.
apt_get_cleanup
