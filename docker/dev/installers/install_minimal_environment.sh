#! /bin/bash
set -euo pipefail

CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1090,SC1091
. "${CURR_DIR}/installer_base.sh"

GEOLOC="${GEOLOC:-cn}"
ARCH="$(uname -m)"

apt_get_update_and_install ca-certificates

# APT sources.list settings
if [[ "${ARCH}" == "x86_64" ]]; then
  if [[ "${GEOLOC}" == "cn" ]]; then
    cp -f "${RCFILES_DIR}/sources.list.cn.x86_64" /etc/apt/sources.list
  fi
else # AArch64
  if [[ "${GEOLOC}" == "cn" ]]; then
    cp -f "${RCFILES_DIR}/sources.list.cn.aarch64" /etc/apt/sources.list
  fi
fi

apt_get_update_and_install \
  curl \
  file \
  gawk \
  gnupg2 \
  less \
  python3 \
  python3-pip \
  python3-distutils \
  sed \
  software-properties-common \
  sudo \
  unzip \
  wget \
  zip \
  xz-utils

# https://git-scm.com/download/linux
add-apt-repository -y ppa:git-core/ppa
add-apt-repository ppa:ubuntu-toolchain-r/test

apt_get_update_and_install \
  git \
  gcc-9 \
  g++-9 \
  libstdc++-9-dev \
  libc6-dev

add-apt-repository -y --remove ppa:git-core/ppa
add-apt-repository -y --remove ppa:ubuntu-toolchain-r/test

apt_get_update_and_install \
  bash-completion \
  build-essential \
  autoconf \
  automake \
  libtool \
  gcc \
  g++ \
  gdb \
  patch \
  vim \
  python3 \
  python3-pip \
  python3-dev \
  pkg-config

update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 90

apt_get_cleanup
##----------------##
##    SUDO        ##
##----------------##
sed -i /etc/sudoers -re 's/^%sudo.*/%sudo ALL=(ALL:ALL) NOPASSWD: ALL/g'

##----------------##
## default shell  ##
##----------------##
chsh -s /bin/bash
ln -s /bin/bash /bin/sh -f

##----------------##
## Python Settings |
##----------------##
update-alternatives --install /usr/bin/python python /usr/bin/python3 36
