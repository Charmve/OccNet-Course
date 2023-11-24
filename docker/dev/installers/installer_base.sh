#! /bin/bash
set -euo pipefail

_BOLD='\033[1m'
_RED='\033[0;31m'
_GREEN='\033[32m'
_WHITE='\033[34m'
_YELLOW='\033[33m'
_NO_COLOR='\033[0m'

function info() {
  (echo >&2 -e "[${_WHITE}${_BOLD}INFO${_NO_COLOR}] $*")
}

function error() {
  (echo >&2 -e "[${_RED}ERROR${_NO_COLOR}] $*")
}

function warning() {
  (echo >&2 -e "${_YELLOW}[WARNING] $*${_NO_COLOR}")
}

function ok() {
  (echo >&2 -e "[${_GREEN}${_BOLD} OK ${_NO_COLOR}] $*")
}

export SYSROOT_DIR="/usr/local"
export RCFILES_DIR="/opt/rcfiles"
export LOCAL_HTTP_ADDR="http://${DOCKER_BUILD_IP:-172.17.0.1}:8388"

function py3_version() {
  local version
  # major.minor.rev (e.g. 3.6.9) expected
  version="$(python3 --version | awk '{print $2}')"
  echo "${version%.*}"
}

function pip3_install() {
  python3 -m pip install --timeout 30 --no-cache-dir "$@"
}

function apt_get_update_and_install() {
  apt-get -y update \
    && apt-get -y install --no-install-recommends "$@"
}

# Remove a package completely
function apt_get_purge() {
  apt-get -y purge --autoremove "$@"
}

function apt_get_cleanup() {
  apt-get clean && rm -rf /var/lib/apt/lists/*
}

function make_sure_maiwei_environ_initialized() {
  [[ -d "${SYSROOT_DIR}/bin" ]] || mkdir -p "${SYSROOT_DIR}/bin"
  [[ -d "${SYSROOT_DIR}/include" ]] || mkdir -p "${SYSROOT_DIR}/include"
  [[ -d "${SYSROOT_DIR}/lib" ]] || mkdir -p "${SYSROOT_DIR}/lib"
  [[ -d "${SYSROOT_DIR}/share" ]] || mkdir -p "${SYSROOT_DIR}/share"
}

function _local_http_cached() {
  /usr/bin/curl -sfI --connect-timeout 2 "${LOCAL_HTTP_ADDR}/$1"
}

function _checksum_check_pass() {
  local pkg="$1"
  local expected_cs="$2"
  local actual_cs
  actual_cs="$(/usr/bin/sha256sum "${pkg}" | awk '{print $1}')"
  if [[ "${actual_cs}" == "${expected_cs}" ]]; then
    true
  else
    warning "${pkg}: checksum mismatch, ${expected_cs}" \
      "expected, got: ${actual_cs}"
    false
  fi
}

function download_if_not_cached {
  local pkg_name="$1"
  local expected_cs="$2"
  local url="$3"

  if _local_http_cached "${pkg_name}"; then
    local local_addr="${LOCAL_HTTP_ADDR}/${pkg_name}"
    info "Local http cache hit ${pkg_name}..."
    curl -fsSL "${local_addr}" -o "${pkg_name}"
    if _checksum_check_pass "${pkg_name}" "${expected_cs}"; then
      ok "Successfully downloaded ${pkg_name} from ${LOCAL_HTTP_ADDR}," \
        "will use it."
      return
    else
      warning "Found ${pkg_name} in local http cache, but checksum mismatch."
      rm -f "${pkg_name}"
    fi
  fi # end http cache check

  info "Start to download $pkg_name from ${url} ..."
  curl -fsSL "$url" -o "$pkg_name"
  ok "Successfully downloaded $pkg_name"
}

function run() {
  info "$*"
  "$@"
}

make_sure_maiwei_environ_initialized
