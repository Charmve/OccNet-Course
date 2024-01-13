#! /bin/bash
TOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
source "${TOP_DIR}/scripts/maiwei_base.sh"

##### EXPORTED CONSTANTS AND VARIABLES #####
export X86_64_DEV_REPO="charmve/maiwei-dev-x86_64-20231116"
export AARCH64_DEV_REPO="global/maiwei-aarch64-dev"

############################################################
# Get Dev Docker tag for arch defined in the docker/TAG file
# Globals:
#   TOP_DIR => MAIWEI_ROOT_DIR
# Arguments:
#   an optional architecture name, e.g. x86_64, aarch64
# Returns:
#   Dev Docker tag for the arch
############################################################
function dev_docker_tag_default() {
  local arch="${1:-}"
  if [[ -z "${arch}" ]]; then
    arch="$(uname -m)"
  fi

  local tag_id
  if [[ "${arch}" == "x86_64" ]]; then
    tag_id="${arch}"
  else
    local model
    model="$(determine_arm64_model)"
    if [[ "${model}" == "jetson-orin" ]]; then
      tag_id="${arch}-${model}"
    else
      tag_id="${arch}"
    fi
  fi
  awk -F'=' -v tag_id="${tag_id}" '$1 == tag_id {print $2}' "${TOP_DIR}/docker/TAG"
}

function runtime_docker_tag_default() {
  local arch="${1:-}"
  if [[ -z "${arch}" ]]; then
    arch="$(uname -m)"
  fi
  awk -F'=' -v key="runtime-${arch}" '$1 == key {print $2}' "${TOP_DIR}/docker/TAG"
}

function ml_docker_tag_default() {
  local arch="${1:-}"
  if [[ -z "${arch}" ]]; then
    arch="$(uname -m)"
  fi

  awk -F'=' -v key="ml-${arch}" '$1 == key {print $2}' "${TOP_DIR}/docker/TAG"
}

function dev_docker_image() {
  local tag="${1:-}"
  local arch="${2:-}"
  if [[ -z "${arch}" ]]; then
    arch="$(uname -m)"
  fi

  if [[ -z "${tag}" ]]; then
    tag="$(dev_docker_tag_default "${arch}")"
  fi

  if [[ "${arch}" == "x86_64" ]]; then
    echo "${X86_64_DEV_REPO}:${tag}"
  else
    echo "${AARCH64_DEV_REPO}:${tag}"
  fi
}

function docker_image_size() {
  docker inspect -f "{{ .Size }}" "$1" | numfmt --to si --format "%.1f"
}

function prod_docker_tag_default() {
  local arch
  arch="${1:-$(uname -m)}"
  awk -F'=' -v key="prod-${arch}" '$1 == key {print $2}' "${TOP_DIR}/docker/TAG"
}

function dev_docker_image() {
  local tag="${1:-}"

  local arch="${2:-$(uname -m)}"

  if [[ -z "${tag}" ]]; then
    tag="$(dev_docker_tag_default "${arch}")"
  fi

  if [[ "${arch}" == "x86_64" ]]; then
    echo "${X86_64_DEV_REPO}:${tag}"
  else
    echo "${AARCH64_DEV_REPO}:${tag}"
  fi
}

function docker_image_size() {
  docker inspect -f "{{ .Size }}" "$1" | numfmt --to si --format "%.1f"
}

