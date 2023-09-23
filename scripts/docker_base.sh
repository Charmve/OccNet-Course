#! /bin/bash
TOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "${TOP_DIR}/scripts/mway_base.sh"

##### EXPORTED CONSTANTS AND VARIABLES #####
export ALIYUN_REGISTRY="registry.mwayai.com"
export X86_64_DEV_REPO="global/mway-dev"
export AARCH64_DEV_REPO="global/mway-aarch64-dev"

function docker_login_aliyun_registry_ro() {
  # NOTE:
  # Ref: https://mway.atlassian.net/servicedesk/customer/portal/8/article/1288962533
  echo "mway@2021" | docker login --username=registry-ro@1429873127699649 \
    --password-stdin registry.mwayai.com
}

function docker_login_aliyun_registry() {
  echo "1055d3e698d289f2af8" | docker login --username=registry-tmp@1429873127699649 \
    --password-stdin registry.mwayai.com
}

function docker_login_aliyun_registry_rw() {
  if [[ -z "${ALIYUN_REGISTRY_USERNAME:-}" || -z "${ALIYUN_REGISTRY_PASSWORD:-}" ]]; then
    error "Please set the following Environment variables for access to push to AliYun Registry:"
    error "  export ALIYUN_REGISTRY_USERNAME=XXX"
    error "  export ALIYUN_REGISTRY_PASSWORD=YYY"
    return 1
  fi
  echo "${ALIYUN_REGISTRY_PASSWORD}" | docker login --username "${ALIYUN_REGISTRY_USERNAME}" \
    --password-stdin registry.mwayai.com
}

############################################################
# Get Dev Docker tag for arch defined in the docker/TAG file
# Globals:
#   TOP_DIR => MWAY_ROOT_DIR
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

function check_logined_aliyun_repo() {
  if [ ! -f ~/.docker/config.json ]; then
    echo "false"
    return
  fi
  local TOKEN
  TOKEN=$(jq '.auths."registry.mwayai.com"' ~/.docker/config.json)
  if [ "$TOKEN" == "null" ]; then
    echo "false"
  else
    echo "true"
  fi
}

function check_image_exist_in_aliyun_repo() {
  local IMAGE="$1"
  local ALIYUN_LOGINED

  docker_login_aliyun_registry > /dev/null 2>&1
  ALIYUN_LOGINED=$(check_logined_aliyun_repo)

  if [ "$ALIYUN_LOGINED" != "true" ]; then
    echo "false"
    return
  fi
  if docker manifest inspect "$IMAGE" > /dev/null 2>&1; then
    echo "true"
  else
    echo "false"
  fi
}

function push_image_to_aliyun_repo() {
  local IMAGE=$1
  local ALIYUN_LOGINED
  docker_login_aliyun_registry > /dev/null 2>&1
  ALIYUN_LOGINED=$(check_logined_aliyun_repo)

  if [ "$ALIYUN_LOGINED" != "true" ]; then
    error "******** Push To ALIYUN Repo Failed, Please login********"
  else
    info "Pushing image $IMAGE"
    docker push "$IMAGE"
    info 'Pushing image Done'
  fi
}
