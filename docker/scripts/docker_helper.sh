#!/bin/bash
set -euo pipefail

MAIWEI_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
source "${MAIWEI_ROOT_DIR}/docker/scripts/docker_base.sh"

_BOLD='\033[1m'
_RED='\033[0;31m'
_GREEN='\033[32m'
_WHITE='\033[34m'
_YELLOW='\033[33m'
_NO_COLOR='\033[0m'

SUPPORTED_ARCHS=(
  x86_64
  aarch64
)

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

function container_status() {
	docker inspect -f '{{.State.Status}}' "$1" 2>/dev/null
}

function remove_container() {
	local container="$1"
	docker stop "${container}" &>/dev/null
	docker rm -v -f "${container}" &>/dev/null
}

function ensure_one_dev_docker() {
	local container="$1"
	local enforce="$2"
	local status
	if status="$(container_status "${container}")"; then
		if [[ "${status}" == "running" ]]; then
			warning "Another container named ${container} already running."
			if [[ "${enforce}" == false ]]; then
				warning "  Consider starting Dev Docker with the '-f/--force' option. Exiting."
				exit 1
			else
				info "Remove existing container with name [${container}] ..."
			fi
		fi
		remove_container "${container}"
	else
		ok "No previous ${container} found."
	fi
}

function run() {
	echo "${@}" >&2
	"${@}"
}

function docker_image_exists() {
	local img="$1"
	docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${img}$"
}

function sigint_handler() {
  ok "Interrupted. Exit."
  exit 1
}

function _check_arch_support() {
  local host_arch="$1"
  for arch in "${SUPPORTED_ARCHS[@]}"; do
    if [[ "${arch}" == "${host_arch}" ]]; then
      return 0
    fi
  done
  return 1
}

function check_host_environment() {
  local host_os
  host_os="$(uname -s)"
  if [[ "${host_os}" != "Linux" ]]; then
    warning "Running Maiwei Dev Docker container on ${host_os} is not supported. Exiting..."
    exit 1
  fi

  local host_arch
  host_arch="$(uname -m)"

  if ! _check_arch_support "${host_arch}"; then
    error "Unsupported CPU arch: ${host_arch}"
    exit 1
  fi

  local status
  status="$(systemctl is-active docker)"
  if [[ "${status}" != "active" ]]; then
    error "It seems that Docker is not running. You can run \"systemctl start docker\" and then try again."
    exit 1
  fi
}

function determine_gpu_use_host() {
  local -n _use_gpu="$1"
  local -n _run_cmd="$2"
  _use_gpu=0
  _run_cmd=("docker" "run")

  local host_arch
  host_arch="$(uname -m)"

  if [[ "${host_arch}" == "aarch64" ]]; then
    #TODO(Charmve): Refer to Xavier:/etc/systemd/nv.sh
    _use_gpu=1
  elif [[ "${host_arch}" == "x86_64" ]]; then
    if [[ ! -x "$(command -v nvidia-smi)" ]]; then
      warning "nvidia-smi not found. CPU will be used."
    elif [[ -z "$(nvidia-smi)" ]]; then
      warning "No GPU device found. CPU will be used."
    else
      _use_gpu=1
    fi
  else
    error "Unsupported CPU architecture: ${host_arch}"
    return 1
  fi

  local nv_docker_doc="https://github.com/NVIDIA/nvidia-docker/blob/master/README.md"
  if [[ "${_use_gpu}" -eq 1 ]]; then
    if [[ -x "$(which nvidia-docker)" ]]; then
      _run_cmd=("nvidia-docker" "run")
    elif [[ -x "$(which nvidia-container-toolkit)" ]]; then
      local docker_version
      docker_version="$(docker version --format '{{.Server.Version}}')"
      if dpkg --compare-versions "${docker_version}" "ge" "19.03"; then
        _run_cmd=("docker" "run" "--gpus" "all")
	  else
        warning "Please upgrade to docker-ce 19.03+ to access GPU from container."
        _use_gpu=0
      fi
    else
      _use_gpu=0
      warning "Cannot access GPU from within container. Please install latest Docker" \
        "and NVIDIA Container Toolkit as described by: "
      warning "  ${nv_docker_doc}"
    fi
  fi
}

function run() {
  echo "${@}" >&2
  "${@}"
}

function setup_user_bazelrc() {
  local container="$1"
  local office="$2"
  local use_gpu="$3"
  my_cmd=(
    "/maiwei/docker/scripts/setup_user_bazelrc.py"
    "--office=${office}"
  )
  if [[ "${use_gpu}" -gt 0 ]]; then
    my_cmd+=("--use_gpu")
  fi
  docker exec -u "${USER}" "${container}" "${my_cmd[@]}"
}

function git_lfs_check() {
  if inside_git; then
    if [[ ! -x "$(command -v git-lfs)" ]]; then
      warning "git-lfs not found. You may experience issues with Git operations on host."
      info "To install git-lfs, please refer to:"
      info "  https://docs.github.com/en/github/managing-large-files/versioning-large-files/installing-git-large-file-storage"
    fi
  fi
}