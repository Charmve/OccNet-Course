#! /usr/bin/env bash
set -u

MWAY_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
source "${MWAY_ROOT_DIR}/scripts/docker_base.sh"

BAYAREA_HABOR_REPO="docker-img.mwayai.com"
readonly BAYAREA_HABOR_REPO

SUPPORTED_ARCHS=(
  x86_64
  aarch64
)

##########################################################################
# Check whether the specified Docker image exist
# Arguments:
#   img  The specified Docker image to check for existence
# Returns:
#   0 if found, non-zero otherwise
##########################################################################
function docker_image_exist() {
  local img="$1"
  docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${img}$"
}

function docker_tag_image() {
  docker image tag "$1" "$2"
}

function docker_rename_image() {
  docker image tag "$1" "$2"
  docker rmi "$1"
}

function container_status() {
  docker inspect -f '{{.State.Status}}' "$1"
}

function remove_container() {
  local container="$1"
  docker stop "${container}" &> /dev/null
  docker rm -v -f "${container}" &> /dev/null
}

#############################################
# Pull Docker image
# Globals:
#   ALIYUN_REGISTRY
# Arguments:
#   $1 image to pull
# Returns:
#   0 on success, non-zero otherwise
#############################################
function docker_pull_image() {
  local image="$1"
  local registry_image
  registry_image="${BAYAREA_HABOR_REPO}/${image}"

  if [[ -n "${registry_image}" ]]; then
    if docker pull "${registry_image}"; then
      docker_rename_image "${registry_image}" "${image}"
      ok "Successfully pulled Docker image ${image} Habor."
      return 0
    else
      warning "Failed to pull Docker image ${image} Habor."
      warning "Now switch to AliYun Registry."
    fi
  fi

  registry_image="${ALIYUN_REGISTRY}/${image}"
  docker_login_aliyun_registry_ro
  if docker pull "${registry_image}"; then
    ok "Successfully pulled Docker image ${image} from AliYun Registry"
    docker_rename_image "${registry_image}" "${image}"
    return 0
  else
    error "Failed to pull Docker image ${image} from AliYun Registry."
    return 1
  fi
}

##########################################
# Resolve Dev Docker container conflicts
# Globals:
#   MWAY_ROOT_DIR
#######################################
function resolve_container_conflict() {
  local container="$1"
  local enforce="$2"

  local existing_containers
  readarray -t existing_containers < <(docker ps -a --filter "name=^mway_dev_" --format '{{.Names}}')

  if [[ "${#existing_containers[@]}" -eq 0 ]]; then
    info "No other Dev container(s) found."
    return 0
  fi

  info "Found existing Dev container(s): ${existing_containers[*]}"
  # Found an existing container with the same name
  if array_contains "${container}" "${existing_containers[@]}"; then
    local status
    status="$(container_status "${container}")"
    if [[ "${status}" == "running" ]]; then
      warning "Container conflict found: another container also named [${container}]"
      if [[ "${enforce}" == false ]]; then
        warning "  Consider starting Dev Docker with the '-f/--force' option. Exiting."
        exit 1
      else
        info "Remove existing container with name [${container}] ..."
      fi
    fi
    remove_container "${container}"
  fi

  for entry in "${existing_containers[@]}"; do
    # Note: Already processed.
    if [[ "${entry}" == "${container}" ]]; then
      continue
    fi

    local status
    status="$(container_status "${entry}")"
    if [[ "${status}" == "exited" ]]; then
      remove_container "${entry}"
      continue
    fi

    local root_dir_host
    root_dir_host="$(docker exec "${entry}" printenv MWAY_ROOT_DIR_HOST)"
    if [[ "${root_dir_host}" == "${MWAY_ROOT_DIR}" ]]; then
      warning "Container conflict found, another container [${entry}] already in workspace [${MWAY_ROOT_DIR}](host)"
      if [[ "${enforce}" == false ]]; then
        warning "  Consider starting Dev Docker with the '-f/--force' option. Exiting."
        exit 1
      fi
      info "Force removal of [${entry}] in workspace [${MWAY_ROOT_DIR}]..."
      remove_container "${entry}"
    fi
  done
}

##########################################
# Add user(sudoer) in docker
# Globals:
#   IS_CI
#######################################
function docker_start_user() {
  local container="$1"
  local user="$2"
  local uid="$3"
  local group="$4"
  local gid="$5"
  if [[ "${user}" != "root" ]]; then
    local work_dir="/mway"
    docker exec -u root "${container}" \
      bash -c "${work_dir}/docker/scripts/docker_start_user.sh ${user} ${uid} ${group} ${gid}"
  fi
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
    warning "Running QCraft Dev Docker container on ${host_os} is not supported. Exiting..."
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

function determine_host_map_dir() {
  local -n _map_dir="$1"
  local curr_dir="$2"

  # Try-and-mount local map git repo first
  if [[ -d "${curr_dir}/mway-maps" || -d "${curr_dir}/mway-maps-china" ]]; then
    ok "Non-shared maps under the [${curr_dir}] directory will be used."
    _map_dir="${curr_dir}"
    return
  fi

  upper_dir="$(dirname "${curr_dir}")"
  if [[ -d "${upper_dir}/mway-maps" ]] || [[ -d "${upper_dir}/mway-maps-china" ]]; then
    ok "Shared maps under the [${upper_dir}] directory will be used."
    _map_dir="${upper_dir}"
    return
  fi
  _map_dir=
}

function determine_display() {
  # Note:
  # DISPLAY was unbounded for remote SSH sessions, we parse it from `w`
  # output for the following scenario:
  # Dev Docker started via remote SSH w/ GUI programs run later from local X env
  local display="${DISPLAY:-}"
  if [[ -z "${display}" ]]; then
    display="$(w -hs | awk -v user="${USER:-mway}" '$1 == user {
        if ($2 ~ /^:[0-9]+$/) {print $2; exit;}
    }')"
    warning "[Env] Fake DISPLAY from TTY: ${display}"
  fi
  echo "${display}"
}

function mount_debugfs() {
  local mount_point="/sys/kernel/debug"
  # Ref:
  # https://serverfault.com/questions/50585/whats-the-best-way-to-check-if-a-volume-is-mounted-in-a-bash-script
  if mountpoint -q "${mount_point}"; then
    info "debugfs already mounted on ${mount_point}"
    return
  fi
  sudo mount -t debugfs none /sys/kernel/debug
  sudo chmod -R 0755 /sys/kernel/debug
}

function install_prebuilt_goofys() {
  wget https://github.com/kahing/goofys/releases/latest/download/goofys \
    && sudo mv goofys /usr/local/bin/ \
    && sudo chmod a+x /usr/local/bin/goofys
}

function determine_gpu_use_host() {
  local -n _use_gpu="$1"
  local -n _run_cmd="$2"
  _use_gpu=0
  _run_cmd=("docker" "run")

  local host_arch
  host_arch="$(uname -m)"

  if [[ "${host_arch}" == "aarch64" ]]; then
    #TODO: Refer to Xavier:/etc/systemd/nv.sh
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
  local use_gpu="$2"
  local my_cmd=(
    "/mway/docker/scripts/setup_user_bazelrc.py"
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

function parse_mount_spec() {
  local spec="$1"
  local -n _result="$2"
  readarray -td':' parts < <(printf "%s" "${spec}")
  local num_parts="${#parts[@]}"
  local src_dir="${parts[0]}"
  if [[ ! -d "${src_dir}" ]]; then
    error "Directory-doesn't-exist: ${src_dir} (from --mount ${spec})"
    return 1
  fi
  if [[ "${num_parts}" -eq 1 ]]; then
    _result+=("type=bind,source=${src_dir},target=${src_dir}")
  elif [[ "${num_parts}" -eq 2 ]]; then
    local dst_dir="${parts[1]}"
    _result+=("type=bind,source=${src_dir},target=${dst_dir}")
  elif [[ "${num_parts}" -eq 3 ]]; then
    local dst_dir="${parts[1]}"
    local mode="${parts[2]}"
    if [[ "${mode}" == "ro" ]]; then
      _result+=("type=bind,source=${src_dir},target=${dst_dir},readonly")
    elif [[ "${mode}" == "rw" ]]; then
      _result+=("type=bind,source=${src_dir},target=${dst_dir}")
    else
      error "Unknown mount mode ${mode} (from '--mount ${spec}'). Should be 'ro' or 'rw'"
      return 1
    fi
  fi
  return 0
}

function check_tools_docker() {
  dev_param=()
  local result=0
  for var in "$@"; do
    if [[ "${var}" != "--tools" ]]; then
      dev_param+=("${var}")
    else
      result=1
    fi
  done
  return ${result}
}

function run_tools_docker_sh() {
  local RUNNING_CONTAINER="mway-tools"
  local existing_containers
  readarray -t existing_containers < <(docker ps -a --filter "name=^mway-tools" --format '{{.Names}}')

  for entry in "${existing_containers[@]}"; do
    if [[ "${entry}" == "${RUNNING_CONTAINER}" ]]; then
      local docker_status
      docker_status="$(container_status "${entry}")"
      if [[ "${docker_status}" == "exited" ]]; then
        local docker_run_cmd
        docker_run_cmd="docker start"
        run ${docker_run_cmd} ${RUNNING_CONTAINER}
        run docker exec -e USER="${USER}" -u "${USER}" -it "${RUNNING_CONTAINER}" /mway/scripts/"$1".sh "${dev_param[@]}"
        exit
      fi
      if [[ "${docker_status}" == "running" ]]; then
        run docker exec -e USER="${USER}" -u "${USER}" -it "${RUNNING_CONTAINER}" /mway/scripts/"$1".sh "${dev_param[@]}"
        exit
      fi
    fi
  done
  echo "setup $1 failed"
  echo "1.exit dev_docker: exit"
  echo "2.init tools_docker: OFFICE=\${OFFICE} dev-env/scripts/init_tool.sh"
  exit
}
