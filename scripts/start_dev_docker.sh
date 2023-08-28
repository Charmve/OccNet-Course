#!/bin/bash
set -u

CHARMVE_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
# shellcheck disable=SC1090,SC1091
source "${CHARMVE_ROOT_DIR}/docker/scripts/docker_helper.sh"
# shellcheck disable=SC1090,SC1091
source "${CHARMVE_ROOT_DIR}/scripts/util/util.sh"

TOP_DIR="${CHARMVE_ROOT_DIR}"

HOST_ARCH="$(uname -m)"
IS_CI="${IS_CI:-false}"
IS_ORIN_CROSS_COMPILE="${IS_ORIN_CROSS_COMPILE:-false}"

if [[ "${IS_ORIN_CROSS_COMPILE}" == false && "${HOST_ARCH}" == "x86_64" && "${IS_CI}" == true ]]; then
  # Go two levels up for CI shell-runner to run forked repos
  TOP_DIR="$(cd "${TOP_DIR}/../.." && pwd -P)"
fi
info "TOP_DIR: [${TOP_DIR}], CHARMVE_ROOT_DIR: [${CHARMVE_ROOT_DIR}]"

# Note(jiaming): So that users can use CTRL+C to interrupt
trap sigint_handler INT

readonly INDENT="    " # 4 spaces
readonly CAR_ID="${CAR_ID:-}"
readonly LAPTOP_IP="${LAPTOP_IP:-192.168.5.125}"

readonly DEV_CONTAINER_DEFAULT="charmve_dev_${USER}"
DEV_CONTAINER="${DEV_CONTAINER_DEFAULT}"

OFFICE="${OFFICE:-}"
HOME_IN="/home/${USER}"

DOCKER_IMG=
SHM_SIZE=
MEM_SIZE=

MEM_LIMITED=false # determined by SHM_SIZE and MEM_SIZE

HOSTNAME_HOST="$(hostname)"
MOUNT_DEBUGFS="${MOUNT_DEBUGFS:-true}"
ENABLE_CORE_DUMP="${ENABLE_CORE_DUMP:-true}"
MOUNT_NFS="${MOUNT_NFS:-true}"
USE_GOOFYS="${USE_GOOFYS:-true}"

# NOTE(jiaming): Use of Env vars is discouraged for cmdline execution.
BAZEL_DIST_DIR="${HOME}/.distdir"
USE_GPU_HOST=true
USE_LOCAL_IMAGE=false
ASSUME_YES=false
UNIQUE_VIM=false
ENFORCE=false
ONBOARD_YES=false
PID_NAMESPACE=""

CUSTOM_DIR_MOUNTS=()

ARM64_MODEL=
if [[ "${HOST_ARCH}" == "aarch64" ]]; then
  ARM64_MODEL="$(determine_arm64_model)"
fi

function create_bazel_cache_dirs() {
  for dirent in "bazel" "repository_cache" "distdir"; do
    dirent="${CHARMVE_ROOT_DIR}/.cache/${dirent}"
    [[ -d "${dirent}" ]] || mkdir -p "${dirent}"
  done
}

function determine_local_volumes() {
  local -n _volumes="$1"
  local map_dir="$2"
  if [[ ! -e "${HOME}/charmve_data/core/backup" ]]; then
    mkdir -p "${HOME}/charmve_data/core/backup" || true
  fi
  _volumes=(
    "-v" "${TOP_DIR}:/charmve"
    "-v" "${TOP_DIR}/onboard/params/run_params/vehicles:/vehicles"
    "-v" "/media:/media"
    "-v" "/usr/src:/usr/src"
    "-v" "/var/run/dbus/system_bus_socket:/var/run/dbus/system_bus_socket"
    "-v" "/etc/localtime:/etc/localtime:ro"
    "-v" "/etc/machine-id:/etc/machine-id"
    "-v" "/lib/modules:/lib/modules"
    "-v" "/var/run/docker.sock:/var/run/docker.sock:rw"
    "-v" "/tmp/.X11-unix:/tmp/.X11-unix:rw"
    "-v" "${HOME}:/hosthome:rw"
    "-v" "${CHARMVE_ROOT_DIR}/.gitconfig:/etc/gitconfig"
    "-v" "${HOME}/.ssh:${HOME_IN}/.ssh"
    "-v" "${HOME}/.aws:${HOME_IN}/.aws"
    "-v" "${HOME}/.bash_history:${HOME_IN}/.bash_history"
    "-v" "/tmp/docker_files:/tmp/docker_files:rw"
    "-v" "/lib/systemd/system:/lib/systemd/system:rw"
    "-v" "/etc/network:/etc/network:rw"
    "-v" "/etc/netplan:/etc/netplan:rw"
    "-v" "/usr/bin/nvidia-docker:/usr/bin/nvidia-docker"
    "-v" "/home/charmve/charmve_data/core:/core"
  )

  if [[ -n "${map_dir}" ]]; then
    if [[ -d "${map_dir}/charmve-maps" ]]; then
      _volumes+=("-v" "${map_dir}/charmve-maps:/charmve-maps")
    fi
    if [[ -d "${map_dir}/charmve-maps-china" ]]; then
      _volumes+=("-v" "${map_dir}/charmve-maps-china:/charmve-maps-china")
    fi
  fi

  if [[ "${MEM_LIMITED}" == false ]]; then
    _volumes+=("-v" "/dev:/dev")
  fi

  if [[ "${HOST_ARCH}" == "x86_64" ]]; then
    _volumes+=(
      "-v" "/sys/kernel/debug:/sys/kernel/debug:rw"
      "-v" "${HOME}/release_home:/release_home"
    )
  else
    _volumes+=(
      "-v" "/usr/include/EGL:/usr/include/EGL:ro"
      "-v" "/usr/include/GLES2:/usr/include/GLES2:ro"
      "-v" "/etc/nv_tegra_release:/etc/nv_tegra_release:ro"
      "-v" "/etc/configure-camera:/etc/configure-camera:rw"
    )
    if [[ "${ARM64_MODEL}" == "jetson-xavier" ]]; then
      _volumes+=("-v" "/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra")
    fi
  fi

  if [[ "${ONBOARD_YES}" == true ]]; then
    # TODO(liyu): remove it if we start dev docker by tugboat
    local tugboat_env="/usr/local/lib/tugboat/env"
    if [[ -f "${tugboat_env}" ]]; then
      chmod a+rw "${tugboat_env}"
      # shellcheck disable=SC1090,SC1091
      source "${tugboat_env}"
    fi

    local qpad_dir="${HOME}/qpad"
    [[ -d "${qpad_dir}" ]] || mkdir -p "${qpad_dir}"
    _volumes+=("-v" "${HOME}/qpad:/qpad:rw")
  fi

  # Mount per user bashrc so that it can be sync'ed inside/outside Docker
  # NOTE(jiaming): Differences between '-v' and '--mount' behavior:
  # A) If you use '-v' or '--volume' to bind-mount a file or directory that
  #    does not yet exist on the Docker host, -v creates the endpoint for you.
  #    It is always created as a directory.
  # B) If you use '--mount' to bind-mount a file or directory that does not
  #    yet exist on the Docker host, Docker does not automatically create it
  #    for you, but generates an error.
  # Ref: https://docs.docker.com/storage/bind-mounts/#differences-between--v-and---mount-behavior
  local custom_bashrc="${HOME}/.bashrc_docker"
  if [[ -f "${custom_bashrc}" ]]; then
    chmod a+rw "${custom_bashrc}"
    _volumes+=("--mount" "type=bind,source=${custom_bashrc},target=${HOME_IN}/.personal_bashrc")
  fi

  # Load personal vimrc
  local custom_vimrc="${HOME}/.vimrc_docker"
  if [[ -f "${custom_vimrc}" ]]; then
    chmod a+rw "${custom_vimrc}"
    _volumes+=("--mount" "type=bind,source=${custom_vimrc},target=${HOME_IN}/.personal_vimrc")
  fi

  local custom_vim_dir
  if [[ "${UNIQUE_VIM}" == true ]]; then
    custom_vim_dir="${HOME}/.vim_docker"
  else
    custom_vim_dir="${HOME}/.vim"
  fi
  if [[ -d "${custom_vim_dir}" ]]; then
    _volumes+=("--mount" "type=bind,source=${custom_vim_dir},target=${HOME_IN}/.vim")
  fi

  local personal_gitconfig="${HOME}/.gitconfig"
  if [[ -f "${personal_gitconfig}" ]]; then
    chmod a+rw "${personal_gitconfig}"
    _volumes+=("--mount" "type=bind,source=${personal_gitconfig},target=${personal_gitconfig}")
  fi

  # Support custom binaries users may need.
  local custom_binary_dir="${HOME}/.bin_docker"
  if [[ -d "${custom_binary_dir}" ]]; then
    _volumes+=("--mount" "type=bind,source=${custom_binary_dir},target=${HOME_IN}/bin")
  fi

  for ent in "${CUSTOM_DIR_MOUNTS[@]}"; do
    _volumes+=("--mount" "${ent}")
  done

  # Support vscode-server
  local vscode_server_dir="${CHARMVE_ROOT_DIR}/.vscode-server"
  if [[ -d "${vscode_server_dir}" ]]; then
    _volumes+=("--mount" "type=bind,source=${vscode_server_dir},target=${HOME_IN}/.vscode-server")
  fi

  # Support git worktree
  local git_dir="${CHARMVE_ROOT_DIR}/.git"
  if [[ -f "${git_dir}" ]]; then
    local git_worktree_dir
    git_worktree_dir="$(head -n 1 "${git_dir}" | awk -F': ' '{print $2}')"
    git_worktree_dir="$(dirname "$(dirname "${git_worktree_dir}")")"
    _volumes+=("--mount" "type=bind,source=${git_worktree_dir},target=${git_worktree_dir}")
  fi

  # Note(jiaming):
  # 1) multiple workspaces share the same distdir
  # 2) disable writes to distdir from inside Docker to avoid deletion by mistake
  if [[ "${HOST_ARCH}" == "aarch64" || "${IS_CI}" == false ]]; then
    create_bazel_cache_dirs
    _volumes+=("-v" "${CHARMVE_ROOT_DIR}/.cache:/charmve_cache")
    if [[ -d "${BAZEL_DIST_DIR}" ]]; then
      _volumes+=("-v" "${BAZEL_DIST_DIR}:/charmve_cache/distdir:ro")
    fi
  else
    _volumes+=(
      "-v" "${HOME}/.cache:${HOME_IN}/.cache"
      "-v" "${HOME}/.distdir:${HOME_IN}/.distdir"
      "-v" "${HOME}/.repository_cache:${HOME_IN}/.repository_cache"
    )
  fi
}

function determine_docker_env_vars() {
  local -n _env_vars="$1"
  local map_dir="$2"
  local use_gpu_host="$3"
  local display="$4"
  _env_vars=(
    "-e" "DOCKER_USER=$USER"
    "-e" "DOCKER_USER_ID=$USER_ID"
    "-e" "DOCKER_GRP=$GRP"
    "-e" "DOCKER_GRP_ID=$GRP_ID"
    "-e" "DOCKER_IMG=${DOCKER_IMG}"
    "-e" "OFFICE=${OFFICE}"
    "-e" "CAR_ID=${CAR_ID}"
    "-e" "LAPTOP_IP=${LAPTOP_IP}"
    "-e" "MACHINE_HOSTNAME=${HOSTNAME_HOST}"
    "-e" "CHARMVE_ROOT_DIR_HOST=${CHARMVE_ROOT_DIR}"
    "-e" "USE_GPU_HOST=${use_gpu_host}"
    "-e" "OMP_NUM_THREADS=1"
    "-e" "NVIDIA_DRIVER_CAPABILITIES=video,compute,utility,display"
  )
  if [[ -n "${map_dir}" ]]; then
    _env_vars+=("-e" "HOST_MAP_DIR=${map_dir}")
  fi

  # vantage or other ui app on omc
  if [[ "${HOST_ARCH}" == "x86_64" ]]; then
    # NOTE(jiaming): QT_* was introduced by Da Fang in MR 4304
    _env_vars+=(
      "-e" "DISPLAY=${display}"
      "-e" "QT_X11_NO_MITSHM=1"
      "-e" "QT_GRAPHICSSYSTEM=native"
    )
  elif [[ "${ARM64_MODEL}" == "jetson-orin" ]]; then
    # NOTE(sweif): temporary fix for workaround SEGV due to Jetpack libnvjpeg
    #  internal symbols conflict to those in libjpeg.  most of modules do not
    #  use nvjpeg but they may use libjpeg from OpenCV.  default to preload
    #  libjpeg for the majority.  however modules do use nvjpeg have to unset
    #  this env, e.g. ImageForwardModule
    _env_vars+=("-e" "LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjpeg.so.8")
  fi

  if [[ "${ONBOARD_YES}" == true ]]; then
    local device
    if [[ "${HOST_ARCH}" == "x86_64" ]]; then
      device="omc"
    else
      device="xavier"
    fi
    _env_vars+=("-e" "DEVICE=${device}")
  fi
}

function usage() {
  local dev_docker_repo
  if [[ "${HOST_ARCH}" == "x86_64" ]]; then
    dev_docker_repo="${X86_64_DEV_REPO}"
  else
    dev_docker_repo="${AARCH64_DEV_REPO}"
  fi

  cat << EOF
Usage: $0 [options] ...
OPTIONS:
    -n, --name   <NAME>     Specify Docker container name as charmve_dev_<NAME>
    -o, --office <OFFICE>   Specify office location (bayarea-scott/beijing-tf/suzhou-tc/shenzhen-yx)
    -t, --tag    <TAG>      Specify Dev image named ${dev_docker_repo}:<TAG> to start
    -l, --local             Use local Docker image if possible. Usually used in
                              combination with the '-t' option.
    --distdir <DIR>         Specify Host-side Bazel distribution directory <DIR>
    -f, --force             Force removal of conflicting running container(s)
    --cpu                   Startup Dev Docker container in CPU-ONLY mode
    --unique_vim            Don't use \$HOME/.vim from host.
    --mount <MOUNT_SPEC>    Extra bind mount. <MOUNT_SPEC> should be of format
                              SRC_DIR[:TGT_DIR[:PERM]] . TGT_DIR is assumed to be equal to
                              SRC_DIR if not specified. PERM can be one of ["rw", "ro"] if specified.
                              Can be specified multiple times.
    -s, --shm  <SHM_SIZE>   Size of /dev/shm, e.g., 512m, 4g.
    -m, --mem  <MEM_SIZE>   Hard limit of memory access for Docker container, e.g., 512m, 2g, 16g.
    -y, --yes               Assume "yes" as answer to all prompts and run non-interactively
    --onboard               Start Dev Docker with onboard service
    --pid  <PID_NAMESPACE>  Set PID namespace e.g., host
    -h, --help              Show this message and exit
EOF

  if [[ "${HOST_ARCH}" == "x86_64" ]]; then
    cat << EOF
    --ml                    Use ML Docker image specified in docker/TAG
    --nomount_nfs           Whether to mount NFS Share
    --nouse_goofys          Mount local AWS S3 buckets without goofys.
                              Doesn't work if '--nomount_nfs' is present
EOF
  fi

  cat << EOF
E.g.
    $0 -n 2nd -f            Force start a container named charmve_dev_2nd
    $0 -l -t bazel-compl    Start a Dev container based on local image tagged 'bazel-compl'
EOF
}

function parse_cmdline_args() {
  local office=
  local name=
  local distdir=
  local image_tag=
  local shm_size=
  local mem_size=
  local mount_spec=
  local use_ml=false

  while [[ $# -gt 0 ]]; do
    local opt="$1"
    shift
    # Note(jiaming): sort options alphabetically
    case "${opt}" in
      --distdir)
        distdir="$1"
        shift
        optarg_check_for_opt "${opt}" "${distdir}"
        ;;
      -f | --force)
        ENFORCE=true
        ;;
      -l | --local)
        USE_LOCAL_IMAGE=true
        ;;
      --cpu)
        # Run Docker in CPU-ONLY mode
        USE_GPU_HOST=false
        ;;
      -s | --shm | --shm-size)
        shm_size="$1"
        shift
        optarg_check_for_opt "${opt}" "${shm_size}"
        ;;
      -m | --mem | --memory)
        mem_size="$1"
        shift
        optarg_check_for_opt "${opt}" "${mem_size}"
        ;;
      --mount)
        mount_spec="$1"
        shift
        if ! parse_mount_spec "${mount_spec}" CUSTOM_DIR_MOUNTS; then
          exit 1
        fi
        ;;
      -n | --name)
        name="$1"
        shift
        optarg_check_for_opt "${opt}" "${name}"
        ;;
      --nomount_nfs)
        MOUNT_NFS=false
        ;;
      -o | --office)
        office="$1"
        shift
        optarg_check_for_opt "${opt}" "${office}"
        ;;
      --ml)
        use_ml=true
        ;;
      -t | --tag)
        image_tag="$1"
        shift
        optarg_check_for_opt "${opt}" "${image_tag}"
        ;;
      --pid)
        PID_NAMESPACE="$1"
        shift
        optarg_check_for_opt "${opt}" "${PID_NAMESPACE}"
        ;;
      --nouse_goofys)
        USE_GOOFYS=false
        ;;
      --unique_vim)
        UNIQUE_VIM=true
        ;;
      -y | --yes)
        ASSUME_YES=true
        ;;
      --onboard)
        ONBOARD_YES=true
        ;;
      -h | --help)
        usage
        exit 1
        ;;
      *)
        error "Unknown option '${opt}'"
        if [[ "${opt}" == *"="* ]]; then
          error "Please separate option name and its value with SPACE rather than '='"
        fi
        usage
        exit 1
        ;;
    esac
  done

  [[ -z "${office}" ]] || OFFICE="${office}"
  [[ -z "${name}" ]] || DEV_CONTAINER="charmve_dev_${name}"
  [[ -z "${shm_size}" ]] || SHM_SIZE="${shm_size}"
  [[ -z "${mem_size}" ]] || MEM_SIZE="${mem_size}"
  if [[ -n "${SHM_SIZE}" || -n "${MEM_SIZE}" ]]; then
    MEM_LIMITED=true
  fi

  if [[ -n "${distdir}" ]]; then
    [[ -d "${distdir}" ]] || mkdir -p "${distdir}"
    BAZEL_DIST_DIR="${distdir}"
    ok "Use ${distdir} as Host-side Bazel Distribution Files Directory (--distdir)"
  fi

  if [[ -z "${image_tag}" && "${use_ml}" == true ]]; then
    image_tag="$(ml_docker_tag_default "${HOST_ARCH}")"
  fi

  DOCKER_IMG="$(dev_docker_image "${image_tag}" "${HOST_ARCH}")"
}

function interactive_office_setup() {
  local success=false

  while [[ "${success}" == false ]]; do
    local answer
    print_green "Please type a number to select your office interactively:"
    echo "  1: Beijing"
    echo "  2: Suzhou"
    echo "  3: Shenzhen"
    echo "  4: Bayarea"
    echo "  5: Guangzhou"
    read -p "Your choice is: " -r -n1 answer
    case "${answer}" in
      1)
        OFFICE="beijing-tf" # TODO(jiaming): Canonical office name
        success=true
        ;;
      2)
        OFFICE="suzhou-tc"
        success=true
        ;;
      3)
        OFFICE="shenzhen-yx"
        success=true
        ;;
      4)
        OFFICE="bayarea-scott"
        success=true
        ;;
      5)
        OFFICE="guangzhou"
        success=true
        ;;
      *) ;;
    esac
    echo
    if [[ "${success}" == true ]]; then
      ok "Your choice is [${answer}:${OFFICE}]"
    else
      warning "Choice of [${answer}] is invalid. Please try again."
    fi
  done
}

function main() {
  if [[ "$(id -u)" == 0 ]]; then
    error "Start Dev Docker as ROOT is prohibited."
    exit 0
  fi
  check_host_environment
  git_lfs_check

  parse_cmdline_args "$@"

  if [[ "${IS_CI}" == false ]]; then
    # make sure ${CHARMVE_ROOT_DIR}/.gitconfig is included by local repo
    git config --local include.path "../.gitconfig"
  fi

  local map_dir=
  determine_host_map_dir map_dir "${TOP_DIR}"

  local use_gpu_host
  local docker_run_cmd
  if [[ "${USE_GPU_HOST}" == false ]]; then
    use_gpu_host=0
    docker_run_cmd=("docker" "run")
  else
    determine_gpu_use_host use_gpu_host docker_run_cmd
  fi

  # NOTE(jiaming): iff office is not set and in interactive mode
  if [[ -z "${OFFICE}" && "${ASSUME_YES}" == true ]]; then
    error "OFFICE is empty. Exiting..."
    exit 1
  elif [[ -z "${OFFICE}" ]]; then
    interactive_office_setup
  fi

  local office
  office="$(canonical_office "${OFFICE}")"
  if ! validate_office "${office}"; then
    error "OFFICE=${OFFICE} is invalid. Exiting..."
    exit 1
  fi

  local country
  country="$(country_of_office "${office}")"
  if [[ "${country}" == "us" ]] && [[ "${USE_GOOFYS}" == true ]]; then
    warning "goofys will not be used in US"
    USE_GOOFYS=false
  fi

  readonly office

  # Resolve running container conflicts
  resolve_container_conflict "${DEV_CONTAINER}" "${ENFORCE}"

  local should_pull_image=true

  if [[ "${USE_LOCAL_IMAGE}" == true ]]; then
    if docker_image_exist "${DOCKER_IMG}"; then
      ok "Docker image ${DOCKER_IMG} found locally, will be used"
      should_pull_image=false
    else
      warning "Docker image ${DOCKER_IMG} not found locally although '-l/--local' option specified."
    fi
  fi

  if [[ "${should_pull_image}" == true ]]; then
    info "Pull Docker image ${DOCKER_IMG} from registry."
    if ! docker_pull_image "${DOCKER_IMG}" "${office}"; then
      error "Failed to pull Dev Docker image: ${DOCKER_IMG}"
      exit 1
    fi
  fi

  if [[ "${IS_CI}" == true ]]; then
    [[ -d "${HOME}/.cache" ]] || mkdir "${HOME}/.cache"
  fi

  local display
  display="$(determine_display)"

  # change docker.sock mode to push simulation's image.
  sudo chmod o+rw /var/run/docker.sock

  [[ -d /tmp/docker_files ]] || sudo mkdir -p /tmp/docker_files
  sudo chmod 777 /tmp/docker_files

  if [[ "${HOST_ARCH}" == "x86_64" ]]; then
    IncNetWorkBufSize

    if [[ "${MOUNT_NFS}" == true ]]; then
      if [[ ! -x "$(command -v goofys)" ]]; then
        warning "No binary named \"goofys\" found in your PATH(s) although USE_GOOFYS set."
        if [[ "${ASSUME_YES}" == true ]]; then
          print_green "Installing prebuilt goofys..."
          install_prebuilt_goofys
        else
          warning "Please refer to the following link on how to install it yourself:"
          print_green "    => https://github.com/kahing/goofys#installation"
          warning "Or use the \"-y\" option to have goofys installed automatically."
          exit 1
        fi
      fi
      # Note(jiaming): to avoid duplicate check there, which is unnecessary.
      bash -c ". ${CHARMVE_ROOT_DIR}/scripts/mount_nfs.sh && mount_nfs ${office} ${USE_GOOFYS}"
    fi
    if [[ "${MOUNT_DEBUGFS}" == true ]]; then
      mount_debugfs
    fi
    if [[ "${ENABLE_CORE_DUMP}" == true ]]; then
      EnableCoreDump
    fi
  fi

  if [[ "${HOST_ARCH}" == "aarch64" && "${IS_CI}" == false ]]; then
    bash "${TOP_DIR}/docker/scripts/aarch64/nv_setup_board.sh" "${ARM64_MODEL}"
  fi

  # init docker runtime environment
  USER_ID=$(id -u)
  GRP=$(id -g -n)
  GRP_ID=$(id -g)

  info "Starting docker container \"${DEV_CONTAINER}\" ..."
  echo "============ VARS BEGIN ==============="
  echo "OFFICE: ${OFFICE}"
  echo "USE_GPU_HOST: ${use_gpu_host}"
  echo "DOCKER_RUN_CMD: ${docker_run_cmd[*]}"
  echo "DOCKER_IMG: ${DOCKER_IMG}"
  echo "DISPLAY: ${display}"
  echo "SHM_SIZE: ${SHM_SIZE:-unlimited}"
  echo "MEM_SIZE: ${MEM_SIZE:-unlimited}"
  echo "USER: ${USER}(uid=${USER_ID},gid=${GRP_ID},home=${HOME})"
  echo "HOST_MAP_DIR: ${map_dir}"
  echo "MOUNT_NFS: ${MOUNT_NFS}"
  echo "USE_GOOFYS: ${USE_GOOFYS}"
  echo "============ VARS END ==============="

  local env_vars
  determine_docker_env_vars env_vars "${map_dir}" "${use_gpu_host}" "${display}"
  local volumes
  determine_local_volumes volumes "${map_dir}"

  local docker_run_cmd_with_opts=(
    "${docker_run_cmd[@]}"
    -itd
    --privileged
    --name="${DEV_CONTAINER}"
    --workdir=/charmve
  )

  local hostname_in="${DEV_CONTAINER//_/-}"
  if [[ "${MEM_LIMITED}" == false ]]; then
    docker_run_cmd_with_opts+=("--ipc=host")
  else
    if [[ -n "${MEM_SIZE}" ]]; then
      docker_run_cmd_with_opts+=("--memory=${MEM_SIZE}")
    fi
    if [[ -n "${SHM_SIZE}" ]]; then
      docker_run_cmd_with_opts+=("--shm-size=${SHM_SIZE}")
    fi
  fi

  if [[ -n "${PID_NAMESPACE}" ]]; then
    docker_run_cmd_with_opts+=("--pid=${PID_NAMESPACE}")
  fi

  docker_run_cmd_with_opts+=(
    --net=host
    --device /dev/snd
    --add-host "${hostname_in}:127.0.0.1"
    --add-host "${HOSTNAME_HOST}:127.0.0.1"
    --hostname="${hostname_in}"
    "${env_vars[@]}"
    "${volumes[@]}"
    "${DOCKER_IMG}"
  )

  if ! run "${docker_run_cmd_with_opts[@]}" /bin/bash > /dev/null; then
    error "Failed to start Docker container [${DEV_CONTAINER}] based on image [${DOCKER_IMG}]"
    exit 1
  fi

  info "Setup user account and grant permissions for ${USER}..."
  docker_start_user "${DEV_CONTAINER}" "${USER}" "${USER_ID}" "${GRP}" "${GRP_ID}"

  if [[ "${IS_CI}" == false ]]; then
    setup_user_bazelrc "${DEV_CONTAINER}" "${office}" "${use_gpu_host}"
  fi

  if [[ "${ONBOARD_YES}" == true ]]; then
    docker exec -it -u "${USER}" "${DEV_CONTAINER}" /charmve/scripts/start_service.sh
  fi

  ok "Successfully started Docker container [${DEV_CONTAINER}] based on image: [${DOCKER_IMG}]"
  ok "To login, please run: "
  if [[ "${DEV_CONTAINER}" == "${DEV_CONTAINER_DEFAULT}" ]]; then
    echo "${INDENT}${INDENT}scripts/goto_dev_docker.sh"
  else
    echo "${INDENT}${INDENT}scripts/goto_dev_docker.sh --name ${DEV_CONTAINER#charmve_dev_}"
  fi

  ok "Enjoy!"
}

main "$@"
