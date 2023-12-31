#! /bin/bash
set -euo pipefail

CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
TOP_DIR="${CURR_DIR}/.."

# shellcheck disable=SC1090,SC1091
source "${TOP_DIR}/docker/scripts/docker_helper.sh"

DOCKER_USER="${USER:-charmve}"

DEV_CONTAINER="maiwei_dev_${DOCKER_USER}"
ENFORCE=false

function usage() {
	cat <<EOF
Usage: $0 [options] ...
OPTIONS:
    -f, --force             Force removal of conflicting running container(s)
    -h, --help              Show this message and exit
EOF
}

function parse_cmdline_args() {
	while [[ $# -gt 0 ]]; do
		local opt="$1"
		shift
		case "${opt}" in
		-f | --force)
			ENFORCE=true
			;;
		-h | --help)
			usage
			exit 0
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
}

function determine_dev_docker_image() {
	local tag_file
	tag_file="${CURR_DIR}/../docker/TAG"
	local result=
	if [[ ! -f "${tag_file}" ]]; then
		error "Docker TAG file not found: ${tag_file}"
		exit 0
	else
		local tag_id
		tag_id="$(uname -m)"
		tag="$(awk -F'=' -v tag_id="${tag_id}" '$1 == tag_id {print $2}' "${tag_file}")"
		if [[ -n "${tag}" ]]; then
			result="charmve/maiwei-dev-x86_64-20231116:$tag"
		else
			result="charmve/maiwei-dev-x86_64-20231116:latest"
		fi
	fi
	echo "${result}"
}

function determine_volume_mounts() {
	local -n _ret="$1"
	local top_dir
	top_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
	_ret+=(
		"${top_dir}:/maiwei"
		"${top_dir}/.cache:/maiwei_cache"
		"${HOME}:/hosthome:rw"
		"/media:/media"
		"/var/run/dbus/system_bus_socket:/var/run/dbus/system_bus_socket"
		"/etc/localtime:/etc/localtime"
		"/etc/machine-id:/etc/machine-id"
		"/sys/kernel/debug:/sys/kernel/debug"
	)
}

function generate_docker_run_command() {
	local -n _result="$1"
	local image="$2"

	readonly env_vars=(
		"USER=${DOCKER_USER}"
		"DOCKER_IMG=${image}"
	)

	declare -a vmounts
	determine_volume_mounts vmounts

	local hostname_dev
	hostname_dev="${DEV_CONTAINER//_/-}"

	_result+=(
		nvidia-docker
		run
		-itd
		--privileged
		"--ipc=host"
		"--name=""${DEV_CONTAINER}"
		"--workdir=/maiwei"
		"--hostname=""${hostname_dev}"
	)

	for var in "${env_vars[@]}"; do
		_result+=("-e" "${var}")
	done

	for vol in "${vmounts[@]}"; do
		_result+=("-v" "${vol}")
	done

	_result+=(
		--shm-size 16g
	)

	_result+=(
		"${image}"
		"/bin/bash"
	)
}

function docker_start_user() {
	local container="$1"
	local user="$2"

	local uid group gid
	uid="$(id -u)"
	group="$(id -g -n)"
	gid="$(id -g)"

	docker exec -u root "${container}" \
		bash -c "/maiwei/docker/scripts/docker_start_user.sh ${user} ${uid} ${group} ${gid}"
}

function main() {
	parse_cmdline_args "$@"
	local dev_image
	# dev_image="$(determine_dev_docker_image)"
	dev_image="charmve/maiwei-dev-x86_64-20231116:latest"

	if docker_image_exists "${dev_image}"; then
		info "Dev Docker image ${dev_image} found, will be used."
	elif docker pull ${dev_image}; then
		docker tag ${dev_image} ${dev_image}
	elif [ -f "$TOP_DIR"/docker/TAG ]; then
		cd "$TOP_DIR"/code/cuda-quant/CUDA-BEVFusion/bevfusion/docker
		docker build . -t "${DEV_CONTAINER}"
	else
		error "Dev Docker image '${dev_image}' not found locally."
		error "For MAIWEIers, please run the following commands:"
		error "  docker pull ${dev_image}"
		error "  docker tag ${dev_image} ${dev_image}"
		error "And then RETRY"
		return 1
	fi

	ensure_one_dev_docker "${DEV_CONTAINER}" "${ENFORCE}"

	declare -a docker_cmd
	generate_docker_run_command docker_cmd "${dev_image}"
	if run "${docker_cmd[@]}"; then
		ok "Dev Docker for started successfully. [${dev_image}]"
	else
		error "Failed to start Dev Docker. Exiting ..."
		return 1
	fi

	info "Setup user account and grant permissions for ${DOCKER_USER} ..."
	docker_start_user "${DEV_CONTAINER}" "${DOCKER_USER}"
	ok "To login, run: "
	ok "  scripts/goto_dev_docker.sh"
}

main "$@"
