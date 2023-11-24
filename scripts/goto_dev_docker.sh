#! /bin/bash
set -u

TOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
# shellcheck disable=SC1091,SC1090
source "${TOP_DIR}/docker/scripts/docker_base.sh"

DEFAULT_CONTAINER="maiwei_dev_${USER}"
readonly DEFAULT_CONTAINER

DEV_CONTAINER="${DEFAULT_CONTAINER}"
USER="${USER:-"charmve"}"

function usage() {
	cat <<EOF
Usage: $0 [options] ...
OPTIONS:
    -n, --name   <NAME>     Specify Docker container name as maiwei_dev_NAME
    -h, --help              Show this message and exit
EOF
}

function parse_cmdline_args() {
	local name=
	while [[ $# -gt 0 ]]; do
		local opt="$1"
		shift
		case "${opt}" in
		-n | --name)
			name="$1"
			shift
			optarg_check_for_opt "${opt}" "${name}"
			;;
		-h | --help)
			usage
			exit 1
			;;
		esac
	done
	[[ -n "${name}" ]] && DEV_CONTAINER="maiwei_dev_${name}"
}

function _compare_timestamp_of_docker_image() {
	local current="${1##*-}"
	local latest="${2##*-}"
	if [[ "${#current}" != "${#latest}" || "${current}" == "${latest}" ]]; then
		echo 0
	elif [[ "${current}" < "${latest}" ]]; then
		echo -1
	else
		echo 1
	fi
}

function check_image_updates() {
	local latest
	latest="$(dev_docker_image)"
	local current
	current="$(docker exec "${DEV_CONTAINER}" bash -c 'echo "${DOCKER_IMG}"')"
	if [[ "${latest}" != "${current}" ]]; then
		local result
		result="$(_compare_timestamp_of_docker_image "${current}" "${latest}")"
		if (("${result}" < 0)); then
			warning "===================================================================="
			warning "The tag of your currently running Dev Docker image is:"
			warning "  ${current##*:}"
			warning "However, a newer image tagged ${latest##*:} is available."
			warning "Please consider restarting Dev Docker at your earliest convenience."
			warning "===================================================================="
		elif (("${result}" > 0)); then
			ok "It seems that you are running a newer Docker image than the latest one."
			ok "    Currently Running:        ${current##*:}"
			ok "    Latest (see docker/TAG):  ${latest##*:}"
		else
			warning "Unable to check if newer Docker image is available."
			warning "    Currently Running:       ${current##*:}"
			warning "    Latest(see docker/TAG):  ${latest##*:}"
		fi
	else
		ok "==================================================="
		ok "Your Dev Docker image is up-to-date."
		ok "==================================================="
	fi
}

function is_container_running() {
	local container="$1"
	local status
	status="$(docker inspect -f '{{.State.Status}}' "${container}" 2>/dev/null)"
	[[ "${status}" == "running" ]]
}

function main() {
	parse_cmdline_args "$@"

	local container="${DEV_CONTAINER}"
	if ! is_container_running "${container}"; then
		error "No running Docker container named ${container}"
		exit 1
	fi

	check_image_updates

	if [ ! -d ~/workspace/OccNet-Course ]; then
		error "OccNet-Course should place at ~/workspace as default."
		exit 0
	fi

	xhost +local:"${USER}" &>/dev/null

	nvidia-docker run \
		-it \
		-v "/media:/media" \
		-v "${HOME}:/hosthome:rw" \
		-v "$HOME/workspace/OccNet-Course:/maiwei" \
		--hostname=maiwei-dev \
		--workdir=/maiwei \
		--shm-size 16g \
		"$X86_64_DEV_REPO" /bin/bash
	# charmve/maiwei-dev-x86_64-20231116 /bin/bash

	xhost -local:"${USER}" 1>/dev/null 2>&1
}

main "$@"
