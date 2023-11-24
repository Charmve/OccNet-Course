#!/bin/bash
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
