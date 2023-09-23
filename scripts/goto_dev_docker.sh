#! /usr/bin/env bash
###############################################################################
# Copyright 2017 The Apollo Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################
TOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
# shellcheck disable=SC1091
source "${TOP_DIR}/scripts/docker_base.sh"

DEFAULT_CONTAINER="mway_dev_${USER}"
readonly DEFAULT_CONTAINER

DEV_CONTAINER="${DEFAULT_CONTAINER}"

function usage() {
  cat << EOF
Usage: $0 [options] ...
OPTIONS:
    -n, --name   <NAME>     Specify Docker container name as mway_dev_NAME
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
  [[ -n "${name}" ]] && DEV_CONTAINER="mway_dev_${name}"
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
    local result="$(_compare_timestamp_of_docker_image "${current}" "${latest}")"
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
  status="$(docker inspect -f '{{.State.Status}}' "${container}" 2> /dev/null)"
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
  xhost +local:"${USER}" &> /dev/null

  docker exec \
    -e HIST_FILE="${HOME}/.bash_history" \
    -e USER="${USER}" \
    -u "${USER}" \
    -it "${container}" \
    /bin/bash
  xhost -local:"${USER}" 1> /dev/null 2>&1
}

main "$@"
