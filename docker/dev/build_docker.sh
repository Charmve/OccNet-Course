#!/bin/bash
set -euo pipefail

function determine_docker0_ip() {
  local ip_raw
  ip_raw="$(ip addr show docker0 2> /dev/null | awk '{
    if ($1=="inet") { print $2 }
    }')"
  if [[ -z "${ip_raw}" ]]; then
    echo "127.0.0.1"
  else
    echo "${ip_raw%%/*}"
  fi
}

function run() {
  echo >&2 "$*"
  "$@"
}

function main() {
  local timestamp
  timestamp="$(date +%Y%m%d_%H%M)"
  local cache_ip
  cache_ip="$(determine_docker0_ip)"
  run docker build \
    -t "charmve/maiwei-dev:linux-x86_64-${timestamp}" \
    --build-arg DOCKER_BUILD_IP="${cache_ip}" \
    -f dev.x86_64.dockerfile .
}

main "$@"
