#! /usr/bin/env bash
# Usage:
#   buildifier.sh <path/to/BUILD/files/or/dirs>
set -euo pipefail

TOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "${TOP_DIR}/scripts/maiwei_base.sh"

function check_existence_of_buildtools() {
  if [[ -z "$(which buildifier)" ]]; then
    error "Command 'buildifier' not found in PATH. Please install it manually from"
    error "    https://github.com/bazelbuild/buildtools"
    return 1
  fi
  if [[ -z "$(which buildozer)" ]]; then
    error "Command 'buildozer' not found in PATH. Please install it manually from"
    error "    https://github.com/bazelbuild/buildtools"
    return 1
  fi
}

function replace_rules_py_bzl() {
  local fn="$1"
  local abspath=
  if [[ "${fn}" == /* ]]; then
    abspath="${fn}"
  else
    abspath="${PWD}/${fn}"
  fi

  buildozer 'replace_load //bazel:rules_py.bzl py_library' "${fn}" 2> /dev/null || true
  buildozer 'replace_load //bazel:rules_py.bzl py_binary' "${fn}" 2> /dev/null || true
  buildozer 'replace_load //bazel:rules_py.bzl py_test' "${fn}" 2> /dev/null || true
}

function replace_rules_cc_bzl() {
  local fn="$1"
  local abspath=
  if [[ "${fn}" == /* ]]; then
    abspath="${fn}"
  else
    abspath="${PWD}/${fn}"
  fi
  if [[ "${abspath}" == "${TOP_DIR}"/cyber* || "${abspath}" == "${TOP_DIR}"/third_party/s2geometry* ]]; then
    warning "Replacement of cc_* rules skipped: ${abspath}"
    return
  fi
  buildozer 'replace_load //bazel:rules_cc.bzl cc_library' "${fn}" 2> /dev/null || true
  buildozer 'replace_load //bazel:rules_cc.bzl cc_binary' "${fn}" 2> /dev/null || true
  buildozer 'replace_load //bazel:rules_cc.bzl cc_test' "${fn}" 2> /dev/null || true
}

function run_build_tools() {
  local fn="$1"
  if [[ "${fn}" == */BUILD || "${fn}" == "BUILD" || "${fn}" == */BUILD.bazel ]]; then
    replace_rules_cc_bzl "${fn}"
    replace_rules_py_bzl "${fn}"
  fi

  buildifier -lint=fix --warnings=all "${fn}"
}

function find_bazel_files_beneath_dir() {
  local dir="$1"
  find "${dir}" -type f \( -name "BUILD" \
    -or -name "WORKSPACE" \
    -or -name "*.BUILD" \
    -or -name "*.bzl" \
    -or -name "*.bazel" \)
}

function run_buildifier() {
  for target in "$@"; do
    if [[ -f "${target}" ]]; then
      if bazel_ext "${target}"; then
        run_build_tools "${target}"
        ok "Done formatting Bazel file ${target}"
      fi
    elif [[ -d "${target}" ]]; then
      while read -r fn; do
        run_build_tools "${fn}"
      done < <(find_bazel_files_beneath_dir "${target}")
      ok "Done formatting Bazel files beneath '${target}'"
    else
      error "No such file or directory: ${target}"
      exit 1
    fi
  done
}

function main() {
  if [ $# -eq 0 ]; then
    error "Usage: $0 <path/to/BUILD/files/or/dirs>"
    exit 1
  fi
  check_existence_of_buildtools
  run_buildifier "$@"
}

main "$@"
