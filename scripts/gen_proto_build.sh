#! /usr/bin/env bash
# Usage:
#   scripts/gen_proto_build.sh --gen-build
#
set -euo pipefail

TOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "${TOP_DIR}/scripts/maiwei_base.sh"

IS_CI="${IS_CI:-false}"

function print_usage() {
  echo -e "${RED}Usage${NO_COLOR}:
    ${BOLD}$0${NO_COLOR} --gen-build ${BLUE}# Gen build file${NO_COLOR}"
  echo -e "\n${RED}Options${NO_COLOR}:
  ${BLUE}--gen-build          ${NO_COLOR}Gen build file.
  ${BLUE}-h|--help            ${NO_COLOR}Show this message and exit."
}

function run_genbuild() {
  pushd ${TOP_DIR} > /dev/null
  for arg in "$@"; do
    if [[ -f "${arg}" ]]; then
      if build_ext "${arg}"; then
        "${TOP_DIR}/scripts/proto_build_generator.py" "${arg}"
      fi
      if proto_ext "${arg}"; then
        "${TOP_DIR}/scripts/proto_add_go_pkg.py" "${arg}"
        build_file_path=$(dirname ${arg})"/BUILD"
        "${TOP_DIR}/scripts/proto_build_generator.py" "${build_file_path}"
      fi
    fi
  done
  popd > /dev/null
}

function main() {
  if [ "$#" -eq 0 ]; then
    print_usage
    exit 1
  fi

  while [[ $# -gt 0 ]]; do
    local opt="$1"
    case "${opt}" in
      --gen-build)
        GEN_BUILD=1
        shift
        ;;
      -h | --help)
        print_usage
        exit 1
        ;;
      *)
        if [[ "${opt}" == -* ]]; then
          print_usage
          exit 1
        fi
        ;;
    esac
  done
  # Note:
  # 1) Exclude deleted files, Ref:
  #    https://stackoverflow.com/questions/6894322/how-to-make-git-diff-and-git-log-ignore-new-and-deleted-files
  # 2) git-clang-format Ref:
  #    https://github.com/llvm/llvm-project/blob/release/12.x/clang/tools/clang-format/git-clang-format
  local what_to_diff
  what_to_diff="${CI_MERGE_REQUEST_DIFF_BASE_SHA:-HEAD~1}"

  if [[ "${GEN_BUILD}" -eq 1 ]]; then
    while read -r one_change; do
      run_genbuild "${one_change}"
    done < <(git diff --ignore-submodules --diff-filter=d --name-only "${what_to_diff}")
  fi
}

main "$@"
