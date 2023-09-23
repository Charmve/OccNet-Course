#!/bin/bash
# set -u

export BOLD='\033[1m'
export RED='\033[0;31m'
export BLUE='\033[0;34m'
export GREEN='\033[32m'
export WHITE='\033[34m'
export YELLOW='\033[33m'
export NO_COLOR='\033[0m'

# Note(jiaming): Readonly.
# Not using the `readonly` keyword here as this file may be sourced multiple times
AVAILABLE_OFFICES=(
  bayarea
  beijing
  shenzhen
  suzhou
  guangzhou
)

function info() {
  (echo >&2 -e "[${WHITE}${BOLD}INFO${NO_COLOR}] $*")
}

function error() {
  (echo >&2 -e "${RED}[ERROR] $*${NO_COLOR}")
}

function warning() {
  (echo >&2 -e "${YELLOW}[WARNING] $*${NO_COLOR}")
}

function ok() {
  (echo >&2 -e "[${GREEN}${BOLD} OK ${NO_COLOR}] $*")
}

function print_green() {
  echo >&2 -e "${GREEN} $*${NO_COLOR}"
}

function print_yellow() {
  echo >&2 -e "${YELLOW} $*${NO_COLOR}"
}

function print_red() {
  echo >&2 -e "${RED} $*${NO_COLOR}"
}

function file_ext() {
  local filename
  filename="${1##*/}"
  local actual_ext
  actual_ext="${filename##*.}"
  if [[ "${actual_ext}" == "${filename}" ]]; then
    actual_ext=""
  fi
  echo "${actual_ext}"
}

# Get composite file extension for file with extensions such as
#  ".pb.txt", ".pb.bin", ".pb.h", ".pb.cc"
# Examples:
# abc => <nil>
# .abc => abc
# abc.txt => txt
# .abc.txt => abc.txt
# a.pb.txt => pb.txt
# a.b.pb.txt => pb.txt
# Ref: https://stackoverflow.com/questions/10586153/how-to-split-a-string-into-an-array-in-bash
function file_comp_ext() {
  local filename
  filename="$(basename "$1")"
  readarray -d . -t fds <<< "${filename}."
  unset "fds[-1]"
  local n="${#fds[@]}"
  case "${n}" in
    1)
      echo ""
      ;;
    2)
      echo "${fds[1]}"
      ;;
    *)
      echo "${fds[-2]}.${fds[-1]}"
      ;;
  esac
}

function array_contains() {
  local match="$1"
  shift
  for _ent in "$@"; do
    if [[ "${_ent}" == "${match}" ]]; then
      return 0
    fi
  done
  return 1
}

# Checks if the file has a c/c++/cuda extension.
function c_family_ext() {
  local actual_ext
  actual_ext="$(file_ext "$1")"
  local _c_family_exts=("h" "hh" "hxx" "hpp" "c" "cc" "cxx" "cpp" "cu" "cuh")
  if array_contains "${actual_ext}" "${_c_family_exts[@]}"; then
    return 0
  else
    return 1
  fi
}

function c_family_header() {
  local ext
  ext="$(file_ext "$1")"
  [[ "${ext}" == "h" ||
    "${ext}" == "hh" ||
    "${ext}" == "hpp" ||
    "${ext}" == "hxx" ||
    "${ext}" == "cuh" ]]

}

# Checks if the specified argument is a proto file.
function proto_ext() {
  [[ "$(file_ext "$1")" == "proto" || "$(file_comp_ext "$1")" == "pb.txt" ]]
}

function go_ext() {
  [[ "$(file_ext "$1")" == "go" ]]
}

function build_ext() {
  local fname
  fname="$(basename "$1")"
  [[ "${fname}" == "BUILD" ]]
}

function bazel_ext() {
  local fname
  fname="$(basename "$1")"
  if [[ "${fname}" == "BUILD" || "${fname}" == "WORKSPACE" ]]; then
    return 0
  fi

  local __ext
  __ext="$(file_ext "${fname}")"
  for ext in "bzl" "bazel" "BUILD"; do
    if [ "${ext}" == "${__ext}" ]; then
      return 0
    fi
  done
  return 1
}

function plain_py_ext() {
  local ext
  ext="$(file_ext "$1")"
  # shellcheck disable=SC2076
  [[ "${ext}" == "py" ]] && ! [[ "$1" =~ "_pb2.py" || "$1" =~ "_pb2_grpc.py" ]]
}

function bash_ext() {
  local ext
  ext="$(file_ext "$1")"
  [[ "${ext}" == "sh" || "${ext}" == "bash" ]]
}

function optarg_check_for_opt() {
  local opt="$1"
  local optarg="$2"
  if [[ -z "${optarg}" || "${optarg}" =~ ^-.* ]]; then
    error "Missing parameter for ${opt}. Exiting..."
    exit 1
  fi
}

function read_one_line_file() {
  local fpath="$1"
  local text
  read -r text < "${fpath}"
  echo "${text}"
}

function inside_docker() {
  [[ -f /.dockerenv ]]
}

function inside_git() {
  [[ "$(git rev-parse --is-inside-work-tree 2> /dev/null)" == true ]]
}

# Returns the current git branch.
function current_git_branch() {
  git rev-parse --abbrev-ref HEAD
}

###########################################################
# Get canonical office name
# Arguments:
#   office name as cmdline argument or env variable
# Outputs:
#   Write canonical name to stdout
###########################################################
function canonical_office() {
  local result=
  local office="$1"
  case "${office}" in
    beijing-tf | beijing | bj)
      result="beijing"
      ;;
    shenzhen-yx | shenzhen)
      result="shenzhen"
      ;;
    suzhou-tc | suzhou)
      result="suzhou"
      ;;
    bayarea-scott | bayarea | us)
      result="bayarea"
      ;;
    guangzhou)
      result="guangzhou"
      ;;
    *)
      result="${office}"
      ;;
  esac
  echo "${result}"
}

#############################################
# Get the country of the "canonical" office
# Arguments:
#   Canonical office name
# Outputs:
#   Write country name to stdout
#############################################
function country_of_office() {
  local country=
  case "$1" in
    beijing | shenzhen | suzhou | wuhan | guangzhou)
      country="cn"
      ;;
    bayarea)
      country="us"
      ;;
    *)
      country="unknown"
      ;;
  esac
  echo "${country}"
}

#############################################
# Check if the "canonical" office is valid
# Globals:
#   AVAILABLE_OFFICES
# Arguments:
#   A canonical office name
# Returns:
#   0 if office name is valid, non-zero otherwise
#############################################
function validate_office() {
  local office="$1"
  for cand in "${AVAILABLE_OFFICES[@]}"; do
    if [[ "${cand}" == "${office}" ]]; then
      return 0
    fi
  done
  return 1
}

function determine_arm64_model() {
  # https://qcraft.atlassian.net/wiki/spaces/~603857569/pages/1689845769/NV+machines
  local arch
  arch="$(uname -m)"
  if [[ "${arch}" == "x86_64" ]]; then
    return 1
  fi
  local prod
  if [[ -f /etc/nv_tegra_release ]]; then
    prod="jetson"
  else
    prod="drive"
  fi
  local soc
  soc="$(tr -d '\0' < /proc/device-tree/compatible | awk -F',' '{print $NF}')"
  case "${soc}" in
    tegra194)
      echo "${prod}-xavier"
      ;;
    tegra23*)
      echo "${prod}-orin"
      ;;
    *)
      echo "" # E.g. arm64 machines on CI
      ;;
  esac
}

# sudo not exists and default user is root on j5
if ! command -v sudo > /dev/null 2>&1; then
  function sudo() {
    local -a args=()
    local flag=0
    while [[ "$#" -gt 0 ]]; do
      if [ $flag -eq 1 ]; then
        args+=("$1")
        shift
      else
        case "$1" in # ignore args of sudo
          -*)
            shift
            ;;
          *)
            args+=("$1")
            flag=1
            shift
            ;;
        esac
      fi
    done
    "${args[@]}"
  }
fi

function q_sudo() {
  cmd="sudo ${*}"
  eval "${cmd}"
}
