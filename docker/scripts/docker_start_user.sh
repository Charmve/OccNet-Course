#! /bin/bash
# Warning: Don't run this script directly!!!
set -euo pipefail

function create_user_account() {
  local user_name="$1"
  local uid="$2"
  local group_name="$3"
  local gid="$4"
  addgroup --gid "${gid}" "${group_name}"

  adduser --disabled-password --force-badname --gecos '' \
    "${user_name}" --uid "${uid}" --gid "${gid}"

  echo "export PS1=\"\\e[0;32m[\\u@\\h:\\w]\\e[m\\$ \"" \
    > "/home/${user_name}/.bash_aliases"

  chown -R "${user_name}:${user_name}" "/home/${user_name}"
  usermod -aG sudo "${user_name}"
}

function setup_user_account_if_non_exist() {
  local user_name="$1"
  local uid="$2"
  local group_name="$3"
  local gid="$4"
  if grep -q "^${user_name}:" /etc/passwd; then
    echo "User ${user_name} already exist. Skip setting user account."
    return
  fi
  create_user_account "$@"
}

function sudoer_without_password() {
  local user_name="$1"
  echo -e "\n${user_name}  ALL=(ALL:ALL)  NOPASSWD:ALL" >> /etc/sudoers
}

function docker_start_user() {
  local user_name="$1"
  local uid="$2"
  local group_name="$3"
  local gid="$4"

  if [ "${uid}" != "${gid}" ]; then
    echo "Warning: uid(${uid}) != gid(${gid}) found."
  fi
  if [ "${user_name}" != "${group_name}" ]; then
    echo "Warning: user_name(${user_name}) != group_name(${group_name}) found."
  fi
  setup_user_account_if_non_exist "$@"

  sudoer_without_password "${user_name}"
}

docker_start_user "$@"
