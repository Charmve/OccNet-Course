#! /bin/bash
set -euo pipefail
CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

bash "${CURR_DIR}/install_osqp.sh"
