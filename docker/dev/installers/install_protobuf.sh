#! /bin/bash
set -euo pipefail

CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1090,SC1091
. "${CURR_DIR}/installer_base.sh"

VERSION="3.19.4"

PKG_NAME="protobuf-${VERSION}.tar.gz"
CHECKSUM="d0f5f605d0d656007ce6c8b5a82df3037e1d8fe8b121ed42e536f569dec16113"
DOWNLOAD_LINK="https://github.com/protocolbuffers/protobuf/archive/v${VERSION}.tar.gz"

download_if_not_cached "$PKG_NAME" "$CHECKSUM" "$DOWNLOAD_LINK"

tar xzf ${PKG_NAME}

pushd protobuf-${VERSION}
mkdir cmake/build && cd cmake/build

cmake .. \
  -DBUILD_SHARED_LIBS=ON \
  -Dprotobuf_BUILD_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release

# ./configure --prefix=/usr
make -j"$(nproc)"
make install

# cd ../../python
# python setup.py install --cpp_implementation
popd > /dev/null

ldconfig

# echo -e "\nexport PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp"
ok "Successfully installed protobuf, VERSION=${VERSION}"

# Clean up.
rm -fr ${PKG_NAME} protobuf-${VERSION}
