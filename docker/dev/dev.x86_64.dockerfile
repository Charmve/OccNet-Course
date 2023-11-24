FROM ubuntu:18.04 
ARG DOCKER_BUILD_IP
ENV DOCKER_BUILD_IP "${DOCKER_BUILD_IP}"

COPY rcfiles /opt/rcfiles
COPY installers/installer_base.sh \
     installers/install_minimal_environment.sh \
     \
     installers/install_cmake.sh \
     installers/install_llvm_clang.sh \
     installers/install_bazel.sh \
     installers/install_buildifier.sh \
     installers/install_buildozer.sh \
     installers/install_shfmt.sh \
     installers/install_shellcheck.sh \
     installers/install_toolchain_deps.sh \
     \
     /tmp/installers/

RUN bash /tmp/installers/install_minimal_environment.sh

ENV PATH="/opt/llvm/bin:$PATH"
RUN bash /tmp/installers/install_toolchain_deps.sh

COPY installers/install_gflags.sh \
     installers/install_glog.sh \
     \
     installers/install_bzip2.sh \
     installers/install_lz4.sh \
     installers/install_zstd.sh \
     installers/install_libxml2.sh \
     installers/install_libarchive.sh \
     installers/install_boost.sh \
     \
     installers/install_double_conversion.sh \
     installers/install_eigen.sh \
     installers/install_yaml_cpp.sh \
     installers/install_protobuf.sh \
     installers/install_common_deps.sh \
     \
     installers/install_gperftools.sh \
     \
     installers/install_osqp.sh \
     installers/install_pnc_deps.sh \
     \
     /tmp/installers/

RUN bash /tmp/installers/install_common_deps.sh
RUN bash /tmp/installers/install_gperftools.sh
RUN bash /tmp/installers/install_pnc_deps.sh

COPY installers/install_ffmpeg.sh \
     installers/install_opencv.sh \
     /tmp/installers/
RUN bash /tmp/installers/install_ffmpeg.sh
RUN bash /tmp/installers/install_opencv.sh
