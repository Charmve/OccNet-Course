#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Replacement for setup_user_bazelrc.sh to support X-Docker builds
# It attempts to create Bazel directories under workspace, with
# user.bazelrc modified accordingly.
# Needed by `scripts/start_dev_docker.sh` for non-CI users

import argparse
import logging
import multiprocessing
import os
import platform
import re
import shutil
from pathlib import Path

# NOTE(Charmve):
# We only parse lines of the following formats:
# 1) stage --key=value
# 2) stage --key value
RC_LINE_PATT = re.compile(r"(\w+(:\w+)?)\s+--(\w+)[ =](.*)")

X_BUILD_INDICATOR = "# X-BUILD: PLEASE-LEAVE-THIS-LINE-INTACT"

# ===== Helper Functions Start ===== #
def architecture():
    return platform.machine()


def nproc():
    # Ref command: lscpu -a --extended
    return multiprocessing.cpu_count()


def is_x_build():
    return os.getenv("QCRAFT_X_BUILD", "0") == "1"


def determine_remote_cache_address(office):
    address = None
    if office == "beijing":
        address = "http://bazel-cache-beijing.qcraftai.com:19099"
    elif office == "suzhou":
        address = "http://bazel-cache-suzhou.qcraftai.com:19099"
    elif office == "shenzhen":
        address = "http://bazel-cache-shenzhen.qcraftai.com:19099"
    elif office == "bayarea":
        address = "http://bazel-cache-us-local.qcraftai.com:19099"
    else:
        logging.error(f"Unknown office: {office}")
    return address


def determine_gpu_arch_line(arch, custom_val=None):
    if custom_val:
        gpu_archs = custom_val
    else:
        if arch == "x86_64":
            gpu_archs = "sm_70,sm_75,sm_80,sm_86"
        else:
            gpu_archs = "sm_72"
    return f"build:gpu --cuda_gpu_arch={gpu_archs}"


def determine_concurrency_level():
    num_cpus = nproc()
    return max(num_cpus - 3, 2)


# ===== Helper Functions End ===== #

# This function was not used temporarily
def parse_user_bazelrc(fentry):
    x_build = False
    kvs = dict()
    with open(fentry, encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line.startswith("#"):
                if line.startswith(X_BUILD_INDICATOR):
                    x_build = True
            elif line:
                match = RC_LINE_PATT.match(line)
                if match:
                    kvs[match.group(3)] = match.group(4)
    return (x_build, kvs)


def write_user_bazelrc(
    fentry,
    x_build=False,
    use_gpu=False,
    office="beijing",
):
    def add_seperator_line(lines):
        lines.append("")

    lines = [X_BUILD_INDICATOR] if x_build else []

    lines.extend(
        [
            "startup --output_user_root=/qcraft_cache/bazel",
            # Commands like `bazel sync` will not respect repository_cache settings
            # if "--repository_cache" was set for `build` rather than `common`.
            # Ref: https://docs.bazel.build/versions/main/guide.html#option-defaults
            "common --repository_cache=/qcraft_cache/repository_cache",
            "common --distdir=/qcraft_cache/distdir",
            # extend to prevent cc rules no-cache & no-remote nouse issue
            # Ref: https://github.com/bazelbuild/bazel/issues/8830
            # Ref: https://github.com/bazelbuild/bazel/issues/6038
            "common --experimental_allow_tags_propagation",
        ]
    )
    add_seperator_line(lines)

    lines.extend(
        [
            "build --action_env=DOCKER_REPO_CACHE=/qcraft_cache/docker",
            "build --action_env=GOMODCACHE=/qcraft_cache/go/module",
            "build --action_env=GOCACHE=/qcraft_cache/go/cache",
            "build --action_env=GO_REPOSITORY_USE_HOST_CACHE=1",
        ]
    )

    add_seperator_line(lines)
    arch = architecture()
    if use_gpu:
        gpu_arch_line = determine_gpu_arch_line(arch)
        lines.append(gpu_arch_line)
    else:
        lines.append("build --config=cpu")

    remote_cache_addr = determine_remote_cache_address(office)
    if remote_cache_addr:
        add_seperator_line(lines)
        lines.append(f"build --remote_cache={remote_cache_addr}")

    add_seperator_line(lines)

    num_jobs = determine_concurrency_level()
    lines.extend(
        [
            "build --local_cpu_resources=HOST_CPUS-3",
            "build --local_ram_resources=HOST_RAM*.7",
            f"build --jobs={num_jobs}",
        ]
    )
    add_seperator_line(lines)

    # Extra config
    #   startup --host_jvm_args=-Xmx8g
    #   startup --host_jvm_args=-Xms1g
    #   build --remote_executor=grpcs://buildfarm-all.qcraftai.com:443
    #   build --remote_upload_local_results=false
    with open(fentry, "w", encoding="utf-8") as fout:
        for line in lines:
            fout.write(f"{line}\n")


def main(workspace, use_gpu=False, office="beijing"):
    bazelrc_path = workspace.joinpath("user.bazelrc")
    curr_is_x = is_x_build()

    if not bazelrc_path.exists():
        write_user_bazelrc(
            bazelrc_path, x_build=curr_is_x, use_gpu=use_gpu, office=office
        )
        return

    logging.info("Existing user.bazelrc moved and renamed as user.bazelrc.bak ...")
    bazelrc_bak = workspace.joinpath("user.bazelrc.bak")
    shutil.move(bazelrc_path, bazelrc_bak)
    write_user_bazelrc(bazelrc_path, x_build=curr_is_x, use_gpu=use_gpu, office=office)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Performs gpu build",
    )
    parser.add_argument("--nouse_gpu", dest="use_gpu", action="store_false")
    parser.set_defaults(use_gpu=True)

    parser.add_argument(
        "-o",
        "--office",
        default="beijing",
        type=str,
        metavar="OFFICE",
    )

    args = parser.parse_args()

    workspace = Path(__file__).parent.joinpath("../..").resolve()
    main(workspace, use_gpu=args.use_gpu, office=args.office)
