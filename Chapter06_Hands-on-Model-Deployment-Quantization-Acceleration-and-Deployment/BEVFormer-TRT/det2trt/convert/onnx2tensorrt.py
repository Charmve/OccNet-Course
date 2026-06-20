import os
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def build_engine(
    onnx_file_path,
    engine_file_path,
    dynamic_input=None,
    int8=False,
    fp16=False,
    max_workspace_size=1,
    calibrator=None,
):
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
        EXPLICIT_BATCH
    ) as network, builder.create_builder_config() as config, trt.OnnxParser(
        network, TRT_LOGGER
    ) as parser, trt.Runtime(
        TRT_LOGGER
    ) as runtime:
        # max_workspace_size GB
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, (1 << 32) * max_workspace_size
        )
        # Parse model file
        if not os.path.exists(onnx_file_path):
            print("ONNX file {} not found.".format(onnx_file_path))
            exit(0)
        print("Loading ONNX file from path {}...".format(onnx_file_path))
        with open(onnx_file_path, "rb") as model:
            print("Beginning ONNX file parsing")
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        if dynamic_input is not None:
            profile = builder.create_optimization_profile()
            profile.set_shape(network.get_input(0).name, *dynamic_input)
            config.add_optimization_profile(profile)

        if int8:
            config.set_flag(trt.BuilderFlag.INT8)
            if calibrator is not None:
                config.int8_calibrator = calibrator

        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        print("Completed parsing of ONNX file")
        print(
            "Building an engine from file {}; this may take a while...".format(
                onnx_file_path
            )
        )
        plan = builder.build_serialized_network(network, config)
        engine = runtime.deserialize_cuda_engine(plan)
        print("Completed creating Engine")
        with open(engine_file_path, "wb") as f:
            f.write(plan)
        print(f"TensorRT engine has been saved in {engine_file_path}")
        return engine
