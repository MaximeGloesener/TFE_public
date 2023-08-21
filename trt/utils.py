import sys
sys.path.append('trt')
import common
import tensorrt as trt
import pycuda.driver as cuda
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit


# The Onnx path is used for Onnx models
def build_engine_onnx(TRT_LOGGER, model_file):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        trt_config = builder.create_builder_config()
        trt_config.max_workspace_size = common.GiB(4)
        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(model_file, 'rb') as model:
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None
        return builder.build_engine(network, trt_config)

def build_engine_onnx_int8(TRT_LOGGER, model_file, calib):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.explicit_batch()) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        trt_config = builder.create_builder_config()
        trt_config.max_workspace_size = common.GiB(4)
        trt_config.set_flag(trt.BuilderFlag.INT8)
        trt_config.set_flag(trt.BuilderFlag.FP16)
        trt_config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        trt_config.int8_calibrator = calib

        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(model_file, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return ValueError('Failed to parse the ONNX file.')

        return builder.build_engine(network, trt_config)
    
def build_engine_onnx_qat(TRT_LOGGER, model_file):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        trt_config = builder.create_builder_config()
        trt_config.max_workspace_size = common.GiB(4)
        trt_config.set_flag(trt.BuilderFlag.INT8)
        trt_config.set_flag(trt.BuilderFlag.FP16)
        trt_config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(model_file, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return ValueError('Failed to parse the ONNX file.')
        return builder.build_engine(network, trt_config)

def build_engine_onnx_fp16(TRT_LOGGER, model_file, calib):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        trt_config = builder.create_builder_config()
        trt_config.max_workspace_size = common.GiB(4)
        trt_config.set_flag(trt.BuilderFlag.FP16)
        trt_config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        trt_config.int8_calibrator = calib

        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(model_file, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return ValueError('Failed to parse the ONNX file.')
        return builder.build_engine(network, trt_config)

def load_engine(TRT_LOGGER, model_file):
    with open(model_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())        
    return engine

