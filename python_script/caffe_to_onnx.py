import coremltools
import onnxmltools
import sys


if len(sys.argv) != 3:
    print ("Usage:", sys.argv[0], "test.prototxt  test.caffemodel")
    sys.exit(-1)

# Update your input name and path for your caffe model
proto_file = sys.argv[1]
caffe_model = sys.argv[2]
# Update the output name and path for intermediate coreml model, or leave as is
# output_coreml_model = 'test.mlmodel'
# Change this path to the output name and path for the onnx model
output_onnx_model = 'caffe-coremol-onnx.onnx'


# Convert Caffe model to CoreML 
coreml_model = coremltools.converters.caffe.convert((caffe_model, proto_file))

# Save CoreML model
#coreml_model.save(output_coreml_model)

# Load a Core ML model
#coreml_model = coremltools.utils.load_spec(output_coreml_model)

# Convert the Core ML model into ONNX
onnx_model = onnxmltools.convert_coreml(coreml_model)

# Save as protobuf
onnxmltools.utils.save_model(onnx_model, output_onnx_model)
