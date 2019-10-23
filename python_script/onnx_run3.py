import numpy as np
import onnx
import onnxruntime.backend as backend

# import caffe2.python.onnx.backend as backend
import os,sys

if len(sys.argv) != 2:
    print ("Usage:", sys.argv[0], " OnnxModel")
    sys.exit(-1)

model = onnx.load(sys.argv[1])
session = backend.prepare(model,  strict=False)


# get input data
# x = np.random.randn(1, 3, 224, 224).astype(np.float32)
x=np.load("test_1_3_224_224.npy").astype(np.float32)
print(x.shape)
print(x)

# Run the model on the backend

output = session.run(x)
# output = np.array(output)
# print(output.shape)
print(output)
