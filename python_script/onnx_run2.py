import numpy as np
import onnx
import onnxruntime
import onnxruntime.backend as backend
#import caffe2.python.onnx.backend as backend
import os,sys


if len(sys.argv) != 2:
    print ("Usage:", sys.argv[0], " OnnxModel")
    sys.exit(-1)


model = onnx.load(sys.argv[1])
session = backend.prepare(model)



# get input data
x = np.random.randn(1, 3, 224, 224).astype(np.float32) 
# print(x)

# Run the model on the backend
output = session.run(x)
print(output)
