import numpy as np
import onnx
import os,sys
import cv2



def get_onnx_weights(OnnxModel, outFileDir):
    model = onnx.load(OnnxModel)
    weights = model.graph.initializer
    for weight in weights:
        print(weight.name, weight.dims, weight.data_type)
        data = weight.float_data
        data = np.array(data, np.float32)
        data = np.reshape(data, weight.dims)
        file_name = weight.name.replace('/', '.')
        np.save(outFileDir + file_name, data)
        
    
if __name__ == "__main__":

    if len(sys.argv) != 3:
        print ("Usage:", sys.argv[0], " OnnxModel  outputFile")
        sys.exit(-1)

    get_onnx_weights(sys.argv[1], sys.argv[2])