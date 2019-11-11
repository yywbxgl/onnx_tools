import caffe
import numpy as np
from PIL import Image
import sys, os
# import cv2

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__)) + '/'

PROTO_FILE = 'deploy.prototxt'

def run(test_dir):
    protofile = test_dir + PROTO_FILE
    net = caffe.Net(protofile, caffe.TEST)

    for node in net.blobs:
        print(node)

        if 'data' in node:
            x = np.load(test_dir + "data.npy")
            (c,h,w) = x.shape
            x = np.reshape(x, (1, c, h, w))
            net.blobs['data'].data[...] = x 

        if 'conv' in node:
            w = np.load(test_dir + node + "-weight.npy")
            print(w.shape)
            b = np.load(test_dir + node + "-bias.npy")
            print(b.shape)
            net.params[node][0].data[:] = w
            net.params[node][1].data[:] = b
        
        if 'ip' in node:
            w = np.load(test_dir + node + "-weight.npy")
            print(w.shape)
            b = np.load(test_dir + node + "-bias.npy")
            print(b.shape)
            net.params[node][0].data[:] = w
            net.params[node][1].data[:] = b
    
    # inference
    y1 = net.forward()

    # print output
    for output in y1:
        # print(net.blobs[output].data)
        ret = net.blobs[output].data

    return net, ret


if __name__ == '__main__':
    run(sys.argv[1])