import caffe_pb2
from google.protobuf import text_format
import onnx
import sys


def loadCaffeModel(net_path, model_path):
    # read prototxt
    net = caffe_pb2.NetParameter()
    # 把字符串读如message中
    text_format.Merge(open(net_path).read(), net)
    # print(net.layer)

    # read caffemodel
    model = caffe_pb2.NetParameter()
    f = open(model_path, 'rb')
    # 反序列化
    model.ParseFromString(f.read())
    f.close()
    # print(net.layer)
    print("1.caffe模型加载完成")
    print(model)
    # return net,model


if __name__ == "__main__":
    # loadCaffeModel("caffemodel/test/test.prototxt", "caffemodel/test/test.caffemodel")
    if len(sys.argv) != 2:
        print ("Usage:", sys.argv[0], " caffeModel")
        sys.exit(-1)

    # read caffemodel
    model = caffe_pb2.NetParameter()
    f = open(sys.argv[1], 'rb')
    # 反序列化
    model.ParseFromString(f.read())
    f.close()
    #print(model)
    for node in model.layer:
        print(node.name)
        if node.name == "Prelu1":
            print(node.blobs[0].shape)
            print(node.blobs[0].data)
    # print(model.layer[1].blobs[0].shape)
    # print(len(model.layers))
    # print(model.layers[0])
    # print(model.layers[1].param)
    # print(model.layers[1].convolution_param)
    # print(model.layers[1].blobs[0].shape)
    # print(model.layers[1].blobs[0].diff)
    # print(model.layers[1].blobs[0].num)
    # print(model.layers[1].blobs[0].channels)
    # print(model.layers[1].blobs[0].height)
    # print(model.layers[1].blobs[0].width)

