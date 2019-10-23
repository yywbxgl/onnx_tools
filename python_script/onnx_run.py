import numpy as np
from PIL import Image
import pylab as plt
import onnx
import os
import glob
#import onnxruntime.backend as backend
import caffe2.python.onnx.backend as backend
from PIL import Image
import sys
import cv2
import scipy

from onnx import numpy_helper


if len(sys.argv) != 2:
    print ("Usage:", sys.argv[0], " ONNX Model")
    sys.exit(-1)

model = onnx.load(sys.argv[1])
# model = onnx.load("my_test.onnx")
#model = onnx.load("conv_img.onnx")
tf_rep = backend.prepare(model)

# # input data
# # x = np.zeros([1,3,5,5] , dtype=np.float32)
# x1 = np.arange(25, dtype=np.float32)
# x2 = np.arange(25, dtype=np.float32)
# x3 = np.arange(25, dtype=np.float32)
# x = np.concatenate((x1, x2, x3))
# x = np.reshape(x, (1,3,5,5))
# print(x)


# x = np.arange(25, dtype=np.float32)
# x = np.reshape(x, (1,1,5,5))
# print(x)


# 将numpy 转为图片保存
# tmp = []
# channel = 3
# for i in range(25):
#     for j in range(channel):
#         tmp.append(i)
# print(tmp)
# x = np.array(tmp)
# x = np.reshape(x, (5,5,3))
# print(x)
# cv2.imwrite("conv.jpg", x)
# print("save img conv.jpg")

# # 读取图片数据
img = Image.open("cat.jpg")
# img = cv2.imread("cat.jpg")
# cv2默认为 BGR顺序，而其他软件一般使用RGB，所以需要转换
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2默认为bgr顺序
x = np.array(img, dtype=np.float32)
x = np.reshape(x, (1,3,224,224))
# print(x.shape)
print(x)


# x = np.random.randn(3, 4, 5).astype(np.float32)
# print(x.shape)
# y = np.expand_dims(x,axis=1)
# print(y.shape)

# for i in x[:,0,:]:
#     print (i)

# [rows, cols, channel] = x.shape
# for i in range(rows):
#     for j in range(cols):
#         print(x[i,j])

# for i in x:
#     print(i)
# print(x)

# # Run the model on the backend
output = tf_rep.run(x)
print(output)
print(np.max(output))
# output = backend.run_model(model, img)
