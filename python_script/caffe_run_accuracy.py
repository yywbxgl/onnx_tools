import numpy as np
import caffe
# import caffe2.python.onnx.backend as backend
import os,sys
import cv2

# caffe.set_mode_gpu()

# 图片数据读取为numpy
def get_numpy_from_img(file):
    # img = Image.open(file)
    # x = np.array(img, dtype='float32')
    # x = x.reshape(net.blobs['data'].data.shape)

    img = cv2.imread(file)
    img = cv2.resize(img, (227,227))
    # cv2默认为 BGR顺序，而其他软件一般使用RGB，所以需要转换
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2默认为bgr顺序
    x = np.array(img, dtype=np.float32)

    # print(x)
    # x = np.reshape(x, (1,5,5,3))
    # 矩阵转置换，img读取后的格式为W*H*C 转为model输入格式 C*W*H
    x = np.transpose(x,(2,0,1))
    return x


def caffe_ference(net, test_dir):

    f_val_label = open("my_label.txt", "r")
    val_labels = f_val_label.readlines()
    for val_label in val_labels:
        val_label = val_label.strip('\n')

    results = {}
    #得到文件夹下的所有文件名称
    files= os.listdir(test_dir)
    sorted_files = sorted(files)

    total_num = 0
    top1_num = 0
    top5_num = 0
    
    for file in sorted_files:
        print(file)
        total_num += 1
        
        # cv2 读取后的数据为BGR方式
        x = get_numpy_from_img(test_dir + file)
        (a,b,c) = x.shape
        x = x.reshape(1, a, b, c)

        # mean操作
        x[0,0,:,:] -= 123.68
        x[0,1,:,:] -= 116.779
        x[0,2,:,:] -= 103.939

        net.blobs['data'].data[...] = x
        output = net.forward()

        # print(output)
        # print(output["prob_1"])

        output = np.array(output["prob"][0])
        output = np.array(output)
        output = np.reshape(output,(1000,))
        # print(output)
        # output = output[0]

        # 输出top5
        # print(np.max(output))
        # print(type(output))
        #print(np.argsort(output))

        sort_idx = np.flip(np.squeeze(np.argsort(output)))
        result = str(sort_idx[:5])
        print(result)
        node = {}
        node["result"] = result

        picture_name = file
        picture_name_index = picture_name.split(".")[0].split("_")[-1]
        picture_name_index = int(picture_name_index)
        label = val_labels[picture_name_index-1].strip("\n")
        node["label"] = label

        if label in result:
            node["top5"] = "True"
            top5_num +=1
        else:
            node["top5"] = "False"

        if label in str(sort_idx[:1]):
            node["top1"] = "True"
            top1_num +=1
        else:
            node["top1"] = "False"

        print(node)
        results[file] = node
        
        print("total: ", total_num)
        # print("top1 hit: ", top1_num)
        # print("top5 hit: ", top5_num)
        print("top1_accuracy_rate: ", top1_num/total_num)
        print("top5_accuracy_rate: ", top5_num/total_num)


    total_num = len(results)
    top1_num = 0
    top5_num = 0

    sorted_results = sorted(results.keys())
    for i in sorted_results :
        temp = "img[%03s]  lable[%-3s]  result[%-19s]   top1[%s]  top5[%s]"%\
            (i.split('.')[0],  results[i]["label"], results[i]["result"], results[i]["top1"], results[i]["top5"])
        print(temp)

        
    print("total: ", total_num)
    print("top1 hit: ", top1_num)
    print("top5 hit: ", top5_num)
    print("top1_accuracy_rate: ", top1_num/total_num)
    print("top5_accuracy_rate: ", top5_num/total_num)


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print ("Usage:", sys.argv[0], " caffePrototxt  caffeModel  testFileDir")
        sys.exit(-1)

    net = caffe.Net(sys.argv[1], sys.argv[2], caffe.TEST)
    caffe_ference(net, sys.argv[3])
