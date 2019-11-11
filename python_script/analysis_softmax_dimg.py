# 解析dimg文件，得到top1-5，对推理结果与图片标签对比，统计准确率

import os, sys
import json
import re
import numpy as np

# 查找所有dimg output文件,得到文件夹下的所有文件名称
def get_all_files(dir):
    ret = []
    for root,dirs,files in os.walk(dir):
        for file in files:
            if ".dimg" in file:
                ret.append(os.path.join(root,file))
    return ret

# 解析output文件结果，放入map
def read_all_output_file(input_dir):
    files = get_all_files(input_dir)
    files.sort()
    results = {}

    top2_same = 0
    top3_same = 0
    top4_same = 0
    top5_same = 0
    temp=0
    
    for f in files:
        print(f)
        fin = open(f, 'r')
        value = []
        dimgs = fin.readlines()
        for dimg in dimgs:
            value.append(int(dimg.strip('\n')))
        value = np.array(value)
        sorted_arg = np.flip(np.squeeze(np.argsort(value)))
        top5 = str(sorted_arg[:5])
        
        node = {}
        node['file_name'] = f
        node['result'] = top5
        node['top 1~5 be same'] = "False"
        node['top2 be same'] = "False"
        node['top3 be same'] = "False"
        node['top4 be same'] = "False"
        node['top5 be same'] = "False"
        node['top same index'] = ""
        
        # 判断top5 中前几类的概率是否相同
        if (value[sorted_arg[0]] == value[sorted_arg[1]]):
            node['top 1~5 be same'] = "True"
            node['top2 be same'] = "True"
            node['top same index'] = str(sorted_arg[:2])
            top2_same += 1
            # print(value[sorted_arg[0]], value[sorted_arg[1]])
            # if (value[sorted_arg[0]] == 127):
                # temp += 1
        
        if (value[sorted_arg[0]] == value[sorted_arg[1]]) and \
            (value[sorted_arg[1]] == value[sorted_arg[2]]):
            node['top 1~5 be same'] = "True"
            node['top3 be same'] = "True"
            node['top same index'] = str(sorted_arg[:3])
            top3_same +=1

        if (value[sorted_arg[0]] == value[sorted_arg[1]]) and \
        (value[sorted_arg[1]] == value[sorted_arg[2]]) and \
        (value[sorted_arg[2]] == value[sorted_arg[3]]):
            node['top 1~5 be same'] = "True"
            node['top4 be same'] = "True"
            node['top same index'] = str(sorted_arg[:4])
            top4_same +=1
            
        if (value[sorted_arg[0]] == value[sorted_arg[1]]) and \
        (value[sorted_arg[1]] == value[sorted_arg[2]]) and \
        (value[sorted_arg[2]] == value[sorted_arg[3]]) and \
        (value[sorted_arg[3]] == value[sorted_arg[4]]):
            node['top 1~5 be same'] = "True"
            node['top5 be same'] = "True"
            node['top same index'] = str(sorted_arg[:5])
            top5_same +=1

        file_name = f.split(".jpg")[0].split("/")[-1]
        results[file_name] = node
        
    print("top2 be same : ", top2_same)
    print("top3 be same : ", top3_same)
    print("top4 be same : ", top4_same)
    print("top5 be same : ", top5_same)
    # print("top value equal 127 : ", temp)
    print()

    return results



# 对解析的result与图片标签对比，进行准确率统计
def analyze_accuracy(results):
    f_val_label = open("my_label.txt", "r")
    val_labels = f_val_label.readlines()
    for val_label in val_labels:
        val_label = val_label.strip('\n')
        
    # 添加标签
    for i in results:
        picture_name_index = i.split(".jpg")[0].split("_")[-1]
        picture_name_index = int(picture_name_index)
        label = val_labels[picture_name_index-1].strip("\n")
        results[i]["label"] = label
        results[i]["pic_index"] = picture_name_index

        if label in results[i]["result"]:
            results[i]["top5"] = "True"
        else:
            results[i]["top5"] = "False"

        if label in results[i]["result"].split(' ')[0]:
            results[i]["top1"] = "True"
        else:
            results[i]["top1"] = "False"
        
        # top1 增加概率相同的情况
        if label in results[i]["top same index"]:
            results[i]["top1"] = "True"

    # 统计准确率
    total_num = len(results)
    top1_num = 0
    top1_and_no_same = 0
    top1_and_be_same = 0
    top5_num = 0

    sorted_results = sorted(results.keys())
    for i in sorted_results :
    # for i in results:
        temp = "img[%03s]  lable[%-3s]  result[%-19s]   top1[%s]  top5[%s]"%\
            (i.split('.')[0],  results[i]["label"], results[i]["result"], results[i]["top1"], results[i]["top5"])
        # print(temp)

        if results[i]["top5"] == "True":
            top5_num += 1
        if results[i]["top1"] == "True":
            top1_num += 1
        if  results[i]["top1"] == "True" and results[i]['top 1~5 be same'] == "True":
            top1_and_be_same += 1
        if  results[i]["top1"] == "True" and results[i]['top2 be same'] == "False":
            top1_and_no_same += 1

    print("-------nvdla runtime result------")
    print("total num: ", total_num)
    print("top1 hit : %f [%d]"%(top1_num/total_num, top1_num))
    print("top1 hit and no same: %f [%d]"%(top1_and_no_same/total_num, top1_and_no_same))
    print("top1 hit and be same : %f [%d]"%(top1_and_be_same/total_num, top1_and_be_same))
    print("top5 hit : %f [%d]"%(top5_num/total_num, top5_num))
    
    return results


# 对比nvdla runtime 与onnx runtime 推理的结果
def compare_onnx_run_reslut(nvdla_result):
    onnx_reslut_file = open("inception_v1_onnx_run_reslut.txt", "r")
    onnx_resluts = onnx_reslut_file.readlines()
    
    # 添加onnx run 的结果
    for i in nvdla_result:
        picture_num = results[i]["pic_index"]
        one_result = onnx_resluts[picture_num -1]
        top1_index = one_result.find("top1[")
        if (one_result[top1_index + 5] == "T"):
            nvdla_result[i]["onnx_top1"] = "True"
        elif (one_result[top1_index + 5] == "F"):
            nvdla_result[i]["onnx_top1"] = "False"
        else:
            print("error. can not find top1. ", onnx_resluts)
            sys.exit(-1)

        top5_index = one_result.find("top5[")
        if (one_result[top5_index + 5] == "T"):
            nvdla_result[i]["onnx_top5"] = "True"
        elif (one_result[top5_index + 5] == "F"):
            nvdla_result[i]["onnx_top5"] = "False"
        else:
            print("error. can not find top5. ", onnx_resluts)
            sys.exit(-1)

    # 对比结果
    total_num = len(results)
    onnx_top1_num = 0
    all_top1_num = 0
    onnx_top5_num = 0
    all_top5_num = 0
    top1_and_be_same= 0
    top1_and_no_same= 0
    
    sorted_results = sorted(nvdla_result.keys())
    for i in sorted_results :
        if nvdla_result[i]["onnx_top5"] == "True":
            onnx_top5_num += 1
        if nvdla_result[i]["onnx_top1"] == "True":
            onnx_top1_num += 1
        if nvdla_result[i]["onnx_top5"] == "True" and nvdla_result[i]["top5"] == "True":
            all_top5_num += 1
        if nvdla_result[i]["onnx_top1"] == "True" and nvdla_result[i]["top1"] == "True":
            all_top1_num += 1
        
        if  results[i]["top1"] == "True" and results[i]['top 1~5 be same'] == "True" and nvdla_result[i]["onnx_top1"] == "True":
            top1_and_be_same += 1
        if  results[i]["top1"] == "True" and results[i]['top2 be same'] == "False" and  nvdla_result[i]["onnx_top1"] == "True":
            top1_and_no_same += 1

    print("\n-------onnx runtime result------")
    print("total num: ", total_num)
    print("top1 hit : %f [%d]"%(onnx_top1_num/total_num, onnx_top1_num))
    print("top1 hit and nvdla top1 hit: %f [%d]"%(all_top1_num/total_num, all_top1_num))
    print("top1 hit and nvdla top1 hit and be same: %f [%d]"%(top1_and_be_same/total_num, top1_and_be_same))
    print("top1 hit and nvdla top1 hit and no same: %f [%d]"%(top1_and_no_same/total_num, top1_and_no_same))
    
    print("top5 hit : %f [%d]"%(onnx_top5_num/total_num, onnx_top5_num))
    print("top5 hit and nvdla top5 hit: %f [%d]"%(all_top5_num/total_num, all_top5_num))


if __name__ == "__main__":
    if len(sys.argv) !=2:
        print("Usage: ", sys.argv[0], " result_json_dir")
        sys.exit(-1)
    
    results = read_all_output_file(sys.argv[1])
    results = analyze_accuracy(results)
    compare_onnx_run_reslut(results)