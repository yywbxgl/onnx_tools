# 解析json output文件，对推理结果与图片标签对比，统计准确率

import os, sys
import json
import re

def get_all_files(dir):
    ret = []
    for root,dirs,files in os.walk(dir):
        for file in files:
            if ".json" in file:
                ret.append(os.path.join(root,file))       
    return ret

# 解析json output文件，放入map
def read_all_output_file(input_dir):
    files = get_all_files(input_dir) #得到文件夹下的所有文件名称
    files.sort()
    results = {}

    for f in files:
        print(f)
        fin = open(f, 'rb')
        content = fin.read()
        content = str(content)
        begin = content.find("[[")
        end = content.rfind("]]")
        top6 = content[begin+2:end]
        tops = top6.split(",[")
        tops.pop()
        value_data = ""
        node = {}
        for top in tops:
            value_data = value_data + top.split(',',1)[0] + ' '
         
        node['result'] = value_data.strip(' ')
        results[f] = node
    return results


# 对result与图片标签对比，统计准确率
def analyze_accuracy(results):
    f_val_label = open("my_label.txt", "r")
    val_labels = f_val_label.readlines()
    for val_label in val_labels:
        val_label = val_label.strip('\n')

    for i in results:
        picture_name = i
        picture_name_index = picture_name.split(".jpg")[0].split("_")[-1]
        picture_name_index = int(picture_name_index)
        label = val_labels[picture_name_index-1].strip("\n")
        # print(picture_name)
        # print(picture_name_index)
        # print(results[i]["result"])
        # print(label)
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

    total_num = len(results)
    top1_num = 0
    top5_num = 0

    sorted_results = sorted(results.keys())
    for i in sorted_results :
    # for i in results:
        temp = "img[%03s]  lable[%-3s]  result[%-19s]   top1[%s]  top5[%s]"%\
            (i.split('.')[0],  results[i]["label"], results[i]["result"], results[i]["top1"], results[i]["top5"])
        print(temp)
        if results[i]["top1"] == "True":
            top1_num += 1
        if results[i]["top5"] == "True":
            top5_num += 1
        
    print("total: ", total_num)
    print("top1 hit: ", top1_num)
    print("top5 hit: ", top5_num)
    print("top1_accuracy_rate: ", top1_num/total_num)
    print("top5_accuracy_rate: ", top5_num/total_num)


if __name__ == "__main__":
    if len(sys.argv) !=2:
        print("Usage: ", sys.argv[0], " result_json_dir")
        sys.exit(-1)
    
    results = read_all_output_file(sys.argv[1])
    analyze_accuracy(results)