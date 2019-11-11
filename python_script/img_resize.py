#!/usr/bin/python3

# 提取目录下所有图片,更改尺寸后保存到另一目录
# from PIL import Image
import os.path
import sys, os
import cv2
import numpy as np

#crop_size = 224
crop_size = 224

# 取中心图片
def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]


def convertjpg(inputdir, outdir):

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    files= os.listdir(inputdir) #得到文件夹下的所有文件名称
    sorted_files = sorted(files)
    for file in sorted_files:
        print(file)
        img = cv2.imread(inputdir + '/' + file)
        #cv2读取后的图片数据默认为bgr顺序排列
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        print(img.shape)
        weight = img.shape[0]
        height = img.shape[1]
        if weight < height:
            img = cv2.resize(img, (int(height/weight * crop_size), crop_size))
        else:
            img = cv2.resize(img, (crop_size, int(weight/height * crop_size)))

        print(img.shape)
        img = crop_center(img,crop_size,crop_size)
        (w,h,c) = img.shape
        if w!= crop_size or h!=crop_size or c!=3:
            print("error shape ", img.shape)
            sys.exit(-1)

        save_name = file.split(".")[0] + '.jpg'
        save_file = os.path.join(outdir, os.path.basename(save_name))
        cv2.imwrite(save_file, img)


def convertjpg2(inputdir, outdir, width=crop_size, height=crop_size):

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    files= os.listdir(inputdir) #得到文件夹下的所有文件名称
    for file in files:
        try:
            print(file)
            img = cv2.imread(inputdir + '/' + file)
            img = cv2.resize(img, (width,height))
            save_name = file.split(".")[0] + '.jpg'
            save_file = os.path.join(outdir, os.path.basename(save_name))
            cv2.imwrite(save_file, img)
        except Exception as e:
            print(e)

# squezzenet 图片预处理
def img_preprocess(inputdir, outdir, width=crop_size, height=crop_size):

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    files= os.listdir(inputdir) #得到文件夹下的所有文件名称
    for file in files:
        print(file)
        img = cv2.imread(inputdir + '/' + file)
        img = cv2.resize(img, (256,256))
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #转为RGB方式
        
        img = np.array(img, dtype=np.float32)
        
        # 截取中心 224*224
        img = img[:, 16:240, 16:240]
        
        # 减去mean
        img[0,:,:] -= 123.68
        img[1,:,:] -= 116.28
        img[2,:,:] -= 103.53
        
        # 乘以scale
        img[0,:,:] *= 0.0171
        img[1,:,:] *= 0.0175
        img[2,:,:] *= 0.0174
    
        save_name = file.split(".")[0] + '.jpg'
        save_file = os.path.join(outdir, os.path.basename(save_name))
        cv2.imwrite(save_file, img)


if __name__ == "__main__":

    if len(sys.argv) !=3:
        print("Usage: ", sys.argv[0], " input_path  output_path")
        sys.exit(-1)

    img_preprocess(sys.argv[1], sys.argv[2])
