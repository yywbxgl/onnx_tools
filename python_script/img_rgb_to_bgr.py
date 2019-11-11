
#!/usr/bin/python3

# 提取目录下所有图片, 把RGB的图片修改为BGR图片保存
# from PIL import Image
import os.path
import sys, os
import cv2


def convertjpg(inputdir, outdir):
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    files= os.listdir(inputdir) #得到文件夹下的所有文件名称
    sorted_files = sorted(files)

    for file in sorted_files:
        print(file)
        img = cv2.imread(inputdir + '/' + file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #cv2默认为bgr顺序
        (w,h,c) = img.shape
        if w!= 224 or h!=224 or c!=3:
            print("error shape ", img.shape)
            sys.exit(-1)

        save_name = file.split(".")[0] + '.jpg'
        save_file = os.path.join(outdir, os.path.basename(save_name))
        cv2.imwrite(save_file, img)



if __name__ == "__main__":

    if len(sys.argv) !=3:
        print("Usage: ", sys.argv[0], " input_path  output_path")
        sys.exit(-1)

    convertjpg(sys.argv[1], sys.argv[2])
