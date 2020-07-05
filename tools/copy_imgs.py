# -*-coding:utf-8-*-
import os
import shutil
import glob
import random

data_path = "D:\\rock images\\"
count = 0
for i in range(1):
    i += 1

    data_path_ = data_path # data_path + str(i)
    images_path = os.path.join(data_path_, "*\\*\\*.jpg")

    for image_file in glob.glob(images_path):
        count += 1
        if count % 100 == 0:
            print('copy {} images.'.format(count))
        # print(image_file)
        label_ = image_file.split('\\')[-3]
        if label_ == "ASY":
            img_label = 0
        elif label_ == "CH":
            img_label = 1
        elif label_ == "DJL":
            img_label = 2
        elif label_ == "FS":
            img_label = 3
        elif label_ == "QM":
            img_label = 4
        elif label_ == "GLY":
            img_label = 5
        elif label_ == "HGPM":
            img_label = 6
        elif label_ == "HJL":
            img_label = 7
        elif label_ == "HL":
            img_label = 8
        elif label_ == "HSB":
            img_label = 9
        elif label_ == "HWJ":
            img_label = 10
        elif label_ == "JXNH":
            img_label = 11
        elif label_ == "LW":
            img_label = 12
        elif label_ == "NYY":
            img_label = 13
        elif label_ == "QX":
            img_label = 14
        elif label_ == "SCB":
            img_label = 15
        elif label_ == "SLM":
            img_label = 16
        elif label_ == "SWDL":
            img_label = 17
        elif label_ == "SYY":
            img_label = 18
        elif label_ == "SZP":
            img_label = 19
        elif label_ == "TDZH":
            img_label = 20
        elif label_ == "XCY":
            img_label = 21
        elif label_ == "XHY":
            img_label = 22
        elif label_ == "XRX":
            img_label = 23
        elif label_ == "YY":
            img_label = 24
        elif label_ == "ZC":
            img_label = 25
        elif label_ == "BY":
            img_label = 26
        elif label_ == "SH":
            img_label = 27
        elif label_ == "XK":
            img_label = 28
        elif label_ == "ZYH":
            img_label = 29

        new_name = "D:\\image classification\\datasets\\easy\\trainval\\" + str(
            img_label) + "\\" + image_file.split('\\')[-1]


        shutil.copyfile(image_file, new_name)

