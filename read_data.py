# import tensorflow as tf
import numpy as np
from skimage import io

def read_data():
    train_root = '/data1/data/landmark/training/'
    file_root = '/data1/data/landmark/training/filelist/fpie4kn+multipie78.txt'
    f = open(file_root,'r')
    file = list()
    num = 0
    for line in open(file_root):
        line = f.readline()
        file.append(line)
        num = num + 1
        if(num==50000):
            break
    f.close()
    data_points = np.zeros([num,50])
    data_images = list()
    for i,unit in enumerate(file):
        unit = unit.split()
        image_path = train_root + unit[0]
        img = io.imread(image_path)
        data_images.append(img)
        for j in range(1,51):
            data_points[i,j-1] = unit[j]
    data_images = np.array(data_images)
    return(data_images,data_points)

# x,y = read_data()
# print(x.shape,y.shape)