import cv2
import numpy as np
import os

def over_length(path, save_path):
    for root, dir, files in os.walk(path):
        for file in files:
            # 读入原图片
            img = cv2.imdecode(np.fromfile(os.path.join(root, file), dtype=np.uint8), -1)
            print(img)
            # 将图片高和宽分别x赋值给x，y
            height, width = img.shape[0:2]
            # 显示原图
            cv2.imshow('OriginalPicture', img)
            # (width, int(height / 3)) 元组形式，高度缩放到原来的三分之一
            # img_change1 = cv2.resize(img, (width, int(height / 3)))
            img_change1 = cv2.resize(img, (640, 640), cv2.INTER_LINEAR)
            cv2.imencode('.jpg', img_change1)[1].tofile(save_path + file.split('.')[0]
                                                                  + '_' + 'overlength' + '.jpg')

if __name__ == '__main__':
    over_length("/home/ruoji/code/tengine-lite-yolov5s-tt100k/Tengine/images/",
                '/home/ruoji/code/tengine-lite-yolov5s-tt100k/Tengine/images/')
     
