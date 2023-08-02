# tengine-lite-yolov5s-tt100k



### 一、训练准备

1、yolov5 tag: v6.1

```sh
git clone https://github.com/ultralytics/yolov5.git
git checkout v6.1
```

2、数据集处理

参考帖子：

[YOLO5用于交通标志牌（TT100K数据集）的训练预测（包含数据集格式的转换——TT100K-CoCo格式-YOLO格式，数据集比例的重新划分以及对应图片集的重新划分）_我宿孤栈的博客-CSDN博客](https://blog.csdn.net/qq_37346140/article/details/127122818?spm=1001.2014.3001.5501)

tt100k数据集  下载地址：[TT100K](http://cg.cs.tsinghua.edu.cn/traffic-sign/data_model_code/data.zip)

使用[脚本](./data_preprocess.py)转换后得到数据集

![image-20230802191844345](assets/image-20230802191844345.png)

```
python .\train.py --weights .\yolov5s.pt --cfg .\models\yolov5s.yaml --data .\datasets\tt100k\tt100k_fuxian.yaml --batch-size 8 --name tt100k_1 --epocchs 10
```

