# bert_pretrain_pytorch

本代码是基于pytorch的框架下，对Bert算法进行预训练。


## 训练Bert模型

### 配置环境

``` sh
git clone https://github.com/Tian14267/bert_pretrain_pytorch.git
```

- 安装 Conda:  https://docs.conda.io/en/latest/miniconda.html
- 创建 Conda 环境:

``` sh
conda create -n asr python=3.8
conda activate bert_train
pip install -r requirements.txt
```


### 训练模型

#### 数据准备
使用中文数据。这里推荐部分开源数据如Wudao数据、THUCNews数据。详情如下：
```
wudao数据
链接：https://pan.baidu.com/s/19eEY0jQXgG_H99JUrrN4QQ?pwd=82rs 
提取码：82rs

THUCNews数据
链接：https://pan.baidu.com/s/1W4PJVC_xe-wRxlgH7F51UA?pwd=u9fz 
提取码：u9fz
```

注：
文件夹 ```./data``` 中提供了使用 ```create_pretraining_data.py``` 处理好的训练文件```bert_chinese_fffan_data.eval```，可以直接使用。


#### 训练

##### 训练方案 -1
本方案代码中只加入了Bert对数据mask的任务预测，即掩码语言模型（Masked Language Model，MLM）。而另一个任务 下一句预测（Next Sentence Prediction，NSP）暂不包括在本代码中。
```
单卡训练：

sh pretrain_Bert_fffan_method_1.sh
```

##### 训练方案 -2
先预处理数据，并进行保存。再载入保存的数据，进行模型训练。包含两个任务：MLM和NSP 。
```
单卡训练：

sh pretrain_Bert_fffan_method_2.sh
```
