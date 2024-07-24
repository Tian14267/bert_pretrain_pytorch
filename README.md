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
conda create -n bert_train python=3.8
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

##### 训练方案 -3
**单机多卡**分布式训练。
```
sh pretrain_Bert_distributed_fffan.sh
```
或者
```markdown
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 pretrain_Bert_distributed.py \
                        --model_path "./pretrain_models/bert-base-chinese" \
                        --output_dir "./output_dir/pretrain_base" \
                        --train_batch_size 8
```


##### 训练方案 -4
使用 ```horovod``` 进行单机多卡分布式训练；
```markdown
sh pretrain_Bert_distributed_horovod_fffan.sh
```
或者
```markdown
output_dir=./output_dir/pretrain_fffan
mkdir $output_dir
CUDA_VISIBLE_DEVICES="2,3,4,5" horovodrun -np 4 python pretrain_Bert_distributed_horovod.py \
						  --model_path ./pretrain_models/bert_chinese_fffan \
                          --train_file_path  ./data/bert_chinese_fffan_data.eval \
                          --do_lower_case \
                          --train_batch_size 1 \
                          --output_dir $output_dir  \
                          --learning_rate 5e-5  \
                          --num_train_epochs  3  \
                          --eval_step  20  \
                          --max_seq_len  128  \
                          --save_model_number 1 \
                          --gradient_accumulation_steps  1  3>&2 2>&1 1>&3 | tee $output_dir/bert.log
```

注： ```horovod``` 安装方式：
```markdown
安装 NCCL
安装 horovod：0.27.0
安装方法：HOROVOD_GPU_OPERATIONS=NCCL pip install horovod
使用：CUDA_VISIBLE_DEVICES="1,2,3" horovodrun -np 3 python pretrain_Bert_distributed_horovod.py
https://github.com/horovod/horovod/blob/master/docs/pytorch.rst

```


### 注意
1：当修改 ```max_seq_len``` 参数时，注意 ```./model_info/tokenization.py``` 中的 ```max_len = 512``` 设置是否和max_seq_len一致;

2：注意 预训练模型 配置文件中 ```max_position_embeddings``` 是否与 max_position_embeddings
