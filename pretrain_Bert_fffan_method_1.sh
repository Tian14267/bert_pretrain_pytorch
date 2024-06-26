#:<<EOF

####   方案一：本代码中只加入了Bert对数据mask的任务预测，即掩码语言模型（Masked Language Model，MLM）。
###  而另一个任务 下一句预测（Next Sentence Prediction，NSP）暂不包括在本代码中。

output_dir=./output_dir/pretrain_base
mkdir $output_dir
CUDA_VISIBLE_DEVICES=5 python pretrain_Bert_method_1.py   \
                          --model_path ./pretrain_models/bert-base-chinese \
                          --train_file_path  /data1/fffan/0_data/0_original_data/3_NLP相关数据/0_data_wudao/wudao_data_3B_test.txt \
                          --do_lower_case \
                          --train_batch_size 8 \
                          --output_dir $output_dir  \
                          --learning_rate 5e-5  \
                          --num_train_epochs  3  \
                          --eval_step  20  \
                          --max_seq_len  256  \
                          --save_model_number 1 \
                          --gradient_accumulation_steps  1  3>&2 2>&1 1>&3 | tee $output_dir/bert.log
#EOF