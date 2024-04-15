#:<<EOF

####   方案二：先处理和保存数据。再进行训练

###   处理数据
python create_pretraining_data.py \
			--input_file /data1/fffan/0_data/0_original_data/3_NLP相关数据/0_data_wudao/wudao_data_3B_test.txt \
			--output_file ./output_dir/bert_chinese_fffan_data \
			--model_path ./pretrain_models/bert-base-chinese \
			--max_seq_length 128 \
			--max_predictions_per_seq 20 \
			--random_seed 20 \
			--dupe_factor 10 \
			--masked_lm_prob 0.5 \
			--short_seq_prob 0.1

### 开始训练
output_dir=./output_dir/pretrain_base
mkdir $output_dir
CUDA_VISIBLE_DEVICES=5 python pretrain_Bert_method_2.py   \
                          --model_path ./pretrain_models/bert-base-chinese \
                          --train_file_path  ./output_dir/bert_chinese_fffan_data.train \
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