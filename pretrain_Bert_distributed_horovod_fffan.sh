

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