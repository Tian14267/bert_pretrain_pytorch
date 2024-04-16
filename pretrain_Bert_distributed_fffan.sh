:<<EOF
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 pretrain_Bert_distributed.py \
									--model_path "./pretrain_models/bert-base-chinese" \
									--output_dir "./output_dir/pretrain_base" \
									--train_batch_size 8
EOF


#:<<EOF
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 pretrain_Bert_distributed.py \
									--model_path "./pretrain_models/bert_chinese_fffan" \
									--output_dir "./output_dir/pretrain_fffan" \
									--train_batch_size 2
EOF
