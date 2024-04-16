# coding=utf-8
'''
@Software:PyCharm
@Time:2024/04/07 2:58 下午
@Author: fffan
'''

from __future__ import absolute_import, division, print_function

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import re
import argparse
import logging
import random
import pickle
import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
from collections import namedtuple
from torch.utils.data import (DataLoader, RandomSampler, Dataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from torch.nn import CrossEntropyLoss
import collections
from model_info.file_utils import WEIGHTS_NAME, CONFIG_NAME
from model_info.modeling import BertForPreTraining
from model_info.tokenization import BertTokenizer
from model_info.optimization import BertAdam


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

InputFeatures = namedtuple("InputFeatures", "input_ids input_masks segment_ids masked_lm_positions masked_lm_ids masked_lm_weights")



MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

class PreTrainingDataset(Dataset):
    """ 实际的供给 Dataloader 的数据类 """

    def __init__(self):
        self.data = []

    def add_instance(self, features: collections.OrderedDict):
        self.data.append((
            features["input_ids"],
            features["segment_ids"],
            features["input_mask"],
            features["masked_lm_ids"],
            features["next_sentence_labels"]
        ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_ids, token_type_ids, attention_mask, masked_lm_labels, next_sentence_label = self.data[index]
        return input_ids, token_type_ids, attention_mask, masked_lm_labels, next_sentence_label




class PregeneratedDataset(object):
    def __init__(self, training_path, tokenizer, max_seq_len):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        logger.info('training_path: {}'.format(training_path))
        self.input_ids = []
        self.segment_ids = []
        self.input_masks = []
        self.masked_lm_ids = []
        self.next_sentence_labels = []
        print("#####  开始读取数据：",training_path)

        with open(training_path, 'rb') as f:
            one_pkl_dict = pickle.load(f)

        for i, feature in enumerate(tqdm(one_pkl_dict.data)):
            if not feature:
                continue
            """
            self.input_ids.append(feature.input_ids)
            self.segment_ids.append(feature.segment_ids)
            self.input_masks.append(feature.input_masks)
            self.masked_lm_ids.append(feature.masked_lm_ids)
            self.next_sentence_labels.append(feature.next_sentence_labels)
            """
            self.input_ids.append(feature[0])
            self.segment_ids.append(feature[1])
            self.input_masks.append(feature[2])
            self.masked_lm_ids.append(feature[3])
            self.next_sentence_labels.append(feature[4])

        self.data_size = len(self.input_ids)

    def __len__(self):
        return self.data_size

    def __getitem__(self, item):
        return (
                torch.tensor(self.input_ids[item]),
                torch.tensor(self.input_masks[item]),
                torch.tensor(self.segment_ids[item]),
                torch.tensor(self.masked_lm_ids[item]),
                torch.tensor(self.next_sentence_labels[item]),
                )


def save_model(prefix, model, path):
    logging.info("** ** * Saving  model ** ** * ")
    model_name = "{}_{}".format(prefix, WEIGHTS_NAME)
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(path, model_name)
    output_config_file = os.path.join(path, CONFIG_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file_path", default="./data/bert_chinese_fffan_data.eval", type=str)

    # Required parameters
    parser.add_argument("--model_path", default="./pretrain_models/bert-base-chinese", type=str)
    parser.add_argument("--output_dir", default="./output_dir/pretrain_fffan", type=str)

    # Other parameters
    parser.add_argument("--save_model_number",
                        default=5,
                        type=int, help="The maximum total input sequence length ")
    parser.add_argument("--max_seq_len",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece \n"
                             " tokenization. Sequences longer than this will be truncated, \n"
                             "and sequences shorter than this will be padded.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        # default=True,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=1, type=int, help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--weight_decay',
                        '--wd',
                        default=1e-1,
                        type=float,
                        metavar='W',
                        help='weight decay')
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing \n"
                             "a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--continue_train',
                        action='store_true',
                        help='Whether to train from checkpoints')

    # Additional arguments
    parser.add_argument('--eval_step', type=int, default=5)

    # This is used for running on Huawei Cloud.
    parser.add_argument('--data_url', type=str, default="")

    args = parser.parse_args()

    return args



def main_woker(local_rank, nprocs, args):
    dist.init_process_group(backend='nccl')

    model = BertForPreTraining.from_scratch(args.model_path)
    
    print("#######  local_rank: ",local_rank)
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)

    args.train_batch_size = int(args.train_batch_size / nprocs)
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank])

    #####
    tokenizer = BertTokenizer.from_pretrained(args.model_path, do_lower_case=args.do_lower_case)
    if os.path.exists(os.path.join(args.model_path, "vocab.txt")):
        os.system("cp " + os.path.join(args.model_path, "vocab.txt") + " " + args.output_dir)

    dataset = PregeneratedDataset(args.train_file_path, tokenizer, max_seq_len=args.max_seq_len)
    total_train_examples = len(dataset)
    print("#####   训练数据量：", total_train_examples)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=args.train_batch_size,
                                               num_workers=2,
                                               pin_memory=True,
                                               sampler=train_sampler)
    ########################

    size = 0
    for n, p in model.named_parameters():
        logger.info('n: {}'.format(n))
        logger.info('p: {}'.format(p.nelement()))
        size += p.nelement()

    logger.info('Total parameters: {}'.format(size))

    num_train_optimization_steps = int(total_train_examples / args.train_batch_size /args.num_train_epochs)
    print("#####   训练数据总步数：", num_train_optimization_steps)
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size(
        ) * args.num_train_epochs


    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01
    }, {
        'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0
    }]

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_optimization_steps)

    cudnn.benchmark = True


    global_step = 0
    for one_epoch in range(int(args.num_train_epochs)):
        train_sampler.set_epoch(one_epoch)

        # train for one epoch
        train(train_loader, model,  optimizer, local_rank,
              args, global_step)


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def reduce_value(value, average=True):
    world_size = dist.get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value




def train(train_loader, model, optimizer, local_rank, args, global_step):
    # switch to train mode
    model.train()
    all_save_model_list = []
    for step, batch in enumerate(tqdm(train_loader, desc="# Iteration", ascii=True)):
        #####
        #images = images.cuda(local_rank, non_blocking=True)
        batch = tuple(t.cuda(local_rank, non_blocking=True) for t in batch)
        input_ids, input_mask, segment_ids, masked_lm_ids, next_sentence_labels = batch
        if input_ids.size()[0] != args.train_batch_size:
            continue

        loss = model(input_ids, segment_ids, input_mask, masked_lm_ids, next_sentence_labels)

        #print("#####  global_step:",global_step,"  Loss ", loss)

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        if step % 100 == 0:
            logger.info(f'loss = {loss}')

        #loss = reduce_value(loss, average=True)
        loss = reduce_mean(loss, args.nprocs)
        print("#####  global_step:",global_step,"  Loss ", loss)

        optimizer.step()
        optimizer.zero_grad()
        global_step += 1

        if (global_step + 1) % args.eval_step == 0 and local_rank == 0:
            result = {}
            result['global_step'] = global_step
            result['loss'] = loss

            output_eval_file = os.path.join(args.output_dir, "log.txt")
            with open(output_eval_file, "a") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

            # Save a trained model
            ########################################################################
            prefix = f"step_{global_step}"
            logging.info("** ** * Saving  model ** ** * ")
            model_name = "{}_{}".format(prefix, WEIGHTS_NAME)
            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_file = os.path.join(args.output_dir, model_name)
            output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)

            #####  保存的模型数量超过指定数量，删除模型  #############
            if len(all_save_model_list) == args.save_model_number:
                os.system("rm -rf " + all_save_model_list[0])
                print("####  删除模型：", all_save_model_list[0])
                del all_save_model_list[0]
            ######################################################

            all_save_model_list.append(output_model_file)


def main():
    args = get_parser()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.nprocs = torch.cuda.device_count()
    print("###########    ",args.nprocs)
    main_woker(args.local_rank, args.nprocs, args)


if __name__ == "__main__":
    main()
