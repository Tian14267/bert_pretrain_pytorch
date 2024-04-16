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
from collections import namedtuple
from torch.utils.data import (DataLoader, RandomSampler, Dataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import horovod.torch as hvd
from torch.nn import CrossEntropyLoss
import collections
from torchinfo import summary
from model_info.file_utils import WEIGHTS_NAME, CONFIG_NAME
from model_info.modeling import BertForPreTraining
from model_info.tokenization import BertTokenizer
from model_info.optimization import BertAdam


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

InputFeatures = namedtuple("InputFeatures", "input_ids input_masks segment_ids masked_lm_positions masked_lm_ids masked_lm_weights")

hvd.init()
logger.info("hvd.local_rank:{} ".format(hvd.local_rank()))
logger.info("hvd.rank:{} ".format(hvd.rank()))
logger.info("hvd.local_size:{} ".format(hvd.local_size()))
logger.info("hvd.size:{} ".format(hvd.size()))

torch.cuda.set_device(hvd.local_rank())
os.environ["CUDA_VISIBLE_DEVICES"] = str(hvd.local_rank())
device = 'cuda'



MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        do_whole_word_mask = True
        if (do_whole_word_mask and len(cand_indexes) >= 1 and
                token.startswith("##")):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    rng.shuffle(cand_indexes)

    output_tokens = [t[2:] if len(re.findall('##[\u4E00-\u9FA5]', t))>0 else t for t in tokens]

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index][2:] if len(re.findall('##[\u4E00-\u9FA5]', tokens[index]))>0 else tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)


    return (output_tokens, masked_lm_positions, masked_lm_labels)



def convert_example_to_features(text, tokenizer, max_seq_len):
    """输入text格式：
        1): 单句
        2): 双句，以\t分隔，并且分隔后这两个句子的0,1索引位置
    """
    # 本项目为了方便，以单句的语料为例
    sents = text.split('\t')[:1]
    tokens = ['[CLS]'] + tokenizer.tokenize(sents[0])[:max_seq_len - 2] + ['[SEP]']

    vocab_words = list(tokenizer.vocab.keys())
    rng = random.Random()
    (tokens, masked_lm_positions,
     masked_lm_labels) = create_masked_lm_predictions(tokens, masked_lm_prob=0.1,
                                 max_predictions_per_seq=23, vocab_words=vocab_words, rng=rng)



    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = len(input_ids) * [0]

    if len(sents) > 1:
        token_b = tokenizer.tokenize(sents[1])[:max_seq_len - 2] + ['[SEP]']
        input_ids += tokenizer.convert_tokens_to_ids(token_b)
        segment_ids += len(token_b) * [1]

    input_array = np.zeros(max_seq_len, dtype=np.int)
    input_array[:len(input_ids)] = input_ids

    mask_array = np.zeros(max_seq_len, dtype=np.bool)
    mask_array[:len(input_ids)] = 1

    segment_array = np.zeros(max_seq_len, dtype=np.bool)
    segment_array[:len(segment_ids)] = segment_ids

    masked_lm_positions = list(masked_lm_positions)
    masked_lm_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)
    masked_lm_weights = [1.0] * len(masked_lm_ids)
    max_predictions_per_seq = 23
    while len(masked_lm_positions) < max_predictions_per_seq:
        masked_lm_positions.append(0)
        masked_lm_ids.append(0)
        masked_lm_weights.append(0.0)

    masked_lm_positions = np.array(masked_lm_positions)
    masked_lm_ids = np.array(masked_lm_ids)
    masked_lm_weights = np.array(masked_lm_weights)

    feature = InputFeatures(input_ids=input_array,
                            input_masks=mask_array,
                            segment_ids=segment_array,
                            masked_lm_positions=masked_lm_positions,
                            masked_lm_ids=masked_lm_ids,
                            masked_lm_weights=masked_lm_weights
                            )
    return feature



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


def main():
    args = get_parser()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    torch.backends.cudnn.deterministic = True

    logger.info('args:{}'.format(args))

    tokenizer = BertTokenizer.from_pretrained(args.model_path, do_lower_case=args.do_lower_case)
    if os.path.exists(os.path.join(args.model_path, "vocab.txt")):
        os.system("cp " + os.path.join(args.model_path, "vocab.txt") + " " + args.output_dir)

    model = BertForPreTraining.from_scratch(args.model_path)
    model.to(device)

    # logging
    logger.info(f"[Data]: Reading data...\n")

    dataset = PregeneratedDataset(args.train_file_path, tokenizer, max_seq_len=args.max_seq_len)
    total_train_examples = len(dataset)
    print("#####   训练数据量：", total_train_examples)

    num_train_optimization_steps = int(total_train_examples / args.train_batch_size /
                                       args.gradient_accumulation_steps * args.num_train_epochs)
    print("#####   训练数据总步数：", num_train_optimization_steps)

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=hvd.size(),
                                                                    rank=hvd.rank(), shuffle=True)
    train_params = {
        "batch_size": args.train_batch_size,
        "num_workers": 0,
    }
    # Creation of Dataloaders for testing and validation.
    training_loader = DataLoader(dataset, batch_size=train_params["batch_size"], shuffle=False,
                                 sampler=train_sampler)
    #######################################################
    size = 0
    for n, p in model.named_parameters():
        logger.info('n: {}'.format(n))
        logger.info('p: {}'.format(p.nelement()))
        size += p.nelement()

    logger.info('Total parameters: {}'.format(size))

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
    optimizer = hvd.DistributedOptimizer(optimizer, backward_passes_per_step=args.gradient_accumulation_steps)
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    # Training loop
    logger.info(f"[Initiating Fine Tuning]...\n")
    logger.info("the length of dataloader is: {}".format(len(training_loader)))

    logging.info("***** Running training *****")
    logging.info("  Num examples = {}".format(total_train_examples))
    logging.info("  Batch size = %d", args.train_batch_size)
    logging.info("  Num steps = %d", num_train_optimization_steps)


    model.train()
    global_step = 0
    all_save_model_list = []
    output_model_file = None

    for epoch in tqdm(range(int(args.num_train_epochs)), desc="## Epoch", ascii=True):
        ####
        for step, batch in enumerate(tqdm(training_loader, desc="# Iteration", ascii=True)):
            #####
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids,masked_lm_ids,next_sentence_labels  = batch
            if input_ids.size()[0] != args.train_batch_size:
                continue

            loss = model(input_ids, segment_ids, input_mask, masked_lm_ids,next_sentence_labels)
            loss = loss.mean() # mean() to average on multi-gpu.


            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            if epoch == 0 and step == 0 and hvd.rank() == 0:  ###  只在训练开头打印参数信息，其他时候不打印
               summary(model=model, input_ids=input_ids, segment_ids=segment_ids,input_mask=input_mask, masked_lm_ids=masked_lm_ids, next_sentence_labels=next_sentence_labels,
                        device=device)

            if step % 100 == 0:
                logger.info(f'loss = {loss}')

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if (global_step + 1) % args.eval_step == 0 :
                    result = {}
                    result['global_step'] = global_step
                    result['loss'] = loss

                    output_eval_file = os.path.join(args.output_dir, "log.txt")
                    with open(output_eval_file, "a") as writer:
                        logger.info("***** Eval results *****")
                        for key in sorted(result.keys()):
                            logger.info("  %s = %s", key, str(result[key]))
                            writer.write("%s = %s\n" % (key, str(result[key])))

                    if hvd.rank() == 0:
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


        if output_model_file:
            output_list = output_model_file.split("/")[:-1] + ["pytorch_model.bin"]
            cp_path = "/".join(output_list)
            cp_line = "mv " + output_model_file + " " + cp_path
            print("#####    mv model path: ", cp_line)
            os.system(cp_line)

if __name__ == "__main__":
    main()
