# coding=utf-8
'''
@Software:PyCharm
@Time:2024/04/07 2:58 下午
@Author: fffan
'''

import random
import torch
import pickle
import argparse
import collections
import click
from tqdm import tqdm
from transformer.tokenization import BertTokenizer
from torch.utils.data import Dataset
from loguru import logger


class TrainingInstance(object):
    """ 句子对形式的单个训练数据实例类型 """

    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
                 is_random_next):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels
        self.is_random_next = is_random_next

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join([x for x in self.tokens]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "is_random_next: %s\n" % self.is_random_next
        s += "masked_lm_positions: %s\n" % (" ".join([str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (" ".join([x for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


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


def write_instance_to_example_files(instances, tokenizer, max_seq_length, output_file):
    """ 以 TrainingInstance 创造 Dataset 训练实例"""
    eval_set = PreTrainingDataset()
    train_set = PreTrainingDataset()

    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
        input_ids_copy_for_replace = list(input_ids)  # 用于替换 mask 原字符得到 label
        input_mask = [1] * len(input_ids)
        segment_ids = list(instance.segment_ids)
        assert len(input_ids) <= max_seq_length

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
        # masked_lm_weights = [1.0] * len(masked_lm_ids)

        # 替换以及补齐长度
        for (p, m) in zip(masked_lm_positions, masked_lm_ids):
            input_ids_copy_for_replace[p] = m

        while len(input_ids_copy_for_replace) < max_seq_length:
            input_ids_copy_for_replace.append(-1)

        # while len(masked_lm_positions) < max_predictions_per_seq:
        #     masked_lm_positions.append(0)
        #     masked_lm_ids.append(0)
        #     masked_lm_weights.append(0.0)

        # 这里注意：按照原文思路，随机下一句的 label 为 1，而真实下一句的 label 为 0 。
        next_sentence_label = 1 if instance.is_random_next else 0
        features = collections.OrderedDict()
        features["input_ids"] = create_long_tensor(input_ids)
        features["segment_ids"] = create_long_tensor(segment_ids)
        features["input_mask"] = create_float_tensor(input_mask)
        features["masked_lm_ids"] = create_long_tensor(input_ids_copy_for_replace)
        features["next_sentence_labels"] = create_long_tensor([next_sentence_label])

        if total_written < len(instances) * 0.10:
            eval_set.add_instance(features)
        else:
            train_set.add_instance(features)
        total_written += 1

        # if inst_index < 20:
        #     logger.info("*** Example ***")
        #     logger.info("tokens: %s" % " ".join([x for x in instance.tokens]))

        #     for feature_name in features.keys():
        #         feature = features[feature_name]
        #         values = []
        #         if feature.int64_list.value:
        #             values = feature.int64_list.value
        #         elif feature.float_list.value:
        #             values = feature.float_list.value
        #         logger.info("%s: %s" % (feature_name, " ".join([str(x) for x in values])))

    with open(output_file + '.eval', 'wb') as f:
        pickle.dump(eval_set, f)
    with open(output_file + '.train', 'wb') as f:
        pickle.dump(train_set, f)

    logger.info("Wrote %d total instances" % total_written)


def create_long_tensor(values):
    tensor = torch.LongTensor(list(values))
    return tensor


def create_float_tensor(values):
    tensor = torch.FloatTensor(list(values))
    return tensor


def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng):
    """ 从原始语料生成 TrainingInstance 类实例"""
    all_documents = [[]]

    # 输入语料的数据格式：
    # (1) 每行是一整句完整句子。
    # (2) 每个段落之间用一个空行隔开。
    for input_file in input_files:
        with open(input_file, "r") as reader:
            while True:
                # TODO 这里没有实现判断是否为 unicode 编码
                line = reader.readline()
                if not line:
                    break
                line = line.strip()

                # 如果是个空行，则表示新的段落开始。
                if not line:
                    all_documents.append([])
                tokens = tokenizer.tokenize(line)
                if tokens:
                    all_documents[-1].append(tokens)

    # 去除空段落
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)

    vocab_words = list(tokenizer.vocab.keys())
    instances = []
    for _ in range(dupe_factor):
        for document_index in range(len(all_documents)):
            instances.extend(
                create_instances_from_document(
                    all_documents, document_index, max_seq_length, short_seq_prob,
                    masked_lm_prob, max_predictions_per_seq, vocab_words, rng))

    rng.shuffle(instances)
    return instances


def create_instances_from_document(
        all_documents, document_index, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
    """ 从一个文本段落生成 TrainingInstances 类型实例"""
    document = all_documents[document_index]

    # 因为句子包含 [CLS], [SEP], [SEP] 所以长度减三
    max_num_tokens = max_seq_length - 3

    # 有概率在满足 max_seq_length 的前提下随机决定新的句子总长，
    # 以增加数据的任意性。
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)

    # 这里采取的策略是先预选中最大长度的几个句子片段，然后将片段以完整句子为单位随机分为 A B 两部分，
    # 如果是进入了随机下一句的 case ，则在选完 A 中的句子后，下一句从其他段落中随机挑选句子填满该 instance。
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    #while i < len(document):
    for i in tqdm(range(len(document))):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # a_end 代表从 current_chunk 中选择多少个句子放入 A 片段
                # A 代表第一句话
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                # 随机的 “下一句话”
                if len(current_chunk) == 1 or rng.random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # 随机选取一个其他的段落以摘取一个“任意的”下一句话
                    random_document_index = rng.randint(0, len(all_documents) - 1)
                    for _ in range(10):
                        if random_document_index != document_index:
                            break
                        random_document_index = rng.randint(0, len(all_documents) - 1)

                    random_document = all_documents[random_document_index]
                    random_start = rng.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # 这里由于采取了随机的下一句话，所以原本想当作第二句话的句子都可以放回去等待下一轮使用。
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # 真实的 “下一句话”
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = []
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)

                tokens.append("[SEP]")
                segment_ids.append(0)

                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                (tokens, masked_lm_positions, masked_lm_labels) = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
                instance = TrainingInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels,
                    is_random_next=is_random_next)
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1


    return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
    """ 按照 masked LM 的设定生成输入向量 """

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        cand_indexes.append([i])

    # 先打乱，后面直接去前百分之几做[MASK]
    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            # 80% 的概率替换为 [MASK]
            if rng.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% 的概率保持原始字符
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% 的概率替换为随机字符
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

    return output_tokens, masked_lm_positions, masked_lm_labels


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """ 将两个句子修剪至总长度小于等于设定的最大长度 """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # 以二分之一的概率随机选择从句首或句尾截短当前句子
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",
                        default="/data1/fffan/0_data/0_original_data/3_NLP相关数据/0_data_wudao/wudao_data_3B_test.txt",
                        type=str)

    # Required parameters
    parser.add_argument("--output_file", default="./output_dir/bert_chinese_fffan_data", type=str)
    parser.add_argument("--model_path", default="./pretrain_models/bert-base-chinese", type=str)

    # Other parameters
    parser.add_argument("--max_seq_length", default=128, type=int, help="")
    parser.add_argument("--max_predictions_per_seq", default=20, type=int, help=" ")
    parser.add_argument("--random_seed", default=20, type=int,help=" ")
    parser.add_argument("--dupe_factor", default=10, type=int, help=" ")

    parser.add_argument("--masked_lm_prob", default=0.15, type=float, help=" ")
    parser.add_argument("--short_seq_prob", default=0.1, type=float, help=" ")

    args = parser.parse_args()

    return args



def main():
    #####
    args =  get_parser()

    logger.info("*** Loading the tokenizer ***")
    tokenizer = BertTokenizer.from_pretrained(args.model_path)

    # TODO 这里没有实现原始的给定文件 pattern 来批量匹配文件
    input_files = args.input_file.split(",")
    logger.info("*** Reading from input files ***")
    for args.input_file in input_files:
        logger.info("  %s" % args.input_file)

    rng = random.Random(args.random_seed)
    instances = create_training_instances(
        input_files, tokenizer, args.max_seq_length, args.dupe_factor,
        args.short_seq_prob, args.masked_lm_prob, args.max_predictions_per_seq, rng)

    logger.info("*** Writing to output file ***")
    logger.info("  %s.train  %s.eval" % (args.output_file, args.output_file))

    write_instance_to_example_files(instances, tokenizer, args.max_seq_length, args.output_file)


if __name__ == "__main__":
    main()
