import os
import json
import re
import string
import numpy as np
from tqdm import tqdm
import sys
import random

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


class GTDataset(object):

    def __init__(self, logger, args, data_path, tokenizer, mode):
        self.data_path = data_path
        self.tokenizer = tokenizer

        # Load the word substitution file to add noise
        with open(args.substitution_file, 'r') as f_sub:
            self.synonym = json.load(f_sub)

        # Load the data (labeled or unlabeled)
        with open(self.data_path + '.json', "r") as f:
            self.all_data = json.load(f)

        if args.debug:
            if mode == "train":
                self.all_data = self.all_data[:1000]
            else:
                self.all_data = self.all_data[:100]
        assert type(self.all_data)==list
        assert all(["id" in d for d in self.all_data]), self.all_data[0].keys()
        if type(self.all_data[0]["id"])==int:
            for i in range(len(self.all_data)):
                self.all_data[i]["id"] = str(self.all_data[i]["id"])

        self.data = self.all_data
        self.mode = mode
        self.load = False
        self.logger = logger
        self.args = args
        self.data_type = mode
        self.metric = "BLEU"
        self.max_input_length = self.args.max_input_length
        self.dataset = None
        self.dataloader = None
        self.cache = None
        # List of curriculum segmentation points
        self.curriculum = eval(self.args.curriculum)
        # Current curriculum
        self.curriculum_now = 0
        # triple_num_for_all_data: the number of triples in each data sample
        # entity_set_for_all_data: the set of entities in each data sample
        # linear_length_for_all_data: the number of words in the linearized sequence of each data sample
        self.triple_num_for_all_data, self.entity_set_for_all_data, self.linear_length_for_all_data = \
                                                                        self.get_entity_triple_length_num(self.all_data)
        self.triple_num_for_data, self.entity_set_for_data, self.linear_length_for_data = [], [], []
        self.ppl_ratio = self.args.ppl_ratio
        self.struct_noise = self.args.struct_noise
        self.semantic_noise = self.args.semantic_noise
        self.is_noise = False
        self.curriculum_type = self.args.curriculum_type

    def __len__(self):
        return len(self.data)

    def get_entity_triple_length_num(self, data):
        # Acquire the number of triples / the set of entities in each data sample
        num_list = []
        entity_list = []
        linear_length_list = []
        for data_ele in data:
            triple_cnt = 0
            entity_set = set()
            linear_str = ""
            for _, triple in data_ele['kbs'].items():
                entity_set.add(triple[0])
                for triple_list in triple[2]:
                    triple_cnt += 1
                    entity_set.add(triple_list[1])
                    linear_str += " [head] " + triple[0] + " [relation] " + triple_list[0] + " [tail] " + triple_list[1]
            num_list.append(triple_cnt)
            entity_list.append(entity_set)
            linear_length_list.append(len(linear_str.split()))
        return num_list, entity_list, linear_length_list

    def build_data_from_cur(self):
        # Build the unlabeled dataset at each curriculum
        self.data = []
        if self.curriculum_type == "triple":
            self.triple_num_for_data, self.entity_set_for_data = [], []
            for data_id in range(len(self.all_data)):
                if self.triple_num_for_all_data[data_id] <= self.curriculum[self.curriculum_now]:
                    self.data.append(self.all_data[data_id])
                    self.triple_num_for_data.append(self.triple_num_for_all_data[data_id])
                    self.entity_set_for_data.append(self.entity_set_for_all_data[data_id])
        else:
            self.linear_length_for_data, self.entity_set_for_data = [], []
            for data_id in range(len(self.all_data)):
                if self.linear_length_for_all_data[data_id] <= self.curriculum[self.curriculum_now]:
                    self.data.append(self.all_data[data_id])
                    self.linear_length_for_data.append(self.linear_length_for_all_data[data_id])
                    self.entity_set_for_data.append(self.entity_set_for_all_data[data_id])

    def curriculum_next(self):
        self.curriculum_now += 1

    def set_noise(self, value):
        self.is_noise = value

    def update(self, predictions, gen_probs):
        # Use the generated results of the teacher model to build pseudo-labeled dataset
        assert len(self.data) == len(predictions) == len(gen_probs)
        tmp_data = []
        for _ in range(self.curriculum[self.curriculum_now]+1):
            tmp_data.append([])
        for data_id in range(len(self.data)):
            tmp_ratio = sum([1 if entity in predictions[data_id] else 0 for entity in self.entity_set_for_data[data_id]])
            tmp_ratio /= len(self.entity_set_for_data[data_id])
            # Filter the pseudo-labeled data with coverage
            if tmp_ratio >= self.args.cover_ratio:
                if self.curriculum_type == "triple":
                    tmp_data[self.triple_num_for_data[data_id]].append([self.data[data_id], predictions[data_id], gen_probs[data_id]])
                else:
                    tmp_data[self.linear_length_for_data[data_id]].append([self.data[data_id], predictions[data_id], gen_probs[data_id]])
        # Filter the pseudo-labeled data with generation probability
        for tmp_id in range(1, len(tmp_data)):
            tmp_data[tmp_id] = sorted(tmp_data[tmp_id], key=lambda x:x[2], reverse=True)[:int(len(tmp_data[tmp_id]) * self.ppl_ratio)]
        self.data = []
        for tmp_id in range(1, len(tmp_data)):
            for pair_id in range(len(tmp_data[tmp_id])):
                tmp_data[tmp_id][pair_id][0]['text'] = [tmp_data[tmp_id][pair_id][1]]
                self.data.append(tmp_data[tmp_id][pair_id][0])

    def word_substitution(self, fact_str):
        fact_list = fact_str.split()
        fact_result = []
        for fact_ele in fact_list:
            if random.random() < self.semantic_noise and fact_ele.lower() in self.synonym and len(self.synonym[fact_ele.lower()]) > 0:
                fact_result.append(random.choice(self.synonym[fact_ele.lower()]))
            else:
                fact_result.append(fact_ele)
        return ' '.join(fact_result)

    def __getitem__(self, idx):

        entry = self.data[idx]
        if len(entry['text']) > 0:
            text = random.choice(entry['text'])
        else:
            text = ""

        triple_list = []
        for _, triple in entry['kbs'].items():
            head = triple[0]
            for rel in triple[2]:
                triple_list.append([head, rel[0], rel[1]])

        # 1. structure noise: permute the triple list
        if self.is_noise and random.random() < self.struct_noise:
            random.shuffle(triple_list)

        # 2. semantic noise: substitute the word in entity / relation
        if self.is_noise:
            for triple_id in range(len(triple_list)):
                for fact_id in range(len(triple_list[triple_id])):
                    triple_list[triple_id][fact_id] = self.word_substitution(triple_list[triple_id][fact_id])

        graph_str = ""
        for triple_ele in triple_list:
            graph_str += " [head] " + triple_ele[0] + " [relation] " + triple_ele[1] + " [tail] " + triple_ele[2]
        graph_str = graph_str.strip()

        if self.args.do_lowercase:
            graph_str = graph_str.lower()
            text = text.lower()
        if self.args.append_another_bos and self.args.model_name == "bart":
            graph_str = "<s> " + graph_str
            text = "<s> " + text
        if self.args.model_name == "t5":
            graph_str = graph_str + " </s>"
            text = text + " </s>"

        def pad_to_length(ids, max_length, pad_id):
            if len(ids) >= max_length:
                result_ids = ids[:max_length]
                attn_mask = [1] * max_length
            else:
                result_ids = ids + [pad_id] * (max_length - len(ids))
                attn_mask = [1] * len(ids) + [0] * (max_length - len(ids))
            return result_ids, attn_mask

        graph_input, graph_attn_mask = pad_to_length(self.tokenizer.encode(graph_str), self.args.max_input_length, self.tokenizer.pad_token_id)
        text_input, text_attn_mask = pad_to_length(self.tokenizer.encode(text), self.args.max_output_length, self.tokenizer.pad_token_id)

        graph_input_ids = torch.LongTensor(graph_input)
        graph_attn_mask = torch.LongTensor(graph_attn_mask)
        text_input_ids = torch.LongTensor(text_input)
        text_attn_mask = torch.LongTensor(text_attn_mask)

        return graph_input_ids, graph_attn_mask, text_input_ids, text_attn_mask


def evaluate_bleu(data_ref, data_sys):
    # Compute the BLEU / METEOR / ROUGE metric
    coco_eval = run_coco_eval(data_ref, data_sys)
    scores = {metric: score for metric, score in list(coco_eval.eval.items())}
    return scores["Bleu_4"]


class GTDataLoader(DataLoader):

    def __init__(self, args, dataset, mode):
        if mode == "train":
            sampler = RandomSampler(dataset)
            batch_size = args.train_batch_size
        else:
            sampler = SequentialSampler(dataset)
            batch_size = args.predict_batch_size
        super(GTDataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size, num_workers=args.num_workers)


class GTUnlabelDataLoader(DataLoader):

    def __init__(self, args, dataset, mode):
        if mode == "train":
            sampler = RandomSampler(dataset)
            batch_size = args.train_batch_size_unlabel
        else:
            sampler = SequentialSampler(dataset)
            batch_size = args.predict_batch_size_unlabel
        super(GTUnlabelDataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size, num_workers=args.num_workers)


def run_coco_eval(data_ref, data_sys):
    """Run the COCO evaluator, return the resulting evaluation object (contains both
    system- and segment-level scores."""
    # convert references and system outputs to MS-COCO format in-memory
    coco_ref = create_coco_refs(data_ref)
    coco_sys = create_coco_sys(data_sys)

    print('Running MS-COCO evaluator...', file=sys.stderr)
    coco = COCO()
    coco.dataset = coco_ref
    coco.createIndex()

    coco_res = coco.loadRes(resData=coco_sys)
    coco_eval = COCOEvalCap(coco, coco_res)
    coco_eval.evaluate()

    return coco_eval


def create_coco_refs(data_ref):
    """Create MS-COCO human references JSON."""
    out = {'info': {}, 'licenses': [], 'images': [], 'type': 'captions', 'annotations': []}
    ref_id = 0
    for inst_id, refs in enumerate(data_ref):
        out['images'].append({'id': 'inst-%d' % inst_id})
        for ref in refs:
            out['annotations'].append({'image_id': 'inst-%d' % inst_id,
                                       'id': ref_id,
                                       'caption': ref})
            ref_id += 1
    return out


def create_coco_sys(data_sys):
    """Create MS-COCO system outputs JSON."""
    out = []
    for inst_id, inst in enumerate(data_sys):
        out.append({'image_id': 'inst-%d' % inst_id, 'caption': inst})
    return out
