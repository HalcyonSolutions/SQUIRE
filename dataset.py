import json
import os
import copy
from torch.utils.data import Dataset
from dictionary import Dictionary
import torch
import sys
import numpy as np
import networkx as nx
from tqdm import tqdm
import random
import pandas as pd
from transformers import BertTokenizer
import ast

class Seq2SeqDataset(Dataset):
    _tokenizer = None

    def __init__(self, data_path="FB15K237/", vocab_file="FB15K237/vocab.txt", device="cpu", args=None, split: str = None):
        self.data_path = data_path

        csv_file = getattr(args, "question_file", None)
        self.csv_file =os.path.join(data_path, csv_file) if csv_file else None 
        if self.csv_file is None:
            raise ValueError("args.question_file is required")

        self.data = pd.read_csv(self.csv_file)
        if split is not None:
            self.data = self.data[self.data["SplitLabel"] == split].reset_index(drop=True)

        if Seq2SeqDataset._tokenizer is None:
            Seq2SeqDataset._tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer = Seq2SeqDataset._tokenizer

        self.max_q_len = getattr(args, "max_q_len", 32)
        self.vocab_file = vocab_file
        self.device = device
    
        try:
            self.dictionary = Dictionary.load(vocab_file)
        except FileNotFoundError:
            self.dictionary = Dictionary()
            self._init_vocab()
        self._load_mappings()

        self.padding_idx = self.dictionary.pad()
        self.len_vocab = len(self.dictionary)
        self.smart_filter = args.smart_filter
        self.args = args
    
    def __len__(self):
        return len(self.data)

    def _init_vocab(self):
        self.dictionary.add_symbol('LOOP')
        N = 0
        with open(self.data_path+'relation2id.txt') as fin:
            for line in fin:
                N += 1
        with open(self.data_path+'relation2id.txt') as fin:
            for line in fin:
                r, rid = line.strip().split('\t')
                rev_rid = int(rid) + N # adding reverse relations IDs
                self.dictionary.add_symbol('R'+rid)
                self.dictionary.add_symbol('R'+str(rev_rid))
        with open(self.data_path+'entity2id.txt') as fin:
            for line in fin:
                e, eid = line.strip().split('\t')
                self.dictionary.add_symbol(eid)
        self.dictionary.save(self.vocab_file)
    
    def _load_mappings(self):
        self.entity2id = {}
        with open(self.data_path + "entity2id.txt") as f:
            for line in f:
                e, eid = line.strip().split('\t')
                self.entity2id[e] = eid

        self.relation2id = {}
        with open(self.data_path + "relation2id.txt") as f:
            for line in f:
                r, rid = line.strip().split('\t')
                self.relation2id[r] = 'R' + rid

    def __getitem__(self, index):
        row = self.data.iloc[index]
        question = str(row["Question"])

        paths = ast.literal_eval(row["Paths"])
        tgt_line = []
        for i, hop in enumerate(paths):
            if i == 0:
                tgt_line.extend(hop)
            else:
                tgt_line.extend(hop[1:])
        tgt_line_ids = []
        for i, token in enumerate(tgt_line):
            if i % 2 == 0:  # entity
                try:
                    tgt_line_ids.append(self.entity2id[token])
                except KeyError:
                    raise ValueError(f"Unknown entity: {token}")
            else:  # relation
                try:
                    tgt_line_ids.append(self.relation2id[token])
                except KeyError:
                    raise ValueError(f"Unknown relation: {token}")

        encoded_question = self.tokenizer(
            question,
            padding='max_length',
            truncation=True,
            max_length=self.max_q_len,
            return_tensors='pt'
        )

        target_id = self.dictionary.encode_line(tgt_line_ids)
        l = len(target_id)
        mask = torch.ones_like(target_id)
        for i in range(0, l-2):
            if i % 2 == 0: # do not mask relation
                continue
            if random.random() < self.args.prob: # randomly replace with prob
                target_id[i] = random.randint(0, self.len_vocab - 1)
                mask[i] = 0
        return {
            "id": index,
            "tgt_length": len(target_id),
            "input_ids": encoded_question["input_ids"].squeeze(0),
            "attention_mask": encoded_question["attention_mask"].squeeze(0),
            "target": target_id,
            "mask": mask,
        }

    def collate_fn(self, samples):
        lens = [sample["tgt_length"] for sample in samples]
        max_len = max(lens)
        bsz = len(lens)

        # input_ids = torch.LongTensor(bsz, self.max_q_len)
        # attention_mask = torch.LongTensor(bsz, self.max_q_len)
        input_ids = torch.stack([s["input_ids"] for s in samples])
        attention_mask = torch.stack([s["attention_mask"] for s in samples])

        prev_outputs = torch.LongTensor(bsz, max_len)
        mask = torch.zeros(bsz, max_len)
 
        prev_outputs.fill_(self.dictionary.pad())
        prev_outputs[:, 0].fill_(self.dictionary.bos())
        target = copy.deepcopy(prev_outputs)

        ids =  []
        for idx, sample in enumerate(samples):
            ids.append(sample["id"])
            target_ids = sample["target"]
            input_ids[idx] = sample["input_ids"]
            attention_mask[idx] = sample["attention_mask"]
            prev_outputs[idx, 1:sample["tgt_length"]] = target_ids[: -1]
            target[idx, 0: sample["tgt_length"]] = target_ids
            mask[idx, 0: sample["tgt_length"]] = sample["mask"]

        return {
            "ids": torch.LongTensor(ids).to(self.device),
            "lengths": torch.LongTensor(lens).to(self.device),
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "prev_outputs": prev_outputs.to(self.device),
            "target": target.to(self.device),
            "mask": mask.to(self.device),
        }

    def get_next_valid(self):
        train_valid = dict()
        eval_valid = dict()
        vocab_size = len(self.dictionary)
        eos = self.dictionary.eos()
        with open(self.data_path+'train_triples_rev.txt', 'r') as f:
            for line in tqdm(f):
                h, r, t = line.strip().split('\t')
                hid = self.dictionary.indices[h]
                rid = self.dictionary.indices[r]
                tid = self.dictionary.indices[t]
                e = hid
                er = vocab_size * rid + hid
                if e not in train_valid:
                    if self.smart_filter:
                        train_valid[e] = -30 * torch.ones([vocab_size])
                    else:
                        train_valid[e] = [eos, ]
                if er not in train_valid:
                    if self.smart_filter:
                        train_valid[er] = -30 * torch.ones([vocab_size])
                    else:
                        train_valid[er] = []
                if self.smart_filter:
                    train_valid[e][rid] = 0
                    train_valid[e][eos] = 0
                    train_valid[er][tid] = 0
                else:
                    train_valid[e].append(rid)
                    train_valid[er].append(tid)
        with open(self.data_path+'train_triples_rev.txt', 'r') as f:
            for line in tqdm(f):
                h, r, t = line.strip().split('\t')
                hid = self.dictionary.indices[h]
                rid = self.dictionary.indices[r]
                tid = self.dictionary.indices[t]
                e = hid
                er = vocab_size * rid + hid
                if e not in eval_valid:
                    if self.smart_filter:
                        eval_valid[e] = -30 * torch.ones([vocab_size])
                    else:
                        eval_valid[e] = [eos, ]
                if er not in eval_valid:
                    if self.smart_filter:
                        eval_valid[er] = -30 * torch.ones([vocab_size])
                    else:
                        eval_valid[er] = []
                if self.smart_filter:
                    eval_valid[e][rid] = 0
                    eval_valid[e][eos] = 0
                    eval_valid[er][tid] = 0
                else:
                    eval_valid[e].append(rid)
                    eval_valid[er].append(tid)
        with open(self.data_path+'valid_triples_rev.txt', 'r') as f:
            for line in tqdm(f):
                h, r, t = line.strip().split('\t')
                hid = self.dictionary.indices[h]
                rid = self.dictionary.indices[r]
                tid = self.dictionary.indices[t]
                er = vocab_size * rid + hid
                if er not in eval_valid:
                    if self.smart_filter:
                        eval_valid[er] = -30 * torch.ones([vocab_size])
                    else:
                        eval_valid[er] = []
                if self.smart_filter:
                    eval_valid[er][tid] = 0
                else:
                    eval_valid[er].append(tid)
        with open(self.data_path+'test_triples_rev.txt', 'r') as f:
            for line in tqdm(f):
                h, r, t = line.strip().split('\t')
                hid = self.dictionary.indices[h]
                rid = self.dictionary.indices[r]
                tid = self.dictionary.indices[t]
                er = vocab_size * rid + hid
                if er not in eval_valid:
                    if self.smart_filter:
                        eval_valid[er] = -30 * torch.ones([vocab_size])
                    else:
                        eval_valid[er] = []
                if self.smart_filter:
                    eval_valid[er][tid] = 0
                else:
                    eval_valid[er].append(tid)
        return train_valid, eval_valid
                
class TestDataset(Dataset):
    def __init__(self, data_path="FB15K237/", vocab_file="FB15K237/vocab.txt", device="cpu", src_file=None, args=None, split: str = None):
        self.data_path = data_path
        csv_file = getattr(args, "question_file", None)
        self.csv_file = os.path.join(data_path, csv_file) if csv_file else None
        if self.csv_file is None:
            raise ValueError("args.question_file is required")

        self.data = pd.read_csv(self.csv_file)
        if split is not None:
            self.data = self.data[self.data["SplitLabel"] == split].reset_index(drop=True)
        if Seq2SeqDataset._tokenizer is None:
            Seq2SeqDataset._tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer = Seq2SeqDataset._tokenizer
        self.max_q_len = getattr(args, "max_q_len", 32)
        self.vocab_file = vocab_file
        self.device = device
    
        try:
            self.dictionary = Dictionary.load(vocab_file)
        except FileNotFoundError:
            self.dictionary = Dictionary()
            self._init_vocab()
        self.entity2id = {}
        self.id2entity = {}
        with open(self.data_path + "entity2id.txt") as f:
            for line in f:
                e, eid = line.strip().split('\t')
                self.entity2id[e] = eid
                self.id2entity[eid] = e
        self.relation2id = {}
        self.id2relation = {}
        relation_rows = []
        with open(self.data_path + "relation2id.txt") as f:
            for line in f:
                r, rid = line.strip().split('\t')
                relation_rows.append((r, rid))
        num_relations = len(relation_rows)
        for r, rid in relation_rows:
            self.relation2id[r] = rid
            self.id2relation["R" + rid] = r
            self.id2relation["R" + str(int(rid) + num_relations)] = f"{r} (reverse)"
        self.padding_idx = self.dictionary.pad()
        self.len_vocab = len(self.dictionary)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        question = str(row["Question"])
        answer = str(row["Answer-Entity"])
        head = str(row["Source-Entity"])
        relation = str(row["Query-Relation"])
        encoded_question = self.tokenizer(
            question,
            padding='max_length',
            truncation=True,
            max_length=self.max_q_len,
            return_tensors='pt'
        )
        try:
            answer_id = self.entity2id[answer]
        except KeyError:
            raise ValueError(f"Unknown entity: {answer}")

        try:
            head_id = self.entity2id[head]
        except KeyError:
            raise ValueError(f"Unknown head entity: {head}")

        try:
            relation_id = self.relation2id[relation]
            relation_id = "R"+relation_id
        except KeyError:
            raise ValueError(f"Unknown head entity: {head}")

        head_id = self.dictionary.indices.get(head_id)
        if head_id is None:
            raise ValueError(f"Head token not in dictionary: {head_id}")
        relation_id = self.dictionary.indices.get(relation_id)
        if relation_id is None:
            raise ValueError(f"Relation token not in dictionary: {relation_id}")

        target_id = self.dictionary.encode_line([answer_id])[:-1]
        return {
            "id": index,
            "input_ids": encoded_question["input_ids"].squeeze(0),
            "attention_mask": encoded_question["attention_mask"].squeeze(0),
            "target": target_id,
            "head_id": torch.tensor(head_id, dtype=torch.long),
            "relation_id": torch.tensor(relation_id, dtype=torch.long)
        }

    def collate_fn(self, samples):
        bsz = len(samples)
        input_ids = torch.stack([sample["input_ids"] for sample in samples])
        attention_mask = torch.stack([sample["attention_mask"] for sample in samples])
        target = torch.LongTensor(bsz, 1)
        head_id = torch.LongTensor(bsz)
        relation_id = torch.LongTensor(bsz)

        ids =  []
        for idx, sample in enumerate(samples):
            ids.append(sample["id"])
            target_ids = sample["target"]
            input_ids[idx] = sample["input_ids"]
            attention_mask[idx] = sample["attention_mask"]
            target[idx, 0] = target_ids[0]
            head_id[idx] = sample["head_id"]
            relation_id[idx] = sample["relation_id"]
        
        return {
            "ids": torch.LongTensor(ids).to(self.device),
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "target": target.to(self.device),
            "head_id": head_id.to(self.device),
            "relation_id": relation_id.to(self.device),
        }
