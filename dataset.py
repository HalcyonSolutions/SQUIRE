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
import re


DIRECT_REVERSE_SUFFIX = "_reverse"


def _parse_paths_cell(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def _flatten_path_hops(paths):
    tgt_line = []
    for i, hop in enumerate(paths):
        hop = [str(token) for token in hop]
        if i == 0:
            tgt_line.extend(hop)
        else:
            tgt_line.extend(hop[1:])
    return tgt_line


def _is_wikidata_entity(token):
    return isinstance(token, str) and re.fullmatch(r"Q\d+", token) is not None


def _is_wikidata_relation(token):
    return isinstance(token, str) and re.fullmatch(r"P\d+", token) is not None


def _reverse_relation_token(relation):
    relation = str(relation)
    if relation.endswith(DIRECT_REVERSE_SUFFIX):
        return relation
    return f"{relation}{DIRECT_REVERSE_SUFFIX}"


def _normalize_label_text(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _iter_triple_file(path):
    if not os.path.exists(path):
        return
    with open(path) as fin:
        for line in fin:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            yield tuple(str(part) for part in parts)


def _infer_direct_id_mode(dataframe):
    if "Paths" not in dataframe.columns:
        return False
    for value in dataframe["Paths"]:
        try:
            paths = _parse_paths_cell(value)
        except (ValueError, SyntaxError):
            continue
        if not isinstance(paths, list) or not paths:
            continue
        tgt_line = _flatten_path_hops(paths)
        if len(tgt_line) < 3:
            continue
        return (
            _is_wikidata_entity(tgt_line[0])
            and _is_wikidata_relation(tgt_line[1])
            and _is_wikidata_entity(tgt_line[2])
        )
    return False


def _direct_vocab_is_compatible(dictionary, dataframe):
    sample_tokens = []
    for column in ("Source-Entity", "Answer-Entity"):
        if column not in dataframe.columns:
            continue
        for value in dataframe[column]:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            token = str(value)
            if _is_wikidata_entity(token):
                sample_tokens.append(token)
                break
    if "Paths" in dataframe.columns:
        for value in dataframe["Paths"]:
            try:
                paths = _parse_paths_cell(value)
            except (ValueError, SyntaxError):
                continue
            if not isinstance(paths, list) or not paths:
                continue
            tgt_line = _flatten_path_hops(paths)
            if len(tgt_line) < 3:
                continue
            sample_tokens.extend([tgt_line[0], tgt_line[1], tgt_line[2], _reverse_relation_token(tgt_line[1])])
            break
    return all(token in dictionary.indices for token in sample_tokens)


def _collect_direct_vocab_tokens(data_path, dataframe):
    entities = set()
    relations = set()

    for column in ("Source-Entity", "Answer-Entity"):
        if column not in dataframe.columns:
            continue
        for value in dataframe[column]:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            token = str(value)
            if _is_wikidata_entity(token):
                entities.add(token)

    if "Paths" in dataframe.columns:
        for value in dataframe["Paths"]:
            try:
                paths = _parse_paths_cell(value)
            except (ValueError, SyntaxError):
                continue
            if not isinstance(paths, list):
                continue
            tgt_line = _flatten_path_hops(paths)
            for i, token in enumerate(tgt_line):
                if i % 2 == 0 and _is_wikidata_entity(token):
                    entities.add(token)
                elif i % 2 == 1 and _is_wikidata_relation(token):
                    relations.add(token)

    for file_name in ("train.txt", "valid.txt", "test.txt", "triplets.txt"):
        for h, r, t in _iter_triple_file(os.path.join(data_path, file_name)):
            if _is_wikidata_entity(h):
                entities.add(h)
            if _is_wikidata_relation(r):
                relations.add(r)
            if _is_wikidata_entity(t):
                entities.add(t)

    reverse_relations = {_reverse_relation_token(relation) for relation in relations}
    return entities, relations, reverse_relations

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
        self.direct_id_mode = _infer_direct_id_mode(self.data)

        if Seq2SeqDataset._tokenizer is None:
            Seq2SeqDataset._tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer = Seq2SeqDataset._tokenizer

        self.max_q_len = getattr(args, "max_q_len", 32)
        self.vocab_file = vocab_file
        self.device = device
    
        try:
            self.dictionary = Dictionary.load(vocab_file)
            if self.direct_id_mode and not _direct_vocab_is_compatible(self.dictionary, self.data):
                raise ValueError("Existing vocab file does not match direct QID/PID tokens")
        except (FileNotFoundError, ValueError):
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
        if self.direct_id_mode:
            entities, relations, reverse_relations = _collect_direct_vocab_tokens(self.data_path, self.data)
            for relation in sorted(relations):
                self.dictionary.add_symbol(relation)
            for relation in sorted(reverse_relations):
                self.dictionary.add_symbol(relation)
            for entity in sorted(entities):
                self.dictionary.add_symbol(entity)
            self.dictionary.save(self.vocab_file)
            return
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
        self.id2entity = {}
        if self.direct_id_mode:
            self.relation2id = {}
            self.id2relation = {}

            def add_entity(token, label=None):
                token = str(token)
                if not _is_wikidata_entity(token):
                    return
                self.entity2id[token] = token
                label_text = _normalize_label_text(label)
                if label_text is not None:
                    self.id2entity[token] = label_text
                else:
                    self.id2entity.setdefault(token, token)

            def add_relation(token, label=None):
                token = str(token)
                if not _is_wikidata_relation(token):
                    return
                self.relation2id[token] = token
                label_text = _normalize_label_text(label)
                if label_text is not None:
                    self.id2relation[token] = label_text
                else:
                    self.id2relation.setdefault(token, token)

                rev_token = _reverse_relation_token(token)
                self.relation2id.setdefault(rev_token, rev_token)
                reverse_label = f"{self.id2relation[token]} (reverse)"
                if label_text is not None:
                    self.id2relation[rev_token] = reverse_label
                else:
                    self.id2relation.setdefault(rev_token, reverse_label)

            for _, row in self.data.iterrows():
                add_entity(row.get("Source-Entity"), row.get("Source"))
                add_entity(row.get("Answer-Entity"), row.get("Answer"))

                try:
                    paths = _parse_paths_cell(row.get("Paths"))
                except (ValueError, SyntaxError):
                    paths = []
                try:
                    path_labels = _parse_paths_cell(row.get("Paths-Label"))
                except (ValueError, SyntaxError):
                    path_labels = []

                if not isinstance(paths, list):
                    continue
                if not isinstance(path_labels, list):
                    path_labels = []

                for i, hop in enumerate(paths):
                    if not isinstance(hop, (list, tuple)) or len(hop) < 3:
                        continue
                    label_hop = path_labels[i] if i < len(path_labels) else ()
                    if not isinstance(label_hop, (list, tuple)):
                        label_hop = ()
                    add_entity(hop[0], label_hop[0] if len(label_hop) > 0 else None)
                    add_relation(hop[1], label_hop[1] if len(label_hop) > 1 else None)
                    add_entity(hop[2], label_hop[2] if len(label_hop) > 2 else None)

            for column in ("Source-Entity", "Answer-Entity"):
                if column not in self.data.columns:
                    continue
                for value in self.data[column]:
                    if value is None or (isinstance(value, float) and pd.isna(value)):
                        continue
                    token = str(value)
                    if _is_wikidata_entity(token):
                        self.entity2id[token] = token
                        self.id2entity.setdefault(token, token)

            if "Paths" in self.data.columns:
                for value in self.data["Paths"]:
                    try:
                        paths = _parse_paths_cell(value)
                    except (ValueError, SyntaxError):
                        continue
                    if not isinstance(paths, list):
                        continue
                    tgt_line = _flatten_path_hops(paths)
                    for i, token in enumerate(tgt_line):
                        if i % 2 == 0 and _is_wikidata_entity(token):
                            self.entity2id[token] = token
                            self.id2entity.setdefault(token, token)
                        elif i % 2 == 1 and _is_wikidata_relation(token):
                            self.relation2id[token] = token
                            self.id2relation.setdefault(token, token)
                            rev_token = _reverse_relation_token(token)
                            self.relation2id.setdefault(rev_token, rev_token)
                            self.id2relation.setdefault(rev_token, f"{self.id2relation[token]} (reverse)")
            return

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
            self.relation2id[r] = 'R' + rid
            self.id2relation["R" + rid] = r
            self.id2relation["R" + str(int(rid) + num_relations)] = f"{r} (reverse)"

    def __getitem__(self, index):
        row = self.data.iloc[index]
        question = str(row["Question"])

        paths = _parse_paths_cell(row["Paths"])
        tgt_line = _flatten_path_hops(paths)
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
            if i % 2 == 0:
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

        # Keep worker processes CPU-only. The training loop moves tensor batches
        # to the target device after DataLoader returns them.
        return {
            "ids": torch.LongTensor(ids),
            "lengths": torch.LongTensor(lens),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prev_outputs": prev_outputs,
            "target": target,
            "mask": mask,
        }

    def _add_valid_triple(self, valid_dict, h, r, t, vocab_size, eos):
        hid = self.dictionary.indices[h]
        rid = self.dictionary.indices[r]
        tid = self.dictionary.indices[t]
        e = hid
        er = vocab_size * rid + hid
        if e not in valid_dict:
            if self.smart_filter:
                valid_dict[e] = -30 * torch.ones([vocab_size])
            else:
                valid_dict[e] = [eos, ]
        if er not in valid_dict:
            if self.smart_filter:
                valid_dict[er] = -30 * torch.ones([vocab_size])
            else:
                valid_dict[er] = []
        if self.smart_filter:
            valid_dict[e][rid] = 0
            valid_dict[e][eos] = 0
            valid_dict[er][tid] = 0
        else:
            valid_dict[e].append(rid)
            valid_dict[er].append(tid)

    def get_next_valid(self):
        if self.direct_id_mode:
            train_valid = dict()
            eval_valid = dict()
            vocab_size = len(self.dictionary)
            eos = self.dictionary.eos()

            def add_raw_triples(file_name, valid_dict):
                path = os.path.join(self.data_path, file_name)
                if not os.path.exists(path):
                    return
                with open(path, 'r') as f:
                    for line in tqdm(f):
                        parts = line.strip().split('\t')
                        if len(parts) != 3:
                            continue
                        h, r, t = (str(part) for part in parts)
                        if h not in self.dictionary.indices or r not in self.dictionary.indices or t not in self.dictionary.indices:
                            continue
                        self._add_valid_triple(valid_dict, h, r, t, vocab_size, eos)
                        rev_r = _reverse_relation_token(r)
                        if rev_r in self.dictionary.indices:
                            self._add_valid_triple(valid_dict, t, rev_r, h, vocab_size, eos)

            train_file = "train.txt" if os.path.exists(os.path.join(self.data_path, "train.txt")) else "triplets.txt"
            add_raw_triples(train_file, train_valid)
            add_raw_triples(train_file, eval_valid)
            add_raw_triples("valid.txt", eval_valid)
            add_raw_triples("test.txt", eval_valid)
            return train_valid, eval_valid

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
        self.direct_id_mode = _infer_direct_id_mode(self.data)
        if Seq2SeqDataset._tokenizer is None:
            Seq2SeqDataset._tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer = Seq2SeqDataset._tokenizer
        self.max_q_len = getattr(args, "max_q_len", 32)
        self.vocab_file = vocab_file
        self.device = device
    
        try:
            self.dictionary = Dictionary.load(vocab_file)
            if self.direct_id_mode and not _direct_vocab_is_compatible(self.dictionary, self.data):
                raise ValueError("Existing vocab file does not match direct QID/PID tokens")
        except (FileNotFoundError, ValueError):
            self.dictionary = Dictionary()
            Seq2SeqDataset._init_vocab(self)
        Seq2SeqDataset._load_mappings(self)
        self.padding_idx = self.dictionary.pad()
        self.len_vocab = len(self.dictionary)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        question = str(row["Question"])
        answer = str(row["Answer-Entity"])
        head = str(row["Source-Entity"])
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

        head_token = head_id
        head_id = self.dictionary.indices.get(head_token)
        if head_id is None:
            raise ValueError(f"Head token not in dictionary: {head_token}")

        target_id = self.dictionary.encode_line([answer_id])[:-1]
        return {
            "id": index,
            "input_ids": encoded_question["input_ids"].squeeze(0),
            "attention_mask": encoded_question["attention_mask"].squeeze(0),
            "target": target_id,
            "head_id": torch.tensor(head_id, dtype=torch.long),
        }

    def collate_fn(self, samples):
        bsz = len(samples)
        input_ids = torch.stack([sample["input_ids"] for sample in samples])
        attention_mask = torch.stack([sample["attention_mask"] for sample in samples])
        target = torch.LongTensor(bsz, 1)
        head_id = torch.LongTensor(bsz)

        ids =  []
        for idx, sample in enumerate(samples):
            ids.append(sample["id"])
            target_ids = sample["target"]
            input_ids[idx] = sample["input_ids"]
            attention_mask[idx] = sample["attention_mask"]
            target[idx, 0] = target_ids[0]
            head_id[idx] = sample["head_id"]
        
        # Keep worker processes CPU-only. The evaluation loop moves tensor
        # batches to the target device after DataLoader returns them.
        return {
            "ids": torch.LongTensor(ids),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target": target,
            "head_id": head_id,
        }
