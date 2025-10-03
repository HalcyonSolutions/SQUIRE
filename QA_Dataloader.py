# Reusable supervised QA dataset and collate
from typing import List, Dict, Any, Optional, Tuple, Iterator
import ast
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from functools import partial
from torch.utils.data import IterableDataset, get_worker_info
import csv, io, os

def load_index_column_wise(path: str) -> Tuple[Dict[int, str], Dict[str, int]]:
    # Same behavior as multihopkg.data_utils.load_index_column_wise
    id2item, item2id = {}, {}
    with open(path) as f:
        for line in f:
            item, idx = line.strip().split()
            idx = int(idx)
            id2item[idx] = item
            item2id[item] = idx
    return id2item, item2id

class SupervisedQADataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        entity2id: Dict[str, int],
        relation2id: Dict[str, int],
        question_tokenizer_name: str,
        answer_tokenizer_name: Optional[str] = None,  # defaults to question tokenizer
        split: Optional[str] = None,
        split_column: str = "SplitLabel",
        text_only: bool = False
    ):
        self.df = pd.read_csv(csv_path)
        assert "Question" in self.df.columns and "Answer" in self.df.columns, "Missing Question/Answer columns"
        assert "Source-Entity" in self.df.columns and "Query-Relation" in self.df.columns and "Answer-Entity" in self.df.columns, \
            "Missing Source-Entity/Query-Relation/Answer-Entity columns"
        
        self.text_only = text_only

        # Optional split filtering
        self.split = split.lower() if split else None
        if self.split is not None:
            assert split_column in self.df.columns, f"Missing {split_column} column"
            norm_split = self.df[split_column].astype(str).str.strip().str.lower()
            self.df = self.df[norm_split == self.split].reset_index(drop=True)
            if len(self.df) == 0:
                raise ValueError(f"No rows found for split='{self.split}' in column '{split_column}'")

        if not self.text_only:
            # Tokenizers
            self.qtok = AutoTokenizer.from_pretrained(question_tokenizer_name)
            self.atok = AutoTokenizer.from_pretrained(answer_tokenizer_name or question_tokenizer_name)

            # Tokenize Q/A
            self.df["enc_question"] = self.df["Question"].map(
                lambda x: self.qtok.encode(x, add_special_tokens=True)
            )
            self.df["enc_answer"] = self.df["Answer"].map(
            lambda x: self._safe_encode_answer(x)
            )
        else:
            self.qtok = None
            self.atok = None

        # Map entity/relation ids
        self.df["Source-Entity"] = self.df["Source-Entity"].map(lambda e: entity2id[e])
        self.df["Query-Relation"] = self.df["Query-Relation"].map(lambda r: relation2id[r])
        self.df["Answer-Entity"] = self.df["Answer-Entity"].map(lambda e: entity2id[e])

        # Optional columns
        if "Paths" in self.df.columns:
            # Accept stringified python lists or already-parsed lists
            def _to_path_ids(val):
                hops = ast.literal_eval(val) if isinstance(val, str) else val
                return [[entity2id[h], relation2id[r], entity2id[t]] for (h, r, t) in hops]
            self.df["Paths"] = self.df["Paths"].map(_to_path_ids)
        else:
            self.df["Paths"] = [[] for _ in range(len(self.df))]

        if "Hops" not in self.df.columns:
            self.df["Hops"] = self.df["Paths"].map(lambda hops: len(hops))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        if self.text_only:
            return {
                "question_text": str(row["Question"]),
                "answer_text": str(row["Answer"]),
                "query_ent": int(row["Source-Entity"]),
                "query_rel": int(row["Query-Relation"]),
                "answer_ent": int(row["Answer-Entity"]),
                "paths": row["Paths"],
                "hops": int(row["Hops"]),
            }
        else:
            return {
                "question_ids": torch.tensor(row["enc_question"], dtype=torch.long),
                "answer_ids": torch.tensor(row["enc_answer"], dtype=torch.long),
                "query_ent": int(row["Source-Entity"]),
                "query_rel": int(row["Query-Relation"]),
                "answer_ent": int(row["Answer-Entity"]),
                "paths": row["Paths"],  # List[List[int]] of [head, rel, tail]
                "hops": int(row["Hops"]),
            }

    def _safe_encode_answer(self, answer):
        try:
            # Fallback to CLS/SEP if BOS/EOS are None
            bos = getattr(self.atok, "bos_token_id", None) or getattr(self.atok, "cls_token_id", None)
            eos = getattr(self.atok, "eos_token_id", None) or getattr(self.atok, "sep_token_id", None)
            if bos is None or eos is None:
                raise ValueError("Tokenizer missing BOS/EOS (or CLS/SEP) tokens.")
            encoded = [bos] + self.atok.encode(answer, add_special_tokens=False) + [eos]
            return encoded
        except Exception as e:
            print(f"Error encoding answer: {answer}")
            print(f"Exception: {e}")
            return None

# def make_train_test_dataloaders(
#     csv_path: str,
#     entity2id: Dict[str, int],
#     relation2id: Dict[str, int],
#     question_tokenizer_name: str,
#     answer_tokenizer_name: Optional[str] = None,
#     batch_size: int = 8,
#     num_workers: int = 0,
#     pin_memory: bool = False,
#     text_only: bool = False,
#     ) -> Dict[str, DataLoader]:

#     train_ds = SupervisedQADataset(
#         csv_path, entity2id, relation2id, question_tokenizer_name, answer_tokenizer_name, split="train"
#     )
#     test_ds = SupervisedQADataset(
#         csv_path, entity2id, relation2id, question_tokenizer_name, answer_tokenizer_name, split="test"
#     )
#     train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=qa_collate,
#                         num_workers=num_workers, pin_memory=pin_memory)
#     test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=qa_collate,
#                         num_workers=num_workers, pin_memory=pin_memory)
#     return {"train": train_dl, "test": test_dl}

def make_train_test_dataloaders(
    csv_path: str,
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
    question_tokenizer_name: str,
    answer_tokenizer_name: Optional[str] = None,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = False,
    text_only: bool = False,
    worker_init_fn: Optional[Any] = None,
) -> Dict[str, DataLoader]:

    train_ds = SupervisedQADataset(
        csv_path, entity2id, relation2id, question_tokenizer_name, answer_tokenizer_name,
        split="train", text_only=text_only
    )
    test_ds = SupervisedQADataset(
        csv_path, entity2id, relation2id, question_tokenizer_name, answer_tokenizer_name,
        split="test", text_only=text_only
    )
    
    if not text_only:
        q_pad = train_ds.qtok.pad_token_id or 1   # fallback if None
        a_pad = train_ds.atok.pad_token_id or 1
    else:
        q_pad = 0
        a_pad = 0

    collate = qa_collate_factory(q_pad, a_pad)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                        collate_fn=collate, num_workers=num_workers, pin_memory=pin_memory)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                        collate_fn=collate, num_workers=num_workers, pin_memory=pin_memory)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate,
                          num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=worker_init_fn)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate,
                         num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=worker_init_fn)
    return {"train": train_dl, "test": test_dl}

# def qa_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
#     # Pad Q/A; keep others as lists
#     pad_q = 0  # adjust if your tokenizer uses a different pad id for questions
#     pad_a = 0
#     max_q = max(item["question_ids"].size(0) for item in batch)
#     max_a = max(item["answer_ids"].size(0) for item in batch)

#     q_ids = torch.stack([
#         torch.nn.functional.pad(item["question_ids"], (0, max_q - item["question_ids"].size(0)), value=pad_q)
#         for item in batch
#     ])
#     a_ids = torch.stack([
#         torch.nn.functional.pad(item["answer_ids"], (0, max_a - item["answer_ids"].size(0)), value=pad_a)
#         for item in batch
#     ])

#     return {
#         "question_ids": q_ids,                      # (B, Lq)
#         "answer_ids": a_ids,                        # (B, La) with BOS/EOS
#         "query_ent": torch.tensor([b["query_ent"] for b in batch], dtype=torch.long),
#         "query_rel": torch.tensor([b["query_rel"] for b in batch], dtype=torch.long),
#         "answer_ent": torch.tensor([b["answer_ent"] for b in batch], dtype=torch.long),
#         "paths": [b["paths"] for b in batch],       # ragged: List[List[[h,r,t], ...]]
#         "hops": torch.tensor([b["hops"] for b in batch], dtype=torch.long),
#     }


# def qa_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
#     # Text-only passthrough
#     if "question_text" in batch[0]:
#         return {
#             "question_text": [b["question_text"] for b in batch],
#             "answer_text": [b["answer_text"] for b in batch],
#             "query_ent": torch.tensor([b["query_ent"] for b in batch], dtype=torch.long),
#             "query_rel": torch.tensor([b["query_rel"] for b in batch], dtype=torch.long),
#             "answer_ent": torch.tensor([b["answer_ent"] for b in batch], dtype=torch.long),
#             "paths": [b["paths"] for b in batch],
#             "hops": torch.tensor([b["hops"] for b in batch], dtype=torch.long),
#         }

#     # inside qa_collate (IDs branch)
#     pad_q = getattr(batch[0].get("question_ids"), "new_empty", lambda: None)  # just to avoid unused
#     # Better:
#     # Pass tokenizers into collate or store pad ids with the samples.
#     # Quick fix using the atok/qtok pad id if you thread them in:
#     pad_q = qtok.pad_token_id or 1
#     pad_a = atok.pad_token_id or 1
#     max_q = max(item["question_ids"].size(0) for item in batch)
#     max_a = max(item["answer_ids"].size(0) for item in batch)

#     q_ids = torch.stack([
#         torch.nn.functional.pad(item["question_ids"], (0, max_q - item["question_ids"].size(0)), value=pad_q)
#         for item in batch
#     ])
#     a_ids = torch.stack([
#         torch.nn.functional.pad(item["answer_ids"], (0, max_a - item["answer_ids"].size(0)), value=pad_a)
#         for item in batch
#     ])

#     q_mask = (q_ids != pad_q).long()
#     a_mask = (a_ids != pad_a).long()

#     return {
#         "question_ids": q_ids,
#         "question_attention_mask": q_mask,
#         "answer_ids": a_ids,
#         "answer_attention_mask": a_mask,
#         "query_ent": torch.tensor([b["query_ent"] for b in batch], dtype=torch.long),
#         "query_rel": torch.tensor([b["query_rel"] for b in batch], dtype=torch.long),
#         "answer_ent": torch.tensor([b["answer_ent"] for b in batch], dtype=torch.long),
#         "paths": [b["paths"] for b in batch],
#         "hops": torch.tensor([b["hops"] for b in batch], dtype=torch.long),
#     }

def qa_collate_factory(q_pad_id: int, a_pad_id: int):
    def qa_collate(batch):
        # Text-only passthrough
        if "question_text" in batch[0]:
            return {
                "question_text": [b["question_text"] for b in batch],
                "answer_text": [b["answer_text"] for b in batch],
                "query_ent": torch.tensor([b["query_ent"] for b in batch], dtype=torch.long),
                "query_rel": torch.tensor([b["query_rel"] for b in batch], dtype=torch.long),
                "answer_ent": torch.tensor([b["answer_ent"] for b in batch], dtype=torch.long),
                "paths": [b["paths"] for b in batch],
                "hops": torch.tensor([b["hops"] for b in batch], dtype=torch.long),
            }

        pad_q = q_pad_id
        pad_a = a_pad_id
        max_q = max(item["question_ids"].size(0) for item in batch)
        max_a = max(item["answer_ids"].size(0) for item in batch)

        q_ids = torch.stack([
            torch.nn.functional.pad(item["question_ids"], (0, max_q - item["question_ids"].size(0)), value=pad_q)
            for item in batch
        ])
        a_ids = torch.stack([
            torch.nn.functional.pad(item["answer_ids"], (0, max_a - item["answer_ids"].size(0)), value=pad_a)
            for item in batch
        ])

        q_mask = (q_ids != pad_q).long()
        a_mask = (a_ids != pad_a).long()

        return {
            "question_ids": q_ids,
            "question_attention_mask": q_mask,
            "answer_ids": a_ids,
            "answer_attention_mask": a_mask,
            "query_ent": torch.tensor([b["query_ent"] for b in batch], dtype=torch.long),
            "query_rel": torch.tensor([b["query_rel"] for b in batch], dtype=torch.long),
            "answer_ent": torch.tensor([b["answer_ent"] for b in batch], dtype=torch.long),
            "paths": [b["paths"] for b in batch],
            "hops": torch.tensor([b["hops"] for b in batch], dtype=torch.long),
        }
    return qa_collate

# Example usage
# id2ent, ent2id = load_index_column_wise("data/KinshipHinton/entity2id.txt")
# id2rel, rel2id = load_index_column_wise("data/KinshipHinton/relation2id.txt")
# ds = SupervisedQADataset("data/KinshipHinton/kinship_hinton_qa_2hop.csv", ent2id, rel2id, "facebook/bart-base")
# dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=qa_collate)
def seed_worker(worker_id):
    torch.manual_seed(42)
if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    id2ent, ent2id = load_index_column_wise("/home/nura/projects/SQUIRE/data/kinshiphinton/entity2id.txt")
    id2rel, rel2id = load_index_column_wise("/home/nura/projects/SQUIRE/data/kinshiphinton/relation2id.txt")
    # ds = SupervisedQADataset("/home/nura/projects/SQUIRE/data/kinshiphinton/kinship_hinton_qa_2hop.csv", ent2id, rel2id, "facebook/bart-base")
    # ds = SupervisedQADataset("/home/nura/projects/SQUIRE/data/kinshiphinton/kinship_hinton_qa_2hop.csv", ent2id, rel2id, "distilbert-base-uncased")
    # dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=qa_collate)
    batch_size = 32
    loaders = make_train_test_dataloaders(
        "/home/nura/projects/SQUIRE/data/kinshiphinton/kinship_hinton_qa_2hop.csv",
        ent2id, rel2id,
        question_tokenizer_name="facebook/bart-base",
        answer_tokenizer_name=None,
        batch_size=batch_size,
        text_only=False,
        worker_init_fn=seed_worker,
    )

    # for batch in dl:
    #     print(batch)
    #     break
    for split, dl in loaders.items():
        print(f"{split} batches:")
        for batch in dl:
            print({k: (v.shape if torch.is_tensor(v) else type(v)) for k, v in batch.items()})
            break
    
    print('\ntrain_loader content...\n')
    for train_loader in loaders["train"]:
        train_keys = train_loader.keys()
        print(f"Train loader batch size: {batch_size}")
        print(f"Train loader keys: {train_keys}\n")

        for key in train_keys:
            print(f"Type of {key}: {type(train_loader[key])}")
        print()
        for key in train_keys:
            value = train_loader[key]
            if isinstance(value, torch.Tensor):
                print(f"Shape of {key}: {value.shape}")
            else:
                print(f"Length of {key}: {len(value)}")
        print()
        # print out the first element of each key
        for key in train_keys:
            value = train_loader[key]
            print(f"First element of {key}: {value[0]}")
        break
