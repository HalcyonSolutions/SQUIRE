import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from dataset import Seq2SeqDataset, TestDataset
from model import TransformerModel
import argparse
import numpy as np
import os
import random
from tqdm import tqdm
import logging
import ast
import transformers
import pandas as pd
from iterative_training import Iter_trainer
import math
import matplotlib
from typing import Set, Tuple, Sequence, Dict

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-dim", default=256, type=int)
    parser.add_argument("--hidden-size", default=512, type=int)
    parser.add_argument("--num-layers", default=6, type=int)
    parser.add_argument("--batch-size", default=1024, type=int)
    parser.add_argument("--test-batch-size", default=16, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--weight-decay", default=0, type=float)
    parser.add_argument("--num-epoch", default=20, type=int)
    parser.add_argument("--save-interval", default=10, type=int)
    parser.add_argument("--save-dir", default="model_1")
    parser.add_argument("--ckpt", default="ckpt_30.pt")
    parser.add_argument("--dataset", default="FB15K237")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--label-smooth", default=0.5, type=float)
    parser.add_argument("--l-punish", default=False, action="store_true") # during generation, add punishment for length
    parser.add_argument("--beam-size", default=128, type=int) # during generation, beam size
    parser.add_argument("--no-filter-gen", default=False, action="store_true") # during generation, not filter unreachable next token
    parser.add_argument("--test", default=False, action="store_true") # for test mode
    parser.add_argument("--encoder", default=False, action="store_true") # only use TransformerEncoder
    parser.add_argument("--trainset", default="6_rev_rule")
    parser.add_argument("--loop", default=False, action="store_true") # add self-loop instead of <eos>
    parser.add_argument("--prob", default=0, type=float) # ratio of replaced token
    parser.add_argument("--max-len", default=3, type=int) # maximum number of hops considered
    parser.add_argument("--iter", default=False, action="store_true") # switch for iterative training
    parser.add_argument("--iter-batch-size", default=128, type=int)
    parser.add_argument("--smart-filter", default=False, action="store_true") # more space consumed, less time; switch on when --filter-gen
    parser.add_argument("--warmup", default=3, type=float) # warmup steps ratio
    parser.add_argument("--self-consistency", default=False, action="store_true") # self-consistency
    parser.add_argument("--output-path", default=False, action="store_true") # output top correct path in a file (for interpretability evaluation)
    parser.add_argument("--validate-during-training", dest="validate_during_training", action="store_true", help="run train/valid evaluation during training")
    parser.add_argument("--validate-interval", default=5, type=int, help="run train/valid evaluation every N epochs when validation is enabled")
    
    # question input related
    parser.add_argument("--question-file", default="kinship_hinton_qa_nhop.csv", type=str, help="path to question file for question input csv file")
    parser.add_argument("--max-q-len", default=32, type=int, help="maximum number of tokens for the question") # used for Bert
    parser.add_argument("--num-workers", default=0, type=int, help="number of DataLoader worker processes, CPU-only when > 0; set to 0 to disable multiprocessing")
    parser.add_argument("--eval-preview-count", default=0, type=int, help="number of readable evaluation examples to print per split; set to 0 to disable")
    parser.add_argument("--eval-preview-topk", default=3, type=int, help="number of top predictions to show inside each evaluation preview")
    parser.add_argument("--train-preview-count", default=0, type=int, help="number of readable training examples to print per preview; set to 0 to disable")
    parser.add_argument("--train-preview-interval", default=100, type=int, help="print readable training preview every N optimizer steps when enabled")
    parser.add_argument("--train-preview-topk", default=5, type=int, help="number of top tokens to show for each previewed final answer position")
    ###
    parser.add_argument("--train-paraphrased", default=False, action="store_true", help="Use Paraphrased questions for training.")
    parser.add_argument("--test-paraphrased", default=False, action="store_true", help="Use Paraphrased questions for testing.")

    args = parser.parse_args()
    return args

def safe_lookup(x, rev_dict=None):
    return rev_dict[x] if x in rev_dict else str(x)

def parse_optional_literal(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text
    return value

def relation_edit_distance(
    pred_relations: Sequence[int],
    gt_relations: Sequence[int],
    special_tokens: Set[int],
    inverse_mapping: Dict[int, int],
) -> int:
    """
    Raw relation-sequence edit distance.

    - removes special relation tokens
    - canonicalizes inverse relation tokens
    - computes Levenshtein distance
    - does NOT normalize
    """
    pred_rels = [
        canon_rel(r, inverse_mapping)
        for r in pred_relations
        if r not in special_tokens
    ]
    gt_rels = list(gt_relations)

    dist, _, _ = edit_distance(pred_rels, gt_rels)
    return dist

def canon_rel(r: int, inverse_mapping: Dict[int, int]) -> int:
    return inverse_mapping.get(r, r)

def edit_distance(pred_seq: Sequence[int], gt_seq: Sequence[int]):
    m, n = len(pred_seq), len(gt_seq)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if pred_seq[i - 1] == gt_seq[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[m][n], None, None

def answer_set_f1(predicted_endpoints, gold_answers, eps=1e-8):
    pred_set = set(predicted_endpoints)
    gold_set = set(gold_answers)

    tp = len(pred_set & gold_set)
    precision = tp / (len(pred_set) + eps)
    recall = tp / (len(gold_set) + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return precision, recall, f1

def get_row_text(row, key, default="N/A"):

    if row is None or key not in row:
        return default
    value = row[key]
    if value is None:
        return default

    if key == "Question-Paraphrased":
        value = ast.literal_eval(str(value))[-1]

    if isinstance(value, float) and math.isnan(value):
        return default

    text = str(value).strip()
    return text if text else default

def decode_symbol(symbol, dataset=None):
    if dataset is not None:
        if hasattr(dataset, "id2entity") and symbol in dataset.id2entity:
            return dataset.id2entity[symbol]
        if hasattr(dataset, "id2relation") and symbol in dataset.id2relation:
            return dataset.id2relation[symbol]
    return symbol

def decode_token(token_id, rev_dict, dataset=None):
    symbol = safe_lookup(int(token_id), rev_dict)
    return decode_symbol(symbol, dataset)


def is_relation_symbol(symbol, dataset=None):
    if dataset is not None and hasattr(dataset, "id2relation") and symbol in dataset.id2relation:
        return True
    return isinstance(symbol, str) and symbol.startswith("R")

def format_query_chain(row, relation_fallback):
    if row is None:
        return relation_fallback
    query_relations = parse_optional_literal(row.get("Query-Relations"))
    if isinstance(query_relations, list) and query_relations:
        return " -> ".join(str(relation) for relation in query_relations)
    return get_row_text(row, "Query-Relation", relation_fallback)

def format_gold_path(row):
    if row is None:
        return "N/A"
    paths = parse_optional_literal(row.get("Paths-Label"))
    if not isinstance(paths, list) or not paths:
        paths = parse_optional_literal(row.get("Paths"))
    if not isinstance(paths, list) or not paths:
        return "N/A"

    first_hop = paths[0]
    if not isinstance(first_hop, (list, tuple)) or len(first_hop) < 3:
        return "N/A"

    parts = [str(first_hop[0])]
    for hop in paths:
        if not isinstance(hop, (list, tuple)) or len(hop) < 3:
            continue
        parts.append(f"--{hop[1]}--> {hop[2]}")
    return " ".join(parts) if len(parts) > 1 else "N/A"

def format_generated_path(head_label, path_tokens, rev_dict, dataset, eos, bos):
    if path_tokens is None:
        return "N/A"

    parts = [head_label]
    pending_relation = None

    for token in path_tokens[1:]:
        token_id = int(token)
        if token_id == eos:
            break
        if token_id == bos:
            continue

        symbol = safe_lookup(token_id, rev_dict)
        label = decode_symbol(symbol, dataset)
        if is_relation_symbol(symbol, dataset):
            pending_relation = label
            continue
        if pending_relation is None:
            continue

        if pending_relation.endswith(" (reverse)"):
            relation_name = pending_relation[: -len(" (reverse)")]
            parts.append(f"<-{relation_name}- {label}")
        else:
            parts.append(f"--{pending_relation}--> {label}")
        pending_relation = None

    if pending_relation is not None:
        parts.append(f"--{pending_relation}--> ?")

    return " ".join(parts)

def compute_precision_recall_f1(
    pred: Set,
    gt: Set,
    eps: float = 1e-8,
) -> Tuple[float, float, float]:
    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1

def gt_edge_overlap_f1(
    pred_path: Sequence[Tuple[int, int, int]],
    gt_path: Sequence[Tuple[int, int, int]],
    special_tokens: Set[int],
    inverse_mapping: Dict[int, int],
) -> Tuple[float, float, float]:
    """
    Permutation-invariant edge overlap between predicted and gold paths.

    Mirrors the repo behavior:
    - remove special tokens such as NO_OP / STOP / RESTART
    - canonicalize inverse edges back into forward edges
    - compare edge sets
    """
    pred_edges = {
        canon_edge(h, r, t, inverse_mapping)
        for h, r, t in pred_path
        if r not in special_tokens
    }
    gt_edges = {(h, r, t) for h, r, t in gt_path}
    return compute_precision_recall_f1(pred_edges, gt_edges)

def canon_edge(h: int, r: int, t: int, inverse_mapping: Dict[int, int]) -> Tuple[int, int, int]:
    if r in inverse_mapping:
        return (t, inverse_mapping[r], h)
    return (h, r, t)

def path_tokens_to_edges(path_tokens, eos, bos):
    clean = []
    for token in path_tokens:
        token = int(token)
        if token == eos:
            break
        if token == bos:
            continue
        clean.append(token)

    edges = []
    for idx in range(0, len(clean) - 2, 2):
        edges.append((clean[idx], clean[idx + 1], clean[idx + 2]))

    return edges

def path_tokens_to_relations(path_tokens, eos, bos):
    clean = []
    for token in path_tokens:
        token = int(token)
        if token == eos:
            break
        if token == bos:
            continue
        clean.append(token)

    return [clean[idx] for idx in range(1, len(clean), 2)]

def build_eval_preview(dataset, sample_id, head_id, relation_id, target_id, candidate_ids, candidate_paths, preview_topk, rank_idx, rev_dict, eos, bos, test_paraphrased=False):
    row = None
    if dataset is not None and hasattr(dataset, "data"):
        row = dataset.data.iloc[int(sample_id)]

    source_entity = get_row_text(row, "Source", decode_token(head_id, rev_dict, dataset))
    relation_chain = format_query_chain(row, decode_token(relation_id, rev_dict, dataset))
    gold_answer = get_row_text(row, "Answer", decode_token(target_id, rev_dict, dataset))
    predicted_answer = decode_token(candidate_ids[0], rev_dict, dataset) if candidate_ids else "N/A"
    predicted_path = "N/A"
    if candidate_paths:
        predicted_path = format_generated_path(source_entity, candidate_paths[0], rev_dict, dataset, eos, bos)

    top_predictions = [decode_token(candidate_id, rev_dict, dataset) for candidate_id in candidate_ids[:preview_topk]]
    rank_text = str(rank_idx + 1) if rank_idx is not None else "not ranked"
    status = "correct" if candidate_ids and candidate_ids[0] == target_id else "incorrect"

    if test_paraphrased:
        question_row_text = get_row_text(row, "Question-Paraphrased")
    else:
        question_row_text = get_row_text(row, "Question")
    lines = [
        f"Question: {question_row_text}",
        f"Structured Input: source={source_entity} | relation chain={relation_chain}",
        f"Gold Answer: {gold_answer}",
        f"Predicted Answer: {predicted_answer} ({status}; gold rank={rank_text})",
    ]
    if predicted_path != "N/A":
        lines.append(f"Predicted Path: {predicted_path}")

    gold_path = format_gold_path(row)
    if gold_path != "N/A":
        lines.append(f"Gold Path: {gold_path}")

    if top_predictions:
        lines.append(f"Top-{len(top_predictions)} Predictions: " + " | ".join(top_predictions))

    return "\n".join(lines)

def build_rev_dict(dictionary):
    return {v: k for k, v in dictionary.indices.items()}

def format_token_sequence(token_ids, rev_dict, dataset, length=None):
    if length is not None:
        token_ids = token_ids[:length]
    return " | ".join(decode_token(token_id, rev_dict, dataset) for token_id in token_ids.detach().cpu().tolist())

def format_top_tokens(logit_row, rev_dict, dataset, topk):
    topk = min(topk, logit_row.size(-1))
    probs = F.softmax(logit_row, dim=-1)
    top_probs, top_ids = torch.topk(probs, k=topk)
    pieces = []
    for token_id, prob in zip(top_ids.detach().cpu().tolist(), top_probs.detach().cpu().tolist()):
        pieces.append(f"{decode_token(token_id, rev_dict, dataset)} ({prob:.4f})")
    return " | ".join(pieces)

def build_train_preview(samples, pred, logits, last_idx, dataset, rev_dict, args, step):
    preview_count = min(max(0, args.train_preview_count), pred.size(0))
    preview_topk = max(1, args.train_preview_topk)
    lines = [f"[Train Preview | step {step}]"]

    for i in range(preview_count):
        sample_id = int(samples["ids"][i].detach().cpu().item())
        row = dataset.data.iloc[sample_id] if dataset is not None and hasattr(dataset, "data") else None
        if args.train_paraphrased:
            question = get_row_text(row, "Question-Paraphrased", "N/A")
        else:
            question = get_row_text(row, "Question", "N/A")
        input_len = int(samples["attention_mask"][i].sum().detach().cpu().item())
        input_ids = samples["input_ids"][i, :input_len].detach().cpu().tolist()
        input_text = dataset.tokenizer.decode(input_ids, skip_special_tokens=True) if dataset is not None and hasattr(dataset, "tokenizer") else str(input_ids)
        length = int(samples["lengths"][i].detach().cpu().item()) if "lengths" in samples else int(samples["mask"][i].sum().detach().cpu().item())
        final_pos = int(last_idx[i].detach().cpu().item())

        lines.extend([
            f"Example {i + 1} (dataset id: {sample_id})",
            f"Question: {question}",
            f"Model input text: {input_text}",
            f"input_ids: {input_ids}",
            f"attention_mask length: {input_len}",
            f"prev_outputs (teacher forcing): {format_token_sequence(samples['prev_outputs'][i], rev_dict, dataset, length)}",
            f"target tokens: {format_token_sequence(samples['target'][i], rev_dict, dataset, length)}",
            f"predicted tokens: {format_token_sequence(pred[i], rev_dict, dataset, length)}",
            f"mask: {samples['mask'][i, :length].detach().cpu().tolist()}",
            f"last_idx: {final_pos}",
            f"target_last: {decode_token(samples['target'][i, final_pos].detach().cpu().item(), rev_dict, dataset)}",
            f"pred_last: {decode_token(pred[i, final_pos].detach().cpu().item(), rev_dict, dataset)}",
            f"last-token top-{preview_topk}: {format_top_tokens(logits[i, final_pos], rev_dict, dataset, preview_topk)}",
        ])

    return "\n".join(lines)

def write_tqdm_block(block):
    for line in block.splitlines():
        tqdm.write(line)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)

def build_dataloader_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator

def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved

def evaluate(model, dataloader, device, args, true_triples=None, valid_triples=None, split_name="eval"):
    model.eval()
    beam_size = args.beam_size
    l_punish = args.l_punish
    max_len = 2 * args.max_len + 2
    restricted_punish = -30
    mrr, hit, hit1, hit3, hit5, hit10, edge_f1, relation_edit_distance_sum, answer_f1_sum, answer_p_sum, answer_r_sum, count = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    vocab_size = len(model.dictionary)
    eos = model.dictionary.eos()
    bos = model.dictionary.bos()
    rev_dict = dict()
    lines = []
    dataset = getattr(dataloader, "dataset", None)
    preview_blocks = []
    preview_limit = max(0, getattr(args, "eval_preview_count", 0))
    preview_topk = max(1, getattr(args, "eval_preview_topk", 1))
    split_label = split_name.title()
    for k in model.dictionary.indices.keys():
        v = model.dictionary.indices[k]
        rev_dict[v] = k
    special_tokens = {bos, eos, model.dictionary.pad()}
    loop_token = model.dictionary.indices.get("LOOP")
    if loop_token is not None:
        special_tokens.add(loop_token)
    inverse_mapping = {}
    relation_name_to_id = {}
    if dataset is not None and hasattr(dataset, "id2relation"):
        for token_id, symbol in rev_dict.items():
            if symbol in dataset.id2relation:
                relation_name_to_id[dataset.id2relation[symbol]] = token_id
        for relation_name, token_id in relation_name_to_id.items():
            if relation_name.endswith(" (reverse)"):
                base_name = relation_name[: -len(" (reverse)")]
                if base_name in relation_name_to_id:
                    inverse_mapping[token_id] = relation_name_to_id[base_name]

    # I'm aware that nested function make Code Smell
    # For now they will stay here,
    # they help with debugging.
    # After I'm finished, I remove them.
    def debug_token(token_id):
        token_id = int(token_id)
        return f"{token_id} ({decode_token(token_id, rev_dict, dataset)})"

    def debug_path(token_ids):
        decoded = " | ".join(decode_token(token_id, rev_dict, dataset) for token_id in token_ids)
        return f"{token_ids} ({decoded})"

    def debug_top_values(value_row, topk, use_softmax=False):
        values = F.softmax(value_row, dim=-1) if use_softmax else value_row
        topk = min(topk, values.size(-1))
        top_values, top_ids = torch.topk(values, k=topk)
        ids = top_ids.detach().cpu().tolist()
        scores = top_values.detach().cpu().tolist()
        return "[" + ", ".join(
            f"({token_id}, {score:.6f}, {decode_token(token_id, rev_dict, dataset)})"
            for token_id, score in zip(ids, scores)
        ) + "]"

    def debug_candidate_lines(candidate_scores, topk=10):
        items = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)[:topk]
        if not items:
            return ["[]"]
        return [f"{debug_token(candidate_id)}: {score:.6f}" for candidate_id, score in items]

    def debug_question(sample_id, test_paraphrased=False):
        if dataset is not None and hasattr(dataset, "data"):
            sample_id = int(sample_id)
            if 0 <= sample_id < len(dataset.data):
                if test_paraphrased:
                    return get_row_text(dataset.data.iloc[sample_id], "Question-Paraphrased")
                else:
                    return get_row_text(dataset.data.iloc[sample_id], "Question")
        return "N/A"

    def debug_model_input_lines(input_ids_row, attention_mask_row, prefix_row):
        input_ids_list = input_ids_row.detach().cpu().tolist()
        attention_mask_list = attention_mask_row.detach().cpu().tolist()
        prefix_list = prefix_row.detach().cpu().tolist()
        active_len = int(attention_mask_row.detach().cpu().sum().item())
        active_input_ids = input_ids_list[:active_len]
        if dataset is not None and hasattr(dataset, "tokenizer"):
            input_text = dataset.tokenizer.decode(active_input_ids, skip_special_tokens=True)
            input_tokens = dataset.tokenizer.convert_ids_to_tokens(active_input_ids)
        else:
            input_text = str(active_input_ids)
            input_tokens = active_input_ids
        return [
            "[Model Input]",
            f"input_ids: {input_ids_list}",
            f"attention_mask: {attention_mask_list}",
            f"input interpretation: {input_text}",
            f"input tokens: {input_tokens}",
            f"prefix ids: {prefix_list}",
            f"prefix interpretation: {debug_path(prefix_list)}",
        ]

    def slot_role(prefix_idx):
        return "HEAD/ENTITY" if prefix_idx % 2 == 1 else "RELATION/EOS"

    def row_path_token_to_id(token, is_relation):
        token = str(token)
        if token in model.dictionary.indices:
            return model.dictionary.indices[token]
        if dataset is None:
            return None
        if is_relation and token in relation_name_to_id:
            return relation_name_to_id[token]
        if is_relation and hasattr(dataset, "relation2id") and token in dataset.relation2id:
            mapped = dataset.relation2id[token]
            return model.dictionary.indices.get(mapped)
        if (not is_relation) and hasattr(dataset, "entity2id") and token in dataset.entity2id:
            mapped = dataset.entity2id[token]
            return model.dictionary.indices.get(mapped)
        return None

    def row_to_gt_edges(row):
        if row is None:
            return []
        gt_paths = parse_optional_literal(row.get("Paths"))
        if not isinstance(gt_paths, list) or not gt_paths:
            gt_paths = parse_optional_literal(row.get("Paths-Label"))
        if not isinstance(gt_paths, list) or not gt_paths:
            return []

        gt_path_tokens = [bos]
        for hop_idx, hop in enumerate(gt_paths):
            if not isinstance(hop, (list, tuple)) or len(hop) < 3:
                continue
            h_id = row_path_token_to_id(hop[0], is_relation=False)
            r_id = row_path_token_to_id(hop[1], is_relation=True)
            t_id = row_path_token_to_id(hop[2], is_relation=False)
            if h_id is None or r_id is None or t_id is None:
                continue
            if hop_idx == 0:
                gt_path_tokens.extend([h_id, r_id, t_id])
            else:
                gt_path_tokens.extend([r_id, t_id])
        gt_path_tokens.append(eos)
        return path_tokens_to_edges(gt_path_tokens, eos, bos)

    def row_to_gt_relations(row):
        if row is None:
            return []
        gt_relations = parse_optional_literal(row.get("Query-Relations"))
        if isinstance(gt_relations, list) and gt_relations:
            relation_ids = []
            for relation in gt_relations:
                relation_id = row_path_token_to_id(relation, is_relation=True)
                if relation_id is not None:
                    relation_ids.append(relation_id)
            return relation_ids
        gt_relation = row.get("Query-Relation")
        if gt_relation is None or (isinstance(gt_relation, float) and math.isnan(gt_relation)):
            return []
        relation_id = row_path_token_to_id(gt_relation, is_relation=True)
        return [relation_id] if relation_id is not None else []

    with tqdm(dataloader, desc=f"{split_label} Eval") as pbar:
        for samples in pbar:
            samples = move_batch_to_device(samples, device)
            pbar.set_description(
                "%s Eval | MRR: %f, Hit@1: %f, Hit@3: %f, Hit@5: %f, Hit@10: %f, EdgeF1: %f, RelED: %f, AnswerF1: %f"
                % (split_label, mrr/max(1, count), hit1/max(1, count), hit3/max(1, count), hit5/max(1, count), hit10/max(1, count), edge_f1/max(1, count), relation_edit_distance_sum/max(1, count), answer_f1_sum/max(1, count))
            )
            batch_size = samples["input_ids"].size(0)
            debug_limit = 0
            debug_blocks = []
            if count < 2:
                debug_limit = min(batch_size, 2 - count)
                for i in range(debug_limit):
                    sample_id = samples["ids"][i].detach().cpu().tolist() if "ids" in samples else count + i
                    head_id = samples["head_id"][i].detach().cpu().tolist()
                    target_id = samples["target"][i].detach().cpu().view(-1)[0].tolist()
                    debug_blocks.append([
                        "================ SAMPLE =================",
                        "[INPUT]",
                        f"Sample ID: {sample_id}",
                        f"Question: {debug_question(sample_id, test_paraphrased=args.test_paraphrased)}",
                        f"Head ID: {debug_token(head_id)}",
                        f"Target ID: {debug_token(target_id)}",
                    ])

            candidates = [dict() for i in range(batch_size)]
            candidates_path = [dict() for i in range(batch_size)]
            input_ids = samples["input_ids"].unsqueeze(dim=1).repeat(1, beam_size, 1).to(device)
            attention_mask = samples["attention_mask"].unsqueeze(dim=1).repeat(1, beam_size, 1).to(device)
            # The question encoder input is identical for every beam in the
            # batch, so compute it once and expand the cached states instead of
            # re-running BERT for every beam step.
            question_source = model.encode_question(samples["input_ids"], samples["attention_mask"])
            beam_question_source = question_source.unsqueeze(2).repeat(1, 1, beam_size, 1).reshape(
                question_source.size(0),
                batch_size * beam_size,
                question_source.size(-1),
            )
            prefix = torch.zeros([batch_size, beam_size, max_len], dtype=torch.long).to(device)
            prefix[:, :, 0].fill_(model.dictionary.bos())
            lprob = torch.zeros([batch_size, beam_size]).to(device)
            clen = torch.zeros([batch_size, beam_size], dtype=torch.long).to(device)
            # first token after BOS predicts head_0
            tmp_input_ids = samples["input_ids"]
            tmp_attention_mask = samples["attention_mask"]
            tmp_prefix = torch.zeros([batch_size, 1], dtype=torch.long).to(device)
            tmp_prefix[:, 0].fill_(model.dictionary.bos())
            if count < 2:
                for i in range(debug_limit):
                    debug_blocks[i].extend(debug_model_input_lines(tmp_input_ids[i], tmp_attention_mask[i], tmp_prefix[i]))
            logits = model.logits(
                tmp_input_ids,
                tmp_attention_mask,
                tmp_prefix,
                encoded_source=question_source,
            ).squeeze(1)
            logits = F.log_softmax(logits, dim=-1)
            logits = logits.view(-1, vocab_size)
            argsort = torch.argsort(logits, dim=-1, descending=True)[:, :beam_size]
            prefix[:, :, 1] = argsort[:, :]
            lprob += torch.gather(input=logits, dim=-1, index=argsort)
            clen += 1
            if count < 2:
                debug_logits = logits.view(batch_size, -1)
                for i in range(debug_limit):
                    debug_blocks[i].extend([
                        "[STEP 0 HEAD/ENTITY LOGITS]",
                        "Top-10: " + debug_top_values(debug_logits[i], 10),
                    ])
            target = samples["target"].cpu()
            for l in range(2, max_len):
                tmp_prefix = prefix.unsqueeze(dim=2).repeat(1, 1, beam_size, 1)
                tmp_lprob = lprob.unsqueeze(dim=-1).repeat(1, 1, beam_size)    
                tmp_clen = clen.unsqueeze(dim=-1).repeat(1, 1, beam_size)
                bb = batch_size * beam_size
                if l <= 3 and count < 2:
                    for i in range(debug_limit):
                        debug_blocks[i].append("[Model Input]")
                        for j in range(min(3, beam_size)):
                            debug_blocks[i].append(f"beam {j}:")
                            debug_blocks[i].extend(debug_model_input_lines(input_ids[i][j], attention_mask[i][j], prefix[i][j]))
                all_logits = model.logits(
                    input_ids.view(bb, -1),
                    attention_mask.view(bb, -1),
                    prefix.view(bb, -1),
                    encoded_source=beam_question_source,
                ).view(batch_size, beam_size, max_len, -1)
                logits = torch.gather(input=all_logits, dim=2, index=clen.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, vocab_size)).squeeze(2)
                # relation slots use the previously predicted head; head slots use (head, relation)
                if args.no_filter_gen:
                    logits = F.log_softmax(logits, dim=-1)
                else:
                    restricted = torch.ones([batch_size, beam_size, vocab_size]) * restricted_punish
                    if l % 2 == 0:
                        index = prefix[:, :, l - 1]
                    else:
                        hid = prefix[:, :, l - 2]
                        rid = prefix[:, :, l - 1]
                        index = vocab_size * rid + hid
                    index = index.cpu().numpy()
                    for i in range(batch_size):
                        for j in range(beam_size):
                            if index[i][j] in true_triples:
                                if args.smart_filter:
                                    restricted[i][j] = true_triples[index[i][j]]
                                else:
                                    idx = torch.LongTensor(true_triples[index[i][j]]).unsqueeze(0)
                                    restricted[i][j] = -restricted_punish * torch.zeros(1, vocab_size).scatter_(1, idx, 1) + restricted_punish
                    logits = F.log_softmax(logits+restricted.to(device), dim=-1)
                if l <= 3 and count < 2:
                    for i in range(debug_limit):
                        debug_blocks[i].append(f"[BEAM STEP {l} | {slot_role(l)}]")
                        debug_blocks[i].append("Current prefixes (first 3 beams):")
                        for j in range(min(3, beam_size)):
                            prefix_tokens = prefix[i][j, :l].detach().cpu().tolist()
                            debug_blocks[i].append(f"beam {j}: {debug_path(prefix_tokens)}")
                        debug_blocks[i].append("Logits top-5 for next token:")
                        for j in range(min(3, beam_size)):
                            debug_blocks[i].append(f"beam {j}: {debug_top_values(logits[i][j], 5)}")
                argsort = torch.argsort(logits, dim=-1, descending=True)[:, :, :beam_size]
                tmp_clen = tmp_clen + 1
                tmp_prefix = tmp_prefix.scatter_(dim=-1, index=tmp_clen.unsqueeze(-1), src=argsort.unsqueeze(-1))
                tmp_lprob += torch.gather(input=logits, dim=-1, index=argsort)
                tmp_prefix, tmp_lprob, tmp_clen = tmp_prefix.view(batch_size, -1, max_len), tmp_lprob.view(batch_size, -1), tmp_clen.view(batch_size, -1)
                if l == max_len-1:
                    argsort = torch.argsort(tmp_lprob, dim=-1, descending=True)[:, :(2*beam_size)]
                else:
                    argsort = torch.argsort(tmp_lprob, dim=-1, descending=True)[:, :beam_size]
                prefix = torch.gather(input=tmp_prefix, dim=1, index=argsort.unsqueeze(-1).repeat(1, 1, max_len))
                lprob = torch.gather(input=tmp_lprob, dim=1, index=argsort)
                clen = torch.gather(input=tmp_clen, dim=1, index=argsort)
                # filter out next token after <end>, add to candidates
                for i in range(batch_size):
                    for j in range(beam_size):
                        if l % 2 == 0 and prefix[i][j][l].item() == eos:
                            candidate_pos = l - 1
                            candidate = prefix[i][j][candidate_pos].item()
                            if l_punish:
                                prob = lprob[i][j].item() / max(1, l // 2)
                            else:
                                prob = lprob[i][j].item()
                            path_array = prefix[i][j, :l + 1].detach().cpu().numpy()
                            if count < 2 and i < debug_limit:
                                path_tokens = path_array.tolist()
                                debug_blocks[i].extend([
                                    "[COLLECT CANDIDATE]",
                                    f"Candidate entity: {debug_token(candidate)}",
                                    f"Score: {prob:.6f}",
                                    f"Path: {debug_path(path_tokens)}",
                                ])
                            lprob[i][j] -= 10000
                            if candidate not in candidates[i]:
                                if args.self_consistency:
                                    candidates[i][candidate] = math.exp(prob)
                                else:
                                    candidates[i][candidate] = prob
                                candidates_path[i][candidate] = path_array
                            else:
                                if prob > candidates[i][candidate]:
                                    candidates_path[i][candidate] = path_array
                                if args.self_consistency:
                                    candidates[i][candidate] += math.exp(prob)
                                else:
                                    candidates[i][candidate] = max(candidates[i][candidate], prob)
                # no </s> but reach max_len
                if l == max_len-1:
                    for i in range(batch_size):
                        for j in range(beam_size*2):
                            candidate_pos = l if l % 2 == 1 else l - 1
                            candidate = prefix[i][j][candidate_pos].item()
                            if l_punish:
                                prob = lprob[i][j].item() / max(1, (l - 1) // 2)
                            else:
                                prob = lprob[i][j].item()
                            path_array = prefix[i][j, :candidate_pos + 1].detach().cpu().numpy()
                            if count < 2 and i < debug_limit:
                                path_tokens = path_array.tolist()
                                debug_blocks[i].extend([
                                    "[COLLECT CANDIDATE]",
                                    f"Candidate entity: {debug_token(candidate)}",
                                    f"Score: {prob:.6f}",
                                    f"Path: {debug_path(path_tokens)}",
                                ])
                            if candidate not in candidates[i]:
                                if args.self_consistency:
                                    candidates[i][candidate] = math.exp(prob)
                                else:
                                    candidates[i][candidate] = prob
                                candidates_path[i][candidate] = path_array
                            else:
                                if prob > candidates[i][candidate]:
                                    candidates_path[i][candidate] = path_array
                                if args.self_consistency:
                                    candidates[i][candidate] += math.exp(prob)
                                else:                             
                                    candidates[i][candidate] = max(candidates[i][candidate], prob)
            target = samples["target"].cpu()
            for i in range(batch_size):
                debug_sample = count < 2 and i < debug_limit
                hid = samples["head_id"][i].item()
                
                #! index is irrelevant for new kinship dataset
                #! because it doesn't have Query-Relation information
                # index = vocab_size * rid + hid
                index = None
                if debug_sample:
                    debug_gold = target[i].detach().cpu().view(-1)[0].tolist()
                    debug_blocks[i].extend([
                        "[BEFORE FILTER]",
                        "Top candidates (id, score):",
                    ])
                    debug_blocks[i].extend(debug_candidate_lines(candidates[i]))
                    debug_blocks[i].append(f"Gold: {debug_token(debug_gold)}")

                if index is not None and index in valid_triples:
                    mask = valid_triples[index]
                    for tid in candidates[i].keys():
                        if tid == target[i].item():
                            continue
                        elif args.smart_filter:
                            if mask[tid].item() == 0:
                                candidates[i][tid] -= 100000
                        else:
                            if tid in mask:
                                candidates[i][tid] -= 100000
                if debug_sample:
                    debug_blocks[i].extend([
                        "[AFTER FILTER]",
                        "Top candidates (id, score):",
                    ])
                    debug_blocks[i].extend(debug_candidate_lines(candidates[i]))
                count += 1
                candidate_ = sorted(zip(candidates[i].items(), candidates_path[i].items()), key=lambda x:x[0][1], reverse=True)
                candidate_ids = [pair[0][0] for pair in candidate_]
                candidate_path = [pair[1][1] for pair in candidate_]
                if candidate_ids:
                    candidate = torch.as_tensor(candidate_ids, dtype=torch.long)
                else:
                    candidate = torch.empty(0, dtype=torch.long)
                target_id = target[i].item()
                ranking = (candidate[:] == target_id).nonzero(as_tuple=False)
                rank_idx = None
                row = dataset.data.iloc[int(samples["ids"][i].item())] if dataset is not None and hasattr(dataset, "data") else None
                pred_edges = path_tokens_to_edges(candidate_path[0], eos, bos) if candidate_path else []
                pred_relations = path_tokens_to_relations(candidate_path[0], eos, bos) if candidate_path else []
                gt_edges = row_to_gt_edges(row)
                gt_relations = row_to_gt_relations(row)
                predicted_endpoints = candidate_ids
                gold_answers = [target_id]
                if row is not None:
                    row_answers = parse_optional_literal(row.get("Answers"))
                    if isinstance(row_answers, list) and row_answers:
                        parsed_gold_answers = []
                        for answer in row_answers:
                            answer_id = row_path_token_to_id(answer, is_relation=False)
                            if answer_id is not None:
                                parsed_gold_answers.append(answer_id)
                        gold_answers = parsed_gold_answers
                answer_p, answer_r, answer_f1 = answer_set_f1(predicted_endpoints, gold_answers)
                answer_f1_sum += answer_f1
                answer_p_sum += answer_p
                answer_r_sum += answer_r
                if gt_edges:
                    _, _, sample_edge_f1 = gt_edge_overlap_f1(pred_edges, gt_edges, special_tokens, inverse_mapping)
                    edge_f1 += sample_edge_f1
                relation_edit_distance_sum += relation_edit_distance(pred_relations, gt_relations, special_tokens, inverse_mapping)
                if args.test_paraphrased:
                    question_text = get_row_text(row, "Question-Paraphrased")
                else:
                    question_text = get_row_text(row, "Question")

                head_label = get_row_text(row, "Source", decode_token(hid, rev_dict, dataset))
                target_label = get_row_text(row, "Answer", decode_token(target_id, rev_dict, dataset))
                path_token = f"{question_text}\t{head_label} | {target_label}\t"

                if ranking.nelement() != 0:
                    rank_idx = ranking[0].item()
                    path = candidate_path[rank_idx]
                    path_token += format_generated_path(head_label, path, rev_dict, dataset, eos, bos) + '\t'
                    path_token += str(rank_idx)
                    ranking_value = 1 + rank_idx
                    mrr += (1 / ranking_value)
                    hit += 1
                    if ranking_value <= 1:
                        hit1 += 1
                    if ranking_value <= 3:
                        hit3 += 1
                    if ranking_value <= 5:
                        hit5 += 1
                    if ranking_value <= 10:
                        hit10 += 1
                else:
                    path_token += "wrong"
                lines.append(path_token+'\n')
                if debug_sample:
                    rank_text = rank_idx + 1 if rank_idx is not None else None
                    top1 = candidate_ids[0] if candidate_ids else None
                    debug_blocks[i].extend([
                        "[RANKING]",
                        f"Sorted candidates: {candidate_ids}",
                        "Sorted decoded: " + " | ".join(debug_token(candidate_id) for candidate_id in candidate_ids),
                        f"Gold: {debug_token(target_id)}",
                        f"Rank: {rank_text}",
                        f"Top-1: {debug_token(top1) if top1 is not None else None}",
                        "========================================",
                    ])
                    # uncommend this line if you want to see the debug blocks 
                    # for the first 2 samples (will print a lot of info about model input and logits)
                    # write_tqdm_block("\n".join(debug_blocks[i]))

                if len(preview_blocks) < preview_limit:
                    preview_blocks.append(
                        build_eval_preview(
                            dataset=dataset,
                            sample_id=samples["ids"][i].item(),
                            head_id=hid,
                            relation_id=rid,
                            target_id=target_id,
                            candidate_ids=candidate_ids,
                            candidate_paths=candidate_path,
                            preview_topk=preview_topk,
                            rank_idx=rank_idx,
                            rev_dict=rev_dict,
                            eos=eos,
                            bos=bos,
                            test_paraphrased=args.test_paraphrased
                        )
                    )
    
    if args.output_path and split_name=="test":
        with open(os.path.join(args.save_dir,"test_output_squire.txt"), "w") as f:
            f.writelines(lines)
    metric_denominator = max(1, count)
    if preview_blocks:
        for idx, block in enumerate(preview_blocks, start=1):
            write_tqdm_block(f"[{split_name.upper()} Example {idx}]")
            write_tqdm_block(block)
            if idx != len(preview_blocks):
                tqdm.write("")
    summary = "[%s] MRR: %.6f, Hit@1: %.6f, Hit@3: %.6f, Hit@5: %.6f, Hit@10: %.6f, EdgeF1: %.6f, RelED: %.6f, AnswerF1: %.6f" % (
        split_name.upper(),
        mrr/metric_denominator,
        hit1/metric_denominator,
        hit3/metric_denominator,
        hit5/metric_denominator,
        hit10/metric_denominator,
        edge_f1/metric_denominator,
        relation_edit_distance_sum/metric_denominator,
        answer_f1_sum/metric_denominator,
    )
    tqdm.write(summary)
    logging.info(summary)
    return mrr/metric_denominator, hit1/metric_denominator, hit3/metric_denominator, hit5/metric_denominator, hit10/metric_denominator


def plot_epoch_metrics(metric_history, save_dir):
    epochs = metric_history["epoch"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
    metric_specs = [
        ("mrr", "MRR"),
        ("hit1", "Hit@1"),
        ("hit3", "Hit@3"),
        ("hit5", "Hit@5"),
        ("hit10", "Hit@10"),
    ]

    for ax, (key, title) in zip(axes.flat, metric_specs):
        ax.plot(epochs, metric_history[f"train_{key}"], label="Train", linewidth=2)
        ax.plot(epochs, metric_history[f"valid_{key}"], label="Valid", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.grid(True, alpha=0.3)
        ax.legend()

    for ax in axes.flat[len(metric_specs):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "training_metrics.png"), dpi=200)
    plt.close(fig)

def train(args):
    args.dataset = os.path.join('data', args.dataset)
    save_path = os.path.join('models', args.save_dir)
    ckpt_path = os.path.join(save_path, 'checkpoint')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)
    logging.basicConfig(level=logging.DEBUG,
                    filename=save_path+'/train.log',
                    filemode='w',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )
    logging.info(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader_kwargs = {
        "num_workers": max(0, args.num_workers),
        "pin_memory": device == "cuda",
        "worker_init_fn": seed_worker,
    }
    train_set = Seq2SeqDataset(data_path=args.dataset+"/", vocab_file=args.dataset+"/vocab.txt", device=device, split="train", args=args)
    valid_set = TestDataset(data_path=args.dataset+"/", vocab_file=args.dataset+"/vocab.txt", device=device, src_file="valid_triples.txt", split="dev", args=args)
    train_eval_set = TestDataset(data_path=args.dataset+"/", vocab_file=args.dataset+"/vocab.txt", device=device, src_file="train_triples.txt", split="train", args=args)
    train_valid, eval_valid = train_set.get_next_valid()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, collate_fn=train_set.collate_fn, shuffle=True, generator=build_dataloader_generator(args.seed), **loader_kwargs)
    valid_loader = DataLoader(valid_set, batch_size=args.test_batch_size, collate_fn=valid_set.collate_fn, shuffle=True, generator=build_dataloader_generator(args.seed + 1), **loader_kwargs)
    train_eval_loader = DataLoader(train_eval_set, batch_size=args.test_batch_size, collate_fn=valid_set.collate_fn, shuffle=False, generator=build_dataloader_generator(args.seed + 2), **loader_kwargs)

    model = TransformerModel(args, train_set.dictionary).to(device)
    train_preview_rev_dict = build_rev_dict(train_set.dictionary)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = len(train_loader)
    total_step_num = len(train_loader) * args.num_epoch
    warmup_steps = total_step_num / args.warmup
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, warmup_steps, total_step_num)
    
    #! iterative training isn't tested
    # if args.iter: 
    #     iter_trainer = Iter_trainer(args.dataset, args.iter_batch_size, 32, 4)
    #     iter_epoch = []
    #     max_len = args.max_len
    #     total = 0
    #     for i in range(1, max_len+1):
    #         total += (1/i)
    #     epochs = 0
    #     for i in range(1, max_len+1):
    #         iter_epoch.append(int(args.num_epoch/(total*i)))
    #         epochs += int(args.num_epoch/(total*i))
    #     iter_epoch[-1] += (args.num_epoch-epochs)
    #     curr_iter = -1
    #     curr_iter_epoch = 0
    #     logging.info(
    #                 "[Iter0: %d] [Iter1: %d] [Iter2: %d]"
    #                 % (iter_epoch[0], iter_epoch[1], iter_epoch[2])
    #                 )

    best_hit1 = -float("inf")
    best_epoch = -1
    metric_history = {
        "epoch": [],
        "train_mrr": [],
        "valid_mrr": [],
        "train_hit1": [],
        "valid_hit1": [],
        "train_hit3": [],
        "valid_hit3": [],
        "train_hit5": [],
        "valid_hit5": [],
        "train_hit10": [],
        "valid_hit10": [],
    }
    steps = 0
    for epoch in range(args.num_epoch):

        #! iterative training isn't tested
        # if args.iter:
        #     if curr_iter_epoch == 0: # start next iteration
        #         curr_iter += 1
        #         curr_iter_epoch = iter_epoch[curr_iter]
        #         # label new dataset
        #         if curr_iter > 0:
        #             logging.info("--------Iterating--------")
        #             (src_lines, tgt_lines) = iter_trainer.get_iter(model, curr_iter)
        #             train_set.src_lines += src_lines
        #             train_set.tgt_lines += tgt_lines
        #             train_loader = DataLoader(train_set, batch_size=args.batch_size, collate_fn=train_set.collate_fn, shuffle=True)
        #         # new scheduler
        #         step_num = len(train_loader) * curr_iter_epoch
        #         warmup_steps = step_num / args.warmup
        #         if curr_iter != 0:
        #             optimizer = optim.Adam(model.parameters(), lr=args.lr / 5, weight_decay=args.weight_decay) # fine-tuning with smaller lr
        #             warmup_steps = 0
        #         scheduler = transformers.get_linear_schedule_with_warmup(optimizer, warmup_steps, step_num)
        #     curr_iter_epoch -= 1

        model.train()
        with tqdm(train_loader, desc="training") as pbar:
            losses = []
            token_accs = []
            last_accs = []
            for batch_idx, samples in enumerate(pbar):
                samples = move_batch_to_device(samples, device)
                optimizer.zero_grad()
                loss = model.get_loss(**samples)
                loss.backward()
                optimizer.step()
                scheduler.step()
                steps += 1
                losses.append(loss.item())

                with torch.no_grad():
                    logits = model.logits(
                        samples["input_ids"],
                        samples["attention_mask"],
                        samples["prev_outputs"]
                    )

                    pred = logits.argmax(dim=-1)

                    target = samples["target"]
                    mask = samples["mask"]

                    correct = (pred == target) & mask.bool()
                    token_acc = correct.sum().float() / mask.sum().float()

                    lengths = mask.sum(dim=1).long()
                    last_idx = lengths - 2
                    batch_indices = torch.arange(pred.size(0), device=pred.device)

                    pred_last = pred[batch_indices, last_idx]
                    target_last = target[batch_indices, last_idx]

                    last_acc = (pred_last == target_last).float().mean()

                    if args.train_preview_count > 0 and (batch_idx == 0 or steps % max(1, args.train_preview_interval) == 0):
                        preview_block = build_train_preview(
                            samples,
                            pred,
                            logits,
                            last_idx,
                            train_set,
                            train_preview_rev_dict,
                            args,
                            steps
                        )
                        write_tqdm_block(preview_block)
                        logging.info("\n%s", preview_block)

                token_accs.append(token_acc.item())
                last_accs.append(last_acc.item())
                pbar.set_description(
                    f"Epoch: {epoch+1}, Loss: {np.mean(losses):.4f}, TokenAcc: {token_acc:.4f}, LastAcc: {last_acc:.4f}, lr: {optimizer.param_groups[0]['lr']:.6f}"
                )
        logging.info(
                "[Epoch %d/%d] [train loss: %f] [token acc: %f] [last acc: %f]"
                % (epoch + 1, args.num_epoch, np.mean(losses), np.mean(token_accs), np.mean(last_accs))
                )
        validate_interval = max(1, args.validate_interval)
        should_validate = args.validate_during_training and ((epoch + 1) % validate_interval == 0)
        if should_validate:
            with torch.no_grad():
                train_mrr, train_hit1, train_hit3, train_hit5, train_hit10 = evaluate(model, train_eval_loader, device, args, train_valid, eval_valid, split_name="train")
                valid_mrr, valid_hit1, valid_hit3, valid_hit5, valid_hit10 = evaluate(model, valid_loader, device, args, train_valid, eval_valid, split_name="valid")

            metric_history["epoch"].append(epoch + 1)
            metric_history["train_mrr"].append(train_mrr)
            metric_history["valid_mrr"].append(valid_mrr)
            metric_history["train_hit1"].append(train_hit1)
            metric_history["valid_hit1"].append(valid_hit1)
            metric_history["train_hit3"].append(train_hit3)
            metric_history["valid_hit3"].append(valid_hit3)
            metric_history["train_hit5"].append(train_hit5)
            metric_history["valid_hit5"].append(valid_hit5)
            metric_history["train_hit10"].append(train_hit10)
            metric_history["valid_hit10"].append(valid_hit10)

            logging.info(
                "[Epoch %d Metrics] [Train MRR: %.6f Hit@1: %.6f Hit@3: %.6f Hit@5: %.6f Hit@10: %.6f] "
                "[Valid MRR: %.6f Hit@1: %.6f Hit@3: %.6f Hit@5: %.6f Hit@10: %.6f]",
                epoch + 1,
                train_mrr,
                train_hit1,
                train_hit3,
                train_hit5,
                train_hit10,
                valid_mrr,
                valid_hit1,
                valid_hit3,
                valid_hit5,
                valid_hit10,
            )

            if valid_hit1 > best_hit1:
                best_hit1 = valid_hit1
                best_epoch = epoch + 1
                torch.save(model.state_dict(), ckpt_path + "/best_model.pt".format(best_epoch))
                logging.info("[Checkpoint Saved] [Epoch: %d] [Best Hit@1: %f]", best_epoch, best_hit1)
            else:
                logging.info(
                    "[Checkpoint Skipped] [Epoch: %d] [Hit@1: %f] [Best Epoch: %d] [Best Hit@1: %f]",
                    epoch + 1,
                    valid_hit1,
                    best_epoch,
                    best_hit1,
                )

            plot_epoch_metrics(metric_history, save_path)
        else:
            logging.info(
                "[Epoch %d/%d] train/valid evaluation skipped. validate_during_training=%s validate_interval=%d",
                epoch + 1,
                args.num_epoch,
                args.validate_during_training,
                validate_interval,
            )

def checkpoint(args):
    args.dataset = os.path.join('data', args.dataset)
    save_path = os.path.join('models', args.save_dir)
    ckpt_path = os.path.join(save_path, 'checkpoint')
    if not os.path.exists(ckpt_path):
        print("Invalid path!")
        return
    logging.basicConfig(level=logging.DEBUG,
                    filename=save_path+'/test.log',
                    filemode='w',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader_kwargs = {
        "num_workers": max(0, args.num_workers),
        "pin_memory": device == "cuda",
        "worker_init_fn": seed_worker,
    }
    train_set = Seq2SeqDataset(data_path=args.dataset+"/", vocab_file=args.dataset+"/vocab.txt", device=device, split="train", args=args)
    test_set = TestDataset(data_path=args.dataset+"/", vocab_file=args.dataset+"/vocab.txt", device=device, src_file="test_triples.txt", split="test", args=args)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, collate_fn=test_set.collate_fn, shuffle=True, generator=build_dataloader_generator(args.seed + 3), **loader_kwargs)
    train_valid, eval_valid = train_set.get_next_valid()
    model = TransformerModel(args, train_set.dictionary)
    model.load_state_dict(torch.load(os.path.join(ckpt_path, args.ckpt)))
    model.args = args
    model = model.to(device)
    with torch.no_grad():
        evaluate(model, test_loader, device, args, train_valid, eval_valid, split_name="test")
    

if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    if args.test:
        checkpoint(args)
    else:
        train(args)
