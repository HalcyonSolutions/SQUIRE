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
from tqdm import tqdm
import logging
import ast
import transformers
from iterative_training import Iter_trainer
import math
import matplotlib

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
    parser.add_argument("--eval-preview-count", default=0, type=int, help="number of readable evaluation examples to print per split; set to 0 to disable")
    parser.add_argument("--eval-preview-topk", default=3, type=int, help="number of top predictions to show inside each evaluation preview")
    parser.add_argument("--train-preview-count", default=0, type=int, help="number of readable training examples to print per preview; set to 0 to disable")
    parser.add_argument("--train-preview-interval", default=100, type=int, help="print readable training preview every N optimizer steps when enabled")
    parser.add_argument("--train-preview-topk", default=5, type=int, help="number of top tokens to show for each previewed final answer position")
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

def get_row_text(row, key, default="N/A"):
    if row is None or key not in row:
        return default
    value = row[key]
    if value is None:
        return default
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
        if symbol.startswith("R"):
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

def build_eval_preview(dataset, sample_id, head_id, relation_id, target_id, candidate_ids, candidate_paths, preview_topk, rank_idx, rev_dict, eos, bos):
    row = None
    if dataset is not None and hasattr(dataset, "data"):
        row = dataset.data.iloc[int(sample_id)]

    source_entity = get_row_text(row, "Source-Entity", decode_token(head_id, rev_dict, dataset))
    relation_chain = format_query_chain(row, decode_token(relation_id, rev_dict, dataset))
    gold_answer = get_row_text(row, "Answer-Entity", decode_token(target_id, rev_dict, dataset))
    predicted_answer = decode_token(candidate_ids[0], rev_dict, dataset) if candidate_ids else "N/A"
    predicted_path = "N/A"
    if candidate_paths:
        predicted_path = format_generated_path(source_entity, candidate_paths[0], rev_dict, dataset, eos, bos)

    top_predictions = [decode_token(candidate_id, rev_dict, dataset) for candidate_id in candidate_ids[:preview_topk]]
    rank_text = str(rank_idx + 1) if rank_idx is not None else "not ranked"
    status = "correct" if candidate_ids and candidate_ids[0] == target_id else "incorrect"

    lines = [
        f"Question: {get_row_text(row, 'Question')}",
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

def evaluate(model, dataloader, device, args, true_triples=None, valid_triples=None, split_name="eval"):
    model.eval()
    beam_size = args.beam_size
    l_punish = args.l_punish
    max_len = 2 * args.max_len + 1
    restricted_punish = -30
    mrr, hit, hit1, hit3, hit10, count = (0, 0, 0, 0, 0, 0)
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

    def debug_question(sample_id):
        if dataset is not None and hasattr(dataset, "data"):
            sample_id = int(sample_id)
            if 0 <= sample_id < len(dataset.data):
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

    with tqdm(dataloader, desc=f"{split_label} Eval") as pbar:
        for samples in pbar:
            pbar.set_description(
                "%s Eval | MRR: %f, Hit@1: %f, Hit@3: %f, Hit@10: %f"
                % (split_label, mrr/max(1, count), hit1/max(1, count), hit3/max(1, count), hit10/max(1, count))
            )
            batch_size = samples["input_ids"].size(0)
            debug_limit = 0
            debug_blocks = []
            if count < 3:
                debug_limit = min(batch_size, 3 - count)
                for i in range(debug_limit):
                    sample_id = samples["ids"][i].detach().cpu().tolist() if "ids" in samples else count + i
                    head_id = samples["head_id"][i].detach().cpu().tolist()
                    relation_id = samples["relation_id"][i].detach().cpu().tolist()
                    target_id = samples["target"][i].detach().cpu().view(-1)[0].tolist()
                    debug_blocks.append([
                        "================ SAMPLE =================",
                        "[INPUT]",
                        f"Sample ID: {sample_id}",
                        f"Question: {debug_question(sample_id)}",
                        f"Head ID: {debug_token(head_id)}",
                        f"Relation ID: {debug_token(relation_id)}",
                        f"Target ID: {debug_token(target_id)}",
                    ])

            # if count < 5:
            #     debug_prefix = torch.zeros([batch_size, 3], dtype=torch.long).to(device)
            #     debug_prefix[:, 0] = model.dictionary.bos()
            #     debug_prefix[:, 1] = samples["head_id"]
            #     debug_prefix[:, 2] = samples["relation_id"]

            #     debug_logits = model.logits(
            #         samples["input_ids"],
            #         samples["attention_mask"],
            #         debug_prefix
            #     )[:, -1, :]  # last position predicts entity

            #     probs = F.softmax(debug_logits, dim=-1)
            #     topk = torch.topk(probs, k=10, dim=-1)

            #     for i in range(min(3, batch_size)):
            #         gold = samples["target"][i].item()

            #         sorted_ids = torch.argsort(probs[i], descending=True)
            #         rank = (sorted_ids == gold).nonzero(as_tuple=False)
            #         rank = rank.item() if rank.numel() > 0 else None

            #         print("\n[LOGITS ENTITY DEBUG]")
            #         print("Gold:", gold)
            #         print("Top-10:", topk.indices[i].tolist())
            #         print("Gold rank:", rank)

            candidates = [dict() for i in range(batch_size)]
            candidates_path = [dict() for i in range(batch_size)]
            input_ids = samples["input_ids"].unsqueeze(dim=1).repeat(1, beam_size, 1).to(device)
            attention_mask = samples["attention_mask"].unsqueeze(dim=1).repeat(1, beam_size, 1).to(device)
            prefix = torch.zeros([batch_size, beam_size, max_len], dtype=torch.long).to(device)
            prefix[:, :, 0].fill_(model.dictionary.bos())
            lprob = torch.zeros([batch_size, beam_size]).to(device)
            clen = torch.zeros([batch_size, beam_size], dtype=torch.long).to(device)
            # first token: choose beam_size from only vocab_size, initiate prefix
            tmp_input_ids = samples["input_ids"]
            tmp_attention_mask = samples["attention_mask"]
            tmp_prefix = torch.zeros([batch_size, 1], dtype=torch.long).to(device)
            tmp_prefix[:, 0].fill_(model.dictionary.bos())
            if count < 3:
                for i in range(debug_limit):
                    debug_blocks[i].extend(debug_model_input_lines(tmp_input_ids[i], tmp_attention_mask[i], tmp_prefix[i]))
            logits = model.logits(tmp_input_ids, tmp_attention_mask, tmp_prefix).squeeze()
            if args.no_filter_gen:
                logits = F.log_softmax(logits, dim=-1)
            else:
                restricted = torch.ones([batch_size, vocab_size]) * restricted_punish
                # index = tmp_input_ids[:, 1].cpu().numpy()
                index = samples["head_id"].cpu().numpy()
                for i in range(batch_size):
                    if index[i] in true_triples:
                        if args.smart_filter:
                            restricted[i] = true_triples[index[i]]
                        else:
                            idx = torch.LongTensor(true_triples[index[i]]).unsqueeze(0)
                            restricted[i] = -restricted_punish * torch.zeros(1, vocab_size).scatter_(1, idx, 1) + restricted_punish
                logits = F.log_softmax(logits+restricted.to(device), dim=-1) # batch_size * vocab_size
            logits = logits.view(-1, vocab_size)
            argsort = torch.argsort(logits, dim=-1, descending=True)[:, :beam_size]
            prefix[:, :, 1] = argsort[:, :]
            lprob += torch.gather(input=logits, dim=-1, index=argsort)
            clen += 1
            # if count < 3:
            #     debug_logits = logits.view(batch_size, -1)
            #     for i in range(debug_limit):
            #         debug_blocks[i].extend([
            #             "[STEP 0 LOGITS]",
            #             "Top-10: " + debug_top_values(debug_logits[i], 10, use_softmax=True),
            #         ])
            target = samples["target"].cpu()
            for l in range(2, max_len):
                tmp_prefix = prefix.unsqueeze(dim=2).repeat(1, 1, beam_size, 1)
                tmp_lprob = lprob.unsqueeze(dim=-1).repeat(1, 1, beam_size)    
                tmp_clen = clen.unsqueeze(dim=-1).repeat(1, 1, beam_size)
                bb = batch_size * beam_size
                if l <= 3 and count < 3:
                    for i in range(debug_limit):
                        debug_blocks[i].append("[Model Input]")
                        for j in range(min(3, beam_size)):
                            debug_blocks[i].append(f"beam {j}:")
                            debug_blocks[i].extend(debug_model_input_lines(input_ids[i][j], attention_mask[i][j], prefix[i][j]))
                all_logits = model.logits(input_ids.view(bb, -1), attention_mask.view(bb, -1), prefix.view(bb, -1)).view(batch_size, beam_size, max_len, -1)
                logits = torch.gather(input=all_logits, dim=2, index=clen.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, vocab_size)).squeeze(2)
                # restrict to true_triples, compute index for true_triples
                if args.no_filter_gen:
                    logits = F.log_softmax(logits, dim=-1)
                else:
                    restricted = torch.ones([batch_size, beam_size, vocab_size]) * restricted_punish
                    hid = prefix[:, :, l-2]
                    if l == 2:
                        hid = samples["head_id"].unsqueeze(1).repeat(1, beam_size)
                    rid = prefix[:, :, l-1]
                    if l % 2 == 0:
                        index = vocab_size * rid + hid
                    else:
                        index = rid
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
                if l <= 3 and count < 3:
                    for i in range(debug_limit):
                        debug_blocks[i].append(f"[BEAM STEP {l}]")
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
                        if prefix[i][j][l].item() == eos:
                            candidate = prefix[i][j][l-1].item()
                            if l_punish:
                                prob = lprob[i][j].item() / int(l / 2)
                            else:
                                prob = lprob[i][j].item()
                            if count < 3 and i < debug_limit:
                                path_tokens = prefix[i][j, :l + 1].detach().cpu().tolist()
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
                                candidates_path[i][candidate] = prefix[i][j].cpu().numpy()
                            else:
                                if prob > candidates[i][candidate]:
                                    candidates_path[i][candidate] = prefix[i][j].cpu().numpy()
                                if args.self_consistency:
                                    candidates[i][candidate] += math.exp(prob)
                                else:
                                    candidates[i][candidate] = max(candidates[i][candidate], prob)
                # no <end> but reach max_len
                if l == max_len-1:
                    for i in range(batch_size):
                        for j in range(beam_size*2):
                            candidate = prefix[i][j][l].item()
                            if l_punish:
                                prob = lprob[i][j].item() / int(max_len/2)
                            else:
                                prob = lprob[i][j].item()
                            if count < 3 and i < debug_limit:
                                path_tokens = prefix[i][j, :l + 1].detach().cpu().tolist()
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
                                candidates_path[i][candidate] = prefix[i][j].cpu().numpy()
                            else:
                                if prob > candidates[i][candidate]:
                                    candidates_path[i][candidate] = prefix[i][j].cpu().numpy()
                                if args.self_consistency:
                                    candidates[i][candidate] += math.exp(prob)
                                else:                             
                                    candidates[i][candidate] = max(candidates[i][candidate], prob)
            target = samples["target"].cpu()
            for i in range(batch_size):
                debug_sample = count < 3 and i < debug_limit
                hid = samples["head_id"][i].item()
                rid = samples["relation_id"][i].item()
                index = vocab_size * rid + hid
                if debug_sample:
                    debug_gold = target[i].detach().cpu().view(-1)[0].tolist()
                    debug_blocks[i].extend([
                        "[BEFORE FILTER]",
                        "Top candidates (id, score):",
                    ])
                    debug_blocks[i].extend(debug_candidate_lines(candidates[i]))
                    debug_blocks[i].append(f"Gold: {debug_token(debug_gold)}")
                if index in valid_triples:
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

                # path_token = rev_dict[hid] + " " + rev_dict[rid] + " " + rev_dict[target[i].item()] + '\t'
                path_token = safe_lookup(hid, rev_dict) + " " + safe_lookup(rid, rev_dict) + " " + safe_lookup(target_id, rev_dict) + '\t'

                if ranking.nelement() != 0:
                    rank_idx = ranking[0].item()
                    path = candidate_path[rank_idx]
                    for token in path[1:-1]:
                        path_token += (rev_dict[token]+' ')
                    path_token += (rev_dict[path[-1]]+'\t')
                    path_token += str(rank_idx)
                    ranking_value = 1 + rank_idx
                    mrr += (1 / ranking_value)
                    hit += 1
                    if ranking_value <= 1:
                        hit1 += 1
                    if ranking_value <= 3:
                        hit3 += 1
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
                    write_tqdm_block("\n".join(debug_blocks[i]))

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
                        )
                    )
    
    if args.output_path:
        with open("test_output_squire.txt", "w") as f:
            f.writelines(lines)
    metric_denominator = max(1, count)
    if preview_blocks:
        for idx, block in enumerate(preview_blocks, start=1):
            write_tqdm_block(f"[{split_name.upper()} Example {idx}]")
            write_tqdm_block(block)
            if idx != len(preview_blocks):
                tqdm.write("")
    summary = "[%s] MRR: %.6f, Hit@1: %.6f, Hit@3: %.6f, Hit@10: %.6f" % (
        split_name.upper(),
        mrr/metric_denominator,
        hit1/metric_denominator,
        hit3/metric_denominator,
        hit10/metric_denominator,
    )
    tqdm.write(summary)
    logging.info(summary)
    return mrr/metric_denominator, hit1/metric_denominator, hit3/metric_denominator, hit10/metric_denominator


def plot_epoch_metrics(metric_history, save_dir):
    epochs = metric_history["epoch"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    metric_specs = [
        ("mrr", "MRR"),
        ("hit1", "Hit@1"),
        ("hit3", "Hit@3"),
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
    train_set = Seq2SeqDataset(data_path=args.dataset+"/", vocab_file=args.dataset+"/vocab.txt", device=device, split="train", args=args)
    valid_set = TestDataset(data_path=args.dataset+"/", vocab_file=args.dataset+"/vocab.txt", device=device, src_file="valid_triples.txt", split="test", args=args) # in kinship there's no valid set
    test_set = TestDataset(data_path=args.dataset+"/", vocab_file=args.dataset+"/vocab.txt", device=device, src_file="test_triples.txt", split="test", args=args)
    train_eval_set = TestDataset(data_path=args.dataset+"/", vocab_file=args.dataset+"/vocab.txt", device=device, src_file="train_triples.txt", split="train", args=args)
    train_valid, eval_valid = train_set.get_next_valid()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, collate_fn=train_set.collate_fn, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.test_batch_size, collate_fn=test_set.collate_fn, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, collate_fn=test_set.collate_fn, shuffle=True)
    train_eval_loader = DataLoader(train_eval_set, batch_size=args.test_batch_size, collate_fn=test_set.collate_fn, shuffle=False)
    
    model = TransformerModel(args, train_set.dictionary).to(device)
    train_preview_rev_dict = build_rev_dict(train_set.dictionary)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = len(train_loader)
    total_step_num = len(train_loader) * args.num_epoch
    warmup_steps = total_step_num / args.warmup
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, warmup_steps, total_step_num)
    
    if args.iter:
        iter_trainer = Iter_trainer(args.dataset, args.iter_batch_size, 32, 4)
        iter_epoch = []
        max_len = args.max_len
        total = 0
        for i in range(1, max_len+1):
            total += (1/i)
        epochs = 0
        for i in range(1, max_len+1):
            iter_epoch.append(int(args.num_epoch/(total*i)))
            epochs += int(args.num_epoch/(total*i))
        iter_epoch[-1] += (args.num_epoch-epochs)
        curr_iter = -1
        curr_iter_epoch = 0
        logging.info(
                    "[Iter0: %d] [Iter1: %d] [Iter2: %d]"
                    % (iter_epoch[0], iter_epoch[1], iter_epoch[2])
                    )
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
        "train_hit10": [],
        "valid_hit10": [],
    }
    steps = 0
    for epoch in range(args.num_epoch):
        if args.iter:
            if curr_iter_epoch == 0: # start next iteration
                curr_iter += 1
                curr_iter_epoch = iter_epoch[curr_iter]
                # label new dataset
                if curr_iter > 0:
                    logging.info("--------Iterating--------")
                    (src_lines, tgt_lines) = iter_trainer.get_iter(model, curr_iter)
                    train_set.src_lines += src_lines
                    train_set.tgt_lines += tgt_lines
                    train_loader = DataLoader(train_set, batch_size=args.batch_size, collate_fn=train_set.collate_fn, shuffle=True)
                # new scheduler
                step_num = len(train_loader) * curr_iter_epoch
                warmup_steps = step_num / args.warmup
                if curr_iter != 0:
                    optimizer = optim.Adam(model.parameters(), lr=args.lr / 5, weight_decay=args.weight_decay) # fine-tuning with smaller lr
                    warmup_steps = 0
                scheduler = transformers.get_linear_schedule_with_warmup(optimizer, warmup_steps, step_num)
            curr_iter_epoch -= 1
        model.train()
        with tqdm(train_loader, desc="training") as pbar:
            losses = []
            token_accs = []
            last_accs = []
            for batch_idx, samples in enumerate(pbar):
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
                train_mrr, train_hit1, train_hit3, train_hit10 = evaluate(model, train_eval_loader, device, args, train_valid, eval_valid, split_name="train")
                valid_mrr, valid_hit1, valid_hit3, valid_hit10 = evaluate(model, valid_loader, device, args, train_valid, eval_valid, split_name="valid")

            metric_history["epoch"].append(epoch + 1)
            metric_history["train_mrr"].append(train_mrr)
            metric_history["valid_mrr"].append(valid_mrr)
            metric_history["train_hit1"].append(train_hit1)
            metric_history["valid_hit1"].append(valid_hit1)
            metric_history["train_hit3"].append(train_hit3)
            metric_history["valid_hit3"].append(valid_hit3)
            metric_history["train_hit10"].append(train_hit10)
            metric_history["valid_hit10"].append(valid_hit10)

            logging.info(
                "[Epoch %d Metrics] [Train MRR: %.6f Hit@1: %.6f Hit@3: %.6f Hit@10: %.6f] "
                "[Valid MRR: %.6f Hit@1: %.6f Hit@3: %.6f Hit@10: %.6f]",
                epoch + 1,
                train_mrr,
                train_hit1,
                train_hit3,
                train_hit10,
                valid_mrr,
                valid_hit1,
                valid_hit3,
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
    train_set = Seq2SeqDataset(data_path=args.dataset+"/", vocab_file=args.dataset+"/vocab.txt", device=device, args=args)
    test_set = TestDataset(data_path=args.dataset+"/", vocab_file=args.dataset+"/vocab.txt", device=device, src_file="test_triples.txt", args=args)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, collate_fn=test_set.collate_fn, shuffle=True)
    train_valid, eval_valid = train_set.get_next_valid()
    model = TransformerModel(args, train_set.dictionary)
    model.load_state_dict(torch.load(os.path.join(ckpt_path, args.ckpt)))
    model.args = args
    model = model.to(device)
    with torch.no_grad():
        evaluate(model, test_loader, device, args, train_valid, eval_valid, split_name="test")
    

if __name__ == "__main__":
    args = get_args()
    if args.test:
        checkpoint(args)
    else:
        train(args)
