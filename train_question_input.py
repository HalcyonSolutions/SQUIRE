# train_question_input.py
import os, ast, csv
import math
import logging
import argparse
from typing import Tuple
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import transformers

from dictionary import Dictionary
from dataset_question_input import qa_batch_to_squire_samples, load_dictionary_or_init
from model_question_input import Question2PathModel
from dataset import Seq2SeqDataset as _LegacySeq2Seq  # for valid mask builders
# from QA_Dataloader import make_train_test_dataloaders_streaming as make_train_test_dataloaders
from QA_Dataloader import make_train_test_dataloaders
from QA_Dataloader import load_index_column_wise  # <- your loader


if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.85, device=0)

def get_args():
    p = argparse.ArgumentParser()
    # Core SQUIRE params
    p.add_argument("--dataset", required=True, help="Dataset folder under data/, e.g. kinshiphinton")
    p.add_argument("--save-dir", default="model_q_1")
    p.add_argument("--embedding-dim", type=int, default=256)
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--label-smooth", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--test-batch-size", type=int, default=32)
    p.add_argument("--num-epoch", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--save-interval", type=int, default=5)
    p.add_argument("--beam-size", type=int, default=128)
    p.add_argument("--max-len", type=int, default=3)
    p.add_argument("--l-punish", action="store_true")
    p.add_argument("--no-filter-gen", action="store_true")
    p.add_argument("--smart-filter", action="store_true")
    p.add_argument("--warmup", type=float, default=3.0)
    p.add_argument("--self-consistency", action="store_true")
    p.add_argument("--output-path", action="store_true")
    p.add_argument("--tail-only", action="store_true")

    # Question encoder
    p.add_argument("--question-tokenizer", default="distilbert-base-uncased")
    p.add_argument("--answer-tokenizer", default=None)
    p.add_argument("--unfreeze-text-encoder", action="store_true")

    # CSV + indices
    p.add_argument("--csv-path", required=True, help="QA CSV with Question/Answer/... columns")
    p.add_argument("--entity2id", required=True)
    p.add_argument("--relation2id", required=True)

    # misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-only", action="store_true")
    p.add_argument("--load-ckpt", type=str, default="", help="Path to checkpoint to eval (e.g. models_new/..../ckpt_30.pt)")
    p.add_argument("--FULL", action="store_true")
    return p.parse_args()

def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def sanity_check_vocab(dictionary, ent2id, rel2id):
    idx = dictionary.indices  # str -> int
    # Entities in KG vocab are stringified numbers; relations are 'R<id>'
    missing_e = [e for e in ent2id.values() if str(e) not in idx]
    missing_r = [r for r in rel2id.values() if f"R{r}" not in idx]
    if missing_e:
        print(f"[WARN] {len(missing_e)} entity ids missing from vocab.txt (e.g., {missing_e[:10]})")
    if missing_r:
        print(f"[WARN] {len(missing_r)} relation ids missing from vocab.txt (e.g., {missing_r[:10]})")


def _build_train_valid_masks(args, device) -> Tuple[dict, dict, Dictionary]:
    """
    Reuse original dataset code just to construct reachable-next-token masks (true_triples).
    This expects data/<dataset>/ files (entity2id.txt, relation2id.txt, *_triples_rev.txt, etc.)
    to exist — same requirement as SQUIRE.
    """
    ds_path = os.path.join("data", args.dataset)
    # Dummy args holder for the Seq2SeqDataset
    class _Obj: pass
    a = _Obj()
    a.trainset = "6_rev"            # name doesn't matter here; we only call get_next_valid()
    a.loop = False
    a.smart_filter = args.smart_filter
    train_set = _LegacySeq2Seq(data_path=ds_path + "/", vocab_file=ds_path + "/vocab.txt", device=device, args=a)
    train_valid, eval_valid = train_set.get_next_valid()
    return train_valid, eval_valid, train_set.dictionary

def _build_train_valid_masks_light(args, device):
    """
    Build sparse adjacency dicts, RAM-safe.

    true_triples[key] -> list[int] of valid next tokens
    valid_triples[key] -> set[int] of tails to filter during ranking

    Keys are the ones your evaluate() already uses:
      - first step (predict relation after BOS): key = head entity id
      - later when predicting next token given (h, r): key = vocab_size * r + h
    """
    ds_path = os.path.join("data", args.dataset)
    vocab_file = os.path.join(ds_path, "vocab.txt")

    dictionary = Dictionary.load(vocab_file)
    vocab_size = len(dictionary)

    def _read_triples(fname):
        triples = []
        with open(os.path.join(ds_path, fname), "r", encoding="utf-8") as f:
            for line in f:
                h, r, t = line.strip().split()
                # entities are plain ints in vocab, relations are "R<id>"
                h = int(h)
                t = int(t)
                assert r[0] == "R"
                r = int(r[1:])
                triples.append((h, r, t))
        return triples

    train_triples = _read_triples("train_triples_rev.txt")

    # Build sparse adjacency
    head2rels = {}      # h -> {r}
    hr2tails  = {}      # (h,r) -> {t}
    for h, r, t in train_triples:
        head2rels.setdefault(h, set()).add(r)
        hr2tails.setdefault((h, r), set()).add(t)

    true_triples = {}
    valid_triples = {}

    # First step: allowed relations for each head entity
    for h, rels in head2rels.items():
        true_triples[h] = sorted(rels)

    # Subsequent steps: allowed tails for (h, r)
    for (h, r), tails in hr2tails.items():
        key = vocab_size * r + h
        lst = sorted(tails)
        true_triples[key] = lst
        valid_triples[key] = set(lst)

    return true_triples, valid_triples, dictionary

@torch.no_grad()
def evaluate(model, dl: DataLoader, device, args, true_triples: dict, valid_triples: dict, dictionary: Dictionary):
    model.eval()
    beam_size = args.beam_size
    l_punish = args.l_punish
    max_len = 2 * args.max_len + 1
    restricted_punish = -30
    mrr = hit = hit1 = hit3 = hit5 = hit10 = hit20 = count = 0
    vocab_size = len(dictionary)
    eos = dictionary.eos()
    bos = dictionary.bos()

    rev_dict = {v: k for k, v in dictionary.indices.items()}
    lines = []

    with tqdm(dl, desc="testing") as pbar:
        for batch in pbar:
            # Adapt batch to SQUIRE tensors + pass-through question tensors
            samples = qa_batch_to_squire_samples(batch, dictionary, device, tail_only=args.tail_only)
            # Unpack
            source = samples["source"]

            vocab_size = len(dictionary)
            assert source.max().item() < vocab_size, f"OOB in source: {source.max()} >= {vocab_size}"

            
            target_gold = samples["target"].cpu()
            question_ids = samples["question_ids"]
            question_mask = samples["question_attention_mask"]

            batch_size = source.size(0)
            candidates = [dict() for _ in range(batch_size)]
            candidates_path = [dict() for _ in range(batch_size)]

            # Prepare prefix buffers
            prefix = torch.zeros([batch_size, beam_size, max_len], dtype=torch.long, device=device)
            prefix[:, :, 0].fill_(bos)
            lprob = torch.zeros([batch_size, beam_size], device=device)
            clen = torch.zeros([batch_size, beam_size], dtype=torch.long, device=device)

            # First step: choose relations given head (filtered)
            tmp_source = source
            tmp_prefix = torch.zeros([batch_size, 1], dtype=torch.long, device=device)
            tmp_prefix[:, 0].fill_(bos)

            logits = model.logits(
                tmp_source, tmp_prefix,
                question_ids=question_ids, question_attention_mask=question_mask
            ).squeeze(1)  # [B, V]
            if args.no_filter_gen:
                logits = F.log_softmax(logits, dim=-1)
            else:
                restricted = torch.ones([batch_size, vocab_size], device=device) * restricted_punish
                index = tmp_source[:, 1].detach().cpu().numpy()  # head
                for i in range(batch_size):
                    if index[i] in true_triples:
                        if args.smart_filter:
                            restricted[i] = true_triples[index[i]].to(device)
                        else:
                            idx = torch.LongTensor(true_triples[index[i]]).unsqueeze(0)
                            restricted[i] = -restricted_punish * torch.zeros(1, vocab_size, device=device)\
                                .scatter_(1, idx.to(device), 1) + restricted_punish
                logits = F.log_softmax(logits + restricted, dim=-1)

            argsort = torch.argsort(logits, dim=-1, descending=True)[:, :beam_size]
            assert argsort.max().item() < vocab_size, f"OOB in step-1 argsort: {argsort.max()} >= {vocab_size}"

            prefix[:, :, 1] = argsort
            lprob += torch.gather(input=logits, dim=-1, index=argsort)
            clen += 1

            for L in range(2, max_len):
                tmp_prefix = prefix.unsqueeze(2).repeat(1, 1, beam_size, 1)
                tmp_lprob = lprob.unsqueeze(-1).repeat(1, 1, beam_size)
                tmp_clen = clen.unsqueeze(-1).repeat(1, 1, beam_size)

                bb = batch_size * beam_size
                assert prefix.max().item() < vocab_size, f"OOB in prefix: {prefix.max()} >= {vocab_size}"

                all_logits = model.logits(
                    source.unsqueeze(1).repeat(1, beam_size, 1).view(bb, -1),
                    prefix.view(bb, -1),
                    question_ids=question_ids.unsqueeze(1).repeat(1, beam_size, 1).view(bb, -1),
                    question_attention_mask=question_mask.unsqueeze(1).repeat(1, beam_size, 1).view(bb, -1)
                ).view(batch_size, beam_size, max_len, -1)

                logits = torch.gather(
                    input=all_logits, dim=2,
                    index=clen.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, vocab_size)
                ).squeeze(2)

                # Filtering of reachable next tokens
                if args.no_filter_gen:
                    logits = F.log_softmax(logits, dim=-1)
                else:
                    restricted = torch.ones([batch_size, beam_size, vocab_size], device=device) * restricted_punish
                    hid = prefix[:, :, L-2]
                    if L == 2:
                        hid = source[:, None, 1].repeat(1, beam_size)
                    rid = prefix[:, :, L-1]
                    if L % 2 == 0:
                        index = vocab_size * rid + hid
                    else:
                        index = rid
                    index_np = index.detach().cpu().numpy()
                    for i in range(batch_size):
                        for j in range(beam_size):
                            key = index_np[i][j]
                            if key in true_triples:
                                if args.smart_filter:
                                    restricted[i][j] = true_triples[key].to(device)
                                else:
                                    idx = torch.LongTensor(true_triples[key]).unsqueeze(0)
                                    restricted[i][j] = -restricted_punish * torch.zeros(1, vocab_size, device=device)\
                                        .scatter_(1, idx.to(device), 1) + restricted_punish
                    logits = F.log_softmax(logits + restricted, dim=-1)

                argsort = torch.argsort(logits, dim=-1, descending=True)[:, :, :beam_size]
                tmp_clen = tmp_clen + 1
                tmp_prefix = tmp_prefix.scatter_(dim=-1, index=tmp_clen.unsqueeze(-1), src=argsort.unsqueeze(-1))
                tmp_lprob += torch.gather(input=logits, dim=-1, index=argsort)

                tmp_prefix = tmp_prefix.view(batch_size, -1, max_len)
                tmp_lprob  = tmp_lprob.view(batch_size, -1)
                tmp_clen   = tmp_clen.view(batch_size, -1)

                if L == max_len - 1:
                    pick = torch.argsort(tmp_lprob, dim=-1, descending=True)[:, :(2*beam_size)]
                else:
                    pick = torch.argsort(tmp_lprob, dim=-1, descending=True)[:, :beam_size]

                prefix = torch.gather(tmp_prefix, 1, pick.unsqueeze(-1).repeat(1, 1, max_len))
                lprob  = torch.gather(tmp_lprob,  1, pick)
                clen   = torch.gather(tmp_clen,   1, pick)

                # collect candidates when we emit <eos>
                for i in range(batch_size):
                    for j in range(prefix.size(1)):
                        if prefix[i, j, L].item() == eos:
                            candidate = prefix[i, j, L-1].item()
                            prob = lprob[i, j].item() / int(L/2) if l_punish else lprob[i, j].item()
                            lprob[i, j] -= 10000
                            if candidate not in candidates[i]:
                                score = math.exp(prob) if args.self_consistency else prob
                                candidates[i][candidate] = score
                                candidates_path[i][candidate] = prefix[i, j].detach().cpu().numpy()
                            else:
                                if prob > candidates[i][candidate]:
                                    candidates_path[i][candidate] = prefix[i, j].detach().cpu().numpy()
                                if args.self_consistency:
                                    candidates[i][candidate] += math.exp(prob)
                                else:
                                    candidates[i][candidate] = max(candidates[i][candidate], prob)

                # max_len reached without eos
                if L == max_len - 1:
                    for i in range(batch_size):
                        for j in range(prefix.size(1)):
                            candidate = prefix[i, j, L].item()
                            prob = lprob[i, j].item() / int(max_len/2) if l_punish else lprob[i, j].item()
                            if candidate not in candidates[i]:
                                score = math.exp(prob) if args.self_consistency else prob
                                candidates[i][candidate] = score
                                candidates_path[i][candidate] = prefix[i, j].detach().cpu().numpy()
                            else:
                                if prob > candidates[i][candidate]:
                                    candidates_path[i][candidate] = prefix[i, j].detach().cpu().numpy()
                                if args.self_consistency:
                                    candidates[i][candidate] += math.exp(prob)
                                else:
                                    candidates[i][candidate] = max(candidates[i][candidate], prob)

            # Rank and score
            for i in range(batch_size):
                hid = source[i, 1].item()
                rid = source[i, 2].item()
                index = vocab_size * rid + hid
                if index in valid_triples:
                    mask = valid_triples[index]
                    for tid in list(candidates[i].keys()):
                        # keep gold; downweight filtered duplicates
                        L = int(samples["mask"][i].sum().item())
                        gold_tail_id = target_gold[i, L-2].item()
                        # if tid != target_gold[i, 0].item():
                        if tid != gold_tail_id:
                            if args.smart_filter:
                                if mask[tid].item() == 0:
                                    candidates[i][tid] -= 100000
                            else:
                                if tid in mask:
                                    candidates[i][tid] -= 100000

                count += 1
                # sort by score
                ordered = sorted(candidates[i].items(), key=lambda kv: kv[1], reverse=True)
                candidate_ids = torch.tensor([cid for cid, _ in ordered], dtype=torch.long)
                
                # ranking = (candidate_ids[:] == target_gold[i, 0]).nonzero(as_tuple=False)
                L = int(samples["mask"][i].sum().item())          # how many target positions are real (incl. <eos>)
                gold_tail_id = target_gold[i, L-2].item()         # last token before <eos> is the tail
                ranking = (candidate_ids[:] == gold_tail_id).nonzero(as_tuple=False)

                path_token = f"{rev_dict[hid]} {rev_dict[rid]} {rev_dict[target_gold[i,0].item()]}\t"
                if ranking.nelement() != 0:
                    rank = 1 + ranking.item()
                    mrr += 1.0 / rank
                    if rank <= 1: hit1 += 1
                    if rank <= 3: hit3 += 1
                    if rank <= 5: hit5 += 1
                    if rank <= 10: hit10 += 1
                    if rank <= 20: hit20 += 1

                    hit += 1
                    path_token += str(ranking.item())
                else:
                    path_token += "wrong"
                lines.append(path_token + "\n")

            pbar.set_description(f"MRR: {mrr/max(1,count):.4f} | H@1: {hit1/max(1,count):.4f} | H@3: {hit3/max(1,count):.4f} | H@5: {hit5/max(1,count):.4f} | H@10: {hit10/max(1,count):.4f} | H@20: {hit20/max(1,count):.4f}")

    if args.output_path:
        with open("test_output_squire_question.txt", "w") as f:
            f.writelines(lines)

    logging.info("[MRR: %f] [Hit@1: %f] [Hit@3: %f] [Hit@5: %f] [Hit@10: %f] [Hit@20: %f]", 
                 mrr/max(1,count), hit1/max(1,count), hit3/max(1,count), hit5/max(1,count), hit10/max(1,count), hit20/max(1,count))
    return hit/max(1,count), hit1/max(1,count), hit3/max(1,count), hit5/max(1,count), hit10/max(1,count), hit20/max(1,count)

#! NEW
def run_fixed_demo(model, dictionary, device, args, df, id2ent, id2rel):
    # Select the fixed question
    fixed_question = 'what was the genre of the book authored by x1 known as "the early asimov"?'

    row = df[df["Question"].str.lower().str.strip() == fixed_question.lower().strip()]
    if len(row) == 0:
        print("Fixed question not found!")
        return
    
    row = row.iloc[0]

    q_text = row["Question"]
    start_ent = row["Source-Entity"]
    gt_ent = row["Answer-Entity"]

    # Convert to human-readable names
    # id2ent = dictionary.entityid2entity
    # id2rel = dictionary.relationid2relation
    # id2ent = df_ent_map       # to be passed in
    # id2rel = df_rel_map

    start_name = id2ent[start_ent]
    gt_name = id2ent[gt_ent]

    # Encode question
    enc = model.tokenizer.encode(q_text, return_tensors="pt").to(device)

    # Run generation
    pred, path_ids = model.generate_with_paths(enc)

    pred_name = id2ent.get(pred, "<unk>")

    # Decode path
    toks = path_ids
    path_str = ""
    for i in range(0, len(toks), 3):
        if i + 2 >= len(toks):
            break
        h = toks[i]
        r = toks[i+1]
        t = toks[i+2]
        path_str += f"{id2ent.get(h, '<?>')} --[{id2rel.get(r, '<?>')}]--> {id2ent.get(t, '<?>')}"
        if i + 3 < len(toks):
            path_str += " -- "

    print("\n==============")
    print("2-Hop:")
    print("Question:", q_text)
    print("KG Start :", start_name)
    print("KG GT Ans:", gt_name)
    print("Agent Ans:", pred_name)
    print("Path:", path_str if path_str else "(no path)")
    print("==============\n")

#! New
def build_dictionary_from_csv(csv_path):
    """
    Automatically builds a SQUIRE Dictionary() object from a QA CSV file.
    No external files required.
    """

    entity_ids = set()
    relation_ids = set()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # direct fields
            entity_ids.add(int(row["Source-Entity"]))
            entity_ids.add(int(row["Answer-Entity"]))
            # TODO map it to 9999
            relation_ids.add(0)

            # paths: list[[h, r, t], ...]
            if row["paths"].strip():
                try:
                    steps = ast.literal_eval(row["paths"])
                    for (h, r, t) in steps:
                        entity_ids.add(int(h))
                        entity_ids.add(int(t))
                        relation_ids.add(int(r))
                except:
                    pass

    # Now build dictionary
    dictionary = Dictionary()

    # special tokens
    dictionary.add_symbol("<s>")
    dictionary.add_symbol("</s>")
    dictionary.add_symbol("<pad>")

    # add entity ids
    for e in sorted(entity_ids):
        dictionary.add_symbol(str(e))

    # add relation ids (formatted like SQUIRE expects: R<id>)
    for r in sorted(relation_ids):
        dictionary.add_symbol(f"R{r}")

    return dictionary

# def run_fixed_demo(model, dictionary, device, args, df):
#     """
#     Print a single fixed example with:
#     - Question
#     - KG Start
#     - Ground-truth Answer
#     - Predicted Answer
#     - Decoded Path
#     """
#     model.eval()

#     # ---------- 1. SELECT THE EXAMPLE BY QUESTION TEXT ----------
#     fixed_question = 'what was the genre of the book authored by x1 known as "the early asimov"?'
#     row = df[df["Question"].str.strip().str.lower() == fixed_question.strip().lower()]

#     if len(row) == 0:
#         print("Fixed example question not found in dataset!")
#         return
    
#     row = row.iloc[0]

#     q_text = row["Question"]
#     start_ent = row["Source-Entity"]
#     gt_answer = row["Answer-Entity"]

#     # Convert IDs back to names
#     rev_ent = dictionary.entityid2entity
#     rev_rel = dictionary.relationid2relation

#     start_name = rev_ent[start_ent]
#     gt_name = rev_ent[gt_answer]

#     # ---------- 2. TOKENIZE QUESTION ----------
#     qtok = model.q_encoder.tokenizer
#     enc = qtok.encode(q_text, return_tensors="pt").to(device)

#     # ---------- 3. RUN MODEL GENERATION ----------
#     with torch.no_grad():
#         pred_ids, path_ids = model.generate_with_paths(enc)

#     pred_entity = pred_ids.item()
#     pred_name = rev_ent[pred_entity]

#     # ---------- 4. DECODE PATH ----------
#     path_str = ""
#     toks = path_ids.tolist()
#     # toks = [h, r, t, h, r, t,..., <end>]

#     for i in range(0, len(toks), 3):
#         h = toks[i]
#         if h == dictionary.eos(): break
#         r = toks[i+1]
#         t = toks[i+2]

#         hn = rev_ent.get(h, "<?>")
#         rn = rev_rel.get(r, "<?>")
#         tn = rev_ent.get(t, "<?>")

#         path_str += f"{hn} --[{rn}]--> {tn}"
#         if i + 3 < len(toks):
#             path_str += " -- "

#     # ---------- 5. PRINT OUTPUT ----------
#     print("\n==============")
#     print("2-Hop:")
#     print(f"Question: {q_text}")
#     print(f"KG Start : {start_name}")
#     print(f"KG GT Ans: {gt_name}")
#     print(f"Agent Ans: {pred_name}")
#     print(f"Path: {path_str}")
#     print("==============\n")

# def run_fixed_demo(model, dictionary, device, args, true_triples, valid_triples):
#     """
#     Runs ONE fixed question through the same beam-search logic as evaluate().
#     Prints: question, KG start, GT, prediction, and predicted path.
#     """

#     fixed_question = 'what was the genre of the book authored by x1 known as "the early asimov"?'

#     # ---- 1. Load the exact row from CSV ----
#     with open(args.csv_path, "r", encoding="utf-8") as f:
#         reader = csv.DictReader(f)
#         row = None
#         for r in reader:
#             if r["Question"].strip().lower() == fixed_question.lower():
#                 row = r
#                 break

#     if row is None:
#         print("Fixed question not found in CSV!")
#         return

#     head = int(row["Source-Entity"])
#     rel = int(row["query_rel"])
#     ans = int(row["Answer-Entity"])
#     path = ast.literal_eval(row["paths"])   # list[[h,r,t],...]

#     # ---- 2. Build SAMPLE identical to dataloader ----
#     bos = dictionary.bos()
#     sample = {
#         "source": torch.tensor([[bos, head, rel]]).to(device),
#         "target": torch.tensor([ans]).to(device),
#     }

#     # ---- 3. Run EXACT SAME DECODING as evaluate() ----
#     # Instead of copying evaluate(), we run evaluate() on a single-item dataloader.

#     from torch.utils.data import DataLoader

#     class SingleDataset(torch.utils.data.Dataset):
#         def __len__(self): return 1
#         def __getitem__(self, idx): return sample

#     single_dl = DataLoader(SingleDataset(), batch_size=1)

#     # Trick: evaluate() already returns candidate ranking,
#     # but does not expose predicted path. However:
#     # During evaluate, for target==answer_ent, "lines" contains the path.
#     # So we temporarily enable output_path to capture path.
#     old_output_path = args.output_path
#     args.output_path = True

#     evaluate(
#         model, dictionary,single_dl, device, args, true_triples, valid_triples
#     )

#     args.output_path = old_output_path

#     # Now read the file produced:
#     with open("test_output_squire.txt", "r") as f:
#         line = f.readline().strip()

#     # Format:  H R T <tab> ... path ... <tab> rank
#     parts = line.split("\t")
#     # parts[0] = "H R T"
#     # parts[1] = predicted path tokens
#     # parts[2] = rank (ignore)

#     pred_tail = int(parts[0].split()[-1])
#     pred_path_tokens = parts[1].split()

#     # ---- 4. Decode printable names ----
#     rev = dictionary.idx2token
#     def name(x): return rev[x]

#     print("\n2-Hop:")
#     print("Question:", fixed_question)
#     print("KG Start :", name(head))
#     print("KG GT Ans:", name(ans))
#     print("Agent Ans:", name(pred_tail))
#     print("Path:", " ".join(pred_path_tokens))
#     print("==============\n")

#! NEW 
@torch.no_grad()
def generate_path_for_question(model, dictionary, question_ids, question_mask, max_len=5):
    model.eval()
    device = question_ids.device

    bos = dictionary.bos()
    eos = dictionary.eos()

    prev = torch.tensor([[bos]], dtype=torch.long, device=device)
    generated = []

    for step in range(max_len):
        logits = model.logits(
            source=None,
            prev_outputs=prev,
            question_ids=question_ids,
            question_attention_mask=question_mask,
        )  # [1, T, V]

        step_logits = logits[:, -1, :]
        next_token = torch.argmax(step_logits, dim=-1)

        token_id = next_token.item()
        generated.append(token_id)

        prev = torch.cat([prev, next_token.unsqueeze(0)], dim=1)

        if token_id == eos:
            break

    # Your dictionary has no .string()
    readable = [dictionary.symbols[t] for t in generated]
    return readable



def train(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Build masks + dictionary from the SQUIRE data folder ---
    train_valid, eval_valid, dictionary = _build_train_valid_masks(args, device)
    # train_valid, eval_valid, dictionary = _build_train_valid_masks_light(args, device)

    # --- Load entity/relation maps for your QA CSV ---
    id2ent, ent2id = load_index_column_wise(args.entity2id)
    id2rel, rel2id = load_index_column_wise(args.relation2id)

    sanity_check_vocab(dictionary, ent2id, rel2id)

    # --- Your QA dataloaders ---
    loaders = make_train_test_dataloaders(
        csv_path=args.csv_path,
        entity2id=ent2id,
        relation2id=rel2id,
        # question_tokenizer_name=args.question-tokenizer if hasattr(args, "question-tokenizer") else args.question_tokenizer,
        question_tokenizer_name=args.question_tokenizer,
        answer_tokenizer_name=args.answer_tokenizer,
        batch_size=args.batch_size,
        text_only=False,
        FULL_dataset=args.FULL,
    )
    train_dl = loaders["train"]
    test_dl  = loaders["test"]

    # --- Model ---
    model = Question2PathModel(
        args,
        dictionary,
        text_model_name=args.question_tokenizer,
        freeze_text=not args.unfreeze_text_encoder
    ).to(device)

    #! NEW
    # ----- Your test question -----
    # TEST_QUESTION = "which continent does the country of citizenship of david thewlis belong to?" # 2hop
    # TEST_QUESTION = "what was the genre of the book authored by x1 known as \"the early asimov\"?"
    TEST_QUESTION = "on which continent was the founder of the company that developed windows live messenger born?" # 4hop/nhop

    # Tokenize using the model's tokenizer
    test_tok = model.tokenizer(
        TEST_QUESTION,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )
    test_question_ids = test_tok["input_ids"].to(device)
    test_question_mask = test_tok["attention_mask"].to(device)


    # hf_params = []
    # base_params = []
    # for n,p in model.named_parameters():
    #     if "text_model" in n:
    #         hf_params.append(p)
    #     else:
    #         base_params.append(p)

    # optimizer = optim.Adam(
    #     [
    #     {"params": base_params, "lr": args.lr, "weight_decay": args.weight_decay},
    #     {"params": hf_params,   "lr": min(args.lr, 5e-5), "weight_decay": 0.0},
    #     ]
    # )

    hf_params, base_params = [], []
    for n, p in model.named_parameters():
        if "text_model" in n:
            hf_params.append(p)
        else:
            base_params.append(p)

    optimizer = optim.Adam(
        [
        {"params": base_params, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": hf_params,   "lr": min(args.lr, 5e-5), "weight_decay": 0.0},
        ]
    )
    # --- Optimizer + scheduler ---
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_dl) * args.num_epoch
    warmup_steps = int(total_steps / args.warmup) if args.warmup > 0 else 0
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # --- Train loop ---
    save_path = os.path.join("models", args.save_dir)
    ckpt_path = os.path.join(save_path, "checkpoint")
    os.makedirs(ckpt_path, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        filename=os.path.join(save_path, "train.log"),
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    for epoch in range(1, args.num_epoch + 1):
        model.train()
        losses = []
        with tqdm(train_dl, desc=f"training (epoch {epoch})") as pbar:
            for batch in pbar:
                samples = qa_batch_to_squire_samples(batch, dictionary, device, tail_only=args.tail_only)
                optimizer.zero_grad()
                loss = model.get_loss(**samples)
                loss.backward()
                optimizer.step()
                scheduler.step()
                losses.append(loss.item())
                pbar.set_postfix(loss=sum(losses)/len(losses))

        logging.info("[Epoch %d/%d] [train loss: %.6f]", epoch, args.num_epoch, sum(losses)/len(losses))

        if (epoch % args.save_interval == 0) or (epoch == args.num_epoch):
            torch.save(model.state_dict(), os.path.join(ckpt_path, f"ckpt_{epoch}.pt"))
            evaluate(model, test_dl, device, args, train_valid, eval_valid, dictionary)
            #! NEW
            # run_fixed_demo(model, dictionary, device, args, loaders["test"].dataset.df, id2ent, id2rel)
            path = generate_path_for_question(model, dictionary, test_question_ids, test_question_mask)
            print(f"\n[DEBUG] Generated path for test question at epoch {epoch}:")
            print(" → ".join(path))
            

def main():
    args = get_args()
    if not args.test_only:
        train(args)
        return

    # -------- test-only path --------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # masks + dictionary
    train_valid, eval_valid, dictionary = _build_train_valid_masks(args, device)
    # entity/relation maps
    id2ent, ent2id = load_index_column_wise(args.entity2id)
    id2rel, rel2id = load_index_column_wise(args.relation2id)
    sanity_check_vocab(dictionary, ent2id, rel2id)
    # dataloaders (we only need test)
    loaders = make_train_test_dataloaders(
        csv_path=args.csv_path,
        entity2id=ent2id,
        relation2id=rel2id,
        question_tokenizer_name=getattr(args, "question_tokenizer", "facebook/bart-base"),
        answer_tokenizer_name=args.answer_tokenizer,
        batch_size=max(1, args.test_batch_size),
        text_only=False,
        FULL_dataset=args.FULL,
    )
    test_dl = loaders["test"]
    # model
    model = Question2PathModel(
        args, dictionary,
        text_model_name=args.question_tokenizer,
        freeze_text=not args.unfreeze_text_encoder
    ).to(device)

    assert args.load_ckpt, "--load-ckpt is required with --test-only"
    state = torch.load(args.load_ckpt, map_location=device)
    model.load_state_dict(state, strict=True)
    evaluate(model, test_dl, device, args, train_valid, eval_valid, dictionary)

if __name__ == "__main__":
    main()
