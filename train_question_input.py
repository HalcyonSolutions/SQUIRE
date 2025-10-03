# train_question_input.py
import os
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
    mrr = hit = hit1 = hit3 = hit10 = count = 0
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
                    if rank <= 10: hit10 += 1
                    hit += 1
                    path_token += str(ranking.item())
                else:
                    path_token += "wrong"
                lines.append(path_token + "\n")

            pbar.set_description(f"MRR: {mrr/max(1,count):.4f} | H@1: {hit1/max(1,count):.4f} | H@3: {hit3/max(1,count):.4f} | H@10: {hit10/max(1,count):.4f}")

    if args.output_path:
        with open("test_output_squire_question.txt", "w") as f:
            f.writelines(lines)

    logging.info("[MRR: %f] [Hit@1: %f] [Hit@3: %f] [Hit@10: %f]", mrr/max(1,count), hit1/max(1,count), hit3/max(1,count), hit10/max(1,count))
    return hit/max(1,count), hit1/max(1,count), hit3/max(1,count), hit10/max(1,count)

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
    save_path = os.path.join("models_new", args.save_dir)
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

def main():
    args = get_args()
    train(args) if not args.test_only else None

if __name__ == "__main__":
    main()

# python train_question_input.py \
#   --dataset kinshiphinton \
#   --csv-path data/kinshiphinton/kinship_hinton_qa_2hop.csv \
#   --entity2id data/kinshiphinton/entity2id.txt \
#   --relation2id data/kinshiphinton/relation2id.txt \
#   --question-tokenizer facebook/bart-base \
#   --batch-size 32 --num-epoch 20 --beam-size 128 --max-len 3 \
#   --save-dir model_q_kinship --unfreeze-text-encoder