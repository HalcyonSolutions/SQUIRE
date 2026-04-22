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
    
    # question input related
    parser.add_argument("--question-file", default="kinship_hinton_qa_nhop.csv", type=str, help="path to question file for question input csv file")
    parser.add_argument("--max-q-len", default=32, type=int, help="maximum number of tokens for the question") # used for Bert
    args = parser.parse_args()
    return args

def safe_lookup(x, rev_dict=None):
    return rev_dict[x] if x in rev_dict else str(x)

def evaluate(model, dataloader, device, args, true_triples=None, valid_triples=None):
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
    for k in model.dictionary.indices.keys():
        v = model.dictionary.indices[k]
        rev_dict[v] = k
    with tqdm(dataloader, desc="testing") as pbar:
        for samples in pbar:
            pbar.set_description("MRR: %f, Hit@1: %f, Hit@3: %f, Hit@10: %f" % (mrr/max(1, count), hit1/max(1, count), hit3/max(1, count), hit10/max(1, count)))
            batch_size = samples["input_ids"].size(0)
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
            logits = model.logits(tmp_input_ids, tmp_attention_mask, tmp_prefix).squeeze()
            if args.no_filter_gen:
                logits = F.log_softmax(logits, dim=-1)
            else:
                restricted = torch.ones([batch_size, vocab_size]) * restricted_punish
                index = tmp_input_ids[:, 1].cpu().numpy()
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
            target = samples["target"].cpu()
            for l in range(2, max_len):
                tmp_prefix = prefix.unsqueeze(dim=2).repeat(1, 1, beam_size, 1)
                tmp_lprob = lprob.unsqueeze(dim=-1).repeat(1, 1, beam_size)    
                tmp_clen = clen.unsqueeze(dim=-1).repeat(1, 1, beam_size)
                bb = batch_size * beam_size
                all_logits = model.logits(input_ids.view(bb, -1), attention_mask.view(bb, -1), prefix.view(bb, -1)).view(batch_size, beam_size, max_len, -1)
                logits = torch.gather(input=all_logits, dim=2, index=clen.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, vocab_size)).squeeze(2)
                # restrict to true_triples, compute index for true_triples
                if args.no_filter_gen:
                    logits = F.log_softmax(logits, dim=-1)
                else:
                    restricted = torch.ones([batch_size, beam_size, vocab_size]) * restricted_punish
                    hid = prefix[:, :, l-2]
                    if l == 2:
                        hid = input_ids[:, :, 1]
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
                hid = samples["input_ids"][i][1].item()
                rid = samples["input_ids"][i][2].item()
                index = vocab_size * rid + hid
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
                count += 1
                candidate_ = sorted(zip(candidates[i].items(), candidates_path[i].items()), key=lambda x:x[0][1], reverse=True)
                candidate = [pair[0][0] for pair in candidate_]
                candidate_path = [pair[1][1] for pair in candidate_]
                candidate = torch.from_numpy(np.array(candidate))
                ranking = (candidate[:] == target[i]).nonzero()

                # path_token = rev_dict[hid] + " " + rev_dict[rid] + " " + rev_dict[target[i].item()] + '\t'
                path_token = safe_lookup(hid, rev_dict) + " " + safe_lookup(rid, rev_dict) + " " + safe_lookup(target[i].item(), rev_dict) + '\t'

                if ranking.nelement() != 0:
                    path = candidate_path[ranking]
                    for token in path[1:-1]:
                        path_token += (rev_dict[token]+' ')
                    path_token += (rev_dict[path[-1]]+'\t')
                    path_token += str(ranking.item())
                    ranking = 1 + ranking.item()
                    mrr += (1 / ranking)
                    hit += 1
                    if ranking <= 1:
                        hit1 += 1
                    if ranking <= 3:
                        hit3 += 1
                    if ranking <= 10:
                        hit10 += 1
                else:
                    path_token += "wrong"
                lines.append(path_token+'\n')
    
    if args.output_path:
        with open("test_output_squire.txt", "w") as f:
            f.writelines(lines)
    logging.info("[MRR: %f] [Hit@1: %f] [Hit@3: %f] [Hit@10: %f]" % (mrr/count, hit1/count, hit3/count, hit10/count))
    return mrr/count, hit1/count, hit3/count, hit10/count


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
            for samples in pbar:
                optimizer.zero_grad()
                loss = model.get_loss(**samples)
                loss.backward()
                optimizer.step()
                scheduler.step()
                steps += 1
                losses.append(loss.item())
                pbar.set_description("Epoch: %d, Loss: %0.8f, lr: %0.6f" % (epoch + 1, np.mean(losses), optimizer.param_groups[0]['lr']))
        logging.info(
                "[Epoch %d/%d] [train loss: %f]"
                % (epoch + 1, args.num_epoch, np.mean(losses))
                )
        with torch.no_grad():
            train_mrr, train_hit1, train_hit3, train_hit10 = evaluate(model, train_eval_loader, device, args, train_valid, eval_valid)
            valid_mrr, valid_hit1, valid_hit3, valid_hit10 = evaluate(model, valid_loader, device, args, train_valid, eval_valid)

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
        evaluate(model, test_loader, device, args, train_valid, eval_valid)
    

if __name__ == "__main__":
    args = get_args()
    if args.test:
        checkpoint(args)
    else:
        train(args)
