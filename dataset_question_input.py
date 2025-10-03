# dataset_question_input.py
from typing import Dict, Any, List
import torch
from dictionary import Dictionary

def load_dictionary_or_init(vocab_path: str, data_path: str) -> Dictionary:
    """
    Loads KG vocabulary used by SQUIRE. If missing, raises to surface setup issues early.
    """
    try:
        return Dictionary.load(vocab_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Vocab not found at {vocab_path}. "
            f"Generate it via utils.py for your dataset under {data_path}/."
        ) from e

def _to_dict_id(dictionary: Dictionary, token: str) -> int:
    # Robust index with fallback to <unk> if something slips through.
    return dictionary.index(token)

def _path_to_token_seq(path_steps: List[List[int]]) -> List[str]:
    """
    Convert a path [[h, r, t], [h2, r2, t2], ...] to a sequence:
        Rr1, t1, Rr2, t2, ...   (entities as raw ids, relations as 'R<id>')
    """
    toks: List[str] = []
    for (h, r, t) in path_steps:
        toks.append(f"R{int(r)}")
        toks.append(str(int(t)))
    return toks

def qa_batch_to_squire_samples(
    batch: Dict[str, Any],
    dictionary: Dictionary,
    device: torch.device,
    bos_when_missing_relation: bool = False,
    tail_only: bool = False,
) -> Dict[str, Any]:
    """
    Adapt a QA batch (from QA_Dataloader.make_train_test_dataloaders) to SQUIRE's expected fields,
    while passing question tensors through for the question-encoder model.
    """
    # ---- Build `source` = [<s>, head, relation] (still needed for filtering/eval) ----
    bsz = batch["question_ids"].size(0)
    bos = dictionary.bos()
    # Entities in your CSV are integer ids; in the SQUIRE vocab they are stored as strings.
    head_ids = [ _to_dict_id(dictionary, str(int(e))) for e in batch["query_ent"].tolist() ]
    rel_ids  = [ _to_dict_id(dictionary, f"R{int(r)}") for r in batch["query_rel"].tolist() ]
    source = torch.stack([
        torch.full((bsz,), bos, dtype=torch.long),
        torch.tensor(head_ids, dtype=torch.long),
        torch.tensor(rel_ids, dtype=torch.long),
    ], dim=1).to(device)

    # ---- Build path/target over KG vocab (alternating relation, entity) ----
    targets: List[torch.Tensor] = []
    lengths: List[int] = []
    eos = dictionary.eos()
    pad = dictionary.pad()

    for i in range(bsz):
        if tail_only:
            # 1-token objective: predict only the final tail (then <eos>)
            t = int(batch["answer_ent"][i].item())
            seq_tokens = [str(t)]
        else:
            steps: List[List[int]] = batch["paths"][i]  # [[h, r, t], ...] or []
            if len(steps) == 0:
                # Fall back to a single-hop sequence if no explicit path provided
                r = int(batch["query_rel"][i].item())
                t = int(batch["answer_ent"][i].item())
                seq_tokens = [f"R{r}", str(t)]
            else:
                seq_tokens = _path_to_token_seq(steps)

        # map to vocab ids and append <eos>
        seq_ids = [ _to_dict_id(dictionary, tok) for tok in seq_tokens ]
        seq_ids.append(eos)
        t_i = torch.tensor(seq_ids, dtype=torch.long)
        targets.append(t_i)
        lengths.append(len(seq_ids))

    max_len = max(lengths)
    prev_outputs = torch.full((bsz, max_len), pad, dtype=torch.long)
    target       = torch.full((bsz, max_len), pad, dtype=torch.long)
    mask         = torch.zeros(bsz, max_len, dtype=torch.float)

    for i, t_i in enumerate(targets):
        L = t_i.size(0)
        prev_outputs[i, 0] = bos
        if L > 1:
            prev_outputs[i, 1:L] = t_i[:-1]
        target[i, :L] = t_i
        mask[i, :L] = 1.0

    return {
        # SQUIRE training/eval tensors
        "source":        source.to(device),
        "prev_outputs":  prev_outputs.to(device),
        "target":        target.to(device),
        "mask":          mask.to(device),
        # Question tensors (pass-through to the model)
        "question_ids":            batch["question_ids"].to(device),
        "question_attention_mask": batch["question_attention_mask"].to(device),
        # Keep around for bookkeeping if needed
        "query_ent": batch["query_ent"].to(device),
        "query_rel": batch["query_rel"].to(device),
        "answer_ent": batch["answer_ent"].to(device),
        "hops": batch["hops"].to(device),
    }
