# model_text_input.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import List
import math

# Reuse SQUIRE's GELU + PositionalEncoding to match original behavior
from model import GELU, PositionalEncoding  # existing file

# class Text2PathModel(nn.Module):
#     """
#     Drop-in replacement for TransformerModel that:
#       - Receives the SAME inputs as SQUIRE: source = [<s>, head, relation], prev_outputs (KG tokens)
#       - Internally converts (head, relation) -> text prompt -> AutoModel embeddings for the source side
#       - Keeps the target side exactly the same (embedding over KG vocab + tied output)
#     """
#     def __init__(self, args, dictionary, text_model_name="distilbert-base-uncased", freeze_text=True):
#         super().__init__()
#         from torch.nn import TransformerEncoder, TransformerEncoderLayer

#         self.args = args
#         self.dictionary = dictionary
#         self.ntoken = len(dictionary)
#         self.ninp = args.embedding_dim
#         self.label_smooth = args.label_smooth

#         # Text encoder on source side
#         self.tokenizer = AutoTokenizer.from_pretrained(text_model_name, use_fast=True)
#         self.text_model = AutoModel.from_pretrained(text_model_name)
#         if freeze_text:
#             for p in self.text_model.parameters():
#                 p.requires_grad = False

#         self.text_proj = nn.Linear(self.text_model.config.hidden_size, self.ninp)

#         # Target (decoder-input) embedding over KG vocab (same as original)
#         self.tgt_embed = nn.Embedding(self.ntoken, self.ninp)
#         nn.init.xavier_normal_(self.tgt_embed.weight)

#         # Positional encoding (same class as original)
#         self.pos_enc = PositionalEncoding(self.ninp)

#         # Encoder-only transformer stack (same topology as when --encoder=True)
#         enc_layer = TransformerEncoderLayer(
#             d_model=self.ninp,
#             nhead=4,
#             dim_feedforward=self.args.hidden_size,
#             dropout=self.args.dropout
#         )
#         self.encoder_stack = TransformerEncoder(enc_layer, self.args.num_layers)

#         # Projection + tied output weights
#         self.fc = nn.Linear(self.ninp, self.ninp)
#         self.gelu = GELU()
#         self.out_weight = self.tgt_embed.weight  # weight tying

#         # Build reverse dict for quick id->symbol mapping
#         self.rev_dict = {v: k for k, v in self.dictionary.indices.items()}

class Text2PathModel(nn.Module):
    def __init__(self, args, dictionary, text_model_name="distilbert-base-uncased", freeze_text=True):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel
        from torch.nn import TransformerEncoder, TransformerEncoderLayer

        self.args = args
        self.dictionary = dictionary
        self.ntoken = len(dictionary)
        self.ninp = args.embedding_dim
        self.label_smooth = args.label_smooth

        # --- HF text backbone ---
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name, use_fast=True)
        self.text_model = AutoModel.from_pretrained(text_model_name)
        if freeze_text:
            for p in self.text_model.parameters():
                p.requires_grad = False

        # Build special tokens from KG dictionary
        # dictionary.indices: {symbol(str) -> idx(int)}
        # entities are numeric strings ("123"), relations look like "R45" (also reverse are "R<id+N>")
        add_tokens = set()
        for sym in self.dictionary.indices.keys():
            if sym.isdigit():                  # entity id
                add_tokens.add(f"[E_{sym}]")
            elif sym.startswith("R"):          # relation token
                rid = sym[1:]
                add_tokens.add(f"[R_{rid}]")
        add_tokens.add("[TAIL]")

        # Register with tokenizer and resize LM embeddings
        # self.tokenizer.add_special_tokens({"additional_special_tokens": sorted(list(add_tokens))})
        # self.text_model.resize_token_embeddings(len(self.tokenizer))
        self.tokenizer.add_special_tokens({"additional_special_tokens": sorted(list(add_tokens))})
        self.text_model.resize_token_embeddings(len(self.tokenizer))
        # IMPORTANT: allow the new special-token rows to learn
        # keep encoder blocks frozen, but unfreeze input embeddings
        emb = self.text_model.get_input_embeddings()
        for p in emb.parameters():
            p.requires_grad = True

        # Project text hidden size -> SQUIRE dim
        self.text_proj = nn.Linear(self.text_model.config.hidden_size, self.ninp)
        self.text_ln = nn.LayerNorm(self.ninp)
        self.text_dropout = nn.Dropout(self.args.dropout)

        # Decoder-side (targets) remain SQUIRE-style over KG vocab
        self.tgt_embed = nn.Embedding(self.ntoken, self.ninp)
        nn.init.xavier_normal_(self.tgt_embed.weight)

        self.pos_enc = PositionalEncoding(self.ninp)
        enc_layer = TransformerEncoderLayer(d_model=self.ninp, nhead=4,
                                            dim_feedforward=self.args.hidden_size,
                                            dropout=self.args.dropout)
        self.encoder_stack = TransformerEncoder(enc_layer, self.args.num_layers)

        self.fc = nn.Linear(self.ninp, self.ninp)
        self.gelu = GELU()
        self.out_weight = self.tgt_embed.weight  # weight tying

        # Reverse mapping: idx -> symbol string
        self.rev_dict = {v: k for k, v in self.dictionary.indices.items()}


    # @staticmethod
    # def _generate_square_subsequent_mask(sz: int):
    #     m = (torch.triu(torch.ones(sz, sz)) == 1).t()
    #     m = m.float().masked_fill(m == 0, float('-inf')).masked_fill(m == 1, 0.0)
    #     return m

    # def _verbalize_batch(self, source: torch.Tensor) -> List[str]:
    #     """
    #     source: [bsz, 3] with [<s>, head_token_id, relation_token_id]
    #     Returns list of text prompts like "HID 123 REL 45 TAIL ?"
    #     """
    #     bsz = source.size(0)
    #     texts = []
    #     for i in range(bsz):
    #         head_tok = source[i, 1].item()
    #         rel_tok  = source[i, 2].item()
    #         head_str = self.rev_dict[head_tok]  # entity id string (e.g., "123")
    #         rel_str  = self.rev_dict[rel_tok]   # relation symbol (e.g., "R45")
    #         # parse relation id integer
    #         if rel_str.startswith('R'):
    #             rel_id = rel_str[1:]
    #         else:
    #             rel_id = rel_str
    #         texts.append(f"HID {head_str} REL {rel_id} TAIL ?")
    #     return texts

    # def logits(self, source: torch.Tensor, prev_outputs: torch.Tensor):
    #     """
    #     Matches the signature used everywhere in SQUIRE.
    #     - source: [bsz, 3]  (unchanged interface)
    #     - prev_outputs: [bsz, out_len]  (KG tokens)
    #     Returns: [bsz, out_len, ntoken]
    #     """
    #     device = prev_outputs.device
    #     bsz = source.size(0)

    #     # 1) Build text prompts from (head, relation)
    #     texts = self._verbalize_batch(source)

    #     # 2) Tokenize & encode with HF model
    #     enc = self.tokenizer(
    #         texts,
    #         padding=True,
    #         truncation=True,
    #         max_length=getattr(self.args, "max_text_len", 64),
    #         return_tensors="pt"
    #     )
    #     input_ids = enc["input_ids"].to(device)
    #     attn_mask = enc["attention_mask"].to(device)
    #     text_out = self.text_model(input_ids=input_ids, attention_mask=attn_mask, return_dict=True).last_hidden_state
    #     # [bsz, src_len, H] -> project to ninp
    #     source_emb = self.text_proj(text_out)  # [bsz, src_len, ninp]
    #     src_len = source_emb.size(1)

    #     # 3) Positional encodings & concat with target-side embeddings (like original)
    #     source_emb = source_emb.transpose(0, 1)  # [src_len, bsz, ninp]
    #     source_emb = source_emb + self.pos_enc(bsz, 0, src_len)  # same API as original

    #     out_len = prev_outputs.size(1)
    #     prev_emb = self.tgt_embed(prev_outputs).transpose(0, 1)  # [out_len, bsz, ninp]
    #     prev_emb = prev_emb + self.pos_enc(bsz, src_len, out_len)

    #     # 4) Mask and transformer stack
    #     mask = self._generate_square_subsequent_mask(out_len).to(device)
    #     enmask = torch.zeros(out_len + src_len, out_len + src_len, device=device)
    #     enmask[:, src_len:] = float("-inf")
    #     enmask[src_len:, src_len:] = mask
    #     full = torch.cat([source_emb, prev_emb], dim=0)
    #     enc_out = self.encoder_stack(full, mask=enmask)[src_len:, :, :].transpose(0, 1)  # [bsz, out_len, ninp]

    #     # 5) Output projection with tied weights
    #     glue = self.gelu(self.fc(enc_out)).reshape(-1, self.ninp)  # [bsz*out_len, ninp]
    #     logits = torch.matmul(glue, self.out_weight.t()).view(bsz, out_len, -1)
    #     return logits

    # def get_loss(self, source, prev_outputs, target, mask, **unused):
    #     logits = self.logits(source, prev_outputs)
    #     lprobs = F.log_softmax(logits, dim=-1)
    #     gold = torch.gather(lprobs, -1, target.unsqueeze(-1)).squeeze(-1)
    #     loss = -(self.label_smooth * gold + (1 - self.label_smooth) / (self.ntoken - 1) * lprobs.sum(dim=-1))
    #     loss = (loss * mask).sum() / mask.sum()
    #     return loss

    def _verbalize_batch(self, source: torch.Tensor):
        """
        source: [bsz, 3] = [<s>, head_id_token, rel_id_token] in KG dictionary indices
        returns: list of strings like "[E_123] [R_45] [TAIL]"
        """
        bsz = source.size(0)
        texts = []
        for i in range(bsz):
            head_tok = source[i, 1].item()
            rel_tok  = source[i, 2].item()
            head_str = self.rev_dict[head_tok]      # e.g., "123" (entity id)
            rel_str  = self.rev_dict[rel_tok]       # e.g., "R45"
            # Build special tokens
            e_tok = f"[E_{head_str}]"               # "[E_123]"
            if rel_str.startswith("R"):
                r_tok = f"[R_{rel_str[1:]}]"        # "[R_45]"
            else:
                r_tok = f"[R_{rel_str}]"
            texts.append(f"{e_tok} {r_tok} [TAIL]")  # exactly 3 tokens
        return texts

    @staticmethod
    def _generate_square_subsequent_mask(sz: int):
        m = (torch.triu(torch.ones(sz, sz)) == 1).t()
        m = m.float().masked_fill(m == 0, float('-inf')).masked_fill(m == 1, 0.0)
        return m

    def logits(self, source: torch.Tensor, prev_outputs: torch.Tensor):
        device = prev_outputs.device
        bsz = source.size(0)
        out_len = prev_outputs.size(1)

        # 1) Build text prompts (3 tokens per example)
        texts = self._verbalize_batch(source)

        # 2) Tokenize without adding CLS/SEP so length stays 3
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=False,
            add_special_tokens=False,     # <-- critical
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].to(device)        # [bsz, 3]
        attn_mask = enc["attention_mask"].to(device)   # [bsz, 3]

        # 3) Encode with HF model
        text_out = self.text_model(input_ids=input_ids, attention_mask=attn_mask, return_dict=True).last_hidden_state
        # [bsz, 3, H] -> project to ninp
        src = self.text_proj(text_out)                 # [bsz, 3, ninp]
        src = self.text_ln(self.text_dropout(src))
        src_len = src.size(1)                          # should be 3

        # 4) Positional enc + concat with target embeddings (SQUIRE style)
        source_emb = src.transpose(0, 1)               # [3, bsz, ninp]
        source_emb = source_emb + self.pos_enc(bsz, 0, src_len)

        # print(f"source_emb:\n{source_emb.shape}")
        # choice = input("Continue? (y/n): ")
        # if choice.lower() != 'y': exit(0)

        prev_emb = self.tgt_embed(prev_outputs).transpose(0, 1)  # [out_len, bsz, ninp]
        prev_emb = prev_emb + self.pos_enc(bsz, src_len, out_len)

        # 5) Encoder-only stack with masks identical to original
        mask = self._generate_square_subsequent_mask(out_len).to(device)
        enmask = torch.zeros(out_len + src_len, out_len + src_len, device=device)
        enmask[:, src_len:] = float("-inf")
        enmask[src_len:, src_len:] = mask

        full = torch.cat([source_emb, prev_emb], dim=0)
        enc_out = self.encoder_stack(full, mask=enmask)[src_len:, :, :].transpose(0, 1)  # [bsz, out_len, ninp]

        # 6) Output projection (tied weights over KG vocab)
        glue = self.gelu(self.fc(enc_out)).reshape(-1, self.ninp)
        logits = torch.matmul(glue, self.out_weight.t()).view(bsz, out_len, -1)
        return logits

    def get_loss(self, source, prev_outputs, target, mask, **unused):
        logits = self.logits(source, prev_outputs)
        lprobs = F.log_softmax(logits, dim=-1)
        gold = torch.gather(lprobs, -1, target.unsqueeze(-1)).squeeze(-1)
        loss = -(self.label_smooth * gold + (1 - self.label_smooth) / (self.ntoken - 1) * lprobs.sum(dim=-1))
        loss = (loss * mask).sum() / mask.sum()
        return loss
