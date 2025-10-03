# model_question_input.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Optional
from model import GELU  # reuse utility from original repo
import numpy as np

class PositionalEncoding(nn.Module):
    """
    Learned positional embeddings with a generous cap and clamping.
    Output shape matches your caller: [seq_len, batch, hidden]
    """
    def __init__(self, hidden_size: int, max_positions: int = 4096):
        super().__init__()
        self.max_positions = max_positions
        self.position_encoding = nn.Embedding(max_positions, hidden_size)

    def forward(self, input_pos: torch.Tensor = None, *,
                batch_len: int, start: int, seq_len: int) -> torch.Tensor:
        """
        Either pass `input_pos` (1D LongTensor of positions) or (batch_len, start, seq_len).
        We’ll build positions [start, start+1, ..., start+seq_len-1] and clamp to table size.
        Returns: [seq_len, batch_len, hidden_size]
        """
        if input_pos is None:
            device = self.position_encoding.weight.device
            pos = torch.arange(start, start + seq_len, device=device)
        else:
            pos = input_pos
        pos = pos.clamp_max(self.max_positions - 1)  # avoid OOB
        pe = self.position_encoding(pos)             # [seq_len, hidden]
        pe = pe.unsqueeze(1).expand(pe.size(0), batch_len, pe.size(1))  # [seq_len, batch, hidden]
        return pe

class Question2PathModel(nn.Module):
    """
    Encoder-only SQUIRE variant that encodes a *question* on the source side and
    decodes over the KG vocabulary on the target side (same as SQUIRE).
    """
    def __init__(self, args, dictionary, text_model_name="distilbert-base-uncased", freeze_text=True):
        super().__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer

        self.args = args
        self.dictionary = dictionary
        self.ntoken = len(dictionary)
        self.ninp = args.embedding_dim
        self.label_smooth = args.label_smooth

        # --- Question encoder (HF) ---
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name, use_fast=True)
        self.text_model = AutoModel.from_pretrained(text_model_name)
        if freeze_text:
            for p in self.text_model.parameters():
                p.requires_grad = False

        self.text_proj = nn.Linear(self.text_model.config.hidden_size, self.ninp)
        self.text_ln = nn.LayerNorm(self.ninp)
        self.text_dropout = nn.Dropout(self.args.dropout)

        # --- Target (decoder-input) embedding over KG vocab (tied output) ---
        self.tgt_embed = nn.Embedding(self.ntoken, self.ninp)
        nn.init.xavier_normal_(self.tgt_embed.weight)
        self.out_weight = self.tgt_embed.weight  # tie

        # --- Positional encoding + encoder stack (same topology as --encoder) ---
        self.pos_enc = PositionalEncoding(self.ninp, max_positions=4096)
        enc_layer = TransformerEncoderLayer(d_model=self.ninp, nhead=4,
                                            dim_feedforward=self.args.hidden_size,
                                            dropout=self.args.dropout)
        self.encoder_stack = TransformerEncoder(enc_layer, self.args.num_layers)

        self.fc = nn.Linear(self.ninp, self.ninp)
        self.gelu = GELU()

    @staticmethod
    def _generate_square_subsequent_mask(sz: int):
        m = (torch.triu(torch.ones(sz, sz)) == 1).t()
        m = m.float().masked_fill(m == 0, float("-inf")).masked_fill(m == 1, 0.0)
        return m

    def _encode_questions(self, question_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode question tokens with the HF model and project to SQUIRE dim.
        question_ids: (B, Lq), attention_mask: (B, Lq)
        returns: source_emb of shape [Lq, B, ninp] with position encodings added.
        """
        outputs = self.text_model(input_ids=question_ids, attention_mask=attention_mask, return_dict=True)
        q_hidden = outputs.last_hidden_state                                    # [B, Lq, H]
        src = self.text_proj(q_hidden)                                          # [B, Lq, ninp]
        src = self.text_ln(self.text_dropout(src)).transpose(0, 1)              # [Lq, B, ninp]
        src = src + self.pos_enc(batch_len=question_ids.size(0), start=0, seq_len=src.size(0))
        return src

    def logits(
        self,
        source: torch.Tensor,                # kept for interface compatibility (not used here)
        prev_outputs: torch.Tensor,          # (B, Lt)
        *,
        question_ids: torch.Tensor,          # (B, Lq)
        question_attention_mask: torch.Tensor,  # (B, Lq)
    ) -> torch.Tensor:
        """
        Return logits over KG vocab: [B, Lt, V]
        """
        device = prev_outputs.device
        bsz, out_len = prev_outputs.size(0), prev_outputs.size(1)

        # 1) Encode question to source embeddings
        source_emb = self._encode_questions(question_ids, question_attention_mask)  # [Lq, B, ninp]
        src_len = source_emb.size(0)

        # 2) Embed target prefix tokens (KG vocab) and add positions offset by src_len
        prev_emb = self.tgt_embed(prev_outputs).transpose(0, 1)                     # [Lt, B, ninp]
        prev_emb = prev_emb + self.pos_enc(batch_len=bsz, start=src_len, seq_len=out_len)

        # 3) Encoder-only stack with masking over the target portion
        mask = self._generate_square_subsequent_mask(out_len).to(device)
        enmask = torch.zeros(out_len + src_len, out_len + src_len, device=device)
        enmask[:, src_len:] = float("-inf")
        enmask[src_len:, src_len:] = mask

        full = torch.cat([source_emb, prev_emb], dim=0)                             # [Lq+Lt, B, ninp]
        enc_out = self.encoder_stack(full, mask=enmask)[src_len:, :, :].transpose(0, 1)  # [B, Lt, ninp]

        # 4) Tied output projection
        glue = self.gelu(self.fc(enc_out)).reshape(-1, self.ninp)                   # [B*Lt, ninp]
        logits = torch.matmul(glue, self.out_weight.t()).view(bsz, out_len, -1)     # [B, Lt, V]
        return logits

    def get_loss(
        self,
        source: torch.Tensor,
        prev_outputs: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        *,
        question_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        **_
    ) -> torch.Tensor:
        logits = self.logits(source, prev_outputs,
                              question_ids=question_ids,
                              question_attention_mask=question_attention_mask)
        lprobs = F.log_softmax(logits, dim=-1)
        gold = torch.gather(lprobs, -1, target.unsqueeze(-1)).squeeze(-1)
        loss = -(self.label_smooth * gold +
                 (1 - self.label_smooth) / (self.ntoken - 1) * lprobs.sum(dim=-1))
        loss = (loss * mask).sum() / mask.sum()
        return loss
