from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import model as model_module


class DummyDictionary:
    def __init__(self, vocab_size=32):
        self.vocab_size = vocab_size

    def __len__(self):
        return self.vocab_size

    def pad(self):
        return 0

    def bos(self):
        return 1

    def eos(self):
        return 2


class FakeBert(torch.nn.Module):
    def forward(self, input_ids, attention_mask):
        hidden_size = 768
        basis = torch.arange(hidden_size, device=input_ids.device, dtype=torch.float32)
        hidden = input_ids.float().unsqueeze(-1) + basis.view(1, 1, hidden_size)
        hidden = torch.remainder(hidden, 19.0) / 19.0
        hidden = hidden * attention_mask.unsqueeze(-1).float()
        return SimpleNamespace(last_hidden_state=hidden)


@pytest.fixture
def cpu_model(monkeypatch):
    def fake_from_pretrained(*args, **kwargs):
        return FakeBert()

    def cpu_positional_forward(self, batch_len, start, seq_len):
        input_pos = torch.tensor(
            [list(range(start + 1, start + seq_len + 1)) for _ in range(batch_len)],
            device=self.position_encoding.weight.device,
        )
        return self.position_encoding(input_pos).transpose(0, 1)

    monkeypatch.setattr(model_module.BertModel, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(model_module.PositionalEncoding, "forward", cpu_positional_forward)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    args = SimpleNamespace(
        embedding_dim=16,
        hidden_size=32,
        dropout=0.0,
        num_layers=1,
        encoder=True,
        label_smooth=0.1,
    )
    model = model_module.TransformerModel(args, DummyDictionary())
    model.eval()
    return model


@pytest.fixture
def dummy_batch():
    batch_size, input_len, output_len = 2, 8, 5
    input_ids = torch.tensor(
        [
            [5, 7, 9, 11, 0, 0, 0, 0],
            [4, 3, 2, 1, 8, 6, 0, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0],
        ],
        dtype=torch.long,
    )
    prev_outputs = torch.tensor(
        [
            [1, 4, 5, 6, 2],
            [1, 7, 8, 9, 2],
        ],
        dtype=torch.long,
    )
    return input_ids, attention_mask, prev_outputs


def test_forward_pass_runs_without_errors(cpu_model, dummy_batch):
    input_ids, attention_mask, prev_outputs = dummy_batch

    with torch.no_grad():
        logits = cpu_model.logits(input_ids, attention_mask, prev_outputs)

    assert logits is not None


def test_output_shape_is_correct(cpu_model, dummy_batch):
    input_ids, attention_mask, prev_outputs = dummy_batch

    with torch.no_grad():
        logits = cpu_model.logits(input_ids, attention_mask, prev_outputs)

    bsz, T = input_ids.size(0), prev_outputs.size(1)
    assert logits.shape == (bsz, T, len(cpu_model.dictionary))


def test_output_contains_no_nans(cpu_model, dummy_batch):
    input_ids, attention_mask, prev_outputs = dummy_batch

    with torch.no_grad():
        logits = cpu_model.logits(input_ids, attention_mask, prev_outputs)

    assert not torch.isnan(logits).any()


def test_forward_runs_on_cpu(cpu_model, dummy_batch):
    input_ids, attention_mask, prev_outputs = dummy_batch

    assert next(cpu_model.parameters()).device.type == "cpu"

    with torch.no_grad():
        logits = cpu_model.logits(input_ids, attention_mask, prev_outputs)

    assert logits.device.type == "cpu"


def test_output_is_deterministic_for_same_input(cpu_model, dummy_batch):
    input_ids, attention_mask, prev_outputs = dummy_batch

    with torch.no_grad():
        logits_1 = cpu_model.logits(input_ids, attention_mask, prev_outputs)
        logits_2 = cpu_model.logits(input_ids, attention_mask, prev_outputs)

    torch.testing.assert_close(logits_1, logits_2)

def test_cached_question_encoding_reuses_bert(cpu_model, dummy_batch, monkeypatch):
    input_ids, attention_mask, prev_outputs = dummy_batch
    bert_calls = {"count": 0}
    original_forward = cpu_model.bert.forward

    def counted_forward(*args, **kwargs):
        bert_calls["count"] += 1
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(cpu_model.bert, "forward", counted_forward)

    with torch.no_grad():
        encoded_source = cpu_model.encode_question(input_ids, attention_mask)
        calls_after_encode = bert_calls["count"]
        cached_logits = cpu_model.logits(
            input_ids,
            attention_mask,
            prev_outputs,
            encoded_source=encoded_source,
        )
        calls_after_cached_logits = bert_calls["count"]
        uncached_logits = cpu_model.logits(input_ids, attention_mask, prev_outputs)

    assert calls_after_encode == 1
    assert calls_after_cached_logits == 1
    assert bert_calls["count"] == 2
    torch.testing.assert_close(cached_logits, uncached_logits)

def test_loss_decreases_one_step(cpu_model, dummy_batch):
    input_ids, attention_mask, prev_outputs = dummy_batch
    target = prev_outputs.clone()
    mask = torch.ones_like(prev_outputs, dtype=torch.float32)
    optimizer = torch.optim.Adam(cpu_model.parameters(), lr=1e-3)

    loss1 = cpu_model.get_loss(input_ids, attention_mask, prev_outputs, target, mask)

    optimizer.zero_grad()
    loss1.backward()
    optimizer.step()

    with torch.no_grad():
        loss2 = cpu_model.get_loss(input_ids, attention_mask, prev_outputs, target, mask)

    assert loss2 <= loss1.detach() or torch.isclose(loss2, loss1.detach())
