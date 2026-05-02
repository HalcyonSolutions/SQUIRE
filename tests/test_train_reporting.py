from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train as train_module


class DummyReportingModel(torch.nn.Module):
    def __init__(self, vocab_size=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.training_flags = []

    def logits(self, input_ids, attention_mask, prev_outputs):
        self.training_flags.append(self.training)
        batch_size, seq_len = prev_outputs.shape
        logits = torch.zeros(batch_size, seq_len, self.vocab_size, dtype=torch.float32)
        for batch_idx in range(batch_size):
            for time_idx in range(seq_len):
                token_id = int(prev_outputs[batch_idx, time_idx].item())
                logits[batch_idx, time_idx, token_id] = 1.0
        return logits


def test_train_reporting_stats_use_eval_mode_and_restore_training_state():
    model = DummyReportingModel()
    model.train()

    samples = {
        "input_ids": torch.tensor([[3, 4, 0], [5, 6, 7]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long),
        "prev_outputs": torch.tensor([[1, 4, 2], [1, 5, 2]], dtype=torch.long),
        "target": torch.tensor([[1, 4, 2], [1, 5, 2]], dtype=torch.long),
        "mask": torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float32),
    }

    logits, pred, last_idx, token_acc, last_acc = train_module.compute_train_reporting_stats(model, samples)

    assert model.training is True
    assert model.training_flags == [False]
    assert logits.shape == (2, 3, model.vocab_size)
    torch.testing.assert_close(pred, samples["target"])
    torch.testing.assert_close(last_idx, torch.tensor([1, 1], dtype=torch.long))
    assert token_acc.item() == 1.0
    assert last_acc.item() == 1.0
