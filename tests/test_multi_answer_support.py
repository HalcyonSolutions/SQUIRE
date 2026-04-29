import pandas as pd
import torch

from dataset import Seq2SeqDataset, TestDataset as SquireTestDataset, _get_row_answer_entities
from dictionary import Dictionary
from train import first_correct_candidate_rank, normalize_gold_answer_ids


class DummyTokenizer:
    pad_token_id = 0

    def __call__(self, text, padding, truncation, max_length, return_tensors):
        token_count = min(len(str(text).split()), max_length)
        token_ids = list(range(1, token_count + 1))
        padded_ids = token_ids + [self.pad_token_id] * (max_length - token_count)
        attention_mask = [1] * token_count + [0] * (max_length - token_count)
        return {
            "input_ids": torch.tensor([padded_ids], dtype=torch.long),
            "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
        }


class Args:
    loop = False
    prob = 0
    smart_filter = False
    max_q_len = 8
    question_file = "qa.csv"
    verbose = False
    test_paraphrased = False
    train_paraphrased = False


def build_vocab(vocab_path):
    dictionary = Dictionary()
    dictionary.add_symbol("LOOP")
    dictionary.add_symbol("R0")
    dictionary.add_symbol("R1")
    for entity_id in ("0", "1", "2", "3"):
        dictionary.add_symbol(entity_id)
    dictionary.save(str(vocab_path))
    return dictionary


def build_temp_multi_answer_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(Seq2SeqDataset, "_tokenizer", DummyTokenizer())

    qa_rows = pd.DataFrame(
        [
            {
                "Question": "Who is connected to head one?",
                "Source": "Head One",
                "Source-Entity": "Q1",
                "Answer": "Single Answer",
                "Answer-Entity": "Q2",
                "Answers": None,
                "Answers-Entities": None,
                "Path-Key": "P1",
                "SplitLabel": "test",
            },
            {
                "Question": "Who is connected to head two?",
                "Source": "Head Two",
                "Source-Entity": "Q1",
                "Answer": None,
                "Answer-Entity": "['Q3', 'Q4']",
                "Answers": "['Multi Answer A', 'Multi Answer B']",
                "Answers-Entities": None,
                "Path-Key": "P1->P1",
                "SplitLabel": "test",
            },
        ]
    )
    qa_rows.to_csv(tmp_path / "qa.csv", index=False)

    (tmp_path / "entity2id.txt").write_text("Q1\t0\nQ2\t1\nQ3\t2\nQ4\t3\n", encoding="utf-8")
    (tmp_path / "relation2id.txt").write_text("P1\t0\n", encoding="utf-8")
    build_vocab(tmp_path / "vocab.txt")

    args = Args()
    return SquireTestDataset(
        data_path=f"{tmp_path}/",
        vocab_file=f"{tmp_path}/vocab.txt",
        device="cpu",
        split="test",
        args=args,
    )


def test_answer_parser_supports_alternate_multi_answer_columns():
    row = pd.Series({"Answers-Entities": "['Q3', 'Q4']"})
    assert _get_row_answer_entities(row) == ["Q3", "Q4"]


def test_test_dataset_returns_single_and_multi_gold_answers(tmp_path, monkeypatch):
    dataset = build_temp_multi_answer_dataset(tmp_path, monkeypatch)

    single = dataset[0]
    multi = dataset[1]

    assert single["gold_answers"].tolist() == [dataset.dictionary.indices["Q2"]]
    assert single["target"].tolist() == [single["gold_answers"][0].item()]

    assert multi["gold_answers"].tolist() == [
        dataset.dictionary.indices["Q3"],
        dataset.dictionary.indices["Q4"],
    ]
    assert multi["target"].tolist() == [multi["gold_answers"][0].item()]


def test_test_dataset_collate_fn_pads_gold_answers(tmp_path, monkeypatch):
    dataset = build_temp_multi_answer_dataset(tmp_path, monkeypatch)
    batch = dataset.collate_fn([dataset[0], dataset[1]])

    assert batch["gold_answers"].shape == (2, 2)
    assert batch["gold_answers"][0].tolist() == [dataset.dictionary.indices["Q2"], -1]
    assert batch["gold_answers"][1].tolist() == [
        dataset.dictionary.indices["Q3"],
        dataset.dictionary.indices["Q4"],
    ]


def test_multi_answer_ranking_accepts_any_gold_answer():
    gold_answers = normalize_gold_answer_ids(torch.tensor([17, -1, 23]), fallback_target_id=99)
    rank_idx = first_correct_candidate_rank([5, 9, 23, 42], gold_answers)

    assert gold_answers == [17, 23]
    assert rank_idx == 2
    assert 1 / (rank_idx + 1) == 1 / 3
    assert int(rank_idx is not None and rank_idx < 3) == 1
