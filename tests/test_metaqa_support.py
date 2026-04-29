import pandas as pd
import torch

from dataset import Seq2SeqDataset, Seq2SeqDataset_MetaQA, TestDataset_MetaQA as MetaQATestDataset
from train_metaqa import build_entity_candidate_ids, build_entity_candidate_mask, multi_answer_nll_loss


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
    smart_filter = False
    max_q_len = 8
    question_file = "metaqa.csv"
    train_question_file = None
    eval_question_file = None
    test_paraphrased = False
    train_paraphrased = False


def build_temp_metaqa_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(Seq2SeqDataset, "_tokenizer", DummyTokenizer())

    qa_rows = pd.DataFrame(
        [
            {
                "Question": "Who acted in movie a?",
                "Source": "Movie A",
                "Source-Entity": "movie_a",
                "Answers": "['Actor One', 'Actor Two']",
                "Answers-Entities": "['actor_1', 'actor_2']",
                "SplitLabel": "train",
            },
            {
                "Question": "Who directed movie a?",
                "Source": "Movie A",
                "Source-Entity": "movie_a",
                "Answer": "Director One",
                "Answer-Entity": "director_1",
                "SplitLabel": "test",
            },
        ]
    )
    qa_rows.to_csv(tmp_path / "metaqa.csv", index=False)

    (tmp_path / "entity2id.txt").write_text(
        "movie_a\t0\nactor_1\t1\nactor_2\t2\ndirector_1\t3\n",
        encoding="utf-8",
    )
    (tmp_path / "relation2id.txt").write_text("acted_in\t0\n", encoding="utf-8")

    args = Args()
    train_set = Seq2SeqDataset_MetaQA(
        data_path=f"{tmp_path}/",
        vocab_file=f"{tmp_path}/vocab.txt",
        device="cpu",
        split="train",
        args=args,
    )
    test_args = Args()
    test_args.eval_question_file = "metaqa.csv"
    test_set = MetaQATestDataset(
        data_path=f"{tmp_path}/",
        vocab_file=f"{tmp_path}/vocab.txt",
        device="cpu",
        split="test",
        args=test_args,
    )
    return train_set, test_set


def test_metaqa_train_dataset_returns_endpoint_supervision(tmp_path, monkeypatch):
    train_set, _ = build_temp_metaqa_dataset(tmp_path, monkeypatch)

    sample = train_set[0]

    assert "Paths" not in train_set.data.columns
    assert sample["head_id"].item() == train_set.dictionary.indices["0"]
    assert sample["gold_answers"].tolist() == [
        train_set.dictionary.indices["1"],
        train_set.dictionary.indices["2"],
    ]
    assert sample["target"].item() == sample["gold_answers"][0].item()


def test_metaqa_test_dataset_collate_pads_gold_answers(tmp_path, monkeypatch):
    train_set, test_set = build_temp_metaqa_dataset(tmp_path, monkeypatch)

    batch = test_set.collate_fn(
        [
            {
                **train_set[0],
            },
            test_set[0],
        ]
    )

    assert batch["gold_answers"].shape == (2, 2)
    assert batch["gold_answers"][0].tolist() == [
        train_set.dictionary.indices["1"],
        train_set.dictionary.indices["2"],
    ]
    assert batch["gold_answers"][1].tolist() == [
        train_set.dictionary.indices["3"],
        -1,
    ]
    assert batch["target"].tolist() == [
        train_set.dictionary.indices["1"],
        train_set.dictionary.indices["3"],
    ]


def test_metaqa_split_label_is_optional(tmp_path, monkeypatch):
    monkeypatch.setattr(Seq2SeqDataset, "_tokenizer", DummyTokenizer())

    qa_rows = pd.DataFrame(
        [
            {
                "Question": "Who acted in movie a?",
                "Source-Entity": "movie_a",
                "Answers-Entities": "['actor_1', 'actor_2']",
            }
        ]
    )
    qa_rows.to_csv(tmp_path / "no_split.csv", index=False)
    (tmp_path / "entity2id.txt").write_text("movie_a\t0\nactor_1\t1\nactor_2\t2\n", encoding="utf-8")
    (tmp_path / "relation2id.txt").write_text("acted_in\t0\n", encoding="utf-8")

    args = Args()
    args.question_file = "no_split.csv"
    dataset = Seq2SeqDataset_MetaQA(
        data_path=f"{tmp_path}/",
        vocab_file=f"{tmp_path}/vocab.txt",
        device="cpu",
        split="train",
        args=args,
    )

    assert len(dataset) == 1


def test_metaqa_entity_candidate_mask_and_multi_answer_loss(tmp_path, monkeypatch):
    train_set, test_set = build_temp_metaqa_dataset(tmp_path, monkeypatch)
    entity_candidate_ids = build_entity_candidate_ids(train_set, test_set)
    entity_mask = build_entity_candidate_mask(len(train_set.dictionary), entity_candidate_ids, device="cpu")

    relation_id = train_set.dictionary.indices["R0"]
    assert entity_mask[relation_id].item() is False
    assert entity_mask[train_set.dictionary.pad()].item() is False
    for entity_symbol in ("0", "1", "2", "3"):
        assert entity_mask[train_set.dictionary.indices[entity_symbol]].item() is True

    logits = torch.full((1, len(train_set.dictionary)), -10.0)
    logits[0, train_set.dictionary.indices["1"]] = 0.1
    logits[0, train_set.dictionary.indices["2"]] = 0.2

    multi_loss, _ = multi_answer_nll_loss(
        logits,
        torch.tensor([[train_set.dictionary.indices["1"], train_set.dictionary.indices["2"]]], dtype=torch.long),
        entity_mask,
    )
    single_loss, _ = multi_answer_nll_loss(
        logits,
        torch.tensor([[train_set.dictionary.indices["1"], -1]], dtype=torch.long),
        entity_mask,
    )

    assert multi_loss.item() < single_loss.item()
