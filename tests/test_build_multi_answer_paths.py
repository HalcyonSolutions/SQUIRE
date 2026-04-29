import csv
import os

import torch

from scripts.build_multi_answer_paths import (
    build_relation_index,
    convert_multi_answer_csv,
    find_path_for_relation_sequence,
    parse_path_key,
    parse_text_list,
)
from dataset import Seq2SeqDataset
from dataset import TestDataset as SquireTestDataset


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
    question_file = None
    train_question_file = None
    eval_question_file = "eval.csv"
    verbose = False
    test_paraphrased = False
    train_paraphrased = False


def run_conversion(tmp_path, rows, triples):
    input_csv = tmp_path / "multi.csv"
    output_csv = tmp_path / "recovered.csv"
    kg_dir = tmp_path / "kg"
    kg_dir.mkdir(exist_ok=True)

    with open(input_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Question",
                "Source",
                "Source-Entity",
                "Answer",
                "Answer-Entity",
                "Path-Key",
                "Hops",
                "SplitLabel",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(kg_dir / "train.txt", "w", encoding="utf-8") as f:
        for head, relation, tail in triples:
            f.write(f"{head}\t{relation}\t{tail}\n")

    summary = convert_multi_answer_csv(
        input_csv=str(input_csv),
        kg_dir=str(kg_dir),
        output_csv=str(output_csv),
        triple_files=("train.txt",),
    )

    with open(output_csv, newline="", encoding="utf-8") as f:
        output_rows = list(csv.DictReader(f))

    return output_rows, summary, kg_dir, output_csv


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_parse_text_list_supports_list_and_scalar_values():
    assert parse_text_list("['Q1', 'Q2']") == ["Q1", "Q2"]
    assert parse_text_list("Q1") == ["Q1"]
    assert parse_text_list("") == []


def test_parse_path_key_supports_single_and_chain_values():
    assert parse_path_key("P26") == ["P26"]
    assert parse_path_key("P27->P6") == ["P27", "P6"]
    assert parse_path_key("['P50', 'P20']") == ["P50", "P20"]


def test_find_path_for_relation_sequence_uses_exact_relation_chain():
    relation_index = build_relation_index(
        [
            ("Q1", "P1", "Q2"),
            ("Q2", "P2", "Q3"),
            ("Q2", "P9", "Q4"),
            ("Q1", "P1", "Q5"),
            ("Q5", "P2", "Q6"),
        ]
    )

    assert find_path_for_relation_sequence(relation_index, "Q1", "Q3", ["P1", "P2"]) == [
        ["Q1", "P1", "Q2"],
        ["Q2", "P2", "Q3"],
    ]
    assert find_path_for_relation_sequence(relation_index, "Q1", "Q4", ["P1", "P2"]) is None


def test_convert_multi_answer_csv_recovers_single_hop_path(tmp_path):
    rows, summary, _, _ = run_conversion(
        tmp_path,
        rows=[
            {
                "Question": "Who is connected by P1?",
                "Source": "Source A",
                "Source-Entity": "Q1",
                "Answer": "['Answer One']",
                "Answer-Entity": "['Q2']",
                "Path-Key": "P1",
                "Hops": "1",
                "SplitLabel": "train",
            }
        ],
        triples=[("Q1", "P1", "Q2")],
    )

    assert len(rows) == 1
    assert rows[0]["Paths"] == "[['Q1', 'P1', 'Q2']]"
    assert summary["paths_recovered"] == 1


def test_convert_multi_answer_csv_recovers_two_hop_path(tmp_path):
    rows, summary, _, _ = run_conversion(
        tmp_path,
        rows=[
            {
                "Question": "Who is reached by P1 then P2?",
                "Source": "Source A",
                "Source-Entity": "Q1",
                "Answer": "['Answer One']",
                "Answer-Entity": "['Q3']",
                "Path-Key": "P1->P2",
                "Hops": "2",
                "SplitLabel": "train",
            }
        ],
        triples=[("Q1", "P1", "Q2"), ("Q2", "P2", "Q3")],
    )

    assert len(rows) == 1
    assert rows[0]["Paths"] == "[['Q1', 'P1', 'Q2'], ['Q2', 'P2', 'Q3']]"
    assert summary["paths_recovered"] == 1


def test_convert_multi_answer_csv_expands_multiple_answers_into_multiple_rows(tmp_path):
    rows, summary, _, _ = run_conversion(
        tmp_path,
        rows=[
            {
                "Question": "Who can be reached from Source A?",
                "Source": "Source A",
                "Source-Entity": "Q1",
                "Answer": "['Answer One', 'Answer Two']",
                "Answer-Entity": "['Q3', 'Q4']",
                "Path-Key": "P1->P2",
                "Hops": "2",
                "SplitLabel": "train",
            }
        ],
        triples=[("Q1", "P1", "Q2"), ("Q2", "P2", "Q3"), ("Q2", "P2", "Q4")],
    )

    assert len(rows) == 2
    assert [row["Answer-Entity"] for row in rows] == ["Q3", "Q4"]
    assert summary["paths_recovered"] == 2


def test_convert_multi_answer_csv_skips_wrong_relation_sequence(tmp_path):
    rows, summary, _, _ = run_conversion(
        tmp_path,
        rows=[
            {
                "Question": "Who is reached by the wrong chain?",
                "Source": "Source A",
                "Source-Entity": "Q1",
                "Answer": "['Answer One']",
                "Answer-Entity": "['Q3']",
                "Path-Key": "P1->P2",
                "Hops": "2",
                "SplitLabel": "dev",
            }
        ],
        triples=[("Q1", "P7", "Q2"), ("Q2", "P8", "Q3")],
    )

    assert rows == []
    assert summary["paths_recovered"] == 0
    assert summary["answers_skipped_no_path"] == 1
    assert summary["split"]["dev"]["answers_skipped_no_path"] == 1


def test_convert_multi_answer_csv_counts_missing_paths(tmp_path):
    rows, summary, _, _ = run_conversion(
        tmp_path,
        rows=[
            {
                "Question": "Who can be reached from Source A?",
                "Source": "Source A",
                "Source-Entity": "Q1",
                "Answer": "['Answer One', 'Answer Two', 'Missing Answer']",
                "Answer-Entity": "['Q3', 'Q4', 'Q9']",
                "Path-Key": "P1->P2",
                "Hops": "2",
                "SplitLabel": "train",
            }
        ],
        triples=[("Q1", "P1", "Q2"), ("Q2", "P2", "Q3"), ("Q2", "P2", "Q4"), ("Q1", "P7", "Q9")],
    )

    assert len(rows) == 2
    assert summary["rows_read"] == 1
    assert summary["answer_candidates_attempted"] == 3
    assert summary["paths_recovered"] == 2
    assert summary["answers_skipped_no_path"] == 1
    assert summary["rows_malformed_path_key"] == 0
    assert summary["split"]["train"]["paths_recovered"] == 2


def test_seq2seq_dataset_can_consume_reconstructed_csv(tmp_path, monkeypatch):
    rows, _, kg_dir, output_csv = run_conversion(
        tmp_path,
        rows=[
            {
                "Question": "Who is reached by P1 then P2?",
                "Source": "Source A",
                "Source-Entity": "Q1",
                "Answer": "['Answer One']",
                "Answer-Entity": "['Q3']",
                "Path-Key": "P1->P2",
                "Hops": "2",
                "SplitLabel": "train",
            }
        ],
        triples=[("Q1", "P1", "Q2"), ("Q2", "P2", "Q3")],
    )

    assert len(rows) == 1
    monkeypatch.setattr(Seq2SeqDataset, "_tokenizer", DummyTokenizer())

    args = Args()
    args.question_file = str(output_csv)
    args.train_question_file = str(output_csv)
    dataset = Seq2SeqDataset(
        data_path=f"{kg_dir}/",
        vocab_file=f"{kg_dir}/vocab.txt",
        device="cpu",
        split="train",
        args=args,
    )

    sample = dataset[0]
    decoded = [dataset.dictionary[token_id.item()] for token_id in sample["target"]]
    assert dataset.direct_id_mode is True
    assert dataset.has_paths is True
    assert os.path.basename(dataset.csv_file) == os.path.basename(output_csv)
    assert decoded[:-1] == ["Q1", "P1", "Q2", "P2", "Q3"]


def test_dataset_selection_supports_separate_train_and_eval_question_files(tmp_path, monkeypatch):
    kg_dir = tmp_path / "kg"
    kg_dir.mkdir()

    train_csv = tmp_path / "train_reconstructed.csv"
    eval_csv = tmp_path / "eval_multi.csv"

    write_csv(
        train_csv,
        [
            {
                "Question": "Train question",
                "Source": "Source A",
                "Source-Entity": "Q1",
                "Answer": "Answer One",
                "Answer-Entity": "Q3",
                "Answers": "['Answer One']",
                "Path-Key": "P1->P2",
                "Paths": "[['Q1', 'P1', 'Q2'], ['Q2', 'P2', 'Q3']]",
                "SplitLabel": "train",
            }
        ],
    )
    write_csv(
        eval_csv,
        [
            {
                "Question": "Eval question",
                "Source": "Source A",
                "Source-Entity": "Q1",
                "Answer": "['Answer One']",
                "Answer-Entity": "['Q3']",
                "Path-Key": "P1->P2",
                "SplitLabel": "test",
            }
        ],
    )

    with open(kg_dir / "train.txt", "w", encoding="utf-8") as f:
        f.write("Q1\tP1\tQ2\n")
        f.write("Q2\tP2\tQ3\n")

    monkeypatch.setattr(Seq2SeqDataset, "_tokenizer", DummyTokenizer())

    args = Args()
    args.question_file = None
    args.train_question_file = str(train_csv)
    args.eval_question_file = str(eval_csv)

    train_set = Seq2SeqDataset(
        data_path=f"{kg_dir}/",
        vocab_file=f"{kg_dir}/vocab.txt",
        device="cpu",
        split="train",
        args=args,
    )
    eval_set = SquireTestDataset(
        data_path=f"{kg_dir}/",
        vocab_file=f"{kg_dir}/vocab.txt",
        device="cpu",
        split="test",
        args=args,
    )

    assert os.path.basename(train_set.csv_file) == "train_reconstructed.csv"
    assert os.path.basename(eval_set.csv_file) == "eval_multi.csv"
    assert train_set.data.iloc[0]["Question"] == "Train question"
    assert eval_set.data.iloc[0]["Question"] == "Eval question"


def test_dataset_selection_falls_back_to_question_file_for_both_modes(tmp_path, monkeypatch):
    kg_dir = tmp_path / "kg"
    kg_dir.mkdir()

    shared_csv = tmp_path / "shared.csv"
    write_csv(
        shared_csv,
        [
            {
                "Question": "Shared question",
                "Source": "Source A",
                "Source-Entity": "Q1",
                "Answer": "Answer One",
                "Answer-Entity": "Q3",
                "Path-Key": "P1->P2",
                "Paths": "[['Q1', 'P1', 'Q2'], ['Q2', 'P2', 'Q3']]",
                "SplitLabel": "train",
            }
        ],
    )

    with open(kg_dir / "train.txt", "w", encoding="utf-8") as f:
        f.write("Q1\tP1\tQ2\n")
        f.write("Q2\tP2\tQ3\n")

    monkeypatch.setattr(Seq2SeqDataset, "_tokenizer", DummyTokenizer())

    args = Args()
    args.question_file = str(shared_csv)
    args.train_question_file = None
    args.eval_question_file = None

    train_set = Seq2SeqDataset(
        data_path=f"{kg_dir}/",
        vocab_file=f"{kg_dir}/vocab.txt",
        device="cpu",
        split="train",
        args=args,
    )
    eval_set = SquireTestDataset(
        data_path=f"{kg_dir}/",
        vocab_file=f"{kg_dir}/vocab.txt",
        device="cpu",
        split="train",
        args=args,
    )

    assert os.path.basename(train_set.csv_file) == "shared.csv"
    assert os.path.basename(eval_set.csv_file) == "shared.csv"
