import csv

from scripts.build_multi_answer_paths import (
    build_relation_index,
    convert_multi_answer_csv,
    find_path_for_relation_sequence,
    parse_path_key,
    parse_text_list,
)


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


def test_convert_multi_answer_csv_expands_rows_and_recovers_paths(tmp_path):
    input_csv = tmp_path / "multi.csv"
    output_csv = tmp_path / "recovered.csv"
    kg_dir = tmp_path / "kg"
    kg_dir.mkdir()

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
        writer.writerow(
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
        )

    with open(kg_dir / "train.txt", "w", encoding="utf-8") as f:
        f.write("Q1\tP1\tQ2\n")
        f.write("Q2\tP2\tQ3\n")
        f.write("Q2\tP2\tQ4\n")
        f.write("Q1\tP7\tQ9\n")

    summary = convert_multi_answer_csv(
        input_csv=str(input_csv),
        kg_dir=str(kg_dir),
        output_csv=str(output_csv),
        triple_files=("train.txt",),
    )

    with open(output_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert [row["Answer"] for row in rows] == ["Answer One", "Answer Two"]
    assert rows[0]["Answer-Entity"] == "Q3"
    assert rows[1]["Answer-Entity"] == "Q4"
    assert rows[0]["Answers"] == "['Answer One', 'Answer Two', 'Missing Answer']"
    assert rows[0]["Answers-Entities"] == "['Q3', 'Q4', 'Q9']"
    assert rows[0]["Paths"] == "[['Q1', 'P1', 'Q2'], ['Q2', 'P2', 'Q3']]"
    assert rows[1]["Paths"] == "[['Q1', 'P1', 'Q2'], ['Q2', 'P2', 'Q4']]"

    assert summary["rows_read"] == 1
    assert summary["answer_candidates_attempted"] == 3
    assert summary["paths_recovered"] == 2
    assert summary["answers_skipped_no_path"] == 1
    assert summary["rows_malformed_path_key"] == 0
    assert summary["split"]["train"]["paths_recovered"] == 2
