#!/usr/bin/env python3

import argparse
import ast
import csv
import os
from collections import defaultdict


DEFAULT_TRIPLE_FILES = ("train.txt", "valid.txt", "test.txt", "triplets.txt")


def parse_text_list(value):
    if value is None:
        return []

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = text
    else:
        parsed = value

    if isinstance(parsed, (list, tuple, set)):
        values = []
        for item in parsed:
            item_text = str(item).strip()
            if item_text:
                values.append(item_text)
        return values

    parsed_text = str(parsed).strip()
    return [parsed_text] if parsed_text else []


def parse_path_key(value):
    parts = parse_text_list(value)
    if not parts:
        return []
    if len(parts) == 1 and "->" in parts[0]:
        return [segment.strip() for segment in parts[0].split("->") if segment.strip()]
    return [segment.strip() for segment in parts if segment.strip()]


def read_triples(kg_dir, triple_files):
    triples = []
    used_files = []
    seen = set()

    for file_name in triple_files:
        path = os.path.join(kg_dir, file_name)
        if not os.path.exists(path):
            continue
        used_files.append(file_name)
        with open(path, newline="", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 3:
                    raise ValueError(f"Malformed triple in {path}:{line_no}: {line.rstrip()}")
                triple = tuple(part.strip() for part in parts)
                if triple in seen:
                    continue
                seen.add(triple)
                triples.append(triple)

    if not used_files:
        raise FileNotFoundError(
            f"No triple files found in {kg_dir}. Tried: {', '.join(triple_files)}"
        )

    return triples, used_files


def build_relation_index(triples):
    relation_index = defaultdict(lambda: defaultdict(list))
    seen_tails = defaultdict(set)

    for head, relation, tail in triples:
        key = (head, relation)
        if tail in seen_tails[key]:
            continue
        seen_tails[key].add(tail)
        relation_index[head][relation].append(tail)

    return {
        head: {relation: list(tails) for relation, tails in relations.items()}
        for head, relations in relation_index.items()
    }


def find_path_for_relation_sequence(relation_index, source, target, relation_sequence):
    def dfs(current_entity, relation_idx):
        if relation_idx == len(relation_sequence):
            return [] if current_entity == target else None

        relation = relation_sequence[relation_idx]
        next_entities = relation_index.get(current_entity, {}).get(relation, [])
        for next_entity in next_entities:
            suffix = dfs(next_entity, relation_idx + 1)
            if suffix is not None:
                return [[current_entity, relation, next_entity]] + suffix
        return None

    return dfs(source, 0)


def build_output_fieldnames(input_fieldnames):
    fieldnames = list(input_fieldnames)
    for column in ("Answers", "Answers-Entities", "Paths"):
        if column not in fieldnames:
            fieldnames.append(column)
    return fieldnames


def init_summary():
    return {
        "rows_read": 0,
        "answer_candidates_attempted": 0,
        "paths_recovered": 0,
        "answers_skipped_no_path": 0,
        "rows_malformed_path_key": 0,
        "rows_malformed_answers": 0,
        "rows_written": 0,
        "split": defaultdict(
            lambda: {
                "rows_read": 0,
                "answer_candidates_attempted": 0,
                "paths_recovered": 0,
                "answers_skipped_no_path": 0,
                "rows_malformed_path_key": 0,
                "rows_malformed_answers": 0,
                "rows_written": 0,
            }
        ),
    }


def log_summary(summary, used_files, output_csv):
    print(f"Output CSV: {output_csv}")
    print("Triple files used:", ", ".join(used_files))
    print("Rows read:", summary["rows_read"])
    print("Answer candidates attempted:", summary["answer_candidates_attempted"])
    print("Paths recovered:", summary["paths_recovered"])
    print("Answers skipped because no path found:", summary["answers_skipped_no_path"])
    print("Rows with malformed Path-Key:", summary["rows_malformed_path_key"])
    print("Rows with malformed answer fields:", summary["rows_malformed_answers"])
    print("Rows written:", summary["rows_written"])
    print("Split-wise counts:")
    for split_name in sorted(summary["split"].keys()):
        stats = summary["split"][split_name]
        print(
            f"  {split_name}: rows_read={stats['rows_read']}, "
            f"answer_candidates_attempted={stats['answer_candidates_attempted']}, "
            f"paths_recovered={stats['paths_recovered']}, "
            f"answers_skipped_no_path={stats['answers_skipped_no_path']}, "
            f"rows_malformed_path_key={stats['rows_malformed_path_key']}, "
            f"rows_malformed_answers={stats['rows_malformed_answers']}, "
            f"rows_written={stats['rows_written']}"
        )


def convert_multi_answer_csv(input_csv, kg_dir, output_csv, triple_files):
    triples, used_files = read_triples(kg_dir, triple_files)
    relation_index = build_relation_index(triples)
    summary = init_summary()

    with open(input_csv, newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header: {input_csv}")
        fieldnames = build_output_fieldnames(reader.fieldnames)

        output_dir = os.path.dirname(output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_csv, "w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()

            for row_idx, row in enumerate(reader, start=2):
                split_name = row.get("SplitLabel", "unknown") or "unknown"
                split_summary = summary["split"][split_name]
                summary["rows_read"] += 1
                split_summary["rows_read"] += 1

                source_entity = str(row.get("Source-Entity", "")).strip()
                answer_labels = parse_text_list(row.get("Answer"))
                answer_entities = parse_text_list(row.get("Answer-Entity"))
                relation_sequence = parse_path_key(row.get("Path-Key"))

                if not source_entity or not relation_sequence:
                    summary["rows_malformed_path_key"] += 1
                    split_summary["rows_malformed_path_key"] += 1
                    continue

                if len(answer_labels) != len(answer_entities) or not answer_entities:
                    summary["rows_malformed_answers"] += 1
                    split_summary["rows_malformed_answers"] += 1
                    continue

                summary["answer_candidates_attempted"] += len(answer_entities)
                split_summary["answer_candidates_attempted"] += len(answer_entities)

                for answer_label, answer_entity in zip(answer_labels, answer_entities):
                    path_hops = find_path_for_relation_sequence(
                        relation_index,
                        source_entity,
                        answer_entity,
                        relation_sequence,
                    )
                    if path_hops is None:
                        summary["answers_skipped_no_path"] += 1
                        split_summary["answers_skipped_no_path"] += 1
                        continue

                    output_row = dict(row)
                    output_row["Answer"] = answer_label
                    output_row["Answer-Entity"] = answer_entity
                    output_row["Answers"] = repr(answer_labels)
                    output_row["Answers-Entities"] = repr(answer_entities)
                    output_row["Paths"] = repr(path_hops)

                    writer.writerow(output_row)
                    summary["paths_recovered"] += 1
                    summary["rows_written"] += 1
                    split_summary["paths_recovered"] += 1
                    split_summary["rows_written"] += 1

    log_summary(summary, used_files, output_csv)
    return summary


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, help="path to multi-answer QA CSV")
    parser.add_argument("--kg-dir", required=True, help="directory containing KG triple files")
    parser.add_argument("--output-csv", required=True, help="path to write recovered path-supervised CSV")
    parser.add_argument(
        "--triple-files",
        nargs="+",
        default=list(DEFAULT_TRIPLE_FILES),
        help="KG triple files to scan in order",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    convert_multi_answer_csv(
        input_csv=args.input_csv,
        kg_dir=args.kg_dir,
        output_csv=args.output_csv,
        triple_files=tuple(args.triple_files),
    )


if __name__ == "__main__":
    main()
