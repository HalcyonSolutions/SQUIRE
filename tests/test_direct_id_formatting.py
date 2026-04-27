from dataset import Seq2SeqDataset, _flatten_path_hops, _parse_paths_cell
from train import build_rev_dict, format_generated_path, format_gold_path


class Args:
    loop = False
    prob = 0
    smart_filter = False
    max_q_len = 32
    question_file = "qa_nhop.csv"
    train_paraphrased = True


def build_mquake_dataset():
    args = Args()
    return Seq2SeqDataset(
        data_path="data/mquake_st/",
        vocab_file="data/mquake_st/vocab.txt",
        device="cpu",
        args=args,
    )


def test_direct_id_mappings_use_human_readable_labels():
    dataset = build_mquake_dataset()

    assert dataset.direct_id_mode is True
    assert dataset.id2entity["Q749243"] == "Church of Sweden"
    assert dataset.id2entity["Q46"] == "Europe"
    assert dataset.id2relation["P112"] == "founded by"
    assert dataset.id2relation["P30"] == "continent"
    assert dataset.id2relation["P30_reverse"] == "continent (reverse)"


def test_generated_path_formats_full_direct_id_hops():
    args = Args()
    dataset = build_mquake_dataset()
    
    column = "Question-Paraphrased" if args.train_paraphrased else "Question"
    row = dataset.data[
        dataset.data[column]
        == "Which continent is the country in, where the founder of Church of Sweden is a citizen?"
    ].iloc[0]

    flat_tokens = _flatten_path_hops(_parse_paths_cell(row["Paths"]))
    token_ids = [dataset.dictionary.bos()]
    token_ids.extend(dataset.dictionary.indices[token] for token in flat_tokens)
    token_ids.append(dataset.dictionary.eos())

    rev_dict = build_rev_dict(dataset.dictionary)
    formatted = format_generated_path(
        row["Source"],
        token_ids,
        rev_dict,
        dataset,
        dataset.dictionary.eos(),
        dataset.dictionary.bos(),
    )

    expected = (
        "Church of Sweden --founded by--> Gustav I of Sweden "
        "--country of citizenship--> Sweden --continent--> Europe"
    )
    assert formatted == expected
    assert format_gold_path(row) == expected
