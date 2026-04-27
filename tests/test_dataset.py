from dataset import Seq2SeqDataset, TestDataset
import random
import ast
import numpy as np


class Args:
    loop = False
    prob = 0
    smart_filter = False
    max_q_len = 32
    question_file = "kinship_hinton_qa_nhop.csv"
    verbose = True
    test_paraphrased = True
    train_paraphrased = True


def test_seq2seq_dataset():
    args = Args()

    def vprint(*print_args, **print_kwargs):
        if args.verbose:
            print(*print_args, **print_kwargs)

    dataset = Seq2SeqDataset(
        data_path="data/kinshiphinton_final/",
        vocab_file="data/kinshiphinton_final/vocab.txt",
        device="cpu",
        split="train",
        args=args
    )

    vprint(f"Length of the dataset: {len(dataset)}")
    if args.verbose:
        idx_array = random.sample(range(len(dataset)), 3)
        print(f"Randomly selected indices for testing: {idx_array}")
    else:
        idx_array = np.arange(len(dataset))
        print(f"Testing all samples in the dataset: {len(idx_array)} samples")

    for idx in idx_array:
        sample = dataset[idx]
        row = dataset.data.iloc[idx]
        paths = ast.literal_eval(row["Paths"])
        if args.train_paraphrased:
            true_question = str(row["Question-Paraphrased"]).lower()
        else:
            true_question = row["Question"].lower()
        hops = len(paths)

        # 2*num_hops + 1 = path length
        # +1 add </s> token (end of sentence)
        # hop_length = (2*hops + 1) + 1 
        expected_length = (2 * hops + 1) + 1

        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "target" in sample # values of target should be ids from vocabulary
        assert "tgt_length" in sample

        tokens = [dataset.dictionary[idx.item()] for idx in sample["target"]]
        eos_id = dataset.dictionary.eos()
        assert sample["target"][-1].item() == eos_id, "Missing EOS token"
        assert all(tok != "<unk>" for tok in tokens), "Found <unk> in target"
        for i, tok in enumerate(tokens[:-1]):  # skip EOS
            if i % 2 == 0:
                assert not tok.startswith("R"), f"Expected entity at position {i}, got {tok}"
            else:
                assert tok.startswith("R"), f"Expected relation at position {i}, got {tok}"

        assert sample["tgt_length"] == expected_length
        assert len(sample["input_ids"]) == args.max_q_len
        
        input_ids = sample["input_ids"]
        attention_mask = sample["attention_mask"]
        pad_id = dataset.tokenizer.pad_token_id
        for i in range(len(input_ids)):
            if input_ids[i] == pad_id:
                assert attention_mask[i] == 0
            else:
                assert attention_mask[i] == 1

        vprint("\n=== Dataset Sample ===")
        for key, value in sample.items():
            vprint(f"{key}: {value}")
            try:
                vprint(f"{key} shape: {value.shape}") # we care only about tensors
            except:
                continue
            vprint()

        # check decoding
        # compare true question with decoded question (after removing special tokens)
        decoded = dataset.tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
        assert decoded.strip() == true_question.strip(), f"Decoded question does not match original. Decoded: '{decoded}', Original: '{true_question}'"
        if args.verbose:
            vprint("Original question:", true_question)
            vprint("Decoded question:", decoded)

    print("\nAll Seq2SeqDataset tests passed successfully!")
    return True

def test_test_dataset():
    args = Args()

    def vprint(*print_args, **print_kwargs):
        if args.verbose:
            print(*print_args, **print_kwargs)

    dataset = TestDataset(
        data_path="data/kinshiphinton_final/",
        vocab_file="data/kinshiphinton_final/vocab.txt",
        device="cpu",
        split="test",
        args=args
    )

    vprint(f"Length of the test dataset: {len(dataset)}")
    if args.verbose:
        sample_size = min(3, len(dataset))
        idx_array = random.sample(range(len(dataset)), sample_size)
        print(f"Randomly selected indices for testing: {idx_array}")
    else:
        idx_array = np.arange(len(dataset))
        print(f"Testing all samples in the test dataset: {len(idx_array)} samples")

    for idx in idx_array:
        sample = dataset[idx]
        row = dataset.data.iloc[idx]
        if args.test_paraphrased:
            true_question = str(row["Question-Paraphrased"]).strip().lower()
        else:
            true_question = str(row["Question"]).strip().lower()

        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "target" in sample

        assert len(sample["input_ids"]) == args.max_q_len
        assert len(sample["attention_mask"]) == args.max_q_len
        assert len(sample["target"]) == 1

        input_ids = sample["input_ids"]
        attention_mask = sample["attention_mask"]
        pad_id = dataset.tokenizer.pad_token_id
        for i in range(len(input_ids)):
            if input_ids[i].item() == pad_id:
                assert attention_mask[i].item() == 0

        decoded_target = dataset.dictionary[sample["target"][0].item()]
        assert decoded_target != "<unk>", "Found <unk> in target"

        decoded_question = dataset.tokenizer.decode(sample["input_ids"], skip_special_tokens=True).strip().lower()
        assert decoded_question == true_question, f"Decoded question does not match original. Decoded: '{decoded_question}', Original: '{true_question}'"

        if args.verbose:
            vprint("\n=== TestDataset Sample ===")
            vprint("Original question:", true_question)
            vprint("Decoded question:", decoded_question)
            vprint("Decoded target:", decoded_target)

    print("\nAll TestDataset tests passed successfully!")
    return True


if __name__ == "__main__":
    print("Running Seq2SeqDataset tests...")
    test_seq2seq_dataset()
    print("\n\nRunning TestDataset tests...")
    test_test_dataset()