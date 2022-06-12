from pathlib import Path
from itertools import chain

from tap import Tap
from transformers import AutoTokenizer
from datasets import load_dataset


class Args(Tap):
    input_path: Path  # Path to the text file with the training data (assumes each line is an example)
    output_path: Path  # Folder to write the processed datasets to
    seed: int = 1234  # Random seed to use
    test_proportion: float = 0.1  # Proportion of the input data to use for testing
    tokenizer: str = "gpt2"  # The HuggingFace model name of the tokenizer to use.
    model_context_size: int = (
        512  # The context size of the model that will use this data
    )


# Taken from: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py
def group_texts(examples, block_size: int):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # Drop any extra text that doeesn't fit into the last block.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split into chunks of `block_size`
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result


def main(args: Args):
    # Load and configure the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.model_max_length = args.model_context_size
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens(
            special_tokens_dict=dict(
                pad_token=tokenizer.eos_token,
            )
        )

    # Load the raw text file
    dataset = load_dataset("text", data_files=str(args.input_path))["train"]

    # Tokenize the text and add start tokens
    def tokenize(examples):
        outputs = tokenizer(
            [t.strip() for t in examples["text"]],
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )
        for o in outputs["input_ids"]:
            # Only adding a start token (no stop token)
            # because group_texts will concatenate multiple
            # texts so the start token for the next text will
            # serve as the end token for the previous text.
            o.insert(0, tokenizer.eos_token_id)
        return outputs

    dataset = dataset.map(
        tokenize,
        batched=True,
        num_proc=4,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    # Group the texts into chunks that fill up the model's context window
    dataset = dataset.map(
        group_texts,
        fn_kwargs=dict(
            block_size=args.model_context_size,
        ),
        batched=True,
        num_proc=4,
        batch_size=2400,
        desc="Optimizing data for language modeling",
    )

    # Split the data into train and test sets.
    dataset = dataset.train_test_split(
        test_size=args.test_proportion, shuffle=True, seed=args.seed
    )
    dataset.save_to_disk(args.output_path)


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
