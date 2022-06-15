import csv
from pathlib import Path
from typing import Optional

import torch
from tap import Tap
from transformers import GPT2Config, GPT2LMHeadModel, set_seed

from common import get_tokenizer


class ArgumentParser(Tap):
    checkpoint: Optional[
        Path
    ] = None  # Load model from a checkpoint (loads OpenAI GPT2 Small if not provided)
    prompt: Optional[str] = None  # Provide a single prompt as a script argument
    prompt_file: Optional[Path] = None  # Read prompts from a CSV file
    output: Optional[Path] = None  # Write output to a file
    silent: bool = False  # Do not print output
    n: int = 10  # How many unconditional samples to produce (only used if no prompts are provided)
    seed: int = 1234  # Random seed

    # Sampling options (see: https://huggingface.co/docs/transformers/v4.19.4/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate=)
    do_sample: bool = False  # Defaults to false for greedy sampling.
    temperature: float = 1.0
    num_beams: int = 1  # Defaults to 1 for no beam search
    top_k: int = 50
    top_p: float = 1.0
    typical_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    no_repeat_ngram_size: Optional[int] = None
    max_new_tokens: Optional[int] = None


def main(args: ArgumentParser):
    # Load model and tokenizer
    if args.checkpoint:
        model_config = GPT2Config.from_pretrained(args.checkpoint.parent)
        model = GPT2LMHeadModel(model_config)
        model.load_state_dict(
            torch.load(args.checkpoint / "pytorch_model.bin", map_location="cpu")
        )
    else:
        model_config = GPT2Config.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = get_tokenizer(model_config.n_ctx)

    # Prepare the model inputs (prompts):
    if args.prompt is not None:
        # Use a single prompt provided as an argument
        inputs = [[args.prompt]]
    elif args.prompt_file is not None:
        # Use prompts from a file
        with args.prompt_file.open("r") as f:
            inputs = list(csv.reader(f))
    else:
        # Use blank prompts for unconditional sampling
        inputs = [[""]] * args.n

    # Setup output file if needed
    out_f = None
    writer = None
    if args.output:
        out_f = args.output.open("w")
        writer = csv.writer(out_f)
        writer.writerow(["prompt", "output"])

    # Colors for printing to terminal
    prompt_color = "\033[0;35;08m"
    model_color = "\033[1;33;08m"
    reset_color = "\033[0m"

    # Perform sampling:
    set_seed(args.seed)
    if not args.silent:
        print(f"{prompt_color}PROMPT\t{model_color}MODEL{reset_color}\n")
    for t in inputs:
        prompt = t[0]
        # Convert the input text into tokens
        input_tokens = tokenizer(tokenizer.eos_token + prompt, return_tensors="pt")[
            "input_ids"
        ]
        # Generate a sample from the model
        response_tokens = model.generate(
            input_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            num_beams=args.num_beams,
            top_k=args.top_k,
            top_p=args.top_p,
            typical_p=args.typical_p,
            repitition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        # Trim the prompt tokens from the response
        response_tokens = response_tokens[:, input_tokens.size(1) :].tolist()
        # Decode the tokens back into text
        output = tokenizer.batch_decode(response_tokens, skip_special_tokens=True)[0]

        # Record results
        if not args.silent:
            print(f"{prompt_color}{prompt}{model_color}{output}{reset_color}\n")
        if writer is not None:
            writer.writerow([t[0], output])
    if out_f is not None:
        out_f.close()


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    main(args)
