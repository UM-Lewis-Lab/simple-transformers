import logging
import math
from pathlib import Path
from typing import Optional

import datasets
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import DatasetDict
from tap import Tap
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AdamW,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    default_data_collator,
    get_cosine_schedule_with_warmup,
)

logger = get_logger(__name__)


class ArgumentParser(Tap):
    dataset_path: Path

    # checkpoint: Optional[str] = None  # Path to checkpoint if resuming training
    precision: int = (
        16  # How many bits of precision to use (controls mixed-precision training)
    )
    seed: int = 1234  # Random seed
    n_epochs: int = 1  # How many epochs of training to perform
    batch_size: int = 32  # Batch size for training
    eval_frequency: float = (
        0.2  # How often to calculate test loss (as a proportion of epoch)
    )

    n_layers: int = 12  # Number of layers
    n_heads: int = 12  # Number of attention heads
    embedding_size: int = 768  # Size of embeddings and hidden states
    context_size: int = 512

    lr: float = 0.0001  # Initial learning rate
    warmup_steps: int = 50  # How many warmup steps to use for LR schedule

    b1: float = 0.9  # Beta1 for Adam optimizer
    b2: float = 0.999  # Beta2 for Adam optimizer
    eps: float = 1e-6  # Epsilon for Adam optimizer
    correct_bias: bool = False  # Correct the Adam optimizer bias
    dropout: float = 0.1  # Dropout probability during training
    use_tf32: bool = True  # Whether or not to use TF32 (https://huggingface.co/docs/transformers/main/en/perf_train_gpu_one#tf32=)


def main(args: ArgumentParser):
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = args.use_tf32
    accelerator = Accelerator()

    # Setup logging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Create model and tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.model_max_length = args.context_size
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens(
            special_tokens_dict=dict(
                pad_token=tokenizer.eos_token,
            )
        )
    model_config = GPT2Config(
        n_layer=args.n_layers,
        n_head=args.n_heads,
        n_embd=args.embedding_size,
        n_ctx=args.context_size,
        n_positions=args.context_size,
        vocab_size=tokenizer.vocab_size,
        resid_pdrop=args.dropout,
        embd_pdrop=args.dropout,
        attn_pdrop=args.dropout,
        return_dict=True,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
    )
    model = GPT2LMHeadModel(model_config)

    # Load training data
    dataset = DatasetDict.load_from_disk(args.dataset_path)

    def prepare_batch(batch):
        # Pad the examples and convert them to PyTorch tensors.
        batch = tokenizer.pad(batch, return_tensors="pt", padding="max_length")
        # Add labels for language modeling (labels are just the inputs)
        # HuggingFace models handle shifting the inputs.
        batch["labels"] = batch["input_ids"].detach().clone()
        return batch

    dataset.set_transform(prepare_batch, columns=["input_ids"])
    train_loader = DataLoader(
        dataset["train"],
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        num_workers=12,
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset["test"],
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        num_workers=8,
        pin_memory=True,
    )

    # Setup optimizers
    total_steps = args.n_epochs * len(train_loader)
    eval_every = int(len(train_loader) * args.eval_frequency)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        eps=args.eps,
        betas=(args.b1, args.b2),
        correct_bias=args.correct_bias,
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    # Put data, model, and optimizer onto GPUs.
    model, optimizer, train_loader, test_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, lr_scheduler
    )
    accelerator.wait_for_everyone()

    # Training loop
    model.train()
    pbar = tqdm(
        range(total_steps),
        desc="Training",
        unit="step",
        disable=not accelerator.is_local_main_process,
    )
    status = {}
    for epoch in range(args.n_epochs):
        status["epoch"] = epoch
        model.train()
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(**batch)
            accelerator.backward(outputs.loss)
            optimizer.step()
            lr_scheduler.step()
            pbar.update(1)
            status["loss"] = outputs.loss.detach().float().item()

            if step > 0 and step % eval_every == 0:
                # Calculate test loss
                model.eval()
                losses = []
                for step, batch in enumerate(test_loader):
                    with torch.no_grad():
                        outputs = model(**batch)
                    loss = outputs.loss
                    losses.append(accelerator.gather(loss.repeat(args.batch_size)))
                losses = torch.cat(losses)
                losses = losses[: len(test_loader)]
                test_loss = float("inf")
                try:
                    test_loss = torch.mean(losses).detach().float().item()
                    perplexity = math.exp(test_loss)
                except OverflowError:
                    perplexity = float("inf")
                status.update(
                    dict(
                        test_loss=test_loss,
                        perplexity=perplexity,
                    )
                )
            pbar.set_postfix(status)


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    main(args)
