import logging
import math
from pathlib import Path

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
    GPT2Config,
    GPT2LMHeadModel,
    default_data_collator,
    get_cosine_schedule_with_warmup,
)

from common import get_tokenizer

logger = get_logger(__name__)


class ArgumentParser(Tap):
    run_name: str
    dataset_path: Path
    checkpoint_dir: Path = Path("/checkpoints")
    seed: int = 1234  # Random seed
    n_epochs: int = 1  # How many epochs of training to perform
    per_device_batch_size: int = 12  # Batch size per GPU/CPU used
    eval_frequency: float = (
        1.0  # How often to calculate test loss (as a proportion of epoch)
    )
    checkpoint_frequency: float = (
        1.0  # How often to save a checkpoint (as a proportion of epoch)
    )

    n_layers: int = 12  # Number of layers
    n_heads: int = 12  # Number of attention heads
    embedding_size: int = 768  # Size of embeddings and hidden states
    context_size: int = 1024

    lr: float = 0.0001  # Initial learning rate
    warmup_steps: int = 50  # How many warmup steps to use for LR schedule
    weight_decay: float = 0.01  # Weight decay to use

    b1: float = 0.9  # Beta1 for Adam optimizer
    b2: float = 0.999  # Beta2 for Adam optimizer
    eps: float = 1e-6  # Epsilon for Adam optimizer
    dropout: float = 0.1  # Dropout probability during training
    use_tf32: bool = True  # Whether or not to use TF32 (https://huggingface.co/docs/transformers/main/en/perf_train_gpu_one#tf32=)

    def configure(self):
        self.add_argument("run_name")
        self.add_argument("dataset_path")


def main(args: ArgumentParser):
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = args.use_tf32
    accelerator = Accelerator()

    # Create directory for storing checkpoints
    checkpoint_dir = args.checkpoint_dir / args.run_name
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    # Create model and tokenizer
    tokenizer = get_tokenizer(args.context_size)
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
    if accelerator.is_main_process:
        # Save the model config to disk so we can load it later.
        model_config.save_pretrained(checkpoint_dir)
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
        batch_size=args.per_device_batch_size,
        collate_fn=default_data_collator,
        num_workers=12,
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset["test"],
        batch_size=args.per_device_batch_size,
        collate_fn=default_data_collator,
        num_workers=8,
        pin_memory=True,
    )

    # Calculate schedules
    steps_per_epoch = len(train_loader) // accelerator.num_processes
    total_steps = args.n_epochs * steps_per_epoch
    step_digits = len(str(steps_per_epoch))
    epoch_digits = len(str(args.n_epochs))

    eval_every = int(steps_per_epoch * args.eval_frequency)
    checkpoint_every = int(steps_per_epoch * args.checkpoint_frequency)

    # Setup optimizers
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.lr,
        eps=args.eps,
        betas=(args.b1, args.b2),
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    # Setup logging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

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
    global_step = 0
    for epoch in range(args.n_epochs):
        status["epoch"] = epoch
        model.train()
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(**batch)
            accelerator.backward(outputs.loss)
            optimizer.step()
            lr_scheduler.step()
            status["loss"] = outputs.loss.detach().float().item()

            if (global_step > 0 and global_step % checkpoint_every == 0) or (
                global_step + 1 == total_steps
            ):
                # Save a checkpoint
                str_epoch = str(epoch).zfill(epoch_digits)
                str_step = str(step).zfill(step_digits)
                accelerator.save_state(checkpoint_dir / f"{str_epoch}_{str_step}")

            if global_step > 0 and global_step % eval_every == 0:
                # Calculate test loss
                model.eval()
                losses = []
                for step, batch in enumerate(test_loader):
                    with torch.no_grad():
                        outputs = model(**batch)
                    loss = outputs.loss
                    losses.append(
                        accelerator.gather(loss.repeat(args.per_device_batch_size))
                    )
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
            pbar.update(1)
            global_step += 1


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    main(args)
