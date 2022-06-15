from transformers import GPT2TokenizerFast


def get_tokenizer(model_contex_size: int) -> GPT2TokenizerFast:
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.model_max_length = model_contex_size
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens(
            special_tokens_dict=dict(
                pad_token=tokenizer.eos_token,
            )
        )
    return tokenizer
