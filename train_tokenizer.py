"""
Train a BPE tokenizer for Complexity-I64 (32K vocab).

Usage:
    python train_tokenizer.py --dataset Pacific-Prime/edu-web --vocab-size 32000
    python train_tokenizer.py --data ./local_data/*.txt --vocab-size 32000

Requires:
    pip install complexity-framework datasets

INL - 2025
"""

import argparse
import json
import logging
from pathlib import Path

from complexity.tokenizer import Tokenizer, TokenizerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("complexity_i64.train_tokenizer")


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer (32K)")

    parser.add_argument("--dataset", type=str, default="Pacific-Prime/edu-web")
    parser.add_argument("--data", type=str, default=None, help="Local data files (glob)")
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max-samples", type=int, default=None)

    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--method", type=str, default="bpe",
                        choices=["bpe", "unigram", "wordpiece"])

    parser.add_argument("--output", type=str, default="./tokenizer")
    parser.add_argument("--token", type=str, default=None)

    args = parser.parse_args()

    logger.info("Dataset: %s", args.dataset if not args.data else args.data)
    logger.info("Vocab size: %d", args.vocab_size)
    logger.info("Method: %s", args.method)
    logger.info("Output: %s", args.output)

    config = TokenizerConfig(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        method=args.method,
        format="complexity",
    )

    if args.data:
        tokenizer = Tokenizer.train(args.data, config=config)
    else:
        from datasets import load_dataset

        logger.info("Loading dataset: %s", args.dataset)
        ds = load_dataset(
            args.dataset,
            split=args.split,
            token=args.token,
            streaming=True,
        )

        def text_iterator():
            count = 0
            for example in ds:
                text = example.get(args.text_field) or example.get("text") or example.get("content")
                if text:
                    yield text
                    count += 1
                    if count % 50000 == 0:
                        logger.info("Processed %d samples...", count)
                    if args.max_samples and count >= args.max_samples:
                        break

        tokenizer = Tokenizer.train_from_iterator(text_iterator, config=config)

    # Save
    output_path = Path(args.output)
    tokenizer.save(str(output_path))

    # HF-compatible config
    hf_config = {
        "added_tokens_decoder": {
            "0": {"content": "<|endoftext|>", "special": True},
            "1": {"content": "<|pad|>", "special": True},
            "2": {"content": "<|startoftext|>", "special": True},
            "3": {"content": "<|unk|>", "special": True},
        },
        "bos_token": "<|startoftext|>",
        "eos_token": "<|endoftext|>",
        "pad_token": "<|pad|>",
        "unk_token": "<|unk|>",
        "model_max_length": 1000000000000000019884624838656,
        "tokenizer_class": "PreTrainedTokenizerFast",
    }
    with open(output_path / "tokenizer_config.json", "w") as f:
        json.dump(hf_config, f, indent=2)

    special_map = {
        "bos_token": "<|startoftext|>",
        "eos_token": "<|endoftext|>",
        "pad_token": "<|pad|>",
        "unk_token": "<|unk|>",
    }
    with open(output_path / "special_tokens_map.json", "w") as f:
        json.dump(special_map, f, indent=2)

    logger.info("Saved to: %s", output_path)
    logger.info("Vocab size: %d", len(tokenizer))

    # Test
    test_texts = [
        "The mitochondria is the powerhouse of the cell.",
        "In mathematics, the Pythagorean theorem states that a² + b² = c².",
        "Python is a programming language: def hello(): print('Hello')",
        "La photosynthèse convertit la lumière en énergie.",
    ]

    for text in test_texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        ratio = len(tokens) / len(text.split())
        logger.info("'%s...' -> %d tokens (%.2f tok/word)", text[:40], len(tokens), ratio)


if __name__ == "__main__":
    main()
