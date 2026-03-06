"""
Complexity-I64 :: Datasets

Data pipeline for pre-training and SFT. Tokenizer = 32K vocab.

INL - 2025
"""

import re
import random
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
from torch.utils.data import Dataset, IterableDataset

from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from jinja2 import Template

logger = logging.getLogger(__name__)


# ============================================================================
# PRE-TRAINING DATASETS
# ============================================================================

class PreTokenizedDataset(Dataset):
    """
    Ultra-fast dataset from pre-tokenized parquet shards.
    Zero tokenization overhead during training.
    """

    def __init__(self, data_dir: str, max_length: int = 512):
        import pyarrow.parquet as pq
        from tqdm import tqdm

        self.data_dir = Path(data_dir)
        self.max_length = max_length

        shard_files = sorted(self.data_dir.glob("shard_*.parquet"))
        if not shard_files:
            raise ValueError(f"No shard files found in {data_dir}")

        logger.info("Loading pre-tokenized data from %s...", data_dir)
        logger.info("Found %d shards", len(shard_files))

        self.input_ids = []
        self.labels = []

        for shard_path in tqdm(shard_files, desc="Loading shards"):
            table = pq.read_table(shard_path)
            for row in range(table.num_rows):
                self.input_ids.append(table['input_ids'][row].as_py())
                self.labels.append(table['labels'][row].as_py())

        logger.info("Loaded %d pre-tokenized samples", len(self.input_ids))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class StreamingTextDataset(IterableDataset):
    """Streaming dataset from HuggingFace (memory efficient)."""

    def __init__(
        self,
        dataset_name: str,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 512,
        text_field: str = "text",
        split: str = "train",
        token: Optional[str] = None,
        subset: Optional[str] = None,
        exclude_sources: Optional[list] = None,
    ):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_field = text_field
        self.split = split
        self.token = token
        self.subset = subset
        self.exclude_sources = exclude_sources or []

    def __iter__(self):
        try:
            if self.subset:
                ds = load_dataset(self.dataset_name, self.subset,
                                  split=self.split, streaming=True, token=self.token)
            else:
                ds = load_dataset(self.dataset_name,
                                  split=self.split, streaming=True, token=self.token)
        except Exception as e:
            logger.warning("Could not load %s: %s", self.dataset_name, e)
            return

        if self.exclude_sources:
            ds = ds.filter(
                lambda x: x.get("meta", {}).get("redpajama_set_name", "") not in self.exclude_sources
            )

        buffer = []
        for example in ds:
            text = None
            for field in [self.text_field, "text", "content", "code"]:
                if field in example and example[field]:
                    text = example[field]
                    break
            if not text:
                continue

            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)

            while len(buffer) >= self.max_length + 1:
                chunk = buffer[:self.max_length + 1]
                buffer = buffer[self.max_length:]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels}


# ============================================================================
# CHAT TEMPLATES
# ============================================================================

CHAT_TEMPLATES = {
    "default": """{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] }}

{% set messages = messages[1:] %}{% endif %}{% for message in messages %}{% if message['role'] == 'user' %}User: {{ message['content'] }}

{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}{% if not loop.last %}

{% endif %}{% endif %}{% endfor %}""",

    "chatml": """{% for message in messages %}<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}<|im_start|>assistant
""",

    "simple": """{% for message in messages %}{% if message['role'] == 'user' %}User: {{ message['content'] }}
{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}
{% elif message['role'] == 'system' %}System: {{ message['content'] }}
{% endif %}{% endfor %}""",

    "alpaca": """{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] }}

{% set messages = messages[1:] %}{% endif %}### Instruction:
{{ messages[0]['content'] }}

### Response:
{% if messages|length > 1 %}{{ messages[1]['content'] }}{% endif %}""",
}


# ============================================================================
# FORMAT CONVERTERS
# ============================================================================

def convert_to_messages(example: Dict[str, Any], format_name: str) -> List[Dict[str, str]]:
    """Convert various dataset formats to unified messages format."""

    if format_name == "oasst":
        messages = []
        if "messages" in example:
            for msg in example["messages"]:
                role = msg.get("role", "user")
                content = msg.get("content", msg.get("text", ""))
                messages.append({"role": role, "content": content})
        elif "prompt" in example and "response" in example:
            messages = [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["response"]},
            ]
        return messages

    elif format_name == "sharegpt":
        messages = []
        conversations = example.get("conversations", example.get("messages", []))
        for conv in conversations:
            role_map = {"human": "user", "gpt": "assistant", "system": "system"}
            role = conv.get("from", conv.get("role", "user"))
            role = role_map.get(role, role)
            content = conv.get("value", conv.get("content", ""))
            messages.append({"role": role, "content": content})
        return messages

    elif format_name == "dolphin":
        messages = []
        if example.get("system_prompt"):
            messages.append({"role": "system", "content": example["system_prompt"]})
        if example.get("question"):
            messages.append({"role": "user", "content": example["question"]})
        if example.get("response"):
            messages.append({"role": "assistant", "content": example["response"]})
        return messages

    elif format_name == "alpaca":
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")
        user_content = f"{instruction}\n\nInput: {input_text}" if input_text else instruction
        return [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]

    elif format_name == "qa":
        question = example.get("question", example.get("prompt",
                   example.get("query", example.get("problem", ""))))
        answer = example.get("answer", example.get("response",
                 example.get("solution", example.get("output", ""))))
        messages = []
        if question:
            messages.append({"role": "user", "content": question})
        if answer:
            messages.append({"role": "assistant", "content": answer})
        return messages

    elif format_name == "hh":
        messages = []
        text = example.get("chosen", example.get("text", ""))
        if not text:
            return messages
        turns = re.split(r'\n\n(?=Human:|Assistant:)', text)
        for turn in turns:
            turn = turn.strip()
            if turn.startswith("Human:"):
                content = turn[6:].strip()
                if content:
                    messages.append({"role": "user", "content": content})
            elif turn.startswith("Assistant:"):
                content = turn[10:].strip()
                if content:
                    messages.append({"role": "assistant", "content": content})
        return messages

    elif format_name == "messages":
        return example.get("messages", [])

    elif format_name == "mmlu":
        question = example.get("question", "")
        choices = example.get("choices", [])
        answer_idx = example.get("answer", 0)
        if isinstance(answer_idx, str):
            answer_idx = ord(answer_idx.upper()) - ord('A')
        letters = ["A", "B", "C", "D"]
        choices_text = "\n".join([f"{letters[i]}) {c}" for i, c in enumerate(choices)])
        answer_text = choices[answer_idx] if answer_idx < len(choices) else choices[0]
        return [
            {"role": "user", "content": f"Question: {question}\n\nChoices:\n{choices_text}"},
            {"role": "assistant", "content": f"The answer is {letters[answer_idx]}) {answer_text}"},
        ]

    else:
        # Auto-detect
        if "messages" in example:
            return example["messages"]
        elif "conversations" in example:
            return convert_to_messages(example, "sharegpt")
        elif "instruction" in example:
            return convert_to_messages(example, "alpaca")
        elif "question" in example and "response" in example:
            return convert_to_messages(example, "dolphin")
        elif "question" in example and "answer" in example:
            return convert_to_messages(example, "qa")
        else:
            return []


# ============================================================================
# CONVERSATIONAL SFT DATASET
# ============================================================================

class ConversationalDataset(Dataset):
    """Dataset for conversational SFT with loss masking on user messages."""

    def __init__(
        self,
        dataset_name: str,
        tokenizer: PreTrainedTokenizerFast,
        chat_template: str,
        format_name: str = "auto",
        max_length: int = 2048,
        split: str = "train",
        max_samples: Optional[int] = None,
        token: Optional[str] = None,
        subset: Optional[str] = None,
        mask_user: bool = True,
        _examples: Optional[List] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_user = mask_user
        self.template = Template(chat_template)
        self.format_name = format_name

        if _examples is not None:
            self.examples = _examples
            logger.info("Using %d pre-loaded conversations", len(self.examples))
        else:
            logger.info("Loading dataset: %s", dataset_name)
            try:
                if subset:
                    ds = load_dataset(dataset_name, subset, split=split, token=token)
                else:
                    ds = load_dataset(dataset_name, split=split, token=token)
            except Exception:
                if subset:
                    ds = load_dataset(dataset_name, subset, split=split, token=token, trust_remote_code=True)
                else:
                    ds = load_dataset(dataset_name, split=split, token=token, trust_remote_code=True)

            if max_samples and len(ds) > max_samples:
                ds = ds.select(range(max_samples))

            self.examples = list(ds)
            logger.info("Loaded %d conversations", len(self.examples))

    @classmethod
    def from_multiple_datasets(
        cls,
        datasets_config: List[dict],
        tokenizer: PreTrainedTokenizerFast,
        chat_template: str,
        format_name: str = "auto",
        max_length: int = 2048,
        split: str = "train",
        max_samples: Optional[int] = None,
        token: Optional[str] = None,
        mask_user: bool = True,
    ):
        """Load and combine multiple datasets with weights."""
        all_examples = []
        total_weight = sum(d.get("weight", 1.0) for d in datasets_config)

        logger.info("Loading %d datasets (max_samples=%s)", len(datasets_config), max_samples)

        for ds_config in datasets_config:
            ds_name = ds_config["name"]
            ds_weight = ds_config.get("weight", 1.0) / total_weight
            ds_subset = ds_config.get("subset", None)
            ds_split = ds_config.get("split", split)

            ds_max = int(max_samples * ds_weight) if max_samples else None

            logger.info("[%s] weight=%.2f -> %s samples", ds_name, ds_config.get('weight', 1.0), ds_max or 'all')

            try:
                if ds_subset:
                    ds = load_dataset(ds_name, ds_subset, split=ds_split, token=token)
                else:
                    ds = load_dataset(ds_name, split=ds_split, token=token)
            except Exception as e:
                logger.error("Failed to load %s: %s", ds_name, e)
                continue

            ds_list = list(ds)
            if ds_max and len(ds_list) > ds_max:
                random.shuffle(ds_list)
                ds_list = ds_list[:ds_max]

            ds_format = ds_config.get("format", format_name)
            for ex in ds_list:
                ex["_format"] = ds_format
            all_examples.extend(ds_list)

        random.shuffle(all_examples)
        logger.info("Total combined: %d samples", len(all_examples))

        return cls(
            dataset_name="combined",
            tokenizer=tokenizer,
            chat_template=chat_template,
            format_name=format_name,
            max_length=max_length,
            mask_user=mask_user,
            _examples=all_examples,
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        format_to_use = example.get("_format", self.format_name) if isinstance(example, dict) else self.format_name

        messages = convert_to_messages(example, format_to_use)
        if not messages:
            return self._empty_item()

        full_text = self.template.render(messages=messages)
        tokens = self.tokenizer.encode(full_text, add_special_tokens=True)

        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]

        input_ids = torch.tensor(tokens, dtype=torch.long)

        if self.mask_user:
            labels = self._create_masked_labels(messages, tokens)
        else:
            labels = input_ids.clone()

        return {"input_ids": input_ids, "labels": labels}

    def _create_masked_labels(self, messages: List[Dict], tokens: List[int]) -> torch.Tensor:
        """Labels with -100 for user messages (loss only on assistant content)."""
        labels = [-100] * len(tokens)

        for i, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue

            prefix_messages = messages[:i] + [{"role": "assistant", "content": ""}]
            prefix_text = self.template.render(messages=prefix_messages)
            prefix_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=True)
            start_pos = len(prefix_tokens)

            partial_messages = messages[:i+1]
            partial_text = self.template.render(messages=partial_messages)
            partial_tokens = self.tokenizer.encode(partial_text, add_special_tokens=True)
            end_pos = min(len(partial_tokens), len(tokens))

            if start_pos < end_pos:
                labels[start_pos:end_pos] = tokens[start_pos:end_pos]

            if end_pos >= len(tokens):
                break

        return torch.tensor(labels, dtype=torch.long)

    def _empty_item(self):
        return {
            "input_ids": torch.tensor([self.tokenizer.pad_token_id or 0], dtype=torch.long),
            "labels": torch.tensor([-100], dtype=torch.long),
        }


# ============================================================================
# COLLATE FUNCTIONS
# ============================================================================

def collate_fn(batch):
    """Collate for pre-training (fixed-length sequences)."""
    input_ids = torch.stack([x["input_ids"] for x in batch])
    labels = torch.stack([x["labels"] for x in batch])
    return {"input_ids": input_ids, "labels": labels}


def collate_sft(batch, pad_token_id: int = 0):
    """Collate for SFT (variable-length with padding)."""
    batch = [item for item in batch
             if item["input_ids"].shape[0] > 1 and (item["labels"] != -100).any()]
    if not batch:
        return None

    max_len = max(item["input_ids"].shape[0] for item in batch)

    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].shape[0]
        input_ids[i, :seq_len] = item["input_ids"]
        labels[i, :seq_len] = item["labels"]
        attention_mask[i, :seq_len] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }
