"""Fine-tune Qwen2.5-1.5B-Instruct for Step 4 PII validation with QLoRA.

By default the base model is loaded in 4-bit NF4 and only LoRA adapter weights
are trained. Set ``--use_4bit false`` for full-precision/base-precision LoRA
training on machines with enough VRAM.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def str_to_bool(value: str | bool) -> bool:
  if isinstance(value, bool):
    return value
  normalized = value.strip().lower()
  if normalized in {"1", "true", "yes", "y", "on"}:
    return True
  if normalized in {"0", "false", "no", "n", "off"}:
    return False
  raise argparse.ArgumentTypeError(f"Expected true or false, got {value!r}")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
  parser.add_argument("--train-file", default="data/step4/train.jsonl")
  parser.add_argument("--valid-file", default="data/step4/valid.jsonl")
  parser.add_argument("--output-dir", default="models/qwen_step4_lora")
  parser.add_argument("--learning-rate", type=float, default=1e-4)
  parser.add_argument("--epochs", type=float, default=4)
  parser.add_argument("--batch-size", type=int, default=2)
  parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
  parser.add_argument("--max-length", type=int, default=768)
  parser.add_argument("--lora-r", type=int, default=16)
  parser.add_argument("--lora-alpha", type=int, default=32)
  parser.add_argument("--lora-dropout", type=float, default=0.05)
  parser.add_argument("--use-4bit", "--use_4bit", type=str_to_bool, default=True)
  parser.add_argument("--bnb-4bit-quant-type", default="nf4", choices=["nf4", "fp4"])
  parser.add_argument("--bnb-4bit-use-double-quant", type=str_to_bool, default=True)
  parser.add_argument(
    "--lora-target-modules",
    default="auto",
    help="Comma-separated module names or 'auto' for Qwen projection modules.",
  )
  parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
  parser.add_argument("--early-stopping-patience", type=int, default=3)
  parser.add_argument("--negative-oversample", type=float, default=1.0)
  parser.add_argument("--hard-negative-oversample", type=float, default=1.5)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--push-to-hub", action="store_true")
  parser.add_argument("--hub-model-id", default="")
  return parser.parse_args()


def read_jsonl(path: str) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  with Path(path).open("r", encoding="utf-8") as handle:
    for line in handle:
      if line.strip():
        rows.append(json.loads(line))
  return rows


def oversample_training_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
  sampled = list(rows)
  rng = random.Random(args.seed)
  for row in rows:
    metadata = row.get("metadata", {})
    label = metadata.get("gold_label", "")
    multiplier = 1.0
    if label == "NONE":
      multiplier = args.negative_oversample
    if metadata.get("hard_negative_type"):
      multiplier = max(multiplier, args.hard_negative_oversample)
    extra = max(0, int(multiplier) - 1)
    sampled.extend([row] * extra)
    if multiplier % 1 and rng.random() < (multiplier % 1):
      sampled.append(row)
  rng.shuffle(sampled)
  return sampled


def select_compute_dtype() -> Any:
  import torch

  if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    return torch.bfloat16
  return torch.float16


def resolve_lora_target_modules(model: Any, requested: str) -> list[str]:
  if requested != "auto":
    return [name.strip() for name in requested.split(",") if name.strip()]

  qwen_candidates = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
  ]
  module_names = {name.rsplit(".", 1)[-1] for name, _module in model.named_modules()}
  detected = [name for name in qwen_candidates if name in module_names]
  return detected or qwen_candidates


def main() -> None:
  args = parse_args()
  random.seed(args.seed)

  from datasets import Dataset
  from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
  from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
  )

  train_rows = oversample_training_rows(read_jsonl(args.train_file), args)
  valid_rows = read_jsonl(args.valid_file)
  compute_dtype = select_compute_dtype()

  tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

  def format_row(row: dict[str, Any]) -> str:
    messages = row["messages"]
    if hasattr(tokenizer, "apply_chat_template"):
      return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages)

  def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
    texts = [format_row(json.loads(item)) for item in batch["row_json"]]
    tokenized = tokenizer(texts, truncation=True, max_length=args.max_length, padding=False)
    tokenized["labels"] = [ids[:] for ids in tokenized["input_ids"]]
    return tokenized

  train_ds = Dataset.from_dict({"row_json": [json.dumps(row, ensure_ascii=False) for row in train_rows]}).map(tokenize, batched=True, remove_columns=["row_json"])
  valid_ds = Dataset.from_dict({"row_json": [json.dumps(row, ensure_ascii=False) for row in valid_rows]}).map(tokenize, batched=True, remove_columns=["row_json"])

  quantization_config = None
  if args.use_4bit:
    quantization_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type=args.bnb_4bit_quant_type,
      bnb_4bit_compute_dtype=compute_dtype,
      bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
    )

  model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=None if args.use_4bit else compute_dtype,
    trust_remote_code=True,
  )
  if args.gradient_checkpointing:
    model.gradient_checkpointing_enable()
  if args.use_4bit:
    model = prepare_model_for_kbit_training(model)

  lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=resolve_lora_target_modules(model, args.lora_target_modules),
  )
  model = get_peft_model(model, lora_config)

  training_args = TrainingArguments(
    output_dir=args.output_dir,
    learning_rate=args.learning_rate,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_dir=str(Path(args.output_dir) / "logs"),
    logging_steps=20,
    save_total_limit=3,
    seed=args.seed,
    bf16=str(compute_dtype).endswith("bfloat16"),
    fp16=str(compute_dtype).endswith("float16"),
    report_to=["tensorboard"],
  )

  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
  )
  trainer.train()
  trainer.save_model(args.output_dir)
  tokenizer.save_pretrained(args.output_dir)

  if args.push_to_hub:
    if not args.hub_model_id:
      raise SystemExit("--hub-model-id is required with --push-to-hub")
    trainer.push_to_hub(args.hub_model_id)


if __name__ == "__main__":
  main()
