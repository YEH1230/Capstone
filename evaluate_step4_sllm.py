"""Evaluate Step 4 sLLM predictions with label-level P/R/F1 diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_step4_json(raw_output: str, allowed_labels: set[str] | None = None) -> tuple[dict[str, Any] | None, str]:
  text = (raw_output or "").strip()
  match = JSON_OBJECT_RE.fullmatch(text)
  if match is None:
    return None, "non_json_output"
  try:
    parsed = json.loads(match.group(0))
  except json.JSONDecodeError as error:
    return None, f"json_decode_error:{error.msg}"

  label = str(parsed.get("label", "NONE")).strip()
  if allowed_labels and label not in allowed_labels:
    return None, f"invalid_label:{label}"
  is_pii = bool(parsed.get("is_pii", label != "NONE"))
  if not is_pii:
    label = "NONE"
  confidence = parsed.get("confidence", 0.0)
  try:
    confidence = float(confidence)
  except (TypeError, ValueError):
    confidence = 0.0
  return {
    "is_pii": is_pii,
    "label": label,
    "confidence": confidence,
    "reason": str(parsed.get("reason", "")),
  }, ""


def read_dataset(path: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  with path.open("r", encoding="utf-8") as handle:
    for index, line in enumerate(handle):
      if not line.strip():
        continue
      obj = json.loads(line)
      user = json.loads(obj["messages"][1]["content"])
      assistant_raw = obj["messages"][2]["content"]
      gold, gold_error = parse_step4_json(assistant_raw)
      if gold is None:
        raise ValueError(f"Gold row {index} is not parseable: {gold_error}")
      rows.append(
        {
          "id": obj.get("metadata", {}).get("id", str(index)),
          "sentence": user.get("sentence", ""),
          "candidate": user.get("candidate", {}),
          "gold_label": gold["label"],
          "gold": gold,
          "metadata": obj.get("metadata", {}),
          "prompt_messages": obj["messages"][:2],
        }
      )
  return rows


def read_prediction_file(path: Path) -> list[str]:
  outputs: list[str] = []
  with path.open("r", encoding="utf-8") as handle:
    for line in handle:
      if not line.strip():
        continue
      obj = json.loads(line)
      outputs.append(str(obj.get("raw_output", obj.get("output", ""))))
  return outputs


def gold_as_predictions(rows: list[dict[str, Any]]) -> list[str]:
  return [json.dumps(row["gold"], ensure_ascii=False, separators=(",", ":")) for row in rows]


def predict_with_model(rows: list[dict[str, Any]], model_path: str, max_new_tokens: int) -> list[str]:
  from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

  try:
    from peft import PeftConfig, PeftModel

    peft_config = PeftConfig.from_pretrained(model_path)
    base_id = str(peft_config.base_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(base_id, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, model_path)
  except Exception:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)
  pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
  outputs: list[str] = []
  for row in rows:
    prompt = "\n".join(message["content"] for message in row["prompt_messages"])
    result = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False, return_full_text=False)
    generated = result[0].get("generated_text", "") if result else ""
    outputs.append(str(generated))
  return outputs


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
  precision = tp / (tp + fp) if tp + fp else 0.0
  recall = tp / (tp + fn) if tp + fn else 0.0
  f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
  return {"precision": precision, "recall": recall, "f1": f1}


def evaluate(rows: list[dict[str, Any]], raw_outputs: list[str], allowed_labels: set[str]) -> dict[str, Any]:
  labels = sorted(allowed_labels | {"NONE"})
  confusion: dict[str, Counter[str]] = {label: Counter() for label in labels}
  parse_failures: list[dict[str, Any]] = []
  false_positives: list[dict[str, Any]] = []
  false_negatives: list[dict[str, Any]] = []
  parsed_predictions: list[dict[str, Any]] = []

  for row, raw_output in zip(rows, raw_outputs):
    parsed, error = parse_step4_json(raw_output, allowed_labels)
    if parsed is None:
      parsed = {"is_pii": False, "label": "NONE", "confidence": 0.0, "reason": error}
      parse_failures.append({"sentence": row["sentence"], "raw_output": raw_output, "error": error})
    gold_label = row["gold_label"]
    pred_label = parsed["label"]
    confusion.setdefault(gold_label, Counter())[pred_label] += 1
    parsed_predictions.append(parsed)

    record = {
      "sentence": row["sentence"],
      "candidate": row["candidate"],
      "gold_label": gold_label,
      "predicted_label": pred_label,
      "raw_output": raw_output,
      "parsed": parsed,
      "metadata": row["metadata"],
    }
    if gold_label == "NONE" and pred_label != "NONE":
      false_positives.append(record)
    if gold_label != "NONE" and pred_label == "NONE":
      false_negatives.append(record)

  tag_metrics: dict[str, dict[str, float | int]] = {}
  for label in labels:
    if label == "NONE":
      continue
    tp = confusion.get(label, Counter()).get(label, 0)
    fp = sum(confusion.get(other, Counter()).get(label, 0) for other in labels if other != label)
    fn = sum(count for pred, count in confusion.get(label, Counter()).items() if pred != label)
    metrics = precision_recall_f1(tp, fp, fn)
    tag_metrics[label] = {**metrics, "tp": tp, "fp": fp, "fn": fn, "support": sum(confusion.get(label, Counter()).values())}

  metric_values = list(tag_metrics.values())
  macro = {
    key: sum(float(item[key]) for item in metric_values) / len(metric_values) if metric_values else 0.0
    for key in ("precision", "recall", "f1")
  }
  total_tp = sum(int(item["tp"]) for item in metric_values)
  total_fp = sum(int(item["fp"]) for item in metric_values)
  total_fn = sum(int(item["fn"]) for item in metric_values)
  micro = precision_recall_f1(total_tp, total_fp, total_fn)
  return {
    "samples": len(rows),
    "parse_failures": len(parse_failures),
    "parse_failure_rate": len(parse_failures) / len(rows) if rows else 0.0,
    "micro": micro,
    "macro": macro,
    "tag_metrics": tag_metrics,
    "confusion": {gold: dict(preds) for gold, preds in confusion.items()},
    "false_positives": false_positives,
    "false_negatives": false_negatives,
    "parse_failure_examples": parse_failures[:50],
  }


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / "step4_eval_summary.json").write_text(
    json.dumps({k: v for k, v in result.items() if k not in {"false_positives", "false_negatives"}}, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )
  with (output_dir / "step4_eval_summary.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["label", "precision", "recall", "f1", "tp", "fp", "fn", "support"])
    for label, metrics in result["tag_metrics"].items():
      writer.writerow([label, metrics["precision"], metrics["recall"], metrics["f1"], metrics["tp"], metrics["fp"], metrics["fn"], metrics["support"]])
  labels = sorted(result["confusion"])
  pred_labels = sorted({pred for row in result["confusion"].values() for pred in row} | set(labels))
  with (output_dir / "step4_confusion_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["gold_label", *pred_labels])
    for gold in labels:
      writer.writerow([gold, *[result["confusion"].get(gold, {}).get(pred, 0) for pred in pred_labels]])
  for name in ("false_positives", "false_negatives"):
    with (output_dir / f"step4_{name}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
      for row in result[name]:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dataset", default="data/step4/test.jsonl")
  parser.add_argument("--predictions-jsonl", default="")
  parser.add_argument("--model-path", default="")
  parser.add_argument("--output-dir", default="reports")
  parser.add_argument("--max-new-tokens", type=int, default=128)
  parser.add_argument("--allow-gold-predictions", action="store_true", help="Use gold labels as predictions for smoke tests.")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  rows = read_dataset(Path(args.dataset))
  allowed_labels = {"NONE"} | {row["candidate"].get("label", "NONE") for row in rows} | {row["gold_label"] for row in rows}
  if args.predictions_jsonl:
    outputs = read_prediction_file(Path(args.predictions_jsonl))
  elif args.model_path:
    outputs = predict_with_model(rows, args.model_path, args.max_new_tokens)
  elif args.allow_gold_predictions:
    outputs = gold_as_predictions(rows)
  else:
    raise SystemExit("Provide --predictions-jsonl, --model-path, or --allow-gold-predictions.")
  if len(outputs) != len(rows):
    raise SystemExit(f"Prediction count mismatch: {len(outputs)} != {len(rows)}")
  result = evaluate(rows, outputs, allowed_labels)
  write_outputs(result, Path(args.output_dir))
  print(json.dumps({"micro": result["micro"], "macro": result["macro"], "parse_failure_rate": result["parse_failure_rate"]}, ensure_ascii=False))


if __name__ == "__main__":
  main()
