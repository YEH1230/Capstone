"""Local Qwen LoRA Step 4 verifier with strict JSON parsing."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from loguru import logger

from rag.pii.step3_ner import NERMatch


ALLOWED_LABELS = {
  "NONE",
  "CV_MILITARY_CAMP",
  "CV_POSITION",
  "CV_SEX",
  "FD_MAJOR",
  "LC_ADDRESS",
  "LC_PLACE",
  "OGG_CLUB",
  "OGG_RELIGION",
  "OG_WORKPLACE",
  "PS_NAME",
  "PS_NICKNAME",
  "QT_ALIEN_NUMBER",
  "QT_FOREIGNER_NUMBER",
  "QT_RESIDENT_NUMBER",
  "TM_BLOOD_TYPE",
}

JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_step4_json(
  raw_output: str,
  *,
  allowed_labels: set[str] | None = None,
  confidence_threshold: float = 0.0,
) -> tuple[dict[str, Any], str]:
  """Parse the model JSON response and normalize unsafe outputs to NONE."""
  labels = allowed_labels or ALLOWED_LABELS
  text = (raw_output or "").strip()
  if JSON_OBJECT_RE.fullmatch(text) is None:
    return _none_result(0.0, "non_json_output"), "non_json_output"

  try:
    parsed = json.loads(text)
  except json.JSONDecodeError as error:
    return _none_result(0.0, f"json_decode_error:{error.msg}"), f"json_decode_error:{error.msg}"

  label = str(parsed.get("label", "NONE")).strip()
  if label not in labels:
    return _none_result(0.0, f"invalid_label:{label}"), f"invalid_label:{label}"

  try:
    confidence = float(parsed.get("confidence", 0.0))
  except (TypeError, ValueError):
    confidence = 0.0

  is_pii = bool(parsed.get("is_pii", label != "NONE"))
  if not is_pii or label == "NONE" or confidence < confidence_threshold:
    return {
      "is_pii": False,
      "label": "NONE",
      "confidence": confidence,
      "reason": str(parsed.get("reason", "문맥상 개인정보로 확정하지 않음")),
    }, ""

  return {
    "is_pii": True,
    "label": label,
    "confidence": confidence,
    "reason": str(parsed.get("reason", "")),
  }, ""


def _none_result(confidence: float, reason: str) -> dict[str, Any]:
  return {
    "is_pii": False,
    "label": "NONE",
    "confidence": confidence,
    "reason": reason,
  }


def build_step4_messages(entity_text: str, tag: str, context: str) -> list[dict[str, str]]:
  """Build the unified instruction prompt used by training and inference."""
  payload = {
    "task": "KDPII_STEP4_CLASSIFY",
    "sentence": context,
    "candidate": {"text": entity_text, "label": tag},
    "allowed_labels": sorted(ALLOWED_LABELS),
    "constraints": [
      "Output JSON only.",
      "Choose exactly one label from allowed_labels.",
      "Choose NONE when uncertain.",
      "Judge by context, not by the candidate string alone.",
    ],
  }
  return [
    {
      "role": "system",
      "content": (
        "You are a Korean privacy data validator. Given one sentence and one "
        "PII candidate span with its proposed tag, decide whether the candidate "
        "is real personal information in context. Respond only with compact JSON."
      ),
    },
    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
  ]


class LocalQwenStep4Verifier:
  """Verify Step 3 low-confidence candidates with a local Qwen LoRA model."""

  def __init__(self, config: dict[str, Any], pipeline: Any | None = None) -> None:
    pii_config = config.get("pii", {})
    runtime_config = pii_config.get("runtime", {})
    step4_config = pii_config.get("step4", {})
    legacy_sllm_config = pii_config.get("sllm", {})

    self.enabled = bool(step4_config.get("enabled", runtime_config.get("enable_step4", True)))
    self.provider = "local_qwen"
    self.model_path = str(
      step4_config.get(
        "local_model_path",
        legacy_sllm_config.get("model_path", "models/qwen_step4_lora"),
      )
    )
    self.base_model_path = str(
      step4_config.get(
        "base_model_path",
        legacy_sllm_config.get("base_model_path", "Qwen/Qwen2.5-1.5B-Instruct"),
      )
    )
    self.confidence_threshold = float(step4_config.get("confidence_threshold", 0.75))
    self.max_new_tokens = int(step4_config.get("max_new_tokens", legacy_sllm_config.get("max_new_tokens", 128)))
    self.torch_dtype = step4_config.get("torch_dtype", legacy_sllm_config.get("torch_dtype", "auto"))
    self.device = step4_config.get("device", legacy_sllm_config.get("device", "auto"))
    self.pipeline = pipeline
    self.load_status = "ready" if pipeline is not None else ("not_loaded" if self.enabled else "skipped")
    self.error_message = ""
    self.last_reason = ""
    self.mode = "local_qwen" if self.enabled else "disabled"

  def warm_up(self) -> None:
    if not self.enabled:
      self.load_status = "skipped"
      return
    if self.pipeline is not None:
      self.load_status = "ready"
      return

    try:
      from peft import PeftConfig, PeftModel
      from transformers import AutoModelForCausalLM, AutoTokenizer
      from transformers import pipeline as hf_pipeline

      model_path = Path(self.model_path)
      model_id = str(model_path) if model_path.exists() else self.model_path
      peft_config = PeftConfig.from_pretrained(model_id)
      base_id = self.base_model_path or str(peft_config.base_model_name_or_path)
      tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
      if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
      model = AutoModelForCausalLM.from_pretrained(
        base_id,
        device_map=self.device,
        torch_dtype=self.torch_dtype,
        trust_remote_code=True,
      )
      model = PeftModel.from_pretrained(model, model_id)
      model.eval()
      self.pipeline = hf_pipeline("text-generation", model=model, tokenizer=tokenizer)
      self.load_status = "ready"
      self.error_message = ""
    except Exception as error:
      self.pipeline = None
      self.load_status = "failed"
      self.error_message = str(error)
      logger.warning("Local Qwen Step 4 warm-up failed: {}", error)

  def classify(self, entity_text: str, tag: str, context: str) -> dict[str, Any]:
    if not self.enabled:
      self.last_reason = "disabled"
      return _none_result(0.0, "step4_disabled")
    if self.pipeline is None:
      self.warm_up()
    if self.pipeline is None:
      self.last_reason = "local_qwen_unavailable"
      return _none_result(0.0, self.error_message or "local_qwen_unavailable")

    messages = build_step4_messages(entity_text, tag, context)
    prompt = self._format_prompt(messages)
    try:
      raw = self._call_pipeline(prompt)
      parsed, error = parse_step4_json(
        raw,
        confidence_threshold=self.confidence_threshold,
      )
      self.last_reason = "parse_failed" if error else "verified"
      return parsed
    except Exception as error:
      self.error_message = str(error)
      self.last_reason = "inference_failed"
      logger.warning("Local Qwen Step 4 inference failed: {}", error)
      return _none_result(0.0, str(error))

  def verify(self, entity_text: str, tag: str, context: str) -> bool:
    return bool(self.classify(entity_text, tag, context)["is_pii"])

  def verify_batch(self, matches: list[NERMatch], full_text: str) -> list[NERMatch]:
    if not self.enabled or not matches:
      return []
    verified: list[NERMatch] = []
    for match in matches:
      context_start = max(0, match.start - 100)
      context_end = min(len(full_text), match.end + 100)
      context = full_text[context_start:context_end]
      result = self.classify(match.text, match.tag, context)
      if result["is_pii"]:
        verified.append(match)
    return verified

  def _format_prompt(self, messages: list[dict[str, str]]) -> str:
    tokenizer = getattr(self.pipeline, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
      return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages)

  def _call_pipeline(self, prompt: str) -> str:
    output = self.pipeline(
      prompt,
      max_new_tokens=self.max_new_tokens,
      do_sample=False,
      return_full_text=False,
    )
    first = output[0] if isinstance(output, list) and output else output
    if isinstance(first, dict):
      return str(first.get("generated_text", first.get("summary_text", ""))).strip()
    return str(first).strip()

  def get_runtime_status(
    self,
    *,
    candidate_count: int = 0,
    verified_count: int = 0,
    reason: str = "",
  ) -> dict[str, Any]:
    status = "skipped"
    if self.enabled and self.load_status == "ready":
      status = "ready"
    elif self.enabled and self.load_status == "failed":
      status = "failed"
    elif self.enabled:
      status = "not_loaded"
    return {
      "enabled": self.enabled,
      "provider": self.provider,
      "mode": self.mode,
      "status": status,
      "reason": reason or self.last_reason,
      "model_path": self.model_path,
      "base_model_path": self.base_model_path,
      "load_status": self.load_status,
      "confidence_threshold": self.confidence_threshold,
      "candidate_count": candidate_count,
      "verified_count": verified_count,
      "error": self.error_message,
    }
