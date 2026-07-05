from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from rag.pii.detector import PIIDetector
from rag.pii.step4_local_sllm import LocalQwenStep4Verifier, parse_step4_json


class _MockPipeline:
  tokenizer = None

  def __init__(self, output: str) -> None:
    self.output = output

  def __call__(self, *_args, **_kwargs):
    return [{"generated_text": self.output}]


def test_json_parsing_accepts_valid_json() -> None:
  parsed, error = parse_step4_json(
    '{"is_pii":true,"label":"PS_NAME","confidence":0.88,"reason":"name"}',
    confidence_threshold=0.75,
  )
  assert error == ""
  assert parsed["is_pii"] is True
  assert parsed["label"] == "PS_NAME"


def test_json_parsing_rejects_non_json_text() -> None:
  parsed, error = parse_step4_json("PII")
  assert error == "non_json_output"
  assert parsed["label"] == "NONE"


def test_none_label_when_confidence_below_threshold() -> None:
  parsed, error = parse_step4_json(
    '{"is_pii":true,"label":"PS_NAME","confidence":0.41,"reason":"weak"}',
    confidence_threshold=0.75,
  )
  assert error == ""
  assert parsed["is_pii"] is False
  assert parsed["label"] == "NONE"


def test_ps_name_hard_negative_mock() -> None:
  verifier = LocalQwenStep4Verifier(
    {"pii": {"step4": {"enabled": True, "confidence_threshold": 0.75}}},
    pipeline=_MockPipeline('{"is_pii":false,"label":"NONE","confidence":0.91,"reason":"historical"}'),
  )
  result = verifier.classify("정약용", "PS_NAME", "이번 정책은 정약용의 목민심서 사례를 설명한다.")
  assert result["label"] == "NONE"


def test_lc_place_hard_negative_mock() -> None:
  verifier = LocalQwenStep4Verifier(
    {"pii": {"step4": {"enabled": True, "confidence_threshold": 0.75}}},
    pipeline=_MockPipeline('{"is_pii":false,"label":"NONE","confidence":0.9,"reason":"public place"}'),
  )
  assert verifier.verify("서울역", "LC_PLACE", "서울역 근처 맛집을 묻는 일반 장소 검색 문맥이다.") is False


def test_resident_and_foreigner_number_labels_are_distinct() -> None:
  resident, _ = parse_step4_json(
    '{"is_pii":true,"label":"QT_RESIDENT_NUMBER","confidence":0.96,"reason":"resident"}',
    confidence_threshold=0.75,
  )
  foreigner, _ = parse_step4_json(
    '{"is_pii":true,"label":"QT_FOREIGNER_NUMBER","confidence":0.96,"reason":"foreigner"}',
    confidence_threshold=0.75,
  )
  assert resident["label"] == "QT_RESIDENT_NUMBER"
  assert foreigner["label"] == "QT_FOREIGNER_NUMBER"


def test_step4_provider_switching() -> None:
  local = PIIDetector({"pii": {"runtime": {"enable_step4": True}, "step4": {"provider": "local_qwen"}}})
  assert local.sllm_verifier.provider == "local_qwen"

  openai = PIIDetector({"pii": {"runtime": {"enable_step4": True}, "step4": {"provider": "openai"}, "sllm": {"provider": "openai"}}})
  assert openai.sllm_verifier.provider == "openai"


def test_evaluate_script_smoke(tmp_path: Path) -> None:
  dataset = tmp_path / "test.jsonl"
  output_dir = tmp_path / "reports"
  row = {
    "messages": [
      {"role": "system", "content": "system"},
      {
        "role": "user",
        "content": json.dumps(
          {
            "task": "KDPII_STEP4_CLASSIFY",
            "sentence": "서울역 근처 맛집을 알려줘.",
            "candidate": {"text": "서울역", "label": "LC_PLACE"},
          },
          ensure_ascii=False,
        ),
      },
      {
        "role": "assistant",
        "content": '{"is_pii":false,"label":"NONE","confidence":0.9,"reason":"public"}',
      },
    ],
    "metadata": {"gold_label": "NONE"},
  }
  dataset.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
  script = Path(__file__).resolve().parents[1] / "evaluate_step4_sllm.py"
  subprocess.run(
    [
      sys.executable,
      str(script),
      "--dataset",
      str(dataset),
      "--output-dir",
      str(output_dir),
      "--allow-gold-predictions",
    ],
    check=True,
  )
  assert (output_dir / "step4_eval_summary.json").exists()
