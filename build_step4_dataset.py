"""Build a precision-focused Step 4 PII validation dataset.

The legacy Step 4 data contains only positive CONFIRM examples. This builder
keeps those positives, converts every row to the unified JSON instruction
format, adds tag-specific hard negatives, and performs a sentence-grouped split
to avoid train/valid/test leakage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ALLOWED_LABELS = [
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
]

SYSTEM_PROMPT = (
  "You are a Korean privacy data validator. Given one sentence and one PII "
  "candidate span with its proposed tag, decide whether the candidate is real "
  "personal information in context. Respond only with compact JSON."
)


@dataclass(frozen=True)
class Step4Sample:
  sentence: str
  candidate_text: str
  candidate_label: str
  gold_label: str
  is_pii: bool
  reason: str
  source: str
  hard_negative_type: str = ""

  @property
  def sentence_key(self) -> str:
    normalized = " ".join(self.sentence.split())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def build_user_payload(sample: Step4Sample) -> dict[str, Any]:
  begin = sample.sentence.find(sample.candidate_text)
  end = begin + len(sample.candidate_text) if begin >= 0 else -1
  return {
    "task": "KDPII_STEP4_CLASSIFY",
    "sentence": sample.sentence,
    "candidate": {
      "text": sample.candidate_text,
      "label": sample.candidate_label,
      "begin": begin,
      "end": end,
    },
    "allowed_labels": ALLOWED_LABELS,
    "constraints": [
      "Output JSON only.",
      "Choose exactly one label from allowed_labels.",
      "Choose NONE when uncertain.",
      "Judge by context, not by the candidate string alone.",
    ],
  }


def to_chat_row(sample: Step4Sample) -> dict[str, Any]:
  assistant = {
    "is_pii": sample.is_pii,
    "label": sample.gold_label,
    "confidence": 0.95 if sample.is_pii else 0.92,
    "reason": sample.reason,
  }
  return {
    "messages": [
      {"role": "system", "content": SYSTEM_PROMPT},
      {"role": "user", "content": json.dumps(build_user_payload(sample), ensure_ascii=False)},
      {"role": "assistant", "content": json.dumps(assistant, ensure_ascii=False, separators=(",", ":"))},
    ],
    "metadata": {
      "source": sample.source,
      "gold_label": sample.gold_label,
      "candidate_label": sample.candidate_label,
      "sentence_key": sample.sentence_key,
      "hard_negative_type": sample.hard_negative_type,
    },
  }


def parse_legacy_row(row: dict[str, Any]) -> Step4Sample:
  user = json.loads(row["messages"][1]["content"])
  candidate = user["candidate"]
  label = str(candidate["label"])
  return Step4Sample(
    sentence=str(user["sentence"]),
    candidate_text=str(candidate["text"]),
    candidate_label=label,
    gold_label=label,
    is_pii=True,
    reason="문맥상 후보 표현이 해당 개인정보 태그로 사용됨",
    source="legacy_positive",
  )


def load_legacy_samples(source_dir: Path) -> list[Step4Sample]:
  samples: list[Step4Sample] = []
  for split in ("train", "valid", "test"):
    path = source_dir / f"{split}.jsonl"
    if not path.exists():
      continue
    with path.open("r", encoding="utf-8") as handle:
      for line in handle:
        if line.strip():
          samples.append(parse_legacy_row(json.loads(line)))
  return samples


def hard_negative_templates() -> list[Step4Sample]:
  rows = [
    ("PS_NAME", "정약용", "이번 정책은 정약용의 목민심서 사례를 설명한다.", "historical_name"),
    ("PS_NAME", "해리포터", "사용자는 해리포터 같은 판타지 인물을 예로 들었다.", "character_name"),
    ("PS_NAME", "삼성", "삼성 서비스센터 위치를 알려 달라는 일반 기관명 문맥이다.", "org_fragment"),
    ("PS_NAME", "민수", "민수기 1장을 읽었다는 성경 책 이름 문맥이다.", "common_noun_like_name"),
    ("PS_NICKNAME", "고수", "이 제품은 초보자보다 고수에게 적합하다는 일반 표현이다.", "generic_alias"),
    ("PS_NICKNAME", "막내", "팀에서 막내 역할을 맡은 사람을 일반적으로 부르는 표현이다.", "generic_alias"),
    ("PS_NICKNAME", "왕초보", "왕초보도 따라 할 수 있는 튜토리얼이라는 등급 표현이다.", "generic_alias"),
    ("LC_PLACE", "서울역", "서울역 근처 맛집을 묻는 일반 장소 검색 문맥이다.", "public_place"),
    ("LC_PLACE", "부산", "부산 여행 코스를 추천해 달라는 일반 지역명 문맥이다.", "public_place"),
    ("LC_PLACE", "강남", "강남 상권 분석 보고서의 지역명으로 쓰였다.", "public_place"),
    ("OG_WORKPLACE", "카카오", "카카오 주가 변동을 설명하는 공개 기업명 문맥이다.", "public_org"),
    ("OG_WORKPLACE", "세종대학교", "세종대학교 입학 요강을 요약하는 공개 기관명 문맥이다.", "public_org"),
    ("OG_WORKPLACE", "병원", "가까운 병원을 찾는 일반 장소명 문맥이다.", "generic_org"),
    ("OGG_CLUB", "독서모임", "독서모임을 만드는 방법을 설명하는 일반 모임명이다.", "generic_group"),
    ("OGG_CLUB", "축구동아리", "축구동아리 운영 규칙 예시를 설명하는 일반 표현이다.", "generic_group"),
    ("OGG_CLUB", "동문회", "동문회 행사를 기획하는 일반 단체 유형명이다.", "generic_group"),
    ("QT_RESIDENT_NUMBER", "900101-1234567", "예시 주민등록번호 형식을 설명하기 위한 더미 값이다.", "dummy_identifier"),
    ("QT_ALIEN_NUMBER", "900101-5234567", "외국인등록번호 형식 예시에 쓰인 가상 번호다.", "dummy_identifier"),
    ("QT_FOREIGNER_NUMBER", "900101-5234567", "외국인등록번호 형식 예시에 쓰인 가상 번호다.", "dummy_identifier"),
  ]
  samples: list[Step4Sample] = []
  for label, text, sentence, kind in rows:
    samples.append(
      Step4Sample(
        sentence=sentence,
        candidate_text=text,
        candidate_label=label,
        gold_label="NONE",
        is_pii=False,
        reason="문맥상 특정 개인을 식별하는 개인정보가 아님",
        source="hard_negative_seed",
        hard_negative_type=kind,
      )
    )
  return samples


def expand_hard_negatives(base: list[Step4Sample], copies: int) -> list[Step4Sample]:
  expanded: list[Step4Sample] = []
  wrappers = [
    "RAG 응답 초안: {sentence}",
    "검색 결과 요약에는 다음 문장이 포함된다. {sentence}",
    "보안 진단 중 모델 응답에서 확인된 문장: {sentence}",
    "{sentence} 이 문맥은 특정 개인의 원본 정보 공개가 아니다.",
  ]
  for sample in base:
    expanded.append(sample)
    for index in range(max(0, copies - 1)):
      sentence = (
        wrappers[index % len(wrappers)].format(sentence=sample.sentence)
        + f" [hard-negative-{sample.hard_negative_type}-{index + 1}]"
      )
      expanded.append(
        Step4Sample(
          sentence=sentence,
          candidate_text=sample.candidate_text,
          candidate_label=sample.candidate_label,
          gold_label="NONE",
          is_pii=False,
          reason=sample.reason,
          source=sample.source,
          hard_negative_type=sample.hard_negative_type,
        )
      )
  return expanded


def group_split(
  samples: list[Step4Sample],
  *,
  seed: int,
  train_ratio: float,
  valid_ratio: float,
) -> dict[str, list[Step4Sample]]:
  rng = random.Random(seed)
  groups: dict[str, list[Step4Sample]] = defaultdict(list)
  for sample in samples:
    groups[sample.sentence_key].append(sample)

  label_to_groups: dict[str, list[str]] = defaultdict(list)
  group_primary_label: dict[str, str] = {}
  for key, rows in groups.items():
    counts = Counter(row.gold_label for row in rows)
    primary = counts.most_common(1)[0][0]
    group_primary_label[key] = primary
    label_to_groups[primary].append(key)

  split_keys = {"train": set(), "valid": set(), "test": set()}
  for label, keys in label_to_groups.items():
    rng.shuffle(keys)
    n = len(keys)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    if n >= 3:
      n_train = max(1, n_train)
      n_valid = max(1, n_valid)
    split_keys["train"].update(keys[:n_train])
    split_keys["valid"].update(keys[n_train : n_train + n_valid])
    split_keys["test"].update(keys[n_train + n_valid :])

  output = {"train": [], "valid": [], "test": []}
  assigned = set()
  for split, keys in split_keys.items():
    for key in keys:
      if key in assigned:
        continue
      output[split].extend(groups[key])
      assigned.add(key)
  for key, rows in groups.items():
    if key not in assigned:
      output["train"].extend(rows)
  return output


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8", newline="\n") as handle:
    for row in rows:
      handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_dataset_card(path: Path, stats: dict[str, Any]) -> None:
  lines = [
    "# Step 4 PII Validation Dataset",
    "",
    "This dataset rebuilds the legacy Qwen Step 4 corpus for precision-focused",
    "PII validation. Legacy positive examples are converted to the unified JSON",
    "schema and augmented with NONE hard negatives for high-confusion labels.",
    "",
    "## Splits",
    "",
    "| split | samples | positive | negative |",
    "| --- | ---: | ---: | ---: |",
  ]
  for split in ("train", "valid", "test"):
    item = stats["splits"][split]
    lines.append(
      f"| {split} | {item['samples']} | {item['positive']} | {item['negative']} |"
    )
  lines.extend(
    [
      "",
      "## Unified Output Schema",
      "",
      "```json",
      '{"is_pii": true, "label": "PS_NAME", "confidence": 0.87, "reason": "..."}',
      "```",
      "",
      "Labels must be one of `allowed_labels`; uncertain cases must use `NONE`.",
    ]
  )
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
  source_dir = Path(args.source_dir)
  output_dir = Path(args.output_dir)
  positives = load_legacy_samples(source_dir)
  negatives = expand_hard_negatives(hard_negative_templates(), args.hard_negative_copies)
  samples = positives + negatives
  splits = group_split(
    samples,
    seed=args.seed,
    train_ratio=args.train_ratio,
    valid_ratio=args.valid_ratio,
  )

  stats: dict[str, Any] = {
    "source_dir": str(source_dir),
    "total_samples": len(samples),
    "allowed_labels": ALLOWED_LABELS,
    "splits": {},
  }
  for split, split_samples in splits.items():
    rows = [to_chat_row(sample) for sample in split_samples]
    write_jsonl(output_dir / f"{split}.jsonl", rows)
    labels = Counter(sample.gold_label for sample in split_samples)
    stats["splits"][split] = {
      "samples": len(split_samples),
      "positive": sum(1 for sample in split_samples if sample.is_pii),
      "negative": sum(1 for sample in split_samples if not sample.is_pii),
      "labels": dict(sorted(labels.items())),
      "unique_sentences": len({sample.sentence_key for sample in split_samples}),
    }

  (output_dir / "label_stats.json").write_text(
    json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )
  write_dataset_card(output_dir / "dataset_card.md", stats)
  return stats


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source-dir", default="../data/qwen_step4")
  parser.add_argument("--output-dir", default="data/step4")
  parser.add_argument("--hard-negative-copies", type=int, default=80)
  parser.add_argument("--train-ratio", type=float, default=0.8)
  parser.add_argument("--valid-ratio", type=float, default=0.1)
  parser.add_argument("--seed", type=int, default=42)
  return parser.parse_args()


def main() -> None:
  stats = build_dataset(parse_args())
  print(json.dumps(stats["splits"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
