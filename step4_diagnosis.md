# STEP 4 sLLM Diagnosis

## Scope

This diagnosis inspected the extracted legacy dataset at `D:\sllm\data\qwen_step4`
because `qwen_step4.zip` was not present in the workspace. The available files are
`train.jsonl`, `valid.jsonl`, `test.jsonl`, and `statistics.json`.

## Current Dataset Structure

Each legacy row is an instruction-tuning chat sample:

- system: Korean privacy validator instruction
- user: JSON payload with `task`, `sentence`, and one `candidate`
- assistant: compact JSON with `decision`, `label`, and `text`
- metadata: sentence id, pii id, and source label

The legacy assistant schema is:

```json
{"decision":"CONFIRM","label":"PS_NAME","text":"..."}
```

The new pipeline uses this unified schema instead:

```json
{"is_pii":true,"label":"PS_NAME","confidence":0.87,"reason":"..."}
```

## Label Distribution

Legacy total samples: 11,514. All inspected assistant labels are positive
`CONFIRM` examples.

| label | total |
| --- | ---: |
| PS_NAME | 1,911 |
| PS_NICKNAME | 1,306 |
| OGG_CLUB | 1,199 |
| LC_PLACE | 1,114 |
| CV_POSITION | 1,078 |
| LC_ADDRESS | 1,078 |
| OG_WORKPLACE | 1,033 |
| FD_MAJOR | 638 |
| OGG_RELIGION | 477 |
| CV_SEX | 467 |
| CV_MILITARY_CAMP | 408 |
| TM_BLOOD_TYPE | 405 |
| QT_ALIEN_NUMBER | 200 |
| QT_RESIDENT_NUMBER | 200 |

## Precision Risk Findings

1. Negative data is missing.
   The legacy dataset has no `NONE` examples. A model trained on this data is
   structurally encouraged to confirm every candidate, which directly explains
   high recall with low precision.

2. Hard negatives are missing for the most ambiguous labels.
   Labels such as `PS_NAME`, `PS_NICKNAME`, `LC_PLACE`, `OG_WORKPLACE`, and
   `OGG_CLUB` require context to distinguish personal data from public names,
   generic group names, organization names, and examples. The legacy data does
   not teach these boundaries.

3. Train/validation/test leakage exists at sentence level.
   Exact sentence overlap was found between train/valid and train/test. The
   overlap is small, but it confirms the split was not sentence-grouped.

4. Training and inference formats were inconsistent.
   The existing project inference prompt asked for one token, `PII` or
   `NOT_PII`, while training data asked for `CONFIRM` JSON. This mismatch makes
   parsing and calibration unreliable.

5. The legacy output schema is too weak for thresholding.
   It has no `confidence` and no explicit `NONE` label, so evaluation cannot
   distinguish invalid JSON, uncertainty, false positives, and tag confusion.

6. Tag confusion is likely concentrated in context-dependent labels.
   The riskiest groups are person-like strings (`PS_NAME` vs `PS_NICKNAME` vs
   non-PII names), place/workplace/club labels (`LC_PLACE`, `OG_WORKPLACE`,
   `OGG_CLUB`), and resident/foreigner registration number variants.

## Rebuilt Dataset

`build_step4_dataset.py` converts the legacy positives to the unified JSON
schema and adds `NONE` hard negatives. Current generated counts:

| split | samples | positive | negative |
| --- | ---: | ---: | ---: |
| train | 10,394 | 9,180 | 1,214 |
| valid | 1,327 | 1,175 | 152 |
| test | 1,313 | 1,159 | 154 |

The split is sentence-grouped so exact sentence duplicates do not cross split
boundaries. Label statistics are stored in `data/step4/label_stats.json`.

## Conclusion

The current Hugging Face model should not be used as-is as the STEP 4 verifier.
The likely primary cause of low precision is data design, not simply too few
epochs: the model saw almost only positive examples and a different inference
format. The next valid experiment is to train with the rebuilt dataset, compare
hard-negative/NONE sampling settings, and evaluate tag-wise F1 plus JSON parsing
failure rate before replacing the OpenAI fallback.
