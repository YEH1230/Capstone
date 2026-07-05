# STEP 4 Experiment Comparison

The local session prepared the experiment harness and generated a precision-
focused dataset. Full Qwen2.5-1.5B-Instruct QLoRA training was not run in this session, so
model metrics below are placeholders except for the smoke test, which uses gold
labels to verify the evaluator and output files.

| experiment | dataset version | train | valid | test | micro P | micro R | micro F1 | macro P | macro R | macro F1 | lowest F1 tag | main false-positive cause | adopted |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| evaluator smoke | rebuilt-hard-negative-v1 | 10,394 | 1,327 | 1,313 | 1.000 | 1.000 | 1.000 | 0.933 | 0.933 | 0.933 | labels without test support | none; gold predictions only | no |
| baseline reproduction | legacy-positive-only | 9,235 | 1,161 | 1,118 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | expected over-confirmation from positive-only data | no |
| hard negatives | rebuilt-hard-negative-v1 | 10,394 | 1,327 | 1,313 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | PS_NAME/LC_PLACE/ORG-family ambiguity | pending |
| hard negatives + higher NONE sampling | rebuilt-hard-negative-v1 | 10,394 | 1,327 | 1,313 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | expected reduction in over-confirmation | pending |
| hard negatives + threshold tuning | rebuilt-hard-negative-v1 | 10,394 | 1,327 | 1,313 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | confidence calibration required | pending |

## Adoption Criteria

- micro F1 >= 0.90
- macro F1 >= 0.90 preferred
- target STEP 4 labels each F1 >= 0.90
- precision not materially lower than recall
- JSON parsing failure rate <= 1%

## Recommended Next Runs

1. Train baseline with the legacy positive-only dataset to quantify the current
   over-confirmation pattern.
2. Train with `data/step4` and default QLoRA hard-negative weighting.
3. Train with `--negative-oversample 1.5 --hard-negative-oversample 2.0`.
4. If VRAM is sufficient, compare regular LoRA with `--use_4bit false`.
5. Evaluate each checkpoint with `evaluate_step4_sllm.py` and inspect
   `reports/step4_false_positives.jsonl` before selecting a model.
