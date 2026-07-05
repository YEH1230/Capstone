# Step 4 PII Validation Dataset

This dataset rebuilds the legacy Qwen Step 4 corpus for precision-focused
PII validation. Legacy positive examples are converted to the unified JSON
schema and augmented with NONE hard negatives for high-confusion labels.

## Splits

| split | samples | positive | negative |
| --- | ---: | ---: | ---: |
| train | 10394 | 9180 | 1214 |
| valid | 1327 | 1175 | 152 |
| test | 1313 | 1159 | 154 |

## Unified Output Schema

```json
{"is_pii": true, "label": "PS_NAME", "confidence": 0.87, "reason": "..."}
```

Labels must be one of `allowed_labels`; uncertain cases must use `NONE`.
