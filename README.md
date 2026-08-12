# Crisis Text Classifier — Explainable Demo

A small web demo that classifies a message and shows **which words drove the
decision**, colour-coded by how much each one contributed.

---

## ⚠️ This is a stand-in model, not published research

**The model behind this demo is not the model from "Harnessing Natural Language
Processing for Disaster Response and Crisis Management" (NHSJS, 2025).**

The original fine-tuned checkpoint was not available — the
[source repository](https://github.com/abho7/NLP-Disaster-Response) contains
training and evaluation *scripts* but no trained weights, and no training data
(the README there notes the tweets are withheld under Twitter's policy). So this
demo trains its own substitute on a public dataset.

Two consequences worth being explicit about:

1. **The metrics below are not the paper's metrics** and are not comparable to
   them. Different model, different data, different task.
2. **The task itself differs.** The public labels mark whether a tweet is *about
   a real disaster*. They do not encode *urgency*. The paper's framing
   (`urgent` / `not urgent`, or the `high`/`medium`/`low` in its sample CSV) is
   a different problem, so this demo's classes are named for what they actually
   are: `disaster-related` / `not disaster-related`.

The disclaimer is served from the API (`notice` on every `/predict` response)
and rendered on the page, so it travels with the output rather than living only
in this file.

### Actual measured performance

Held-out test split (1,523 of 7,613 rows), stand-in model:

| Metric | Value |
|---|---|
| Accuracy | **0.8024** |
| F1 (macro) | **0.7986** |
| F1 (positive class) | **0.7711** |
| ROC-AUC | **0.8705** |

Regenerate with `python training/train_standin.py`; they are written to
`artifacts/metrics.json` and served at `GET /metrics`.

---

## How it works

**Model.** TF-IDF (1–2 grams) → logistic regression, trained on
[`venetis/disaster_tweets`](https://huggingface.co/datasets/venetis/disaster_tweets),
a public mirror of Kaggle's *Real or Not? NLP with Disaster Tweets* (7,613 rows).

Chosen over fine-tuning a BERT for two reasons: it mirrors the TF-IDF/Naive-Bayes
half of the pipeline the paper describes, and — more importantly — it is linear,
which makes the explanations **exact**.

**Explanations.** For a linear model, SHAP values have a closed form:

```
phi_j = w_j * (x_j - E[x_j])        base = w · E[x] + b
sum_j phi_j + base = logit(x)                      (exactly)
```

No sampling, no approximation — the highlighted contributions are the true ones.
A test (`test_closed_form_matches_shap_linear_explainer`) asserts these agree
with `shap.LinearExplainer` to within 1e-8. `shap` is therefore a *test*
dependency only; serving does not need it.

Feature-level values are mapped back onto spans of the raw input, splitting each
bigram's contribution across its two words, so the frontend can highlight the
text the user actually typed — whitespace and punctuation preserved.

### Why the highlighting matters

```
"This new album is an absolute disaster lol"
   → not disaster-related (82.0%)
   disaster  +0.83     ← the word alone points to a real disaster
   new       −0.85     ← context outvotes it
   lol       −0.56
```

The label alone would tell you nothing about *why*. This is the feature.

---

## Layout

```
crisis-nlp-demo/
├── app/
│   ├── main.py            FastAPI: /predict, /metrics, /health
│   ├── model_adapter.py   Predictor protocol + linear stand-in + exact SHAP
│   └── static/index.html  single-page UI
├── training/
│   └── train_standin.py   trains the stand-in, writes artifacts/
├── tests/                 16 tests: attribution math + API contract
├── artifacts/             standin_model.joblib (~0.5 MB), metrics.json
├── nlp-model/             read-only clone of the original repo (gitignored)
├── Dockerfile             HF Spaces + Render compatible
└── requirements.txt       serving deps only (no torch, no shap)
```

`nlp-model/` is an **unmodified** clone of the original project, present for
reference only. Nothing in this demo imports from it — there is nothing
importable in it — and nothing writes to it.

### Swapping in the real model

`app/model_adapter.py` defines a `Predictor` protocol. If the fine-tuned BERT
checkpoint is recovered, add a `BertPredictor` satisfying the same protocol
(returning `Segment` contributions from attention or gradient attribution) and
change the one line in `lifespan()` that constructs the predictor. The API and
frontend need no changes.

---

## Running locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements-dev.txt

python training/train_standin.py        # writes artifacts/
uvicorn app.main:app --reload           # http://127.0.0.1:8000
pytest
```

## Deploying

**Render (free tier)** — `render.yaml` in this repo is a Blueprint; point Render
at the repo and it configures itself. Render injects `$PORT`, which the
Dockerfile's `CMD` honours, and `/health` gates traffic until the model has
loaded.

Measured resident memory while serving is **107 MB** against the free tier's
512 MB limit. Free instances sleep after ~15 minutes idle, so the first request
after a quiet period takes ~50s.

**Hugging Face Spaces** — note that Docker Spaces now require a **PRO
subscription**; only Static Spaces are free. Running here for free would mean
porting inference to the browser, which is viable (the model is linear and the
weights are ~0.5 MB as JSON) but is not what this repo currently does.

## API

`POST /predict` — `{"text": "..."}` →

```jsonc
{
  "predicted_class": "disaster-related",
  "confidence": 0.754,
  "class_probabilities": { "disaster-related": 0.754, "not disaster-related": 0.246 },
  "segments": [ { "text": "Roof", "contribution": 0.1955, "is_token": true }, ... ],
  "base_value": -0.42,          // w · E[x] + b
  "absent_contribution": 0.09,  // features not present in this text
  "logit": 1.1199,              // base + absent + sum(contributions)
  "is_standin": true,
  "notice": "Stand-in model trained on the public Kaggle ..."
}
```

`GET /metrics` — held-out metrics. `GET /health` — liveness.
