# W5D3 — NLP Pipelines, SQL Logging & Pakistan Dataset Experiment

This repository contains the Week 5 Day 3 assignment for the NLP Pipeline course.

---

## Files

| File | Description |
|------|-------------|
| `nlp_utils.py` | Reusable NLP module — 5 pipeline functions with DB logging |
| `queries_day3.sql` | 5 SQL queries on the `pipeline_results` table |
| `pakistan_dataset_experiment.py` | Roman Urdu sentiment experiment (50 samples) |
| `nlp_log.db` | Auto-generated SQLite database (created on first run) |
| `pakistan_dataset_results.csv` | Auto-generated CSV of 50 experiment results |

---

## How to Run

```bash
# Install dependencies
pip install transformers torch sentencepiece datasets pandas

# Run the NLP utils demo (creates nlp_log.db)
python nlp_utils.py

# Run the Pakistan dataset experiment
python pakistan_dataset_experiment.py

# Query the log database
sqlite3 nlp_log.db < queries_day3.sql
```

---

## Pakistan Dataset Experiment

### Dataset Chosen

**Dataset:** [`Aimlab/Sentiment-Analysis-Roman-Urdu`](https://huggingface.co/datasets/Aimlab/Sentiment-Analysis-Roman-Urdu)
**Model used:** `distilbert-base-uncased-finetuned-sst-2-english`

**Why this dataset?**
- Contains real Roman Urdu text from Pakistani social media — realistic for e-commerce or social-listening use cases.
- Labels are Positive / Negative — directly matching the binary sentiment task.
- Roman Urdu (Urdu written in English alphabet) is the dominant language for Pakistani online reviews and tweets.
- Tests how well an English-trained model handles a completely different language, which is a key learning objective of this task.

---

### Sample of 10 Inputs and Their Pipeline Outputs

| # | Input Text (Roman Urdu) | Model Output | Confidence |
|---|------------------------|--------------|-----------|
| 1 | bohat acha product hai, mujhe pasand aya | POSITIVE | 0.81 |
| 2 | bilkul bekar cheez hai, waste of money | NEGATIVE | 0.78 |
| 3 | delivery time theek tha lekin quality kharab thi | NEGATIVE | 0.63 |
| 4 | quality bohat achi hai, price bhi reasonable | POSITIVE | 0.84 |
| 5 | mujhe yeh bilkul pasand nahi aya | POSITIVE | 0.55 ❌ |
| 6 | original product mila, bahut khush hoon | POSITIVE | 0.91 |
| 7 | 3 hafton mein delivery aayi, acceptable nahi | POSITIVE | 0.52 ❌ |
| 8 | seller ne galat size bheja, wapas karna pada | POSITIVE | 0.61 ❌ |
| 9 | excellent product, highly recommended | POSITIVE | 0.998 |
| 10 | packaging toot ke aayi, bohat bura laga | NEGATIVE | 0.69 |

*❌ = model predicted wrong (negative text classified as positive)*

**Approximate accuracy on 50 samples: ~58%**
*(vs ~95% accuracy on English SST-2 test set)*

---

### Observation: Does an English-trained Model Work on Urdu Text?

**Short answer: Partially — and only for the easiest cases.**

**Why it partially works:**
The DistilBERT SST-2 model was trained only on English movie reviews. When it gives a "correct" result on Roman Urdu input, it is picking up on:
1. **Borrowed English words** in the text (e.g. "excellent", "quality", "delivery", "product") — words the model actually knows.
2. **Surface signals** like exclamation marks and capitalization that correlate with positive sentiment in its training data.
3. **Majority-class bias** — the model defaults toward POSITIVE, which accidentally aligns with many positive reviews.

**Why it fails:**
- Roman Urdu words like `kharab` (bad), `bekar` (useless), `ghatiya` (low quality), `nahi` (not/no) are **completely out-of-vocabulary**. The model treats them as noise and ignores them.
- **Negation** in Roman Urdu ("pasand nahi" = don't like, "acha nahi" = not good) is not understood at all — so clear negative sentences get labelled POSITIVE.
- **Average confidence on Urdu inputs: ~0.65** vs **~0.97 on English** — the model is essentially guessing.
- Accuracy of ~58% is barely above random chance (50%), making it unusable for production.

**Conclusion:**
An English-only model is **not suitable** for a Pakistani e-commerce review classifier. For a real product, the right approach would be:
- Fine-tune `xlm-roberta-base` on a labelled Urdu/Roman Urdu sentiment dataset
- Use `bert-base-multilingual-cased` with Pakistani review data
- Or use a dedicated Roman Urdu model trained on Pakistani social-media text

Even 2,000–3,000 labelled examples would give dramatically better results than this zero-shot English model approach.

---

### HuggingFace Dataset Screenshot

![Dataset Screenshot](<images/Screenshot 2026-04-22 at 5.55.10 PM.png>)

---

## Model Card Summary: DistilBERT SST-2

**Model page:** https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english

### Training Data — What is SST-2?

The model was fine-tuned on **SST-2 (Stanford Sentiment Treebank, version 2)**, a binary sentiment classification benchmark derived from movie review snippets collected from Rotten Tomatoes. Each sentence was manually annotated by crowd workers as either Positive or Negative. SST-2 is one of the nine tasks in the **GLUE benchmark** — the standard multi-task evaluation suite for English NLP.

Key facts about SST-2:
- ~67,000 training sentences, ~870 validation sentences
- All text is **English only** — single sentences, short (10–30 tokens typically)
- Domain is narrowly **film criticism** (movie reviews)
- Binary labels only: Positive or Negative (no Neutral class)

### Evaluation Metrics and Scores

| Metric | Score |
|--------|-------|
| **Accuracy (SST-2 validation set)** | **91.3%** |

DistilBERT achieves this after being distilled from BERT-base (which scores ~93% on SST-2). DistilBERT is ~40% smaller and ~60% faster while retaining ~97% of BERT's accuracy on this task. The model card reports only accuracy — no F1, precision, or recall breakdown — because SST-2 is a balanced binary task where accuracy is the standard GLUE metric.

### Known Limitations and Biases

1. **Domain limitation:** Trained on movie reviews only. Sentiment language in e-commerce, news, medical, or social media text differs significantly in vocabulary and style.
2. **Language limitation:** English only. Non-English, code-switched, or romanised text is tokenised as unknown tokens — predictions become unreliable.
3. **Social biases from pre-training:** DistilBERT is distilled from BERT, pre-trained on English Wikipedia and BookCorpus. These corpora carry documented biases related to gender, race, and religion. Fine-tuning on SST-2 does not remove these biases.
4. **No Neutral class:** Every input is forced into POSITIVE or NEGATIVE — mixed or neutral reviews cannot be expressed.
5. **Short text assumption:** SST-2 sentences average 10–30 tokens. Performance degrades on long documents without pre-chunking.

### Intended Uses and Out-of-Scope Uses

**Intended uses:**
- Binary positive/negative classification of English sentences
- Benchmarking and research — comparing distillation techniques
- Quick prototyping where approximate English sentiment is acceptable

**Out-of-scope uses:**
- Any language other than English
- High-stakes decisions (hiring, credit scoring, legal analysis) — bias caveats apply
- Neutral sentiment detection
- Fine-grained sentiment (star ratings, aspect-level)
- Long documents without pre-chunking

### Personal Conclusion: Is This Model Suitable for a Pakistani E-commerce Review Classifier?

**No — it is not suitable for production use in this context**, for three reasons:

**1. Language mismatch (critical flaw):** Pakistani e-commerce platforms like Daraz receive reviews in Urdu script, Roman Urdu, and Pakistani English. As demonstrated in the experiment above, accuracy drops to ~58% on Roman Urdu — barely above random. The model cannot understand Urdu vocabulary or grammar at all.

**2. Domain mismatch (significant flaw):** SST-2 is built from film reviews. Product reviews use fundamentally different language — shipping times, packaging quality, price-to-value ratio, seller trust. These concepts simply do not appear in the training data.

**3. No neutral class (practical flaw):** A large portion of real product reviews are mixed or neutral ("product is okay, nothing special"). Forcing every review into POSITIVE or NEGATIVE produces distorted analytics and incorrect seller ratings.

**What I would use instead:** Fine-tune `xlm-roberta-base` on a labelled Urdu/Roman Urdu e-commerce review dataset. Even 2,000–5,000 examples would yield dramatically better and trustworthy results for a real Pakistani application.