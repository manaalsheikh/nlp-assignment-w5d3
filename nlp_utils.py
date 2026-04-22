"""
nlp_utils.py
============
A reusable NLP utility module that wraps five Hugging Face Transformers
pipelines and automatically logs every call to a local SQLite database
(nlp_log.db) for later analysis.

Supported tasks
---------------
1. sentiment_analysis()   -- Binary positive/negative classification
2. zero_shot_classify()   -- Label-free classification with custom labels
3. summarize_text()       -- Abstractive text summarisation (AutoModel)
4. translate_text()       -- English to French translation
5. generate_text()        -- Open-ended text generation (custom function)

All pipeline calls are cached in memory so models are loaded only once per
session, and every result is logged to the `pipeline_results` SQLite table.

Dependencies
------------
    pip install transformers torch sentencepiece

Author : Umed
Course : W5D3 NLP Pipeline Assignment
"""

import os
import json
import sqlite3

# Force PyTorch backend — avoids TensorFlow/protobuf conflicts
os.environ["USE_TF"]    = "0"
os.environ["USE_TORCH"] = "1"

from transformers import (
    pipeline as hf_pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------
DB_PATH = "nlp_log.db"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS pipeline_results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task        TEXT     NOT NULL,
    model       TEXT     NOT NULL,
    input_text  TEXT     NOT NULL,
    output_json TEXT     NOT NULL,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""


def _get_conn() -> sqlite3.Connection:
    """Open a SQLite connection and ensure the schema exists."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(_CREATE_TABLE_SQL)
    conn.commit()
    return conn


def _log(task: str, model: str, input_text: str, output) -> None:
    """
    Persist one pipeline call to the database.

    Parameters
    ----------
    task : str
        HuggingFace pipeline task name (e.g. 'sentiment-analysis').
    model : str
        Model identifier used for this call.
    input_text : str
        Raw text passed to the pipeline.
    output : any
        Pipeline output — serialised to JSON.
    """
    conn = _get_conn()
    conn.execute(
        "INSERT INTO pipeline_results (task, model, input_text, output_json) "
        "VALUES (?, ?, ?, ?)",
        (task, model, input_text, json.dumps(output, ensure_ascii=False)),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Pipeline cache — avoids reloading models on every call
# ---------------------------------------------------------------------------
_CACHE: dict = {}


def _get_pipeline(task: str, model: str = None):
    """
    Return a cached HuggingFace pipeline, loading it only on first call.

    Parameters
    ----------
    task : str
        Pipeline task string (e.g. 'text-generation').
    model : str, optional
        Model checkpoint. Uses HuggingFace default for the task if None.

    Returns
    -------
    transformers.Pipeline
        A ready-to-use pipeline object.
    """
    key = (task, model)
    if key not in _CACHE:
        _CACHE[key] = hf_pipeline(task, model=model) if model else hf_pipeline(task)
    return _CACHE[key]


# ---------------------------------------------------------------------------
# 1. Sentiment Analysis
# ---------------------------------------------------------------------------
def sentiment_analysis(
    text: str,
    model: str = "distilbert-base-uncased-finetuned-sst-2-english",
) -> dict:
    """
    Run binary sentiment analysis on English text.

    Parameters
    ----------
    text : str
        Input sentence or short paragraph.
    model : str, optional
        HuggingFace model checkpoint. Defaults to DistilBERT SST-2.

    Returns
    -------
    dict
        {"label": "POSITIVE" | "NEGATIVE", "score": float}

    Example
    -------
    >>> result = sentiment_analysis("I love this product!")
    >>> print(result)  # {'label': 'POSITIVE', 'score': 0.9998}
    """
    pipe = _get_pipeline("sentiment-analysis", model)
    output = pipe(text[:512])
    _log("sentiment-analysis", model, text, output)
    return output[0]


# ---------------------------------------------------------------------------
# 2. Zero-Shot Classification
# ---------------------------------------------------------------------------
def zero_shot_classify(
    text: str,
    labels: list,
    model: str = "facebook/bart-large-mnli",
) -> dict:
    """
    Classify text into one of the given labels without task-specific training.

    Parameters
    ----------
    text : str
        The input text to classify.
    labels : list of str
        Candidate class names, e.g. ["sports", "politics", "tech"].
    model : str, optional
        Zero-shot classification model checkpoint.

    Returns
    -------
    dict
        Contains sequence, labels (sorted by score), and scores.

    Example
    -------
    >>> result = zero_shot_classify("PSL match was exciting", ["sports", "politics"])
    >>> print(result["labels"][0])  # sports
    """
    pipe = _get_pipeline("zero-shot-classification", model)
    output = pipe(text, labels)
    _log("zero-shot-classification", model, text, output)
    return output


# ---------------------------------------------------------------------------
# 3. Text Summarisation
# Uses AutoTokenizer + AutoModelForSeq2SeqLM directly because
# pipeline("summarization") was removed in newer transformers versions.
# ---------------------------------------------------------------------------
def summarize_text(
    text: str,
    max_length: int = 80,
    min_length: int = 30,
    model: str = "sshleifer/distilbart-cnn-12-6",
) -> str:
    """
    Generate an abstractive summary of the given text.

    Uses AutoTokenizer + AutoModelForSeq2SeqLM directly instead of
    pipeline("summarization") which was dropped in newer transformers versions.

    Parameters
    ----------
    text : str
        Source document. Works best with 100-1000 words.
    max_length : int, optional
        Maximum token length of the summary (default 80).
    min_length : int, optional
        Minimum token length of the summary (default 30).
    model : str, optional
        Seq2Seq summarisation model checkpoint.

    Returns
    -------
    str
        The generated summary string.

    Example
    -------
    >>> summary = summarize_text("Pakistan is located in South Asia...")
    >>> print(summary)
    """
    cache_key = ("summarization_automodel", model)
    if cache_key not in _CACHE:
        tokenizer  = AutoTokenizer.from_pretrained(model)
        model_obj  = AutoModelForSeq2SeqLM.from_pretrained(model)
        _CACHE[cache_key] = (tokenizer, model_obj)

    tokenizer, model_obj = _CACHE[cache_key]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    )
    summary_ids = model_obj.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )
    result = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    _log("summarization", model, text, [{"summary_text": result}])
    return result


# ---------------------------------------------------------------------------
# 4. Translation (English -> French)
# Uses AutoTokenizer + AutoModelForSeq2SeqLM directly because
# pipeline("translation_en_to_fr") was removed in newer transformers versions.
# ---------------------------------------------------------------------------
def translate_text(
    text: str,
    model: str = "Helsinki-NLP/opus-mt-en-fr",
) -> str:
    """
    Translate English text to French using a MarianMT model.

    Uses AutoTokenizer + AutoModelForSeq2SeqLM directly instead of
    pipeline("translation_en_to_fr") which was dropped in newer transformers.

    Parameters
    ----------
    text : str
        English source sentence or paragraph.
    model : str, optional
        MarianMT EN->FR model checkpoint.

    Returns
    -------
    str
        French translation string.

    Example
    -------
    >>> fr = translate_text("Hello, how are you?")
    >>> print(fr)  # Bonjour, comment allez-vous ?
    """
    cache_key = ("translation_automodel", model)
    if cache_key not in _CACHE:
        tokenizer = AutoTokenizer.from_pretrained(model)
        model_obj  = AutoModelForSeq2SeqLM.from_pretrained(model)
        _CACHE[cache_key] = (tokenizer, model_obj)

    tokenizer, model_obj = _CACHE[cache_key]

    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    translated = model_obj.generate(**inputs)
    result = tokenizer.decode(translated[0], skip_special_tokens=True)

    _log("translation_en_to_fr", model, text, [{"translation_text": result}])
    return result


# ---------------------------------------------------------------------------
# 5. Text Generation — Custom function added for W5D3
# ---------------------------------------------------------------------------
def generate_text(
    prompt: str,
    max_length: int = 80,
    num_return_sequences: int = 1,
    model: str = "gpt2",
) -> str:
    """
    Generate a text continuation from a given prompt using GPT-2.

    This is the custom function added for W5D3. Useful for building
    simple autocomplete or story-generation features.

    Parameters
    ----------
    prompt : str
        Starting text that the model will continue.
    max_length : int, optional
        Total token length of the generated output (default 80).
    num_return_sequences : int, optional
        How many completions to generate (default 1).
    model : str, optional
        Text-generation model checkpoint (default gpt2).

    Returns
    -------
    str
        The generated text string (includes the original prompt).

    Example
    -------
    >>> story = generate_text("Once upon a time in Lahore")
    >>> print(story)
    """
    pipe = _get_pipeline("text-generation", model)
    output = pipe(
        prompt,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        truncation=True,
        pad_token_id=50256,
    )
    _log("text-generation", model, prompt, output)
    return output[0]["generated_text"]


# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("nlp_utils.py — W5D3 Function Demo")
    print("=" * 60)

    # 1. Sentiment Analysis
    print("\n[1] Sentiment Analysis")
    tests = [
        "I absolutely love this product, best purchase ever!",
        "The delivery was terrible and packaging was broken.",
        "Yeh product bohat acha hai, mujhe pasand aya.",
    ]
    for t in tests:
        r = sentiment_analysis(t)
        print(f"  Input : {t[:60]}")
        print(f"  Result: {r['label']}  ({r['score']:.4f})\n")

    # 2. Zero-Shot Classification
    print("[2] Zero-Shot Classification")
    zs = zero_shot_classify(
        "PSL cricket match was an exciting game last night.",
        ["sports", "politics", "technology", "weather"],
    )
    print(f"  Input : {zs['sequence']}")
    for lbl, sc in zip(zs["labels"][:3], zs["scores"][:3]):
        print(f"  {lbl:<14}: {sc:.4f}")
    print()

    # 3. Summarisation
    print("[3] Text Summarisation")
    long_text = (
        "Pakistan is a country in South Asia established in 1947. "
        "It has a population of over 220 million people. The country "
        "has diverse geography from Himalayan peaks in the north to "
        "the Arabian Sea coastline in the south. Pakistan's economy "
        "is largely agriculture-based but its technology and services "
        "sectors are growing rapidly in cities like Karachi and Lahore."
    )
    print(f"  Summary: {summarize_text(long_text)}\n")

    # 4. Translation EN -> FR
    print("[4] English -> French Translation")
    en = "Education is the most powerful weapon you can use to change the world."
    fr = translate_text(en)
    print(f"  EN: {en}")
    print(f"  FR: {fr}\n")

    # 5. Text Generation (Custom Function)
    print("[5] Text Generation (Custom Function)")
    story = generate_text("Once upon a time in Lahore", max_length=60)
    print(f"  Generated: {story}\n")

    print("=" * 60)
    print(f"All results logged to '{DB_PATH}'.")
    print("Query with: sqlite3 nlp_log.db < queries_day3.sql")
    print("=" * 60)