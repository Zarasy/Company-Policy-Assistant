
import os
import re
import time
import hashlib
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

POLICY_FILE = BASE_DIR / "company_policies.csv"
CACHE_DIR = BASE_DIR / ".embedding_cache"

# New generation model
GEN_MODEL = "gemini-3.5-flash-lite"

# Gemini Embedding model for vector search
EMBED_MODEL = "gemini-embedding-001"


# =========================================================
# LOAD POLICIES AND API CLIENT
# =========================================================

@lru_cache(maxsize=1)
def policies():
    df = pd.read_csv(POLICY_FILE).fillna("")

    required_columns = {"title", "policy_text"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "The CSV must contain title and policy_text."
        )

    return df


@lru_cache(maxsize=1)
def client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        # Also support Streamlit Cloud secrets
        try:
            import streamlit as st
            api_key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            pass

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. "
            "Add it to .env or Streamlit Secrets."
        )

    return genai.Client(api_key=api_key)


# =========================================================
# API RETRY HANDLING
# =========================================================

def retry_api(operation, attempts=3):
    """
    Retry temporary server errors.

    Do not retry exhausted daily quotas.
    """

    for attempt in range(attempts):
        try:
            return operation()

        except Exception as error:
            message = str(error).lower()

            # Daily quota exhausted: retries will not help
            if (
                "per day limit exceeded" in message
                or "requests per day" in message
                or "quota exceeded" in message
            ):
                raise

            temporary = any(
                code in message
                for code in [
                    "429",
                    "500",
                    "502",
                    "503",
                    "504",
                    "unavailable",
                    "overloaded"
                ]
            )

            if not temporary or attempt == attempts - 1:
                raise

            time.sleep(2 ** attempt)


# =========================================================
# SHARED RESULT FORMAT
# =========================================================

def empty_result(answer, policy="None"):
    return {
        "answer": answer,
        "policy": policy,
        "tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "status": "Completed"
    }


# =========================================================
# METHOD 1: RULES-BASED SEARCH
# =========================================================

STOP_WORDS = {
    "a", "an", "the", "is", "are",
    "what", "how", "can", "could",
    "do", "does", "i", "we",
    "my", "our", "their", "they",
    "to", "for", "and", "of",
    "in", "on", "at", "with",
    "company", "policy", "policies",
    "employees", "employee",
    "about", "please", "tell", "me"
}


def words(text):
    return set(
        re.findall(
            r"[a-z0-9]+",
            str(text).lower()
        )
    ) - STOP_WORDS


def rules(question):
    df = policies()

    question_words = words(question)

    if not question_words:
        return empty_result(
            "No matching policy found."
        )

    candidates = []

    for _, row in df.iterrows():

        title_words = words(row["title"])
        body_words = words(row["policy_text"])

        title_matches = len(
            question_words & title_words
        )

        body_matches = len(
            question_words & body_words
        )

        score = (
            3 * title_matches
            + body_matches
        )

        candidates.append(
            (score, title_matches, row)
        )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    score, title_matches, best_row = candidates[0]

    # Reject weak keyword matches
    if score < 3 or (
        title_matches == 0
        and len(question_words) > 1
    ):
        return empty_result(
            "No sufficiently relevant policy found."
        )

    return empty_result(
        str(best_row["policy_text"]),
        str(best_row["title"])
    )


# =========================================================
# GEMINI EMBEDDINGS
# =========================================================

def policy_documents(df):
    documents = []

    for _, row in df.iterrows():

        document = (
            f"Title: {row['title']}\n"
            f"Department: {row.get('department', '')}\n"
            f"Category: {row.get('category', '')}\n"
            f"Policy: {row['policy_text']}"
        )

        documents.append(document)

    return documents


def embed_texts(texts, task_type):
    vectors = []

    batch_size = 20

    for start in range(0, len(texts), batch_size):

        batch = texts[
            start:start + batch_size
        ]

        response = retry_api(
            lambda: client().models.embed_content(
                model=EMBED_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type
                )
            )
        )

        for item in response.embeddings:
            vectors.append(item.values)

    return np.asarray(
        vectors,
        dtype=np.float32
    )


def get_policy_embeddings(df):
    documents = policy_documents(df)

    fingerprint = hashlib.sha256(
        (
            EMBED_MODEL
            + "\n"
            + "\n".join(documents)
        ).encode("utf-8")
    ).hexdigest()

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    cache_file = (
        CACHE_DIR / f"{fingerprint}.npy"
    )

    if cache_file.exists():
        return np.load(
            cache_file,
            allow_pickle=False
        )

    vectors = embed_texts(
        documents,
        "RETRIEVAL_DOCUMENT"
    )

    np.save(
        cache_file,
        vectors
    )

    return vectors


def vector_search(question, top_k=5):
    df = policies()

    document_vectors = get_policy_embeddings(df)

    question_vector = embed_texts(
        [question],
        "RETRIEVAL_QUERY"
    )[0]

    document_norms = np.linalg.norm(
        document_vectors,
        axis=1
    )

    question_norm = np.linalg.norm(
        question_vector
    )

    denominator = (
        document_norms * question_norm
    )

    similarities = np.divide(
        document_vectors @ question_vector,
        denominator,
        out=np.zeros(len(df)),
        where=denominator != 0
    )

    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    return df.iloc[top_indices].copy()


# =========================================================
# METHOD 2 AND 3: LLM
# =========================================================

def llm(question, indexed=False):
    df = policies()

    if indexed:
        selected = vector_search(
            question,
            top_k=5
        )
    else:
        selected = df

    context_parts = []

    for _, row in selected.iterrows():

        context_parts.append(
            f"POLICY TITLE: {row['title']}\n"
            f"POLICY TEXT: {row['policy_text']}"
        )

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
You are a company policy assistant.

Answer the employee's question using ONLY the
company policies provided below.

Requirements:
1. Do not invent company rules.
2. Do not use external knowledge.
3. Identify the exact supporting policy title.
4. Keep the answer concise and clear.
5. If the policies do not answer the question,
   respond exactly:
   The policy database does not specify.

COMPANY POLICIES:

{context}

EMPLOYEE QUESTION:

{question}
"""

    response = retry_api(
        lambda: client().models.generate_content(
            model=GEN_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0
            )
        )
    )

    usage = response.usage_metadata

    answer = (
        response.text
        or "No response generated."
    )

    if indexed:
        policy = ", ".join(
            selected["title"]
            .astype(str)
            .tolist()
        )
    else:
        policy = "See policy cited in answer"

    if answer.strip().lower().startswith(
        "the policy database does not specify"
    ):
        policy = "None"

    return {
        "answer": answer,
        "policy": policy,
        "tokens": getattr(
            usage,
            "total_token_count",
            None
        ),
        "input_tokens": getattr(
            usage,
            "prompt_token_count",
            None
        ),
        "output_tokens": getattr(
            usage,
            "candidates_token_count",
            None
        ),
        "status": "Completed"
    }


# =========================================================
# COMPARE ALL THREE APPROACHES
# =========================================================

def compare(question):
    methods = [
        (
            "Rules-based",
            lambda: rules(question)
        ),
        (
            "LLM without index",
            lambda: llm(
                question,
                indexed=False
            )
        ),
        (
            "LLM with vector index",
            lambda: llm(
                question,
                indexed=True
            )
        )
    ]

    results = []

    for name, method in methods:

        start_time = time.perf_counter()

        try:
            data = method()

        except Exception as error:
            message = str(error)

            if (
                "429" in message
                and (
                    "per day" in message.lower()
                    or "quota" in message.lower()
                )
            ):
                answer = (
                    "Gemini API daily quota exceeded. "
                    "Please try again after the quota resets "
                    "or select a model with available quota."
                )
            else:
                answer = f"Error: {error}"

            data = {
                "answer": answer,
                "policy": "—",
                "tokens": None,
                "input_tokens": None,
                "output_tokens": None,
                "status": "Error"
            }

        elapsed = round(
            time.perf_counter() - start_time,
            2
        )

        results.append({
            "Approach": name,
            "Answer": data["answer"],
            "Relevant policy": data["policy"],
            "Response time (s)": elapsed,
            "Total tokens": data.get("tokens"),
            "Input tokens": data.get("input_tokens"),
            "Output tokens": data.get("output_tokens"),
            "Status": data.get("status")
        })

    return results
