
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

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env")

POLICY_FILE = BASE / "company_policies.csv"
CACHE = BASE / ".embedding_cache"

GEN_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
EMBED_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"
)

STOP_WORDS = {
    "a", "an", "the", "is", "are", "what", "how",
    "can", "could", "do", "does", "i", "we",
    "my", "our", "their", "they", "to", "for",
    "and", "of", "in", "on", "at", "with",
    "company", "policy", "employees", "employee",
    "about", "please", "tell", "me"
}


@lru_cache(maxsize=1)
def policies():
    df = pd.read_csv(POLICY_FILE).fillna("")
    required = {"title", "policy_text"}

    if not required.issubset(df.columns):
        raise ValueError("Missing policy columns")

    return df


@lru_cache(maxsize=1)
def client():
    key = os.getenv("GEMINI_API_KEY")

    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing from .env"
        )

    return genai.Client(api_key=key)


def retry_api(operation, attempts=3):
    """Retry temporary API failures only."""
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as error:
            message = str(error).lower()

            temporary = any(
                code in message
                for code in [
                    "429", "500", "502", "503", "504",
                    "unavailable", "overloaded"
                ]
            )

            if not temporary or attempt == attempts - 1:
                raise

            time.sleep(2 ** attempt)


def words(text):
    return set(
        re.findall(r"[a-z0-9]+", str(text).lower())
    ) - STOP_WORDS


def result(answer, policy="None"):
    return {
        "answer": answer,
        "policy": policy,
        "tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "status": "Completed"
    }


# METHOD 1: RULES-BASED

def rules(question):
    df = policies()
    q = words(question)

    if not q:
        return result("No matching policy found.")

    candidates = []

    for _, row in df.iterrows():
        title = words(row["title"])
        body = words(row["policy_text"])

        title_matches = len(q & title)
        body_matches = len(q & body)

        score = 3 * title_matches + body_matches
        candidates.append((score, title_matches, row))

    candidates.sort(key=lambda x: x[0], reverse=True)

    score, title_matches, best = candidates[0]

    # Conservative keyword threshold.
    # This is a heuristic, not proof of relevance.
    if score < 3 or (
        title_matches == 0 and len(q) > 1
    ):
        return result(
            "No sufficiently relevant policy found."
        )

    return result(
        str(best["policy_text"]),
        str(best["title"])
    )


# GEMINI EMBEDDINGS

def policy_documents(df):
    return [
        (
            f"Title: {r['title']}\n"
            f"Policy: {r['policy_text']}"
        )
        for _, r in df.iterrows()
    ]


def embed(texts, task):
    vectors = []

    for start in range(0, len(texts), 20):
        batch = texts[start:start + 20]

        response = retry_api(
            lambda: client().models.embed_content(
                model=EMBED_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task
                )
            )
        )

        vectors.extend(
            item.values for item in response.embeddings
        )

    return np.asarray(vectors, dtype=np.float32)


def document_vectors(df):
    docs = policy_documents(df)

    fingerprint = hashlib.sha256(
        (EMBED_MODEL + "\n" + "\n".join(docs)).encode()
    ).hexdigest()

    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{fingerprint}.npy"

    if path.exists():
        return np.load(path, allow_pickle=False)

    vectors = embed(docs, "RETRIEVAL_DOCUMENT")
    np.save(path, vectors)

    return vectors


def vector_search(question, top_k=5):
    df = policies()
    vectors = document_vectors(df)

    query = embed(
        [question], "RETRIEVAL_QUERY"
    )[0]

    denominator = (
        np.linalg.norm(vectors, axis=1)
        * np.linalg.norm(query)
    )

    similarities = np.divide(
        vectors @ query,
        denominator,
        out=np.zeros(len(df)),
        where=denominator != 0
    )

    indices = np.argsort(similarities)[::-1][:top_k]

    return df.iloc[indices]


# METHOD 2 AND 3: LLM

def llm(question, indexed=False):
    df = policies()

    if indexed:
        selected = vector_search(question)
    else:
        selected = df

    context = "\n\n".join(
        f"POLICY TITLE: {r['title']}\n"
        f"POLICY TEXT: {r['policy_text']}"
        for _, r in selected.iterrows()
    )

    prompt = f"""
You are a company policy assistant.

Answer using ONLY the policies provided.

Rules:
- Do not invent policies or requirements.
- Do not use external knowledge.
- Cite the exact supporting policy title.
- If the policies do not answer the question,
  respond exactly:
  The policy database does not specify.
- Be concise and clear.

POLICIES:
{context}

QUESTION:
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

    answer = response.text or "No response generated."

    # Indexed policy field contains candidates,
    # not verified supporting citations.
    policy = (
        ", ".join(selected["title"].astype(str))
        if indexed
        else "See policy cited in answer"
    )

    if answer.strip().lower().startswith(
        "the policy database does not specify"
    ):
        policy = "None"

    return {
        "answer": answer,
        "policy": policy,
        "tokens": getattr(
            usage, "total_token_count", None
        ),
        "input_tokens": getattr(
            usage, "prompt_token_count", None
        ),
        "output_tokens": getattr(
            usage, "candidates_token_count", None
        ),
        "status": "Completed"
    }


# COMPARISON

def compare(question):
    methods = [
        ("Rules-based", lambda: rules(question)),
        (
            "LLM without index",
            lambda: llm(question, False)
        ),
        (
            "LLM with vector index",
            lambda: llm(question, True)
        )
    ]

    output = []

    for name, func in methods:
        start = time.perf_counter()

        try:
            data = func()
        except Exception as error:
            data = {
                "answer": f"Error: {error}",
                "policy": "—",
                "tokens": None,
                "input_tokens": None,
                "output_tokens": None,
                "status": "Error"
            }

        output.append({
            "Approach": name,
            "Answer": data["answer"],
            "Relevant policy": data["policy"],
            "Response time (s)": round(
                time.perf_counter() - start, 2
            ),
            "Total tokens": data.get("tokens"),
            "Input tokens": data.get("input_tokens"),
            "Output tokens": data.get("output_tokens"),
            "Status": data.get("status")
        })

    return output
