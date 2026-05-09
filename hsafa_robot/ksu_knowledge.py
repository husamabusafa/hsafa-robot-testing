"""ksu_knowledge.py — Unified knowledge base from ksu_data/ JSON documents.

Loads all .json files from ksu_data/, extracts text chunks with metadata,
and provides keyword search to answer questions about King Saud University.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import List, Dict
import unicodedata

# Resolve ksu_data directory (repo-relative)
_KSU_DATA_DIR = Path(__file__).parent.parent / "ksu_data"


def _normalize_arabic(text: str) -> str:
    """Light Arabic normalization: remove tashkeel, unify alef/hamza forms."""
    text = unicodedata.normalize("NFKC", text)
    # Remove tashkeel (diacritics)
    text = re.sub(r"[\u064B-\u065F\u0670\u0640]", "", text)
    # Normalize alef variants
    text = text.replace("\u0623", "\u0627").replace("\u0625", "\u0627").replace("\u0622", "\u0627")
    # Normalize hamza on yeh/waw
    text = text.replace("\u0624", "\u0648").replace("\u0626", "\u064a")
    return text


def _extract_chunks(data: dict, doc_name: str) -> List[dict]:
    """Extract text chunks from a single JSON document."""
    chunks = []
    meta = data.get("document_metadata") or {}
    title = meta.get("document_title", doc_name)

    for page in data.get("pages", []):
        page_num = page.get("page_number", 0)
        for elem in page.get("elements", []):
            text = elem.get("text_raw") or elem.get("text", "")
            if not text:
                continue
            chunks.append({
                "document": title,
                "doc_file": doc_name,
                "page": page_num,
                "type": elem.get("type", "text"),
                "text": text,
                "source_pages": elem.get("source_pages", [page_num]),
            })
    return chunks


def _tokenize(text: str) -> set:
    """Tokenize text into normalized keywords."""
    text = _normalize_arabic(text.lower())
    # Keep Arabic letters, English letters, and numbers
    tokens = re.findall(r"[\u0600-\u06FF]+|[a-zA-Z0-9]+", text)
    # Filter very short tokens
    return {t for t in tokens if len(t) >= 2}


class KSUKnowledgeBase:
    """In-memory searchable knowledge base from ksu_data/ documents."""

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else _KSU_DATA_DIR
        self.chunks: List[dict] = []
        self._index: Dict[str, set] = {}  # token -> set of chunk indices
        self._loaded = False

    def load(self) -> "KSUKnowledgeBase":
        """Scan ksu_data/ and build the index. Safe to call multiple times."""
        if self._loaded:
            return self

        if not self.data_dir.exists():
            raise FileNotFoundError(f"ksu_data directory not found: {self.data_dir}")

        for f in sorted(self.data_dir.glob("*.json")):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                doc_chunks = _extract_chunks(data, f.name)
                self.chunks.extend(doc_chunks)
            except Exception as exc:
                print(f"[ksu_knowledge] warning: failed to load {f}: {exc}")

        # Build inverted index
        for idx, chunk in enumerate(self.chunks):
            tokens = _tokenize(chunk["text"])
            for tok in tokens:
                self._index.setdefault(tok, set()).add(idx)

        self._loaded = True
        print(f"[ksu_knowledge] Loaded {len(self.chunks)} chunks from {len(list(self.data_dir.glob('*.json')))} files.")
        return self

    def search(self, query: str, top_k: int = 10) -> List[dict]:
        """Return the most relevant chunks for a natural-language query."""
        if not self._loaded:
            self.load()

        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        scores: Dict[int, float] = {}
        for tok in q_tokens:
            for idx in self._index.get(tok, set()):
                scores[idx] = scores.get(idx, 0.0) + 1.0

        # Boost exact phrase match
        query_norm = _normalize_arabic(query.lower())
        for idx, chunk in enumerate(self.chunks):
            chunk_norm = _normalize_arabic(chunk["text"].lower())
            if query_norm in chunk_norm:
                scores[idx] = scores.get(idx, 0.0) + 5.0

        # Sort by score desc
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [self.chunks[i] for i, _ in ranked[:top_k]]

    def get_stats(self) -> dict:
        """Quick stats about the loaded knowledge base."""
        if not self._loaded:
            self.load()
        docs = {c["doc_file"] for c in self.chunks}
        return {
            "chunks": len(self.chunks),
            "documents": len(docs),
            "unique_tokens": len(self._index),
        }

    def get_document_list(self) -> List[str]:
        """Return list of document titles."""
        if not self._loaded:
            self.load()
        seen = set()
        result = []
        for c in self.chunks:
            t = c["document"]
            if t not in seen:
                seen.add(t)
                result.append(t)
        return result


# Global singleton for reuse across the process
_knowledge: KSUKnowledgeBase | None = None


def get_knowledge() -> KSUKnowledgeBase:
    """Lazy-load the global knowledge base."""
    global _knowledge
    if _knowledge is None:
        _knowledge = KSUKnowledgeBase().load()
    return _knowledge


def query_ksu_knowledge(question: str, top_k: int = 10) -> dict:
    """Public API: answer a KSU question from the knowledge base.

    Returns structured JSON with `results` (relevant chunks),
    `answer_summary`, and `sources`.
    """
    kb = get_knowledge()
    results = kb.search(question, top_k=top_k)

    if not results:
        return {
            "found": False,
            "message": "لم أجد إجابة في الوثائق المتاحة. جرب صياغة السؤال بطريقة أخرى.",
            "results": [],
            "sources": [],
        }

    # De-duplicate sources
    sources = []
    seen = set()
    for r in results:
        key = (r["document"], r["page"])
        if key not in seen:
            seen.add(key)
            sources.append(f"{r['document']} — صفحة {r['page']}")

    return {
        "found": True,
        "results": [
            {
                "text": r["text"],
                "document": r["document"],
                "page": r["page"],
            }
            for r in results
        ],
        "sources": sources,
    }


# ---------------------------------------------------------------------------
# Postgres faculty search
# ---------------------------------------------------------------------------
import os

try:
    import psycopg2
    _HAS_PSYCOPG2 = True
except ImportError:
    psycopg2 = None  # type: ignore
    _HAS_PSYCOPG2 = False


def _get_pg_conn():
    """Return a psycopg2 connection to ksu_faculty DB."""
    if not _HAS_PSYCOPG2:
        raise RuntimeError("psycopg2 not installed")
    return psycopg2.connect(
        dbname="ksu_faculty",
        user=os.environ.get("USER", "postgres"),
    )


def search_ksu_faculty(query: str, limit: int = 10) -> dict:
    """Search the KSU faculty Postgres DB by name, degree, job title, or email.

    Returns structured JSON with faculty matches.
    """
    if not _HAS_PSYCOPG2:
        return {
            "found": False,
            "error": "psycopg2 not installed",
            "results": [],
        }

    conn = _get_pg_conn()
    cur = conn.cursor()

    # Normalize query for LIKE matching
    like_q = f"%{query}%"

    cur.execute(
        """
        SELECT name, academic_degree, job_title, email, phone, profile_url, image_url
        FROM faculty
        WHERE name ILIKE %s
           OR academic_degree ILIKE %s
           OR job_title ILIKE %s
           OR email ILIKE %s
        ORDER BY
            CASE
                WHEN name ILIKE %s THEN 1
                WHEN email ILIKE %s THEN 2
                ELSE 3
            END,
            name
        LIMIT %s
        """,
        (like_q, like_q, like_q, like_q, f"%{query}%", f"%{query}%", limit),
    )

    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        return {
            "found": False,
            "message": f"لم أجد أي عضو هيئة تدريس مطابق لـ \"{query}\" في قاعدة بيانات KSU.",
            "results": [],
        }

    results = []
    for r in rows:
        results.append({
            "name": r[0],
            "academic_degree": r[1],
            "job_title": r[2],
            "email": r[3],
            "phone": r[4],
            "profile_url": r[5],
            "image_url": r[6],
        })

    return {
        "found": True,
        "count": len(results),
        "results": results,
    }


if __name__ == "__main__":
    # Quick smoke test
    kb = get_knowledge()
    print("Stats:", kb.get_stats())
    print("Documents:", kb.get_document_list())
    print()
    res = query_ksu_knowledge("ما هي أنظمة السنة الأولى المشتركة")
    print("Query results:", len(res["results"]))
    for r in res["results"][:3]:
        print(f"  [{r['document']} p{r['page']}]", r["text"][:120])

    print("\n--- Faculty search test ---")
    fres = search_ksu_faculty("Mohamed Hadj")
    print("Found:", fres["found"], "count:", fres.get("count", 0))
    for f in fres.get("results", [])[:3]:
        print(f"  {f['name']} | {f['email']} | {f['profile_url']}")
