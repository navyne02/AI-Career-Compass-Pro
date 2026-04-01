# skill_extractor.py
import re
from typing import Iterable, List, Set

import pdfplumber
from transformers import pipeline
classifier = pipeline(
    "token-classification",
    model="jjzha/jobbert_skill_extraction",
    aggregation_strategy="simple",
)
_TECH_KEYWORDS: List[str] = [
    # Languages
    "python",
    "java",
    "javascript",
    "typescript",
    "c",
    "c++",
    "c#",
    # Web / frameworks
    "html",
    "css",
    "react",
    "angular",
    "node",
    "nodejs",
    "spring",
    "spring boot",
    "springboot",
    "asp.net",
    ".net",
    ".net core",
    "mvc",
    "bootstrap",
    "django",
    "flask",
    # Data / DB
    "sql",
    "t-sql",
    "plsql",
    "sql server",
    "mysql",
    "postgresql",
    "mongodb",
    "oracle",
    # Tools / cloud / ops
    "git",
    "github",
    "gitbash",
    "docker",
    "kubernetes",
    "linux",
    "aws",
    "azure",
    "gcp",
    "postman",
    "visual studio",
    "ssms",
    "sql server management studio",
]

_GENERIC_STOPWORDS: Set[str] = {
    "user",
    "users",
    "project",
    "projects",
    "college",
    "school",
    "place",
    "home",
    "education",
    "skills",
    "objective",
    "profile",
    "developer",
    "working",
    "application",
    "system",
    "design",
    "testing",
    "requests",
    "responsibilities",
    "modules",
    "data",
    "details",
    "name",
    "email",
    "mobile",
}


def extract_text_from_pdf(pdf_file) -> str:
    """Extract raw text from a PDF (Streamlit upload or file path)."""
    text_parts: List[str] = []
    try:
        if hasattr(pdf_file, "seek"):
            pdf_file.seek(0)

        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(page_text)
    except Exception as e:
        print(f"PDF error: {e}")
        return ""

    return "\n".join(text_parts).strip()


def _clean_span(s: str) -> str:
    s = (s or "").replace("Ġ", " ").replace("##", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _chunk_text(text: str, max_chars: int = 1800, overlap: int = 250) -> Iterable[str]:
    """Simple char-based chunking to avoid max-length issues."""
    t = text or ""
    if len(t) <= max_chars:
        yield t
        return
    start = 0
    while start < len(t):
        end = min(len(t), start + max_chars)
        yield t[start:end]
        if end >= len(t):
            break
        start = max(0, end - overlap)


def extract_skills_from_pdf(pdf_file) -> List[str]:
    text = extract_text_from_pdf(pdf_file)
    if not text:
        print("No text extracted from PDF")
        return []

    text_lower = text.lower()
    print(f"Extracted {len(text)} chars text preview: {text_lower[:200]}...")

    # 1) Model-based extraction on full resume text (chunked)
    skills: Set[str] = set()
    model_hits = 0

    for chunk in _chunk_text(text_lower):
        try:
            entities = classifier(chunk)
        except Exception as e:
            print(f"Classifier error: {e}")
            continue

        for ent in entities:
            label = (ent.get("entity_group") or ent.get("entity") or "").upper().strip()
            score = float(ent.get("score", 0.0) or 0.0)
            word = _clean_span(ent.get("word", ""))

            # This model often uses B/I/O. Treat B and I as "skill" tokens/spans.
            is_skill_label = ("SKILL" in label) or (label in {"B", "I", "B-SKILL", "I-SKILL"})
            if not is_skill_label or score < 0.34:
                continue

            w = word.lower().strip(" -–—•·\t\r\n")
            if not w or len(w) < 2 or len(w) > 60:
                continue
            if w in _GENERIC_STOPWORDS:
                continue
            if not re.search(r"[a-z]", w):
                continue

            skills.add(w)
            model_hits += 1

    # 2) Keyword fallback (guarantees core tech skills get detected)
    for kw in _TECH_KEYWORDS:
        if kw in text_lower:
            skills.add(kw)

    skills_list = sorted(skills)
    print(f"Model hits kept: {model_hits}")
    print(f"Found {len(skills_list)} skills: {skills_list[:40]}")
    return skills_list