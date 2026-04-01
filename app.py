import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import re
import os
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
import spacy
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from skill_extractor import extract_skills_from_pdf

try:
    import joblib
    from textblob import TextBlob
    from xgboost import XGBClassifier  # noqa: F401

    HAS_FAKE_JOB_DEPS = True
except ImportError:
    joblib = None
    TextBlob = None
    XGBClassifier = None
    HAS_FAKE_JOB_DEPS = False
def load_css(file_name):
    with open(file_name , encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css("style.css")

# Optional: Sentence-BERT for semantic matching (requires ~500MB download on first run)
USE_SEMANTIC = False
model = None
cosine_similarity = None
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity as _cosine_sim

    cosine_similarity = _cosine_sim
    model = SentenceTransformer("all-MiniLM-L6-v2")
    USE_SEMANTIC = True
except ImportError:
    pass

# ── Page Config ──
st.set_page_config(
    page_title="AI Career Compass Pro",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──
with st.sidebar:
    st.markdown(
        """
    <div style="text-align:center; padding: 1.5rem 0 1.5rem 0;">
        <div style="font-size: 2.8rem; margin-bottom: 0.6rem; filter: drop-shadow(0 0 12px rgba(0,245,255,0.5));">🧭</div>
        <div style="font-size: 1.25rem; font-weight: 700; letter-spacing: -0.01em;
            background: linear-gradient(135deg, #00f5ff, #bf5fff);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
            Career Compass
        </div>
        <div style="font-size: 0.72rem; color: #4a5568; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; margin-top: 4px;">
            AI-Powered Guidance
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown(
        """
    <div style="padding: 0 0.5rem;">
        <p style="font-size: 0.85rem; font-weight: 600; color: #e2e8f0; margin-bottom: 0.75rem;">How it works</p>
        <div style="display: flex; align-items: flex-start; gap: 0.75rem; margin-bottom: 1rem;">
            <div style="min-width: 28px; height: 28px; background: linear-gradient(135deg, #00f5ff, #bf5fff); border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 700; color: #060812;">1</div>
            <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.5;">Upload your resume in PDF format</div>
        </div>
        <div style="display: flex; align-items: flex-start; gap: 0.75rem; margin-bottom: 1rem;">
            <div style="min-width: 28px; height: 28px; background: linear-gradient(135deg, #00f5ff, #bf5fff); border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 700; color: #060812;">2</div>
            <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.5;">AI extracts your skills &amp; matches jobs</div>
        </div>
        <div style="display: flex; align-items: flex-start; gap: 0.75rem; margin-bottom: 1rem;">
            <div style="min-width: 28px; height: 28px; background: linear-gradient(135deg, #00f5ff, #bf5fff); border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 700; color: #060812;">3</div>
            <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.5;">Get personalized job &amp; course recommendations</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    if not USE_SEMANTIC:
        st.warning(
            "**Skill-only mode** — Install `sentence-transformers` for full AI semantic matching.",
            icon="⚠️",
        )
    st.markdown(
        """
    <div style="padding: 0 0.5rem;">
        <p style="font-size: 0.85rem; font-weight: 600; color: #e2e8f0; margin-bottom: 0.5rem;">Powered by</p>
        <div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.8;">
            &#x2022; Sentence-BERT (NLP)<br>
            &#x2022; Random Forest Classifier<br>
            &#x2022; Cosine Similarity Matching<br>
            &#x2022; 1,000+ Job Listings<br>
            &#x2022; 280+ Coursera Courses
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ── Data Loading ──
@st.cache_data
def load_data():
    jobs = pd.read_csv("data/jobs.csv")
    courses = pd.read_csv("data/webautomation_coursera.csv")
    return jobs, courses


jobs_df, courses_df = load_data()


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ── Prepare Job Data ──
jobs_df["full_text"] = (
    jobs_df["Responsibilities"].fillna("")
    + " "
    + jobs_df["Keywords"].fillna("")
    + " "
    + jobs_df["Skills"].fillna("")
)
jobs_df["clean_description"] = jobs_df["full_text"].apply(preprocess_text)
jobs_df_clean = jobs_df.dropna(subset=["job_title", "ExperienceLevel"]).drop_duplicates(
    subset=["job_title", "ExperienceLevel"]
).copy()

# ── Load SBERT Model (or prepare for skill-only mode) ──
job_embeddings = None
if USE_SEMANTIC and model is not None:
    job_embeddings = model.encode(
        jobs_df_clean["clean_description"].tolist(),
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def compute_text_overlap(text_a, text_b):
    """Fallback: simple word-overlap score when SBERT unavailable."""
    words_a = set(re.findall(r"\b[a-z0-9]+\b", text_a.lower()))
    words_b = set(re.findall(r"\b[a-z0-9]+\b", text_b.lower()))
    if not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_b)


def clean_split_skills(text):
    parts = re.split(r"[;,/]", str(text).lower())
    return [p.strip() for p in parts if p.strip()]


# ── Skill → External Resource Mapping (YouTube, docs, etc.) ──
def normalize_skill_name(skill: str) -> str:
    s = str(skill).strip().lower()
    # Remove common suffixes
    s = re.sub(r"\s+(basics?|fundamentals?)$", "", s)
    # Normalizations / aliases
    if s in {"js"}:
        return "javascript"
    if s in {"node.js", "node js"}:
        return "nodejs"
    if s in {"sql server", "t-sql"}:
        return "sql"
    if s in {"powerbi", "power-bi"}:
        return "power bi"
    if s in {"ml"}:
        return "machine learning"
    if s in {"dl"}:
        return "deep learning"
    if s in {"c sharp", "c#"}:
        return "c#"
    if s in {".net", "dotnet", ".net core", ".net framework"}:
        return ".net"
    return s


SKILL_RESOURCES = {
    "python": [
        {
            "title": "Python full course for beginners (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=python+full+course+for+beginners",
            "provider": "YouTube",
        },
        {
            "title": "Official Python tutorial",
            "url": "https://docs.python.org/3/tutorial/",
            "provider": "python.org",
        },
    ],
    "java": [
        {
            "title": "Java programming for beginners (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=java+programming+full+course+for+beginners",
            "provider": "YouTube",
        }
    ],
    "c": [
        {
            "title": "C language beginner course (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=c+programming+full+course",
            "provider": "YouTube",
        }
    ],
    "c++": [
        {
            "title": "C++ programming full course (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=c%2B%2B+programming+full+course",
            "provider": "YouTube",
        }
    ],
    "c#": [
        {
            "title": "C# and .NET for beginners (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=c%23+dotnet+full+course+for+beginners",
            "provider": "YouTube",
        }
    ],
    ".net": [
        {
            "title": ".NET / ASP.NET beginner tutorials (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=asp.net+core+full+course",
            "provider": "YouTube",
        }
    ],
    "javascript": [
        {
            "title": "JavaScript full course (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=javascript+full+course+for+beginners",
            "provider": "YouTube",
        }
    ],
    "html": [
        {
            "title": "HTML & CSS crash course (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=html+css+full+course",
            "provider": "YouTube",
        }
    ],
    "css": [
        {
            "title": "Modern CSS layouts & flexbox (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=css+flexbox+grid+tutorial",
            "provider": "YouTube",
        }
    ],
    "react": [
        {
            "title": "React JS full course (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=react+js+full+course+for+beginners",
            "provider": "YouTube",
        },
        {
            "title": "React official docs",
            "url": "https://react.dev/learn",
            "provider": "react.dev",
        },
    ],
    "nodejs": [
        {
            "title": "Node.js API & backend course (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=node+js+api+rest+full+course",
            "provider": "YouTube",
        }
    ],
    "springboot": [
        {
            "title": "Spring Boot REST API course (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=spring+boot+rest+api+full+course",
            "provider": "YouTube",
        }
    ],
    "flask": [
        {
            "title": "Flask web app tutorial (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=flask+web+app+tutorial",
            "provider": "YouTube",
        }
    ],
    "django": [
        {
            "title": "Django full stack course (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=django+full+stack+course",
            "provider": "YouTube",
        }
    ],
    "sql": [
        {
            "title": "SQL for data analysis (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=sql+for+beginners+full+course",
            "provider": "YouTube",
        }
    ],
    "mysql": [
        {
            "title": "MySQL database tutorial (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=mysql+database+tutorial+for+beginners",
            "provider": "YouTube",
        }
    ],
    "postgresql": [
        {
            "title": "PostgreSQL tutorial (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=postgresql+tutorial+for+beginners",
            "provider": "YouTube",
        }
    ],
    "mongodb": [
        {
            "title": "MongoDB for beginners (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=mongodb+for+beginners",
            "provider": "YouTube",
        }
    ],
    "machine learning": [
        {
            "title": "Machine learning full course (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=machine+learning+full+course",
            "provider": "YouTube",
        }
    ],
    "deep learning": [
        {
            "title": "Deep learning tutorial (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=deep+learning+neural+networks+full+course",
            "provider": "YouTube",
        }
    ],
    "nlp": [
        {
            "title": "NLP with Python course (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=nlp+with+python+tutorial",
            "provider": "YouTube",
        }
    ],
    "data science": [
        {
            "title": "Data science roadmap & course (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=data+science+full+course",
            "provider": "YouTube",
        }
    ],
    "pandas": [
        {
            "title": "Pandas tutorial (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=pandas+python+tutorial",
            "provider": "YouTube",
        }
    ],
    "numpy": [
        {
            "title": "NumPy tutorial (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=numpy+python+tutorial",
            "provider": "YouTube",
        }
    ],
    "scikit learn": [
        {
            "title": "scikit-learn ML course (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=scikit+learn+tutorial",
            "provider": "YouTube",
        }
    ],
    "tensorflow": [
        {
            "title": "TensorFlow 2 tutorial (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=tensorflow+2+tutorial",
            "provider": "YouTube",
        }
    ],
    "keras": [
        {
            "title": "Keras deep learning tutorial (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=keras+deep+learning+tutorial",
            "provider": "YouTube",
        }
    ],
    "pytorch": [
        {
            "title": "PyTorch for deep learning (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=pytorch+for+deep+learning+full+course",
            "provider": "YouTube",
        }
    ],
    "android": [
        {
            "title": "Android app development with Kotlin (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=android+app+development+with+kotlin+full+course",
            "provider": "YouTube",
        }
    ],
    "kotlin": [
        {
            "title": "Kotlin for Android beginners (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=kotlin+android+tutorial",
            "provider": "YouTube",
        }
    ],
    "firebase": [
        {
            "title": "Firebase for web/mobile apps (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=firebase+tutorial",
            "provider": "YouTube",
        }
    ],
    "aws": [
        {
            "title": "AWS cloud practitioner / developer course (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=aws+cloud+practitioner+full+course",
            "provider": "YouTube",
        }
    ],
    "azure": [
        {
            "title": "Microsoft Azure fundamentals (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=azure+fundamentals+full+course",
            "provider": "YouTube",
        }
    ],
    "gcp": [
        {
            "title": "Google Cloud Platform (GCP) basics (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=google+cloud+platform+for+beginners",
            "provider": "YouTube",
        }
    ],
    "docker": [
        {
            "title": "Docker containers & images (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=docker+for+beginners+full+course",
            "provider": "YouTube",
        }
    ],
    "kubernetes": [
        {
            "title": "Kubernetes for developers (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=kubernetes+for+beginners+full+course",
            "provider": "YouTube",
        }
    ],
    "git": [
        {
            "title": "Git & GitHub crash course (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=git+and+github+crash+course",
            "provider": "YouTube",
        }
    ],
    "github": [
        {
            "title": "GitHub basics for developers (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=github+tutorial+for+beginners",
            "provider": "YouTube",
        }
    ],
    "linux": [
        {
            "title": "Linux command line basics (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=linux+command+line+for+beginners",
            "provider": "YouTube",
        }
    ],
    "excel": [
        {
            "title": "Excel for data analysis (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=excel+for+data+analysis+full+course",
            "provider": "YouTube",
        }
    ],
    "power bi": [
        {
            "title": "Power BI beginner to pro (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=power+bi+full+course",
            "provider": "YouTube",
        }
    ],
    "tableau": [
        {
            "title": "Tableau data visualization course (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=tableau+data+visualization+tutorial",
            "provider": "YouTube",
        }
    ],
}


def get_resources_for_skill(skill: str):
    key = normalize_skill_name(skill)
    return SKILL_RESOURCES.get(key, [])


# ── Role-based learning paths (job-title aware) ──
ROLE_RESOURCES = {
    ".net_developer": [
        {
            "title": "🧭 .NET Developer Roadmap (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=.net+developer+roadmap",
            "provider": "YouTube",
        },
        {
            "title": "C# and ASP.NET Core full course (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=c%23+asp.net+core+full+course",
            "provider": "YouTube",
        },
    ],
    "data_scientist": [
        {
            "title": "🧭 Data Scientist Roadmap (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=data+scientist+roadmap",
            "provider": "YouTube",
        },
        {
            "title": "End-to-end data science project tutorials",
            "url": "https://www.youtube.com/results?search_query=data+science+project+end+to+end",
            "provider": "YouTube",
        },
    ],
    "ml_engineer": [
        {
            "title": "🧭 Machine Learning Engineer Roadmap (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=machine+learning+engineer+roadmap",
            "provider": "YouTube",
        },
        {
            "title": "Deploying ML models to production",
            "url": "https://www.youtube.com/results?search_query=deploy+machine+learning+models+to+production",
            "provider": "YouTube",
        },
    ],
    "full_stack": [
        {
            "title": "🧭 Full-Stack Developer Roadmap (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=full+stack+developer+roadmap",
            "provider": "YouTube",
        },
        {
            "title": "MERN / full-stack web app tutorial",
            "url": "https://www.youtube.com/results?search_query=full+stack+web+app+project+mern",
            "provider": "YouTube",
        },
    ],
    "frontend": [
        {
            "title": "🧭 Frontend Developer Roadmap (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=frontend+developer+roadmap",
            "provider": "YouTube",
        },
        {
            "title": "React + modern frontend course",
            "url": "https://www.youtube.com/results?search_query=react+js+frontend+developer+course",
            "provider": "YouTube",
        },
    ],
    "backend": [
        {
            "title": "🧭 Backend Developer Roadmap (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=backend+developer+roadmap",
            "provider": "YouTube",
        },
        {
            "title": "REST API design and best practices",
            "url": "https://www.youtube.com/results?search_query=rest+api+design+best+practices",
            "provider": "YouTube",
        },
    ],
    "android": [
        {
            "title": "🧭 Android Developer Roadmap (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=android+developer+roadmap",
            "provider": "YouTube",
        },
        {
            "title": "Android app with Kotlin full course",
            "url": "https://www.youtube.com/results?search_query=android+app+development+with+kotlin+full+course",
            "provider": "YouTube",
        },
    ],
    "data_analyst": [
        {
            "title": "🧭 Data Analyst Roadmap (YouTube search)",
            "url": "https://www.youtube.com/results?search_query=data+analyst+roadmap",
            "provider": "YouTube",
        },
        {
            "title": "Excel, SQL, and Power BI for data analysis",
            "url": "https://www.youtube.com/results?search_query=excel+sql+power+bi+data+analysis+course",
            "provider": "YouTube",
        },
    ],
}


def get_role_key(job_title: str) -> str:
    t = str(job_title).lower()
    if ".net" in t or "dotnet" in t or "c#" in t:
        return ".net_developer"
    if "data scientist" in t or "data science" in t:
        return "data_scientist"
    if "machine learning engineer" in t or "ml engineer" in t:
        return "ml_engineer"
    if "full stack" in t or "full-stack" in t:
        return "full_stack"
    if "frontend" in t or "front-end" in t or "ui developer" in t:
        return "frontend"
    if "backend" in t or "back-end" in t:
        return "backend"
    if "android" in t or "mobile developer" in t:
        return "android"
    if "data analyst" in t or ("business analyst" in t and "data" in t):
        return "data_analyst"
    return ""


def get_role_resources(job_title: str):
    key = get_role_key(job_title)
    if not key:
        return []
    return ROLE_RESOURCES.get(key, [])


# ── Fake job detection helpers ──
FAKE_MODEL_PATH = "models/fake_job_xgb.pkl"

_FAKE_FRAUD_WORDS = [
    "registration fee",
    "processing fee",
    "urgent hiring",
    "no experience",
    "whatsapp",
    "telegram",
    "easy money",
    "work from home",
    "limited slots",
    "guaranteed income",
    "earn per day",
    "earn per week",
    "instant joining",
]


def _extract_linguistic_features_for_job(text: str):
    text = str(text or "").lower()
    fraud_count = sum(text.count(w) for w in _FAKE_FRAUD_WORDS)
    length = len(text.split())

    if TextBlob is None:
        return [
            fraud_count * 10,
            length / 100,
            0.0,
            0.0,
        ]

    blob = TextBlob(text)
    return [
        fraud_count * 10,
        length / 100,
        blob.sentiment.polarity * 5,
        blob.sentiment.subjectivity * 5,
    ]


def _build_fake_feature_vector(description: str):
    """Build feature vector matching the training pipeline in fakejob.py."""
    if not USE_SEMANTIC or model is None:
        return None
    emb = model.encode([description], convert_to_numpy=True)[0]
    ling_feats = _extract_linguistic_features_for_job(description)
    return np.concatenate([emb, ling_feats])


@st.cache_resource
def load_fake_job_model():
    if not HAS_FAKE_JOB_DEPS:
        return None
    if not os.path.exists(FAKE_MODEL_PATH):
        return None
    try:
        clf = joblib.load(FAKE_MODEL_PATH)
    except Exception:
        return None
    return clf


def predict_fake_probability(description: str):
    """Return probability that a job posting is fraudulent (1 = fake, 0 = real)."""
    vec = _build_fake_feature_vector(description)
    if vec is None:
        return None
    clf = load_fake_job_model()
    if clf is None:
        return None
    try:
        proba = clf.predict_proba(vec.reshape(1, -1))[0, 1]
    except Exception:
        return None
    return float(proba)


# ── Hero Section ──
st.markdown(
    """
<div class="hero-container">
    <div class="hero-title">AI Career Compass Pro</div>
    <div class="hero-subtitle">
        Upload your resume and let AI find your perfect career match.
        Get personalized job recommendations, skill gap analysis, and curated courses.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Upload Section ──
col_pad_l, col_upload, col_pad_r = st.columns([1, 2, 1])
with col_upload:
    uploaded_file = st.file_uploader(
        "Upload your resume (PDF)",
        type=["pdf"],
        label_visibility="collapsed",
    )
    st.markdown(
        '<p style="text-align:center; color:#64748b; font-size:0.82rem; margin-top:0.5rem;">'
        "Drag & drop or click to upload your PDF resume</p>",
        unsafe_allow_html=True,
    )

if uploaded_file is None:
    st.markdown("---")
    feat_cols = st.columns(3)
    features = [
        (
            "🔍",
            "Smart Matching",
            "Semantic AI + skill-based matching for accurate job recommendations",
        ),
        (
            "📊",
            "Gap Analysis",
            "Identify missing skills between your profile and dream roles",
        ),
        (
            "🎓",
            "Course Paths",
            "Get curated Coursera courses to bridge your skill gaps",
        ),
    ]
    for col, (icon, title, desc) in zip(feat_cols, features):
        with col:
            st.markdown(
                f"""
            <div class="stats-card" style="min-height: 160px; text-align: center; display: flex; flex-direction: column; align-items: center; justify-content: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
                <div style="font-size: 1rem; font-weight: 700; color: #e2e8f0; margin-bottom: 0.5rem;">{title}</div>
                <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.5;">{desc}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
    st.stop()

# ── Process Resume ──
with st.spinner("Analyzing your resume..."):
    with pdfplumber.open(uploaded_file) as pdf:
        resume_text = ""
        for page in pdf.pages:
            resume_text += page.extract_text() or ""

    resume_text_raw = resume_text
    resume_text = preprocess_text(resume_text)

    if len(resume_text.strip()) == 0:
        st.error(
            "Could not extract readable text from the PDF. Please try a different file."
        )
        st.stop()

    # Reset file so skill extractor can read from the beginning
    uploaded_file.seek(0)
    resume_skills = extract_skills_from_pdf(uploaded_file)

    if USE_SEMANTIC and model is not None and job_embeddings is not None:
        resume_embedding = model.encode([resume_text], convert_to_numpy=True)
        semantic_scores = cosine_similarity(resume_embedding, job_embeddings)[0]
        jobs_df_clean["semantic_score"] = np.clip(semantic_scores, 0, 1)
    else:
        jobs_df_clean["semantic_score"] = jobs_df_clean[
            "clean_description"
        ].apply(lambda jd: compute_text_overlap(resume_text, jd))

    def compute_skill_score(job_skills):
        job_skill_list = clean_split_skills(job_skills)
        if len(job_skill_list) == 0:
            return 0
        common = set(resume_skills).intersection(set(job_skill_list))
        return len(common) / len(job_skill_list)

    jobs_df_clean["skill_score"] = jobs_df_clean["Skills"].apply(compute_skill_score)

    rel_sem, rel_skill = 0.35, 0.2
    if not USE_SEMANTIC:
        rel_sem, rel_skill = 0.1, 0.15  # Relaxed for text-overlap fallback
    jobs_df_clean["relevant"] = (
        (jobs_df_clean["semantic_score"] > rel_sem)
        | (jobs_df_clean["skill_score"] > rel_skill)
    ).astype(int)

    X = jobs_df_clean[["semantic_score", "skill_score"]]
    y = jobs_df_clean["relevant"]

    rf = RandomForestClassifier(n_estimators=120, random_state=42)

    if len(y.unique()) > 1:
        rf.fit(X, y)
        jobs_df_clean["rf_score"] = rf.predict_proba(X)[:, 1]
    else:
        jobs_df_clean["rf_score"] = jobs_df_clean["semantic_score"]

    top_k = jobs_df_clean.sort_values(by="rf_score", ascending=False).head(3).copy()

    def skill_gap(job_skills):
        job_skill_list = clean_split_skills(job_skills)
        return sorted(list(set(job_skill_list) - set(resume_skills)))

    top_k["missing_skills"] = top_k["Skills"].apply(skill_gap)


def generate_resume_suggestions(resume_text, resume_skills, top_k):
    """Generate structured, rule-based resume improvement suggestions."""
    suggestions = {
        "missing_skills": [],
        "content": [],
        "formatting": [],
    }

    text_lower = str(resume_text).lower()

    # A) Missing Technical Skills
    all_missing = set()
    for _, row in top_k.iterrows():
        for skill in row.get("missing_skills", []):
            if skill:
                all_missing.add(skill)

    if all_missing:
        pretty_skills = ", ".join(sorted({s.title() for s in all_missing}))
        suggestions["missing_skills"].append(
            f"Highlight or start learning these in-demand skills that appear in your top matching roles: {pretty_skills}."
        )
    else:
        suggestions["missing_skills"].append(
            "Your resume already covers most of the key technical skills required for the recommended roles. Consider keeping them clearly grouped in a dedicated Skills section."
        )

    # B) Resume Content Improvements
    word_count = len(re.findall(r"\w+", str(resume_text)))
    if word_count < 300:
        suggestions["content"].append(
            "Your resume appears relatively short. Consider adding more detail about projects, responsibilities, and achievements (aim for at least 300–500 words)."
        )

    if "project" not in text_lower:
        suggestions["content"].append(
            "Add a dedicated 'Projects' section showcasing 2–4 key projects with technologies used, your role, and impact."
        )

    action_verbs = [
        "achieved",
        "developed",
        "built",
        "implemented",
        "designed",
        "created",
        "improved",
        "led",
        "optimized",
        "automated",
    ]
    if not any(verb in text_lower for verb in action_verbs):
        suggestions["content"].append(
            "Use strong action verbs (e.g., 'developed', 'built', 'implemented', 'optimized') at the start of bullet points to make your experience more impactful."
        )

    if not re.search(r"\d|%", str(resume_text)):
        suggestions["content"].append(
            "Include measurable results where possible (e.g., 'improved accuracy by 15%', 'reduced processing time by 30%', 'handled 50+ customer queries per day')."
        )

    # C) Formatting Suggestions (always helpful)
    suggestions["formatting"].extend(
        [
            "Use clear section headings such as 'Summary', 'Skills', 'Experience', 'Projects', and 'Education' to make the resume easy to scan.",
            "Prefer concise bullet points instead of long paragraphs so recruiters and ATS can quickly parse your experience.",
            "Group technical skills by category (e.g., Programming Languages, Frameworks, Databases, Cloud, Tools) for better readability.",
            "Keep the layout ATS-friendly: avoid text inside tables or images, use simple fonts, and ensure consistent alignment and spacing.",
        ]
    )

    return suggestions


# ── ATS Compatibility Analyzer ──
_ATS_STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'shall', 'can', 'need', 'must',
    'we', 'you', 'he', 'she', 'it', 'they', 'i', 'me', 'him', 'her',
    'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their',
    'this', 'that', 'these', 'those', 'what', 'which', 'who', 'whom',
    'not', 'no', 'nor', 'as', 'if', 'then', 'than', 'too', 'very',
    'just', 'about', 'above', 'after', 'again', 'all', 'also', 'any',
    'because', 'before', 'between', 'both', 'each', 'few', 'more',
    'most', 'other', 'over', 'same', 'so', 'some', 'such', 'through',
    'under', 'until', 'up', 'while', 'into', 'out', 'during', 'how',
    'when', 'where', 'why', 'able', 'etc', 'including', 'well',
    'looking', 'role', 'position', 'job', 'company', 'team', 'work',
    'working', 'within', 'across', 'using', 'used', 'use', 'new',
    'good', 'great', 'strong', 'excellent', 'preferred', 'required',
    'requirements', 'responsibilities', 'qualifications', 'experience',
    'years', 'year', 'minimum', 'plus', 'knowledge', 'skills',
    'ability', 'understanding', 'environment', 'opportunity',
}

_ATS_MULTI_WORD_TERMS = [
    'machine learning', 'deep learning', 'data science',
    'natural language processing', 'computer vision',
    'project management', 'version control', 'ci cd',
    'continuous integration', 'continuous deployment',
    'data analysis', 'data engineering', 'web development',
    'mobile development', 'cloud computing', 'big data',
    'artificial intelligence', 'software development',
    'agile methodology', 'scrum master', 'product management',
    'user experience', 'user interface', 'full stack',
    'front end', 'back end', 'rest api', 'unit testing',
    'test driven', 'object oriented', 'problem solving',
    'software engineering', 'system design', 'microservices',
    'distributed systems', 'data structures', 'design patterns',
]

_ATS_POWER_VERBS = [
    'achieved', 'implemented', 'developed', 'managed', 'led',
    'optimized', 'designed', 'built', 'created', 'improved',
    'analyzed', 'collaborated', 'delivered', 'reduced', 'increased',
    'automated', 'architected', 'streamlined', 'mentored', 'launched',
]

_RESUME_SECTIONS = {
    'Contact Information': ['email', 'phone', 'linkedin', 'github', 'address', 'contact', 'mobile'],
    'Professional Summary': ['summary', 'objective', 'profile', 'about me', 'career objective', 'professional summary'],
    'Skills': ['skills', 'technical skills', 'core competencies', 'technologies', 'proficiencies'],
    'Work Experience': ['experience', 'work experience', 'employment', 'professional experience', 'work history', 'internship'],
    'Education': ['education', 'academic', 'degree', 'university', 'college', 'bachelor', 'master', 'b.tech', 'b.e', 'm.tech'],
    'Certifications': ['certification', 'certifications', 'certified', 'certificate', 'licenses', 'credentials'],
}


def analyze_ats_compatibility(raw_resume_text, resume_skills, job_description):
    """Full ATS compatibility analysis of resume against a job description."""
    results = {}
    jd_lower = job_description.lower()
    resume_lower = raw_resume_text.lower()

    # ── Extract keywords from job description ──
    found_multi = set()
    for term in _ATS_MULTI_WORD_TERMS:
        if term in jd_lower:
            found_multi.add(term)

    jd_words = re.findall(r'\b[a-z][a-z0-9#+.]+\b', jd_lower)
    word_freq = {}
    for w in jd_words:
        if w not in _ATS_STOPWORDS and len(w) > 2:
            word_freq[w] = word_freq.get(w, 0) + 1

    sorted_kw = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    top_single = {w for w, _ in sorted_kw[:40]}
    all_jd_keywords = top_single | found_multi

    matched_keywords = {kw for kw in all_jd_keywords if kw in resume_lower}
    missing_keywords = all_jd_keywords - matched_keywords

    recommended = {v for v in _ATS_POWER_VERBS if v in jd_lower and v not in resume_lower}

    results['matched_keywords'] = sorted(matched_keywords)
    results['missing_keywords'] = sorted(missing_keywords)
    results['recommended_keywords'] = sorted(recommended)

    # ── Section structure check ──
    section_scores = {}
    section_suggestions = {}

    for section, keywords in _RESUME_SECTIONS.items():
        found = any(kw in resume_lower for kw in keywords)

        if section == 'Contact Information':
            has_email = bool(re.search(r'[\w.+-]+@[\w-]+\.[\w.-]+', raw_resume_text))
            has_phone = bool(re.search(r'[\+]?[\d\s\-().]{7,}', raw_resume_text))
            if has_email and has_phone:
                score, tip = 100, "Contact information is complete."
            elif has_email or has_phone:
                score, tip = 60, "Add both email and phone number."
            else:
                score, tip = 20, "Add clear contact information (email, phone, LinkedIn)."

        elif section == 'Professional Summary':
            if found:
                score, tip = 85, "Summary detected. Tailor it to the target role with relevant keywords."
            else:
                score, tip = 30, "Add a Professional Summary (3-4 lines) highlighting key qualifications."

        elif section == 'Skills':
            if len(resume_skills) > 10:
                score, tip = 95, "Strong skills section with excellent coverage."
            elif len(resume_skills) > 5:
                score, tip = 75, "Good skills listed. Add more relevant skills from the job description."
            elif len(resume_skills) > 0:
                score, tip = 55, "Skills section is thin. Add more relevant technical skills."
            else:
                score, tip = 20, "Add a dedicated Skills section."

        elif section == 'Work Experience':
            if found:
                has_metrics = bool(re.search(r'\d+\s*%|\d+\+|reduced|increased|improved|saved', resume_lower))
                has_verbs = any(v in resume_lower for v in _ATS_POWER_VERBS[:10])
                if has_metrics and has_verbs:
                    score, tip = 95, "Excellent — quantified achievements with action verbs."
                elif has_verbs:
                    score, tip = 75, "Good action verbs. Add quantified achievements (numbers, %)."
                else:
                    score, tip = 55, "Use action verbs and include measurable results."
            else:
                score, tip = 25, "Add a Work Experience section with titles, companies, dates, and achievements."

        elif section == 'Education':
            if found:
                score, tip = 90, "Education section present."
            else:
                score, tip = 30, "Add an Education section with degree, institution, and graduation year."

        else:  # Certifications
            if found:
                score, tip = 90, "Certifications detected — great addition!"
            else:
                score, tip = 50, "Consider adding relevant certifications."

        section_scores[section] = score
        section_suggestions[section] = tip

    results['section_scores'] = section_scores
    results['section_suggestions'] = section_suggestions

    # ── Formatting assessment ──
    formatting_checks = {}

    word_count = len(re.findall(r'\w+', raw_resume_text))
    if 300 <= word_count <= 1000:
        formatting_checks['Resume Length'] = (True, f"Good length ({word_count} words)")
    elif word_count < 300:
        formatting_checks['Resume Length'] = (False, f"Too short ({word_count} words). Aim for 400-800 words.")
    else:
        formatting_checks['Resume Length'] = (True, f"Detailed resume ({word_count} words). Consider condensing to 2 pages.")

    formatting_checks['ATS-Safe Format'] = (True, "PDF text is extractable — ATS compatible.")

    has_bullets = bool(re.search(r'[•\-\*►▪■]', raw_resume_text))
    formatting_checks['Bullet Points'] = (
        has_bullets,
        "Uses bullet points for readability." if has_bullets else "Add bullet points for experience entries."
    )

    has_dates = bool(re.search(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{4})', resume_lower))
    formatting_checks['Date Formatting'] = (
        has_dates,
        "Dates detected in resume." if has_dates else "Add dates to experience and education entries."
    )

    heading_count = sum(
        1 for keywords in _RESUME_SECTIONS.values()
        if any(kw in resume_lower for kw in keywords)
    )
    formatting_checks['Section Headings'] = (
        heading_count >= 4,
        f"{heading_count}/6 standard sections detected." if heading_count >= 4
        else f"Only {heading_count}/6 standard sections found. Add clear headings."
    )

    results['formatting_checks'] = formatting_checks

    # ── Overall ATS score ──
    keyword_score = (len(matched_keywords) / max(len(all_jd_keywords), 1)) * 30

    resume_skill_lower = {s.lower() for s in resume_skills}
    jd_skill_tokens = set(re.findall(r'\b[a-z][a-z0-9#+.]+\b', jd_lower))
    skill_overlap = len(resume_skill_lower & jd_skill_tokens) / max(len(resume_skill_lower | jd_skill_tokens), 1)
    skill_match_score = skill_overlap * 25

    section_avg = sum(section_scores.values()) / max(len(section_scores), 1)
    section_score = (section_avg / 100) * 20

    fmt_passed = sum(1 for ok, _ in formatting_checks.values() if ok)
    fmt_score = (fmt_passed / max(len(formatting_checks), 1)) * 15

    contact_score = (section_scores.get('Contact Information', 50) / 100) * 10

    total = keyword_score + skill_match_score + section_score + fmt_score + contact_score
    results['overall_score'] = min(round(total), 100)
    results['score_breakdown'] = {
        'Keyword Match (30)': round(keyword_score, 1),
        'Skills Alignment (25)': round(skill_match_score, 1),
        'Section Structure (20)': round(section_score, 1),
        'Formatting (15)': round(fmt_score, 1),
        'Contact Info (10)': round(contact_score, 1),
    }

    # ── Optimization tips with rewrite examples ──
    top_kw = list(matched_keywords)[:3] + list(missing_keywords)[:2]
    kw_display = ', '.join(k.title() for k in top_kw[:5]) or 'relevant technologies'
    skill_display = ', '.join(s.title() for s in list(resume_skills)[:3]) or 'key technologies'
    all_relevant = sorted(set(list(matched_keywords)[:4] + list(resume_skills)[:4]))

    results['optimization_tips'] = [
        {
            'section': 'Professional Summary',
            'before': "Experienced software professional looking for new opportunities.",
            'after': f"Results-driven professional with expertise in {kw_display}, seeking to leverage proven skills in building scalable solutions that drive measurable business impact.",
        },
        {
            'section': 'Experience Bullet Points',
            'before': "Worked on developing web applications using various technologies.",
            'after': f"Developed and deployed 5+ production applications using {skill_display}, resulting in 40% improvement in system performance and user engagement.",
        },
        {
            'section': 'Skills Section',
            'before': "Skills: Python, Java, SQL",
            'after': "Technical Skills: " + ' | '.join(s.title() for s in all_relevant[:8])
                     + "\nTools & Platforms: Git, Docker, AWS, CI/CD Pipeline",
        },
    ]

    return results


# ── Success Banner ──
st.markdown(
    """
<div class="success-banner">
    <span style="font-size: 1.2rem;">&#10003;</span>
    Resume analyzed successfully &mdash; here are your personalized results
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

# ── Quick Stats ──
total_missing = sum(len(row["missing_skills"]) for _, row in top_k.iterrows())
avg_score = top_k["rf_score"].mean()

stat_cols = st.columns(4)
stats = [
    (str(len(resume_skills)), "Skills Detected"),
    (f"{avg_score:.0%}", "Avg Match Score"),
    (str(total_missing), "Skill Gaps Found"),
    ("3", "Jobs Matched"),
]
for col, (num, label) in zip(stat_cols, stats):
    with col:
        st.markdown(
            f"""
        <div class="stats-card">
            <div class="stats-number">{num}</div>
            <div class="stats-label">{label}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

# ── Extracted Skills ──
st.markdown(
    """
<div class="section-header">
    <div class="section-icon">🧠</div>
    <div class="section-title">Your Skills Profile</div>
</div>
""",
    unsafe_allow_html=True,
)

if len(resume_skills) == 0:
    st.warning("No skills were detected. Try uploading a more detailed resume.")
else:
    skills_html = "".join(
        f'<span class="skill-badge">{skill.title()}</span>'
        for skill in sorted(resume_skills)
    )
    st.markdown(
        f'<div style="padding: 0.5rem 0;">{skills_html}</div>',
        unsafe_allow_html=True,
    )

# ── Job Recommendations ──
st.markdown(
    """
<div class="section-header">
    <div class="section-icon">💼</div>
    <div class="section-title">Top Job Recommendations</div>
</div>
""",
    unsafe_allow_html=True,
)

job_cols = st.columns(3)
for idx, (col, (_, row)) in enumerate(zip(job_cols, top_k.iterrows())):
    score = row["rf_score"]
    score_pct = min(score * 100, 100)

    if score >= 0.7:
        bar_color = "linear-gradient(90deg, #10b981, #06b6d4)"
    elif score >= 0.4:
        bar_color = "linear-gradient(90deg, #6366f1, #818cf8)"
    else:
        bar_color = "linear-gradient(90deg, #f59e0b, #fbbf24)"

    exp = row.get("ExperienceLevel", "N/A")
    yrs = row.get("YearsOfExperience", "")
    exp_display = f"{exp}" + (f" ({yrs} yrs)" if yrs else "")

    matched_count = int(
        round(row["skill_score"] * len(clean_split_skills(row["Skills"])))
    )
    total_skills = len(clean_split_skills(row["Skills"]))
    missing_count = len(row["missing_skills"])

    with col:
        st.markdown(
            f"""
        <div class="job-card">
            <div class="job-rank">#{idx + 1}</div>
            <div class="job-title">{row['job_title']}</div>
            <div class="job-meta">
                <span class="meta-tag">&#128188; {exp_display}</span>
                <span class="meta-tag">&#9989; {matched_count}/{total_skills} skills</span>
            </div>
            <div class="score-bar-container">
                <div class="score-label">
                    <span>Match Score</span>
                    <span style="font-weight:600; color:#a5b4fc;">{score:.1%}</span>
                </div>
                <div class="score-bar-bg">
                    <div class="score-bar-fill" style="width:{score_pct}%; background:{bar_color};"></div>
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

# ── Match Score Radar Chart ──
st.markdown(
    """
<div class="section-header">
    <div class="section-icon">📈</div>
    <div class="section-title">Match Breakdown</div>
</div>
""",
    unsafe_allow_html=True,
)

chart_cols = st.columns([2, 1])

with chart_cols[0]:
    fig = go.Figure()

    for _, row in top_k.iterrows():
        fig.add_trace(
            go.Scatterpolar(
                r=[
                    row["semantic_score"],
                    row["skill_score"],
                    row["rf_score"],
                    1
                    - (
                        len(row["missing_skills"])
                        / max(len(clean_split_skills(row["Skills"])), 1)
                    ),
                    row["semantic_score"],
                ],
                theta=[
                    "Semantic Match",
                    "Skill Match",
                    "Overall Score",
                    "Coverage",
                    "Semantic Match",
                ],
                fill="toself",
                name=row["job_title"],
                opacity=0.6,
            )
        )

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(30, 41, 59, 0.5)",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor="rgba(99, 102, 241, 0.15)",
                linecolor="rgba(99, 102, 241, 0.2)",
            ),
            angularaxis=dict(
                gridcolor="rgba(99, 102, 241, 0.15)",
                linecolor="rgba(99, 102, 241, 0.2)",
            ),
        ),
        showlegend=True,
        legend=dict(font=dict(color="#94a3b8", size=11)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        margin=dict(l=40, r=40, t=30, b=30),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

with chart_cols[1]:
    st.markdown('<div style="margin-top: 1rem;">', unsafe_allow_html=True)
    for idx, (_, row) in enumerate(top_k.iterrows()):
        st.markdown(
            f"""
        <div style="background: #1e293b; border-radius: 10px; padding: 0.75rem; margin-bottom: 0.5rem; border: 1px solid rgba(99,102,241,0.1);">
            <div style="font-size: 0.82rem; font-weight: 600; color: #e2e8f0; margin-bottom: 0.5rem;">#{idx+1} {row['job_title']}</div>
            <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #94a3b8; margin-bottom: 2px;">
                <span>Semantic</span><span style="color:#a5b4fc; font-weight:600;">{row['semantic_score']:.3f}</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #94a3b8; margin-bottom: 2px;">
                <span>Skill</span><span style="color:#a5b4fc; font-weight:600;">{row['skill_score']:.3f}</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #94a3b8;">
                <span>Final</span><span style="color:#06b6d4; font-weight:600;">{row['rf_score']:.3f}</span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

# ── Skill Gap Analysis ──
st.markdown(
    """
<div class="section-header">
    <div class="section-icon">📉</div>
    <div class="section-title">Skill Gap Analysis</div>
</div>
""",
    unsafe_allow_html=True,
)

for _, row in top_k.iterrows():
    st.markdown(
        f'<div class="gap-job-header">{row["job_title"]}</div>',
        unsafe_allow_html=True,
    )

    if len(row["missing_skills"]) == 0:
        st.markdown(
            '<div class="no-gaps-badge">&#10024; No major skill gaps — great match!</div>',
            unsafe_allow_html=True,
        )
    else:
        job_skill_list = clean_split_skills(row["Skills"])
        matched_html = ""
        missing_html = ""
        for s in sorted(job_skill_list):
            if s in resume_skills:
                matched_html += (
                    f'<span class="skill-badge matched">{s.title()}</span>'
                )
            elif s in row["missing_skills"]:
                missing_html += (
                    f'<span class="skill-badge missing">{s.title()}</span>'
                )

        combined = matched_html + missing_html
        st.markdown(
            f'<div style="padding: 0.25rem 0 0.75rem 0;">{combined}</div>',
            unsafe_allow_html=True,
        )

        pct = len(row["missing_skills"]) / max(len(job_skill_list), 1)
        st.markdown(
            f"""
        <div style="font-size: 0.78rem; color: #94a3b8; margin-bottom: 0.75rem;">
            <span style="color: #6ee7b7;">&#9679;</span> You have &nbsp;|&nbsp;
            <span style="color: #fca5a5;">&#9679;</span> Missing ({len(row['missing_skills'])} of {len(job_skill_list)} &mdash; {pct:.0%} gap)
        </div>
        """,
            unsafe_allow_html=True,
        )

# ── Course / Learning Recommendations ──
st.markdown(
    """
<div class="section-header">
    <div class="section-icon">🎓</div>
    <div class="section-title">Recommended Learning Resources</div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<p style="color:#94a3b8; font-size:0.88rem; margin-bottom:1rem;">'
    "Curated YouTube searches and official documentation to help you learn the skills required for your target roles.</p>",
    unsafe_allow_html=True,
)

tab_labels = [f"📋 {row['job_title']}" for _, row in top_k.iterrows()]
tab_labels.append("📝 Resume Improvement")
tabs = st.tabs(tab_labels)

job_tabs = tabs[:-1]
improve_tab = tabs[-1]

for tab, (_, row) in zip(job_tabs, top_k.iterrows()):
    with tab:
        if len(row["missing_skills"]) == 0:
            st.markdown(
                '<div class="no-gaps-badge" style="margin: 1rem 0;">&#127881; No skill gaps to fill — you\'re all set!</div>',
                unsafe_allow_html=True,
            )
            continue

        shown = set()
        resource_count = 0
        for skill in row["missing_skills"]:
            resources = get_resources_for_skill(skill)
            for res in resources:
                title = res.get("title", f"Learn {skill.title()}")
                url = res.get("url", "#")
                provider = res.get("provider", "Resource")
                key = (title, url)
                if key in shown:
                    continue
                shown.add(key)
                resource_count += 1

                st.markdown(
                    f"""
                <a href="{url}" target="_blank" class="course-link" style="text-decoration: none;">
                    <div class="course-card">
                        <div class="course-icon">&#128218;</div>
                        <div class="course-info">
                            <div class="course-skill">Learn: {skill.title()}</div>
                            <div class="course-title">{title}</div>
                            <div class="course-provider">
                                {provider}
                            </div>
                        </div>
                    </div>
                </a>
                """,
                    unsafe_allow_html=True,
                )

        if resource_count == 0:
            st.markdown(
                '<p style="color: #64748b; font-style: italic; padding: 1rem 0;">No mapped resources found for these skills yet. You can manually search on YouTube or official docs.</p>',
                unsafe_allow_html=True,
            )

with improve_tab:
    suggestions = generate_resume_suggestions(resume_text, resume_skills, top_k)

    st.markdown(
        '<p style="color:#94a3b8; font-size:0.88rem; margin:0.5rem 0 1rem 0;">'
        "Rule-based tips to make your resume stronger, clearer, and better aligned with your target roles."
        "</p>",
        unsafe_allow_html=True,
    )

    # A) Missing Technical Skills
if suggestions["missing_skills"]:
    bullets = "".join(f"<li>{s}</li>" for s in suggestions["missing_skills"])
    st.markdown(
        f"""
        <div class="card-improve">
            <h4>🔹 Missing Technical Skills</h4>
            <ul>
                {bullets}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="card-improve">
            <h4>🔹 Missing Technical Skills</h4>
            <p>No major missing technical skills detected for your top recommended jobs.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # B) Resume Content Improvements
if suggestions["content"]:
    bullets = "".join(f"<li>{s}</li>" for s in suggestions["content"])
    st.markdown(
        f"""
        <div class="card-improve">
            <h4>🔹 Resume Content Improvements</h4>
            <ul>
                {bullets}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="card-improve">
            <h4>🔹 Resume Content Improvements</h4>
            <p>Your content already looks substantial and action-oriented. Great work!</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # C) Formatting Suggestions
if suggestions["formatting"]:
    bullets = "".join(f"<li>{s}</li>" for s in suggestions["formatting"])
    st.markdown(
        f"""
        <div class="card-improve">
            <h4>🔹 Formatting Suggestions</h4>
            <ul>
                {bullets}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="card-improve">
            <h4>🔹 Formatting Suggestions</h4>
            <p>Formatting suggestions are not available at the moment.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

#
# ── ATS Resume Analyzer Section ──
st.markdown(
    """
<div class="section-header">
    <div class="section-icon">📊</div>
    <div class="section-title">ATS Resume Analyzer</div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<p style="color:#94a3b8; font-size:0.88rem; margin-bottom:1rem;">'
    "Paste a target job description below and click <strong>Analyze ATS Compatibility</strong> "
    "to see how well your resume matches, with keyword analysis, section scores, and optimization tips."
    "</p>",
    unsafe_allow_html=True,
)

ats_job_desc = st.text_area(
    "Paste the target job description",
    placeholder="Paste the full job description here to analyze ATS compatibility...",
    height=180,
    key="ats_jd_input",
)

ats_btn_col1, ats_btn_col2, ats_btn_col3 = st.columns([1, 1, 1])
with ats_btn_col2:
    ats_analyze = st.button("🔍 Analyze ATS Compatibility", use_container_width=True, type="primary")

if ats_analyze:
    if not ats_job_desc.strip():
        st.error("Please paste a non-empty job description to analyze.")
    else:
        with st.spinner("Running ATS analysis..."):
            ats_results = analyze_ats_compatibility(resume_text_raw, resume_skills, ats_job_desc)

        score = ats_results['overall_score']
        if score >= 75:
            score_color = "#10b981"
            score_label = "Excellent"
        elif score >= 50:
            score_color = "#f59e0b"
            score_label = "Good — Needs Improvement"
        else:
            score_color = "#ef4444"
            score_label = "Low — Significant Changes Needed"

        # ── Score display + breakdown ──
        score_col, breakdown_col = st.columns([1, 1.5])

        with score_col:
            st.markdown(
                f"""
            <div class="ats-score-container">
                <div class="ats-score-ring" style="background: conic-gradient({score_color} {score * 3.6}deg, #1e293b {score * 3.6}deg);">
                    <div class="ats-score-inner">
                        <div class="ats-score-value" style="color: {score_color};">{score}%</div>
                        <div class="ats-score-label">ATS Score</div>
                    </div>
                </div>
                <div class="ats-score-verdict" style="color: {score_color};">{score_label}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with breakdown_col:
            st.markdown('<div class="ats-breakdown-title">Score Breakdown</div>', unsafe_allow_html=True)
            for category, pts in ats_results['score_breakdown'].items():
                max_pts = int(re.search(r'\((\d+)\)', category).group(1))
                pct = (pts / max_pts) * 100 if max_pts else 0
                cat_label = re.sub(r'\s*\(\d+\)', '', category)
                if pct >= 70:
                    bar_clr = "#10b981"
                elif pct >= 40:
                    bar_clr = "#f59e0b"
                else:
                    bar_clr = "#ef4444"
                st.markdown(
                    f"""
                <div class="ats-bar-row">
                    <div class="ats-bar-label">{cat_label}</div>
                    <div class="ats-bar-track">
                        <div class="ats-bar-fill" style="width:{pct}%; background:{bar_clr};"></div>
                    </div>
                    <div class="ats-bar-value">{pts}/{max_pts}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        # ── Keyword Analysis ──
        st.markdown(
            """
        <div class="ats-sub-header">🔑 Keyword Analysis</div>
        """,
            unsafe_allow_html=True,
        )

        kw_col1, kw_col2, kw_col3 = st.columns(3)

        with kw_col1:
            matched = ats_results['matched_keywords']
            badges = ''.join(f'<span class="ats-kw-badge matched">{k}</span>' for k in matched) if matched else '<span style="color:#64748b;">None</span>'
            st.markdown(
                f"""
            <div class="ats-kw-card">
                <div class="ats-kw-header matched">✅ Matched Keywords ({len(matched)})</div>
                <div class="ats-kw-body">{badges}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with kw_col2:
            missing = ats_results['missing_keywords']
            badges = ''.join(f'<span class="ats-kw-badge missing">{k}</span>' for k in missing) if missing else '<span style="color:#64748b;">None</span>'
            st.markdown(
                f"""
            <div class="ats-kw-card">
                <div class="ats-kw-header missing">❌ Missing Keywords ({len(missing)})</div>
                <div class="ats-kw-body">{badges}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with kw_col3:
            recommended = ats_results['recommended_keywords']
            badges = ''.join(f'<span class="ats-kw-badge recommended">{k}</span>' for k in recommended) if recommended else '<span style="color:#64748b;">None</span>'
            st.markdown(
                f"""
            <div class="ats-kw-card">
                <div class="ats-kw-header recommended">⚠️ Recommended to Add ({len(recommended)})</div>
                <div class="ats-kw-body">{badges}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # ── Section-wise Score Breakdown ──
        st.markdown(
            '<div class="ats-sub-header">📋 Section-wise Score Breakdown</div>',
            unsafe_allow_html=True,
        )

        sec_cols = st.columns(3)
        for idx, (section, sec_score) in enumerate(ats_results['section_scores'].items()):
            suggestion = ats_results['section_suggestions'][section]
            if sec_score >= 80:
                sec_clr = "#10b981"
                sec_icon = "✅"
            elif sec_score >= 50:
                sec_clr = "#f59e0b"
                sec_icon = "⚠️"
            else:
                sec_clr = "#ef4444"
                sec_icon = "❌"

            with sec_cols[idx % 3]:
                st.markdown(
                    f"""
                <div class="ats-section-card">
                    <div class="ats-section-top">
                        <span>{sec_icon} {section}</span>
                        <span class="ats-section-score" style="color:{sec_clr};">{sec_score}/100</span>
                    </div>
                    <div class="ats-section-bar-bg">
                        <div class="ats-section-bar-fill" style="width:{sec_score}%; background:{sec_clr};"></div>
                    </div>
                    <div class="ats-section-tip">{suggestion}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        # ── Formatting & Structure Evaluation ──
        st.markdown(
            '<div class="ats-sub-header">🔍 Formatting & Structure Evaluation</div>',
            unsafe_allow_html=True,
        )

        for check_name, (passed, detail) in ats_results['formatting_checks'].items():
            icon = "✅" if passed else "❌"
            clr = "#6ee7b7" if passed else "#fca5a5"
            st.markdown(
                f"""
            <div class="ats-check-row">
                <span style="font-size:1.1rem;">{icon}</span>
                <div>
                    <span class="ats-check-name">{check_name}</span>
                    <span class="ats-check-detail" style="color:{clr};">{detail}</span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # ── Resume Optimization Tips (Before / After) ──
        st.markdown(
            '<div class="ats-sub-header">✏️ Resume Optimization Tips — Rewrite Examples</div>',
            unsafe_allow_html=True,
        )

        for tip in ats_results['optimization_tips']:
            st.markdown(
                f"""
            <div class="ats-tip-card">
                <div class="ats-tip-section">{tip['section']}</div>
                <div class="ats-tip-row">
                    <div class="ats-tip-label before">Before</div>
                    <div class="ats-tip-text before">{tip['before']}</div>
                </div>
                <div class="ats-tip-row">
                    <div class="ats-tip-label after">After</div>
                    <div class="ats-tip-text after">{tip['after']}</div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # ── Actionable Improvement Suggestions ──
        st.markdown(
            '<div class="ats-sub-header">🎯 Actionable Improvement Suggestions</div>',
            unsafe_allow_html=True,
        )

        action_items = []
        if ats_results['missing_keywords']:
            top_missing = ', '.join(k.title() for k in ats_results['missing_keywords'][:8])
            action_items.append(f"Add these missing keywords to relevant sections of your resume: <strong>{top_missing}</strong>")

        if ats_results['recommended_keywords']:
            rec_str = ', '.join(k.title() for k in ats_results['recommended_keywords'][:5])
            action_items.append(f"Use these power verbs in your experience bullet points: <strong>{rec_str}</strong>")

        for section, sec_score in ats_results['section_scores'].items():
            if sec_score < 60:
                action_items.append(f"Improve your <strong>{section}</strong> section — {ats_results['section_suggestions'][section]}")

        for check_name, (passed, detail) in ats_results['formatting_checks'].items():
            if not passed:
                action_items.append(f"Fix <strong>{check_name}</strong>: {detail}")

        action_items.append("Quantify achievements with numbers and percentages wherever possible.")
        action_items.append("Mirror the exact job title and key phrases from the job description in your resume.")

        bullets = ''.join(f'<li>{item}</li>' for item in action_items)
        st.markdown(
            f"""
        <div class="ats-action-card">
            <ul>{bullets}</ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

# ── Fake Job Posting Detector ──
st.markdown(
    """
<div class="section-header">
    <div class="section-icon">🚫</div>
    <div class="section-title">Fake Job Posting Detector</div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<p style="color:#94a3b8; font-size:0.88rem; margin-bottom:1rem;">'
    "Paste a job description to estimate how likely it is to be fraudulent, based on the Kaggle fake job postings dataset."
    "</p>",
    unsafe_allow_html=True,
)

if not HAS_FAKE_JOB_DEPS:
    st.warning(
        "Fake job detection is disabled because required libraries are missing. "
        "Install `xgboost`, `textblob`, and `joblib` to enable it.",
        icon="⚠️",
    )
elif not USE_SEMANTIC or model is None:
    st.warning(
        "Fake job detection requires the Sentence-BERT model. "
        "Install `sentence-transformers` to enable it.",
        icon="⚠️",
    )
else:
    job_description = st.text_area(
        "Paste a job description",
        placeholder="Paste the full text of a job posting here...",
        height=180,
    )
    if st.button("Analyze Job Posting"):
        if not job_description.strip():
            st.error("Please paste a non-empty job description.")
        else:
            proba = predict_fake_probability(job_description)
            if proba is None:
                st.error(
                    "Fake job model file not found or could not be loaded. "
                    "Train it by running `python fakejob.py` to create `models/fake_job_xgb.pkl`."
                )
            else:
                label = "Likely Fake" if proba >= 0.5 else "Likely Real"
                color = "#f97373" if proba >= 0.5 else "#22c55e"
                st.markdown(
                    f"""
<div class="card-improve">
    <h4>Risk Assessment</h4>
    <p style="font-size:0.9rem; margin-bottom:0.5rem;">
        Estimated probability of being a fake posting:
        <span style="font-weight:600; color:{color};">{proba:.1%}</span>
        &nbsp;→ <strong>{label}</strong>
    </p>
    <p style="font-size:0.8rem; color:#94a3b8;">
        This score is based on language patterns, sentiment, and suspicious keywords learned from historical fake job postings.
    </p>
</div>
""",
                    unsafe_allow_html=True,
                )

# ── Footer ──
st.markdown("---")
st.markdown(
    """
<div style="text-align: center; padding: 1rem 0 2rem 0;">
    <div style="font-size: 0.82rem; color: #475569;">
        Built with Streamlit &amp; Sentence-BERT &nbsp;&#x2022;&nbsp; AI Career Compass Pro
    </div>
</div>
""",
    unsafe_allow_html=True,
)