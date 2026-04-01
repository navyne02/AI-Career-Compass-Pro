import pandas as pd
import numpy as np
import os
import joblib
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from xgboost import XGBClassifier

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/fake_job_postings.csv")

# Kaggle dataset uses 'fraudulent' as label
df = df.dropna(subset=["description", "fraudulent"])

texts = df["description"].astype(str).tolist()
labels = df["fraudulent"].values  # 1 = fake, 0 = real

# ---------------- FRAUD KEYWORDS ----------------
FRAUD_WORDS = [
    "registration fee", "processing fee", "urgent hiring",
    "no experience", "whatsapp", "telegram",
    "easy money", "work from home",
    "limited slots", "guaranteed income",
    "earn per day", "earn per week", "instant joining"
]

def extract_linguistic_features(text):
    text = text.lower()
    fraud_count = sum(text.count(w) for w in FRAUD_WORDS)
    length = len(text.split())
    blob = TextBlob(text)

    # 🔥 scaled so XGBoost pays attention
    return [
        fraud_count * 10,
        length / 100,
        blob.sentiment.polarity * 5,
        blob.sentiment.subjectivity * 5
    ]


# ---------------- HELPER FUNCTIONS ----------------
def build_feature_matrix(text_list, sbert_model):
    """Encode descriptions and concatenate with linguistic features."""
    print("Encoding descriptions with SBERT...")
    sbert_embeddings = sbert_model.encode(
        text_list,
        batch_size=32,
        show_progress_bar=True,
    )

    features = []
    for i, text in enumerate(text_list):
        ling_feats = extract_linguistic_features(text)
        final_vec = np.concatenate([sbert_embeddings[i], ling_feats])
        features.append(final_vec)

    X = np.array(features)
    print("Final feature shape:", X.shape)  # (N, 388)
    return X


def train_and_save_model(model_path: str = "models/fake_job_xgb.pkl"):
    """Train XGBoost fake‑job classifier and save it to disk."""
    # ---------------- SBERT ENCODING + FEATURE BUILD ----------------
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    X = build_feature_matrix(texts, sbert)

    # ---------------- HANDLE CLASS IMBALANCE ----------------
    pos_weight = (len(labels) - sum(labels)) / sum(labels)

    # ---------------- TRAIN XGBOOST ----------------
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.08,
        scale_pos_weight=pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )

    print("Training XGBoost...")
    model.fit(X, labels)

    # ---------------- SAVE MODEL ----------------
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump(model, model_path)

    print(f"✅ Model trained and saved successfully at {model_path}")


if __name__ == "__main__":
    train_and_save_model()
