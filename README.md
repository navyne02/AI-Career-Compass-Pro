# AI Career Compass Pro (Advanced AI Version)

This project implements **Hybrid Similarity** and **Deep Learning** for high-precision resume analysis and professional career mapping.

---

### 🧠 Core Technologies

1. **SBERT (Sentence-BERT):** A Deep Learning transformer model used for generating **semantic embeddings**. This allows the system to understand context—knowing that "Backend Developer" and "Node.js Engineer" belong to the same professional space.

2. **Hybrid Similarity:** A dual-layer logic that combines **Semantic Cosine Similarity** (for context) with **Ontology-based Lexical matching** (for keyword accuracy).

3. **Skill Ontology:** A hierarchical mapping system that links raw skills to **industry domains**. It understands the "family tree" of technologies (e.g., NumPy -> Data Science -> AI).

4. **Streamlit:** An advanced interactive UI built for a seamless user experience, allowing for real-time resume parsing and score visualization.

---

### 🚀 Advanced Features

* **Contextual Gap Analysis:** Identifies exactly which skills are missing from a resume based on the target job's semantic meaning.
* **Domain Recognition:** Automatically classifies candidates into industry sectors like Fintech, Cloud Computing, or AI Research.
* **Real-time Scoring:** Instant feedback on how well a profile matches a specific job description using the hybrid engine.

---

### 🛠️ Execution

Follow these steps to launch the environment:

1. **Install requirements:**
   ```bash
   pip install -r requirements.txt
