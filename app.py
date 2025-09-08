import os
# Required packages: flask, docx2txt, PyPDF2, scikit-learn, sentence-transformers
from flask import Flask, request, jsonify, render_template, session, flash, redirect, url_for
import docx2txt
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Preprocessing
import re

def preprocess_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', text)
    return text.strip()

# Load SentenceTransformer model once
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

# Extract text from resume files
def extract_text_from_file(file_path):
    text = ""
    if file_path.endswith(".pdf"):
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
    elif file_path.endswith(".docx"):
        text = docx2txt.process(file_path)
    return preprocess_text(text)

# FIXED Similarity Calculation with Relevance Labels
def calculate_similarity(resume_text, jd_text):
    if not resume_text or not jd_text:
        return {
            "Cosine Similarity": 0.0,
            "BERT Semantic Similarity": 0.0,
            "Average Score": 0.0,
            "Relevance": "Not Relevant"
        }

    # TF-IDF cosine similarity
    try:
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=1)
        tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
        cosine_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except Exception:
        cosine_score = 0.0

    # BERT semantic similarity
    try:
        emb1 = sentence_model.encode(resume_text, convert_to_tensor=True)
        emb2 = sentence_model.encode(jd_text, convert_to_tensor=True)
        raw_bert = util.pytorch_cos_sim(emb1, emb2).item()
        bert_score = max(0.0, min((raw_bert + 1.0) / 2.0, 1.0))  # Normalize [0,1]
    except Exception:
        bert_score = 0.0

    # Token overlap ratio
    resume_tokens = set([w for w in resume_text.lower().split() if w not in ENGLISH_STOP_WORDS])
    jd_tokens = set([w for w in jd_text.lower().split() if w not in ENGLISH_STOP_WORDS])
    overlap_ratio = len(resume_tokens & jd_tokens) / max(len(jd_tokens), 1)

    # Weighted average with penalty for unrelated texts
    weighted_avg = (0.6 * bert_score) + (0.3 * cosine_score) + (0.1 * overlap_ratio)
    if overlap_ratio < 0.05 and weighted_avg > 0.6:
        weighted_avg *= 0.5

    # Relevance labels
    if weighted_avg >= 0.7:
        relevance = "Highly Relevant"
    elif weighted_avg >= 0.4:
        relevance = "Moderately Relevant"
    else:
        relevance = "Not Relevant"

    return {
        "Cosine Similarity": round(cosine_score, 3),
        "BERT Semantic Similarity": round(bert_score, 3),
        "Average Score": round(weighted_avg, 3),
        "Relevance": relevance
    }

# Flask App
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")

@app.route("/")
def index():
    return render_template("landing.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/upload")
def upload():
    return render_template("index.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        if not username or not password:
            flash("Username and password are required.", "error")
            return render_template("signup.html")
        if password != confirm:
            flash("Passwords do not match.", "error")
            return render_template("signup.html")
        flash("Signup successful! Please log in.", "success")
        return redirect(url_for("login"))
    return render_template("signup.html")

@app.route("/process", methods=["POST"])
def process():
    jd1 = preprocess_text(request.form.get("job_description1", ""))
    jd2 = preprocess_text(request.form.get("job_description2", ""))
    jd3 = preprocess_text(request.form.get("job_description3", ""))
    jd_texts = [jd1, jd2, jd3]

    resumes = request.files.getlist("resumes")
    results = []
    os.makedirs("uploads", exist_ok=True)

    for resume in resumes:
        path = os.path.join("uploads", resume.filename)
        resume.save(path)
        resume_text = extract_text_from_file(path)

        jd_scores = {}
        for idx, jd_text in enumerate(jd_texts, start=1):
            if jd_text:
                scores = calculate_similarity(resume_text, jd_text)
                jd_scores[f"JD{idx} Avg Score"] = scores["Average Score"]
                jd_scores[f"JD{idx} Relevance"] = scores["Relevance"]
            else:
                jd_scores[f"JD{idx} Avg Score"] = 0.0
                jd_scores[f"JD{idx} Relevance"] = "Not Relevant"

        results.append({"Resume": resume.filename, **jd_scores})

    # Sort by highest score
    results = sorted(
        results,
        key=lambda x: max(
            x.get("JD1 Avg Score", 0),
            x.get("JD2 Avg Score", 0),
            x.get("JD3 Avg Score", 0)
        ),
        reverse=True
    )

    session["last_results"] = results
    return render_template("results.html", results=results)

@app.route("/results")
def results_page():
    results = session.get("last_results")
    if not results:
        flash("No results yet. Please upload resumes and process.", "error")
        return redirect(url_for("index"))
    return render_template("results.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
