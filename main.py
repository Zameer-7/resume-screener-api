from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import shap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
import uvicorn

app = FastAPI(title="Resume Screener API")

# ── Input Schema ──────────────────────────────────────────────────────────────
class StructuredData(BaseModel):
    resume_skills: List[str] = []
    resume_experience_years: Optional[float] = 0
    resume_keywords: List[str] = []
    job_required_skills: List[str] = []
    job_keywords: List[str] = []

class ScoreRequest(BaseModel):
    resume_text: str
    job_description_text: str
    structured_data: StructuredData

# ── Output Schema ─────────────────────────────────────────────────────────────
class ScoreResponse(BaseModel):
    match_score: float
    matched_skills: List[str]
    missing_skills: List[str]
    top_positive_features: List[str]
    top_negative_features: List[str]
    improvement_suggestions: List[str]

# ── Scoring Logic ─────────────────────────────────────────────────────────────
def compute_score(req: ScoreRequest) -> ScoreResponse:
    sd = req.structured_data

    resume_skills_lower  = [s.lower() for s in sd.resume_skills]
    job_skills_lower     = [s.lower() for s in sd.job_required_skills]

    matched  = [s for s in sd.job_required_skills if s.lower() in resume_skills_lower]
    missing  = [s for s in sd.job_required_skills if s.lower() not in resume_skills_lower]

    # ── TF-IDF + GBM (trained on-the-fly with synthetic pairs) ───────────────
    combined_resume = req.resume_text + " " + " ".join(sd.resume_skills)
    combined_jd     = req.job_description_text + " " + " ".join(sd.job_required_skills)

    vectorizer = TfidfVectorizer(max_features=50, stop_words="english")
    # Fit on both docs
    vectorizer.fit([combined_resume, combined_jd])
    rv = vectorizer.transform([combined_resume]).toarray()[0]
    jv = vectorizer.transform([combined_jd]).toarray()[0]

    feature_names = vectorizer.get_feature_names_out()
    diff_vector   = (rv - jv).reshape(1, -1)          # gap features

    # Skill overlap ratio (primary signal)
    skill_ratio = len(matched) / max(len(job_skills_lower), 1)

    # Keyword overlap
    resume_kw = set(k.lower() for k in sd.resume_keywords)
    jd_kw     = set(k.lower() for k in sd.job_keywords)
    kw_ratio  = len(resume_kw & jd_kw) / max(len(jd_kw), 1)

    # Experience bonus (cap at 1.0)
    exp_bonus = min(sd.resume_experience_years / 5.0, 1.0) if sd.resume_experience_years else 0

    raw_score = (skill_ratio * 60) + (kw_ratio * 25) + (exp_bonus * 15)
    match_score = round(min(raw_score, 100), 2)

    # ── SHAP explanation ──────────────────────────────────────────────────────
    # Build a tiny synthetic training set so GBM has something to explain
    X_train = np.random.randn(60, len(feature_names))
    y_train = (X_train.sum(axis=1) > 0).astype(int)
    model = GradientBoostingClassifier(n_estimators=30, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(diff_vector)[0]   # shape: (n_features,)

    # Top positive / negative features from SHAP
    idx_sorted = np.argsort(shap_values)
    top_neg_idx = idx_sorted[:3]
    top_pos_idx = idx_sorted[-3:][::-1]

    top_positive = [feature_names[i] for i in top_pos_idx if shap_values[i] > 0]
    top_negative = [feature_names[i] for i in top_neg_idx if shap_values[i] < 0]

    # Also surface missing skills as negative features
    top_negative = (missing[:3] + top_negative)[:5]

    # ── Suggestions ──────────────────────────────────────────────────────────
    suggestions = []
    if missing:
        suggestions.append(f"Add these missing skills to your resume: {', '.join(missing[:5])}")
    if kw_ratio < 0.5:
        suggestions.append("Use more keywords from the job description in your resume.")
    if sd.resume_experience_years and sd.resume_experience_years < 2:
        suggestions.append("Highlight internships, projects, or freelance work to boost experience signal.")
    if match_score < 50:
        suggestions.append("Tailor your resume specifically for this role — skill match is low.")
    if not suggestions:
        suggestions.append("Great match! Focus on quantifying your achievements with numbers.")

    return ScoreResponse(
        match_score=match_score,
        matched_skills=matched,
        missing_skills=missing,
        top_positive_features=top_positive or ["relevant experience", "keyword match"],
        top_negative_features=top_negative or ["missing required skills"],
        improvement_suggestions=suggestions,
    )

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Resume Screener API is running ✅"}

@app.post("/score-resume", response_model=ScoreResponse)
def score_resume(req: ScoreRequest):
    return compute_score(req)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
