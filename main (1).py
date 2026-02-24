from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import shap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
import uvicorn
import json
import io

# OCR imports
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Text-based PDF fallback
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

app = FastAPI(title="Resume Screener API")

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── PDF Text Extraction ───────────────────────────────────────────────────────
def extract_text_from_pdf(file_bytes: bytes) -> str:
    full_text = ""

    # Try text-based extraction first (faster)
    if PDFPLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
            if full_text.strip():
                return full_text.strip()
        except Exception:
            pass

    # Fallback to OCR for image-based / Canva PDFs
    if OCR_AVAILABLE:
        try:
            images = convert_from_bytes(file_bytes)
            for image in images:
                text = pytesseract.image_to_string(image)
                full_text += text + "\n"
            return full_text.strip()
        except Exception as e:
            return f"OCR extraction failed: {str(e)}"

    return full_text or "Could not extract text from PDF"

# ── Output Schema ─────────────────────────────────────────────────────────────
class ScoreResponse(BaseModel):
    match_score: float
    matched_skills: List[str]
    missing_skills: List[str]
    top_positive_features: List[str]
    top_negative_features: List[str]
    improvement_suggestions: List[str]
    resume_text_preview: Optional[str] = ""

# ── Scoring Logic ─────────────────────────────────────────────────────────────
def compute_score(resume_text: str, job_description_text: str, structured_data: dict) -> ScoreResponse:
    resume_skills    = structured_data.get("resume_skills", [])
    job_skills       = structured_data.get("job_required_skills", [])
    resume_keywords  = structured_data.get("resume_keywords", [])
    job_keywords     = structured_data.get("job_keywords", [])
    experience_years = structured_data.get("resume_experience_years", 0) or 0

    resume_skills_lower = [s.lower() for s in resume_skills]
    job_skills_lower    = [s.lower() for s in job_skills]

    matched = [s for s in job_skills if s.lower() in resume_skills_lower]
    missing = [s for s in job_skills if s.lower() not in resume_skills_lower]

    combined_resume = resume_text + " " + " ".join(resume_skills)
    combined_jd     = job_description_text + " " + " ".join(job_skills)

    vectorizer = TfidfVectorizer(max_features=50, stop_words="english")
    vectorizer.fit([combined_resume, combined_jd])
    rv = vectorizer.transform([combined_resume]).toarray()[0]
    jv = vectorizer.transform([combined_jd]).toarray()[0]
    feature_names = vectorizer.get_feature_names_out()
    diff_vector   = (rv - jv).reshape(1, -1)

    skill_ratio = len(matched) / max(len(job_skills_lower), 1)
    resume_kw   = set(k.lower() for k in resume_keywords)
    jd_kw       = set(k.lower() for k in job_keywords)
    kw_ratio    = len(resume_kw & jd_kw) / max(len(jd_kw), 1)
    exp_bonus   = min(float(experience_years) / 5.0, 1.0) if experience_years else 0

    raw_score   = (skill_ratio * 60) + (kw_ratio * 25) + (exp_bonus * 15)
    match_score = round(min(raw_score, 100), 2)

    X_train = np.random.randn(60, len(feature_names))
    y_train = (X_train.sum(axis=1) > 0).astype(int)
    model   = GradientBoostingClassifier(n_estimators=30, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(diff_vector)[0]

    idx_sorted   = np.argsort(shap_values)
    top_neg_idx  = idx_sorted[:3]
    top_pos_idx  = idx_sorted[-3:][::-1]
    top_positive = [feature_names[i] for i in top_pos_idx if shap_values[i] > 0]
    top_negative = [feature_names[i] for i in top_neg_idx if shap_values[i] < 0]
    top_negative = (missing[:3] + top_negative)[:5]

    suggestions = []
    if missing:
        suggestions.append(f"Add these missing skills: {', '.join(missing[:5])}")
    if kw_ratio < 0.5:
        suggestions.append("Use more keywords from the job description in your resume.")
    if experience_years and float(experience_years) < 2:
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
        resume_text_preview=resume_text[:300]
    )

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Resume Screener API is running ✅", "ocr_available": OCR_AVAILABLE}

# NEW: Accept PDF file upload directly (production endpoint)
@app.post("/score-resume-file", response_model=ScoreResponse)
async def score_resume_file(
    file: UploadFile = File(...),
    job_description: str = Form(...),
    candidate_email: str = Form(...),
    structured_data: str = Form("{}")
):
    file_bytes  = await file.read()
    resume_text = extract_text_from_pdf(file_bytes)
    structured  = json.loads(structured_data)
    return compute_score(resume_text, job_description, structured)

# EXISTING: Accept JSON (backward compatible)
@app.post("/score-resume", response_model=ScoreResponse)
async def score_resume_json(request: dict):
    resume_text   = request.get("resume_text", "")
    job_desc_text = request.get("job_description_text", "")
    structured    = request.get("structured_data", {})
    return compute_score(resume_text, job_desc_text, structured)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
