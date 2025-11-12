# api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
from recommender import CourseRecommender

DATA_PATH = os.environ.get("DATA_PATH", "data/udemy_courses.csv")

app = FastAPI(title="ProfConnect Recommender")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with your frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load dataset and build recommender at startup
if not os.path.exists(DATA_PATH):
    raise RuntimeError(f"Missing data file at {DATA_PATH}. Put udemy_courses.csv in data/")

df = pd.read_csv(DATA_PATH, dtype=str).fillna("")
rec = CourseRecommender(df)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/recommend_minimal/{course_id}")
def recommend_minimal(course_id: str, top_n: int = 6):
    if str(course_id) not in rec.course_id_to_index:
        raise HTTPException(status_code=404, detail="course_id not found")
    results = rec.recommend_minimal(course_id, top_n=top_n)
    return {"course_id": course_id, "recommendations": results}

@app.get("/recommend_by_field")
def recommend_by_field(field: str, duration_months: float = None, top_n: int = 8):
    if not field:
        raise HTTPException(status_code=400, detail="field required")
    recs = rec.recommend_by_field_duration(field, duration_months=duration_months, top_n=top_n)
    out = []
    for r in recs:
        title = r.get('title') or r.get('course_title') or r.get('headline') or ''
        url = r.get('url', '') or ''
        if isinstance(url, str) and url.startswith('/'):
            url = 'https://www.udemy.com' + url
        out.append({
            'title': title,
            'url': url,
            'rating': r.get('rating', ''),
            'course_id': r.get('course_id') or r.get('id') or ''
        })
    return {"field": field, "duration_months": duration_months, "recommendations": out}
