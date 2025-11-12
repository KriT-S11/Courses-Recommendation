# recommender.py
"""
CourseRecommender for Udemy-style CSVs.

Features:
- Normalizes common CSV column names (id, title, url, instructors, etc.)
- Builds TF-IDF on combined text fields (title + subtitle/headline + instructors + subject)
- Methods:
    - recommend_for_course(course_id, top_n)
    - recommend_for_completed(list_of_course_ids, top_n)
    - recommend_minimal(course_id, top_n) -> returns only title, url (absolute), rating
    - recommend_by_field_duration(field, duration_months, top_n, duration_tolerance)
- Persistence: save/load with joblib
"""
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

class CourseRecommender:
    def __init__(self, df: pd.DataFrame, text_fields: Optional[List[str]] = None, max_features: int = 20000):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        self.df = df.copy().reset_index(drop=True).fillna("")

        # Normalize id column
        if 'course_id' not in self.df.columns:
            if 'id' in self.df.columns:
                self.df = self.df.rename(columns={'id': 'course_id'})
            else:
                self.df['course_id'] = self.df.index.astype(str)

        # Normalize title variations
        if 'title' not in self.df.columns:
            if 'course_title' in self.df.columns:
                self.df = self.df.rename(columns={'course_title': 'title'})
            elif 'headline' in self.df.columns:
                self.df = self.df.rename(columns={'headline': 'title'})

        # Normalize instructor column
        if 'instructors' not in self.df.columns and 'instructor' in self.df.columns:
            self.df = self.df.rename(columns={'instructor': 'instructors'})

        # Normalize url
        if 'url' not in self.df.columns and 'link' in self.df.columns:
            self.df = self.df.rename(columns={'link': 'url'})
        if 'url' not in self.df.columns:
            self.df['url'] = ""

        # Ensure rating exists
        if 'rating' not in self.df.columns:
            self.df['rating'] = ""

        # Choose text fields automatically if not provided
        if text_fields is None:
            candidates = ['title', 'subtitle', 'headline', 'description', 'content', 'instructors', 'subject', 'topics']
            text_fields = [c for c in candidates if c in self.df.columns]
            if 'title' in self.df.columns and 'title' not in text_fields:
                text_fields.insert(0, 'title')
        for f in text_fields:
            if f not in self.df.columns:
                self.df[f] = ""
        self.text_fields = text_fields

        # Combined text
        self.df['__text__'] = self.df[self.text_fields].astype(str).agg(' '.join, axis=1)
        if self.df['__text__'].str.strip().replace('', np.nan).isna().all():
            if 'title' in self.df.columns:
                self.df['__text__'] = self.df['title'].astype(str)
            else:
                self.df['__text__'] = self.df['course_id'].astype(str)

        # TF-IDF
        self.vectorizer = TfidfVectorizer(stop_words='english', token_pattern=r'(?u)\b\w+\b', max_features=max_features)
        self.tfidf = self.vectorizer.fit_transform(self.df['__text__'])

        # Mappings
        self.course_id_to_index = {str(cid): idx for idx, cid in enumerate(self.df['course_id'].astype(str).tolist())}
        self.index_to_course_id = {idx: cid for cid, idx in self.course_id_to_index.items()}

    # -------------------------
    # Helpers
    # -------------------------
    def _normalize_url(self, u: str, base_url: str = "https://www.udemy.com") -> str:
        if not isinstance(u, str):
            return ""
        u = u.strip()
        if u == "":
            return ""
        if u.startswith("/"):
            return base_url.rstrip("/") + u
        if u.startswith("course/"):
            return base_url.rstrip("/") + "/" + u
        return u

    # -------------------------
    # Core recommendations
    # -------------------------
    def recommend_for_course(self, course_id: str, top_n: int = 10) -> List[Dict]:
        cid = str(course_id)
        if cid not in self.course_id_to_index:
            return []
        idx = self.course_id_to_index[cid]
        sims = cosine_similarity(self.tfidf[idx], self.tfidf).flatten()
        sims[idx] = -1
        best_idxs = np.argsort(-sims)[:top_n]
        recs = []
        for i in best_idxs:
            row = self.df.iloc[i].to_dict()
            row['score'] = float(sims[i])
            recs.append(row)
        return recs

    def recommend_for_completed(self, completed_course_ids: List[str], top_n: int = 10) -> List[Dict]:
        idxs = [self.course_id_to_index.get(str(c)) for c in completed_course_ids or []]
        idxs = [i for i in idxs if i is not None]
        if not idxs:
            fallback_cols = [c for c in ['num_reviews', 'num_revie', 'num_subs', 'num_subscribers', 'rating'] if c in self.df.columns]
            if fallback_cols:
                df_sorted = self.df.copy()
                try:
                    df_sorted[fallback_cols[0]] = pd.to_numeric(df_sorted[fallback_cols[0]], errors='coerce').fillna(0)
                    df_sorted = df_sorted.sort_values(by=fallback_cols[0], ascending=False)
                except Exception:
                    pass
                return df_sorted.head(top_n).to_dict(orient='records')
            return self.df.head(top_n).to_dict(orient='records')

        user_vec = self.tfidf[idxs].mean(axis=0)
        sims = cosine_similarity(user_vec, self.tfidf).flatten()
        for i in idxs:
            sims[i] = -1
        best_idxs = np.argsort(-sims)[:top_n]
        recs = []
        for i in best_idxs:
            row = self.df.iloc[i].to_dict()
            row['score'] = float(sims[i])
            recs.append(row)
        return recs

    def recommend_minimal(self, course_id: str, top_n: int = 10, base_url: str = "https://www.udemy.com") -> List[Dict]:
        recs = self.recommend_for_course(course_id, top_n=top_n)
        minimal = []
        for r in recs:
            title = r.get('title') or r.get('course_title') or r.get('headline') or r.get('subtitle') or ""
            url = r.get('url', "") or ""
            url = self._normalize_url(url, base_url=base_url)
            rating = r.get('rating', "")
            minimal.append({
                'title': title,
                'url': url,
                'rating': rating
            })
        return minimal

    # -------------------------
    # Field + duration based recommendation
    # -------------------------
    def recommend_by_field_duration(self, field: str, duration_months: float = None, top_n: int = 10, duration_tolerance: float = 2.0) -> List[Dict]:
        """
        Match by field (subject/title) and optionally by duration (months). Returns list of dict rows.
        """
        if not field:
            return self.df.head(top_n).to_dict(orient='records')

        field_low = str(field).strip().lower()
        subject_cols = [c for c in ['subject','topics','category','field','course_subject'] if c in self.df.columns]
        title_col = 'title' if 'title' in self.df.columns else None

        mask = pd.Series(False, index=self.df.index)
        for col in subject_cols:
            mask |= self.df[col].astype(str).str.lower().str.contains(field_low, na=False)
        if title_col:
            mask |= self.df[title_col].astype(str).str.lower().str.contains(field_low, na=False)

        candidates = self.df[mask].copy()
        if candidates.empty:
            vec = self.vectorizer.transform([field_low])
            sims = cosine_similarity(vec, self.tfidf).flatten()
            top_idxs = sims.argsort()[::-1][:top_n]
            recs = []
            for i in top_idxs:
                row = self.df.iloc[i].to_dict()
                row['score'] = float(sims[i])
                recs.append(row)
            return recs

        # Duration handling (best-effort)
        duration_cols = [c for c in ['duration','course_duration','duration_months'] if c in candidates.columns]
        if duration_months is not None and duration_cols:
            dcol = duration_cols[0]
            def parse_duration_to_months(x):
                s = str(x).lower()
                try:
                    if s.replace('.', '', 1).isdigit():
                        return float(s)
                    if 'hour' in s:
                        nums = ''.join(ch if (ch.isdigit() or ch=='.') else ' ' for ch in s).split()
                        if nums:
                            hrs = float(nums[0])
                            return round(hrs / 160.0, 3)
                    if 'week' in s:
                        nums = ''.join(ch if (ch.isdigit() or ch=='.') else ' ' for ch in s).split()
                        if nums:
                            weeks = float(nums[0])
                            return round(weeks / 4.345, 3)
                    if 'month' in s:
                        nums = ''.join(ch if (ch.isdigit() or ch=='.') else ' ' for ch in s).split()
                        if nums:
                            return float(nums[0])
                except:
                    pass
                return np.nan

            candidates['_dur_months'] = candidates[dcol].apply(parse_duration_to_months)
            low = duration_months - duration_tolerance
            high = duration_months + duration_tolerance
            filtered = candidates[(candidates['_dur_months'].notna()) & (candidates['_dur_months'] >= low) & (candidates['_dur_months'] <= high)]
            if not filtered.empty:
                sort_by = None
                if 'rating' in filtered.columns:
                    sort_by = 'rating'
                elif any(c in filtered.columns for c in ['num_reviews','num_revie','num_subs','num_subscribers']):
                    for cc in ['num_reviews','num_revie','num_subs','num_subscribers']:
                        if cc in filtered.columns:
                            sort_by = cc
                            break
                if sort_by:
                    try:
                        filtered[sort_by] = pd.to_numeric(filtered[sort_by], errors='coerce').fillna(0)
                        filtered = filtered.sort_values(by=sort_by, ascending=False)
                    except:
                        pass
                return filtered.head(top_n).to_dict(orient='records')

        # Final fallback: rank candidates by TF-IDF similarity to the field string
        vec = self.vectorizer.transform([field_low])
        sims = cosine_similarity(vec, self.tfidf).flatten()
        cand_idxs = candidates.index.to_list()
        cand_scores = [(i, sims[i]) for i in cand_idxs]
        cand_scores = sorted(cand_scores, key=lambda x: -x[1])[:top_n]
        recs = []
        for i, s in cand_scores:
            row = self.df.iloc[i].to_dict()
            row['score'] = float(s)
            recs.append(row)
        return recs

    # -------------------------
    # Persistence
    # -------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({
            'vectorizer': self.vectorizer,
            'df': self.df
        }, path)

    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        df = data.get('df')
        vectorizer = data.get('vectorizer')
        if df is None or vectorizer is None:
            raise ValueError("Saved file missing required data ('df' or 'vectorizer').")
        inst = cls(df, text_fields=[])
        inst.vectorizer = vectorizer
        inst.tfidf = inst.vectorizer.transform(inst.df['__text__'])
        inst.course_id_to_index = {str(cid): idx for idx, cid in enumerate(inst.df['course_id'].astype(str).tolist())}
        inst.index_to_course_id = {idx: cid for cid, idx in inst.course_id_to_index.items()}
        return inst
