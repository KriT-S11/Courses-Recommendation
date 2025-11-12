# recommender.py
"""
CourseRecommender
- Load a Udemy-style CSV (pandas DataFrame expected)
- Build TF-IDF on combined text fields and compute cosine-similarity recommendations
- Expose:
    - recommend_for_course(course_id, top_n)
    - recommend_for_completed(list_of_course_ids, top_n)
    - recommend_minimal(course_id, top_n) -> only title, url (absolute), rating

Usage:
    from recommender import CourseRecommender
    rec = CourseRecommender(df)          # df is a pandas.DataFrame loaded from CSV
    recs = rec.recommend_minimal('567828', top_n=6)
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
        """
        df: pandas DataFrame loaded from udemy CSV. This constructor will normalize common columns.
        text_fields: list of columns to combine for TF-IDF. If None, auto-selects sensible columns.
        max_features: max TF-IDF features
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        self.df = df.copy().reset_index(drop=True).fillna("")

        # Normalize id column: prefer 'course_id' or 'id'
        if 'course_id' not in self.df.columns:
            if 'id' in self.df.columns:
                self.df = self.df.rename(columns={'id': 'course_id'})
            else:
                # fallback to index-based ids
                self.df['course_id'] = self.df.index.astype(str)

        # Normalize some common alternative column names
        if 'title' not in self.df.columns:
            if 'course_title' in self.df.columns:
                self.df = self.df.rename(columns={'course_title': 'title'})
            elif 'headline' in self.df.columns:
                self.df = self.df.rename(columns={'headline': 'title'})

        # Instructor column normalization
        if 'instructors' not in self.df.columns and 'instructor' in self.df.columns:
            self.df = self.df.rename(columns={'instructor': 'instructors'})

        # Ensure url column exists
        if 'url' not in self.df.columns and 'link' in self.df.columns:
            self.df = self.df.rename(columns={'link': 'url'})
        if 'url' not in self.df.columns:
            self.df['url'] = ""

        # Ensure rating exists
        if 'rating' not in self.df.columns:
            self.df['rating'] = ""

        # Auto-select text fields if not provided
        if text_fields is None:
            candidates = ['title', 'subtitle', 'headline', 'description', 'content', 'instructors', 'subject']
            text_fields = [c for c in candidates if c in self.df.columns]
            # ensure title is included if present
            if 'title' in self.df.columns and 'title' not in text_fields:
                text_fields.insert(0, 'title')
        # Guarantee each text field exists in df
        for f in text_fields:
            if f not in self.df.columns:
                self.df[f] = ""

        self.text_fields = text_fields

        # Combine text
        # convert everything to str (some numeric columns could be present)
        self.df['__text__'] = self.df[self.text_fields].astype(str).agg(' '.join, axis=1)

        # If entire column is empty, fallback to course_id or title
        if self.df['__text__'].str.strip().replace('', np.nan).isna().all():
            if 'title' in self.df.columns:
                self.df['__text__'] = self.df['title'].astype(str)
            else:
                self.df['__text__'] = self.df['course_id'].astype(str)

        # Build TF-IDF
        self.vectorizer = TfidfVectorizer(stop_words='english', token_pattern=r'(?u)\b\w+\b', max_features=max_features)
        self.tfidf = self.vectorizer.fit_transform(self.df['__text__'])

        # Build mappings
        self.course_id_to_index = {str(cid): idx for idx, cid in enumerate(self.df['course_id'].astype(str).tolist())}
        self.index_to_course_id = {idx: cid for cid, idx in self.course_id_to_index.items()}

    # -------------------------
    # Helper methods
    # -------------------------
    def _normalize_url(self, u: str, base_url: str = "https://www.udemy.com") -> str:
        """Ensure url is absolute. If relative like '/course/..' prepend base_url."""
        if not isinstance(u, str):
            return ""
        u = u.strip()
        if u == "":
            return ""
        if u.startswith("/"):
            return base_url.rstrip("/") + u
        # some CSVs have incomplete urls like 'course/...', handle that:
        if u.startswith("course/"):
            return base_url.rstrip("/") + "/" + u
        return u

    # -------------------------
    # Recommendation methods
    # -------------------------
    def recommend_for_course(self, course_id: str, top_n: int = 10) -> List[Dict]:
        """Return top_n similar course rows (full row dicts) for a given course_id. Includes a 'score' field."""
        cid = str(course_id)
        if cid not in self.course_id_to_index:
            return []
        idx = self.course_id_to_index[cid]
        sims = cosine_similarity(self.tfidf[idx], self.tfidf).flatten()
        sims[idx] = -1  # exclude itself
        best_idxs = np.argsort(-sims)[:top_n]
        recs = []
        for i in best_idxs:
            row = self.df.iloc[i].to_dict()
            row['score'] = float(sims[i])
            recs.append(row)
        return recs

    def recommend_for_completed(self, completed_course_ids: List[str], top_n: int = 10) -> List[Dict]:
        """
        Aggregate TF-IDF vectors of completed courses and return top_n courses.
        If completed_course_ids is empty or none found, fall back to popular sorting if available.
        """
        idxs = [self.course_id_to_index.get(str(c)) for c in completed_course_ids or []]
        idxs = [i for i in idxs if i is not None]
        if not idxs:
            # fallback: use num_reviews or rating if present
            fallback_cols = [c for c in ['num_reviews', 'num_revie', 'num_subs', 'num_subscribers', 'rating'] if c in self.df.columns]
            if fallback_cols:
                df_sorted = self.df.copy()
                # Try numeric sorting for first fallback column
                try:
                    df_sorted[fallback_cols[0]] = pd.to_numeric(df_sorted[fallback_cols[0]], errors='coerce').fillna(0)
                    df_sorted = df_sorted.sort_values(by=fallback_cols[0], ascending=False)
                except Exception:
                    df_sorted = df_sorted
                return df_sorted.head(top_n).to_dict(orient='records')
            else:
                return self.df.head(top_n).to_dict(orient='records')

        user_vec = self.tfidf[idxs].mean(axis=0)
        sims = cosine_similarity(user_vec, self.tfidf).flatten()
        for i in idxs:
            sims[i] = -1  # exclude already completed
        best_idxs = np.argsort(-sims)[:top_n]
        recs = []
        for i in best_idxs:
            row = self.df.iloc[i].to_dict()
            row['score'] = float(sims[i])
            recs.append(row)
        return recs

    def recommend_minimal(self, course_id: str, top_n: int = 10, base_url: str = "https://www.udemy.com") -> List[Dict]:
        """
        Return minimal results for frontend: only title, url (absolute), rating.
        """
        recs = self.recommend_for_course(course_id, top_n=top_n)
        minimal = []
        for r in recs:
            # title may be under several names, try common ones
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
    # Persistence
    # -------------------------
    def save(self, path: str):
        """
        Save model metadata (vectorizer and dataframe) so it can be reloaded.
        Note: we re-transform tfidf on load for robustness.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({
            'vectorizer': self.vectorizer,
            'df': self.df
        }, path)

    @classmethod
    def load(cls, path: str):
        """
        Load previously saved model. Rebuilds TF-IDF matrix by transforming stored '__text__'.
        """
        data = joblib.load(path)
        df = data.get('df')
        vectorizer = data.get('vectorizer')
        if df is None or vectorizer is None:
            raise ValueError("Saved file missing required data ('df' or 'vectorizer').")
        # create instance
        inst = cls(df, text_fields=[])
        inst.vectorizer = vectorizer
        inst.tfidf = inst.vectorizer.transform(inst.df['__text__'])
        inst.course_id_to_index = {str(cid): idx for idx, cid in enumerate(inst.df['course_id'].astype(str).tolist())}
        inst.index_to_course_id = {idx: cid for cid, idx in inst.course_id_to_index.items()}
        return inst
