# demo.py
import pandas as pd
from recommender import CourseRecommender

CSV = "data/udemy_courses.csv"

def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    # rename common column names to expected ones
    if 'title' not in df.columns and 'course_title' in df.columns:
        df = df.rename(columns={'course_title': 'title'})
    # make sure url has full domain
    if 'url' in df.columns:
        df['url'] = df['url'].apply(lambda u: 'https://www.udemy.com' + u if isinstance(u, str) and u.startswith('/') else u)
    # if rating missing, fill with 0
    if 'rating' not in df.columns:
        df['rating'] = 0
    return df

if __name__ == "__main__":
    df = load_and_prepare(CSV)
    print("Columns detected:", df.columns.tolist()[:20])

    rec = CourseRecommender(df)
    sample_id = df.iloc[0].get('id') or df.iloc[0].get('course_id')

    print(f"\n🎯 Showing recommendations for course id: {sample_id}\n")
    recs = rec.recommend_for_course(sample_id, top_n=6)

    # --- print only title, url, rating ---
    for r in recs:
        title = r.get('title', 'N/A')
        url = r.get('url', 'N/A')
        rating = r.get('rating', 'N/A')
        print(f"• {title}\n  URL: {url}\n  ⭐ Rating: {rating}\n")
