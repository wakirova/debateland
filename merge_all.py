import json
import pandas as pd

all_data = []

try:
    with open('reddit_posts.json', 'r', encoding='utf-8') as f:
        reddit_posts = json.load(f)
        all_data.extend(reddit_posts)
        print(f"Reddit posts: {len(reddit_posts)}")
except: print("No reddit_posts.json")

try:
    with open('reddit_comments.json', 'r', encoding='utf-8') as f:
        reddit_comments = json.load(f)
        all_data.extend(reddit_comments)
        print(f"Reddit comments: {len(reddit_comments)}")
except: print("No reddit_comments.json")

try:
    with open('youtube_data.json', 'r', encoding='utf-8') as f:
        youtube = json.load(f)
        all_data.extend(youtube)
        print(f"YouTube: {len(youtube)}")
except: print("No youtube_data.json")

try:
    with open('hn_data.json', 'r', encoding='utf-8') as f:
        hn = json.load(f)
        all_data.extend(hn)
        print(f"Hacker News: {len(hn)}")
except: print("No hn_data.json")

df = pd.DataFrame(all_data)


df = df.drop_duplicates(subset=['platform', 'url', 'date', 'text'], keep='first')
print(f"After dedup: {len(df)} rows")

raw_date = df['date'].copy() if 'date' in df.columns else None
df['date'] = pd.NaT

# Reddit: unix seconds in created_utc (avoid overflow: only convert plausible epochs)
if 'created_utc' in df.columns:
    cu = pd.to_numeric(df['created_utc'], errors='coerce')
    sec = cu.where(cu.isna() | (cu <= 1e12), cu / 1000.0)
    ok = sec.notna() & (sec >= 946684800) & (sec <= 4102444800)
    if ok.any():
        parsed = pd.to_datetime(sec[ok], unit='s', errors='coerce', utc=True)
        df.loc[ok, 'date'] = parsed.dt.tz_localize(None)

# YouTube (numeric epoch in date), HN (ISO string in date), etc.
if raw_date is not None:
    missing = df['date'].isna()
    if missing.any():
        sub = raw_date[missing]
        num = pd.to_numeric(sub, errors='coerce')
        unix_ok = num.notna() & (num >= 946684800) & (num <= 4102444800)
        if unix_ok.any():
            p2 = pd.to_datetime(num[unix_ok], unit='s', errors='coerce', utc=True)
            df.loc[sub.index[unix_ok], 'date'] = p2.dt.tz_localize(None)
    missing2 = df['date'].isna()
    if missing2.any():
        sub2 = raw_date[missing2]
        p3 = pd.to_datetime(sub2, errors='coerce', utc=True, format='mixed')
        df.loc[missing2, 'date'] = df.loc[missing2, 'date'].fillna(p3.dt.tz_localize(None))

df['full_text'] = (
    df.get('title', '').fillna('') + ' ' +
    df.get('selftext', '').fillna('') + ' ' +
    df.get('body', '').fillna('') + ' ' +
    df.get('text', '').fillna('')
)
df['full_text'] = df['full_text'].str.strip()

df['content_type'] = 'unknown'
df.loc[df['body'].notna(), 'content_type'] = 'comment'           
df.loc[df['selftext'].notna(), 'content_type'] = 'post'          
df.loc[df['video_id'].notna(), 'content_type'] = 'video_comment' 
df.loc[df['title'].notna() & df['selftext'].isna(), 'content_type'] = 'title_only' 


df = df[~df['full_text'].str.lower().str.contains('deleted|removed', na=False)]
df = df[df['full_text'].str.len() > 10]  
print(f"After removing empty/deleted: {len(df)} rows")

df['total_engagement'] = (
    df.get('likes', 0).fillna(0) +
    df.get('like_count', 0).fillna(0) +
    df.get('score', 0).fillna(0) +
    df.get('points', 0).fillna(0) +
    df.get('num_comments', 0).fillna(0) +
    df.get('comments_count', 0).fillna(0)
)

keep_cols = ['platform', 'content_type', 'date', 'full_text', 'total_engagement', 'url']
df_clean = df[[c for c in keep_cols if c in df.columns]]

df_clean.to_csv('combined_data_cleaned.csv', index=False, encoding='utf-8')
print(f"\n Final cleaned dataset: {len(df_clean)} rows")
print("Saved to combined_data_cleaned.csv")