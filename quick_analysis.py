import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
import re
import nltk

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150  
})

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

df = pd.read_csv('combined_data_cleaned.csv', parse_dates=['date'])
print(f"Loaded {len(df)} rows")

df = df[df['full_text'].notna() & (df['full_text'].str.len() > 20)]
print(f"After filtering: {len(df)} rows")

def get_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity

df['sentiment'] = df['full_text'].apply(get_sentiment)

daily_sentiment = df.groupby(df['date'].dt.date)['sentiment'].mean().reset_index()
daily_sentiment.columns = ['date', 'avg_sentiment']

if not daily_sentiment.empty:
    plt.figure(figsize=(12, 5))
    plt.plot(pd.to_datetime(daily_sentiment['date']), daily_sentiment['avg_sentiment'], marker='o', markersize=3)
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.title('Average sentiment of Claude discourse (daily)')
    plt.xlabel('Date')
    plt.ylabel('Polarity (TextBlob)')
    plt.tight_layout()
    plt.savefig('sentiment_over_time.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(" Saved sentiment_over_time.png")

stop_words = set(stopwords.words('english'))
additional_stops = {'claude', 'ai', 'anthropic', 'like', 'just', 'get', 'can', 'use', 'would', 'also', 'one', 'even', 'though', 'said', 'still', 'got', 'could', 'llm', 'model', 'models', 'code'}
stop_words.update(additional_stops)

vectorizer = TfidfVectorizer(max_features=500, stop_words=list(stop_words))
tfidf = vectorizer.fit_transform(df['full_text'])

num_topics = 5
kmeans = KMeans(n_clusters=num_topics, random_state=42, n_init=10)
kmeans.fit(tfidf)

order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

topic_words = []
for i in range(num_topics):
    top_words = [terms[ind] for ind in order_centroids[i, :10]]
    topic_words.append(', '.join(top_words))

df['topic'] = kmeans.labels_

topic_counts = df['topic'].value_counts().sort_index()

plt.figure(figsize=(10,5))
sns.barplot(x=[f"Topic {i}\n{tw[:40]}" for i,tw in enumerate(topic_words)], y=topic_counts.values)
plt.title('Top 5 Topics Discussed')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.ylabel('Number of posts/comments')
plt.tight_layout()
plt.savefig('topics.png', dpi=150, bbox_inches='tight')
plt.close()
print(" Saved topics.png")

engagement_by_type = None
if 'content_type' in df.columns:
    engagement_by_type = df.groupby('content_type')['total_engagement'].mean().sort_values(ascending=False)
    if not engagement_by_type.empty:
        plt.figure(figsize=(8, 5))
        sns.barplot(x=engagement_by_type.values, y=engagement_by_type.index)
        plt.title('Average Engagement by Content Type')
        plt.xlabel('Avg Total Engagement (likes + comments + upvotes)')
        plt.tight_layout()
        plt.savefig('engagement_by_type.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(" Saved engagement_by_type.png")

platform_engagement = None
if 'platform' in df.columns:
    platform_engagement = df.groupby('platform')['total_engagement'].mean().sort_values(ascending=False)
    if not platform_engagement.empty:
        plt.figure(figsize=(8, 5))
        sns.barplot(x=platform_engagement.values, y=platform_engagement.index)
        plt.title('Average Engagement by Platform')
        plt.xlabel('Avg Total Engagement')
        plt.tight_layout()
        plt.savefig('engagement_by_platform.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved engagement_by_platform.png")

positive_df = df[df['sentiment'] > 0.2]
negative_df = df[df['sentiment'] < -0.2]

def get_common_words(data, n=20):
    text = ' '.join(data['full_text'].astype(str).tolist()).lower()
    words = re.findall(r'\b[a-z]{3,}\b', text)
    words = [w for w in words if w not in stop_words]
    return Counter(words).most_common(n)

pos_words = get_common_words(positive_df)
neg_words = get_common_words(negative_df)

print("\n Most common words in POSITIVE sentiment posts:")
for w, c in pos_words[:10]:
    print(f"  {w}: {c}")

print("\n Most common words in NEGATIVE sentiment posts:")
for w, c in neg_words[:10]:
    print(f"  {w}: {c}")

print("\n" + "="*50)
print("KEY INSIGHTS FROM CLAUDE DISCOURSE")
print("="*50)

if not daily_sentiment.empty:
    max_date = daily_sentiment.loc[daily_sentiment['avg_sentiment'].idxmax(), 'date']
    min_date = daily_sentiment.loc[daily_sentiment['avg_sentiment'].idxmin(), 'date']
    print(f"1. Peak positive sentiment (in this sample) on {max_date} — cross-check vs Anthropic releases / news.")
    print(f"2. Most negative sentiment (in this sample) on {min_date} — sample top posts that day to explain why.")

if engagement_by_type is not None and not engagement_by_type.empty:
    top_type = engagement_by_type.index[0]
    print(f"3. {top_type} posts receive the highest engagement (avg {engagement_by_type.iloc[0]:.1f} interactions)")

if platform_engagement is not None and not platform_engagement.empty:
    top_platform = platform_engagement.index[0]
    print(f"4. {top_platform} drives the most engagement – focus growth efforts there")

topic_counts_sorted = topic_counts.sort_values(ascending=False)
most_common_topic = topic_counts_sorted.index[0]
print(f"5. Most discussed topic: {topic_words[most_common_topic][:60]}...")

print("\n Analysis complete. Charts saved as PNG files.")