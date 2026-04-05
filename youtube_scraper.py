import yt_dlp
import json
import pandas as pd
import time

video_urls = [
    "https://youtu.be/MDN_XVcdCLo?si=zPGLY72A4Le3umnx",  
    "https://youtu.be/K9NdlkcqhY0?si=HX1ai_xjGvq4xSyk",
    "https://youtu.be/RudrWy9uPZE?si=GQLBVXiybvDY1Dnr",
    "https://youtu.be/Ffh9OeJ7yxw?si=1M8V-fvOchepjLEu",
    "https://youtu.be/m54t8xx13Uk?si=U30zXEl1iWrnL7fh"

]

def get_video_comments(video_url):
    ydl_opts = {
    'quiet': True,
    'extract_flat': False,
    'getcomments': True,   
    'writesubtitles': False,
    'writeautomaticsub': False,
}
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        video_id = info.get('id', '')
        title = info.get('title', '')
        upload_date = info.get('upload_date', '')
        views = info.get('view_count', 0)
        likes = info.get('like_count', 0)

        comments = []
        
        if 'comments' in info:
            for comment in info['comments']:
                comments.append({
                    "platform": "YouTube",
                    "video_id": video_id,
                    "video_title": title,
                    "upload_date": upload_date,
                    "views": views,
                    "likes": likes,
                    "author": comment.get('author', ''),
                    "text": comment.get('text', ''),
                    "like_count": comment.get('like_count', 0),
                    "date": comment.get('timestamp', '')
                })
        return comments

all_comments = []
for url in video_urls:
    print(f"Scraping {url}")
    comments = get_video_comments(url)
    all_comments.extend(comments)
    time.sleep(2)  


with open("youtube_data.json", "w", encoding="utf-8") as f:
    json.dump(all_comments, f, ensure_ascii=False, indent=4)

print(f"Saved {len(all_comments)} comments to youtube_data.json")


df = pd.DataFrame(all_comments)
df.to_csv("youtube_data.csv", index=False)
print("Also saved to youtube_data.csv")