import re
import os
import time
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sentence_transformers import SentenceTransformer, util
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import yt_dlp
import streamlit as st
from dotenv import load_dotenv

# Set page config as the first Streamlit command
st.set_page_config(page_title="YouTube Video Recommender", layout="wide")

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load environment variables
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
COOKIES_FILE = r"C:\Users\Raza\Downloads\yt_analyzer\cookies.txt"

def get_authenticated_service(api_key):
    """Authenticate and return YouTube API service."""
    if not api_key:
        st.error("YouTube API key not found in .env file.")
        raise ValueError("API key is required.")
    try:
        return build("youtube", "v3", developerKey=api_key)
    except Exception as e:
        st.error(f"Failed to initialize YouTube API: {e}")
        raise

def clean_text(text):
    """Clean and preprocess text for analysis."""
    if not text:
        return ""
    try:
        text = re.sub(r'\b(uh|um|like|you know)\b', '', text, flags=re.IGNORECASE)
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english')) - {'how', 'make', 'money', 'online'}
        tokens = [word for word in tokens if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return " ".join(tokens)
    except Exception as e:
        st.error(f"Error cleaning text: {e}")
        return ""

def get_video_captions(video_id, lang='en', cookies_file=None, max_retries=3):
    """Fetch and clean auto-generated captions using yt-dlp with retries and optional cookies."""
    ydl_opts = {
        'skip_download': True,
        'writeautomaticsub': True,
        'subtitleslangs': [lang],
        'subtitlesformat': 'vtt',
        'outtmpl': '%(id)s.%(ext)s',
        'quiet': True,
        'no_warnings': True,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    if cookies_file and os.path.exists(cookies_file):
        ydl_opts['cookiefile'] = cookies_file
    
    vtt_file = f"{video_id}.en.vtt"
    
    for attempt in range(max_retries):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
            
            if not os.path.exists(vtt_file):
                st.warning(f"No captions found for video {video_id}.")
                return ""
            
            with open(vtt_file, "r", encoding="utf-8") as f:
                captions = f.read()
            
            st.write(f"Raw captions for video {video_id}: {captions[:200]}...")
            
            captions = re.sub(r'WEBVTT\n.*?\n\n', '', captions)
            captions = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}.*?\n', '', captions)
            captions = re.sub(r'<\d{2}:\d{2}:\d{2}\.\d{3}><c>|</c>', '', captions)
            captions = re.sub(r'\n+', ' ', captions)
            captions = captions.strip()
            return clean_text(captions)
        
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.5)
                continue
            # st.warning(f"Error fetching captions for video {video_id}: {e}")
            return ""
        
        finally:
            if os.path.exists(vtt_file):
                os.remove(vtt_file)

def get_video_comments(_youtube, video_id):
    """Fetch comments and calculate positive comment ratio."""
    try:
        comments_response = _youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=50,
            textFormat="plainText"
        ).execute()
        
        positive_comments = 0
        total_comments = 0
        highest_rated_comment = ""
        highest_likes = -1

        for item in comments_response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]
            comment_text = comment["textDisplay"]
            like_count = comment["likeCount"]
            total_comments += 1
            
            if any(word in comment_text.lower() for word in ["great", "awesome", "good", "excellent", "amazing"]):
                positive_comments += 1
                
            if like_count > highest_likes:
                highest_likes = like_count
                highest_rated_comment = comment_text

        positive_ratio = positive_comments / total_comments if total_comments > 0 else 0
        return positive_ratio, highest_rated_comment
    except HttpError as e:
        if "quota" in str(e).lower():
            st.error(f"Quota exceeded while fetching comments for video {video_id}.")
        return 0, ""
    except Exception as e:
        st.warning(f"Error fetching comments for video {video_id}: {e}")
        return 0, ""

@st.cache_data
def search_videos(_youtube, query, max_results=5):
    """Search YouTube videos and fetch metadata."""
    try:
        search_response = _youtube.search().list(
            q=query,
            part="id,snippet",
            maxResults=max_results,
            type="video"
        ).execute()
        
        videos = []
        for item in search_response.get("items", []):
            video_id = item["id"]["videoId"]
            title = item["snippet"]["title"]
            description = item["snippet"]["description"]
            
            try:
                video_response = _youtube.videos().list(
                    part="statistics",
                    id=video_id
                ).execute()
                
                stats = video_response["items"][0]["statistics"]
                likes = int(stats.get("likeCount", 0))
                views = int(stats.get("viewCount", 0))
                
                videos.append({
                    "video_id": video_id,
                    "title": title,
                    "description": description,
                    "likes": likes,
                    "views": views
                })
            except HttpError as e:
                if "quota" in str(e).lower():
                    st.error(f"Quota exceeded while fetching stats for video {video_id}.")
                    continue
                raise
            time.sleep(random.uniform(0.3, 0.6))
        return videos
    except HttpError as e:
        st.error(f"Error searching videos: {e}")
        if "quota" in str(e).lower():
            st.error("YouTube API quota exceeded. Update the API key in .env or try again later.")
        return []
    except Exception as e:
        st.error(f"Unexpected error searching videos: {e}")
        return []

def analyze_videos(_youtube, query, videos, cookies_file=None):
    """Analyze captions, titles, and descriptions using Sentence-BERT and calculate composite scores."""
    try:
        cleaned_query = clean_text(query)
        video_data = []
        
        for video in videos:
            positive_ratio, top_comment = get_video_comments(_youtube, video["video_id"])
            captions = get_video_captions(video["video_id"], cookies_file=cookies_file)
            combined_text = clean_text(video["title"] + " " + video["description"] + " " + captions) if captions else clean_text(video["title"] + " " + video["description"])
            
            if not combined_text:
                st.warning(f"Skipping video {video['video_id']} due to empty combined text.")
                continue
                
            video_data.append({
                "video_id": video["video_id"],
                "title": video["title"],
                "description": video["description"],
                "captions": combined_text,
                "likes": video["likes"],
                "views": video["views"],
                "positive_ratio": positive_ratio,
                "top_comment": top_comment,
                "used_captions": bool(captions)
            })
            # st.write(f"Video {video['video_id']}: Used captions: {bool(captions)}")
            time.sleep(random.uniform(0.3, 0.6))
        
        if not video_data:
            st.error("No videos with usable content found.")
            return []
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode(cleaned_query)
        caption_embeddings = [model.encode(video["captions"]) for video in video_data]
        similarity_scores = [util.cos_sim(query_embedding, emb).item() for emb in caption_embeddings]
        
        max_score = max(similarity_scores) if similarity_scores else 1
        for i, video in enumerate(video_data):
            relevance_score = (similarity_scores[i] / max_score) * 100
            video["relevance_score"] = relevance_score
            video["composite_score"] = (
                0.6 * relevance_score +
                0.2 * (video["likes"] / (video["views"] + 1)) * 100 +
                0.2 * video["positive_ratio"] * 100
            )
        
        video_data.sort(key=lambda x: x["composite_score"], reverse=True)
        
        # st.write("**Debug Info**")
        # st.write(f"Cleaned Query: {cleaned_query}")
        # if video_data:
            # st.write(f"Sample Combined Text (Video 1): {video_data[0]['captions'][:200]}...")
            # st.write(f"Captions Used: {video_data[0]['used_captions']}")
        
        return video_data[:5]
    except Exception as e:
        st.error(f"Error analyzing videos: {e}")
        return []

def main():
    # Modern dark theme CSS
    st.markdown("""
        <style>
        .main {
            background-color: #1e1e2e;
            color: #cdd6f4;
            padding: 20px;
            font-family: 'Inter', sans-serif;
        }
        .stTextInput>div>input {
            background-color: #313244;
            color: #cdd6f4;
            border: 1px solid #45475a;
            border-radius: 8px;
            padding: 10px;
        }
        .stButton>button {
            background: linear-gradient(90deg, #7aa2f7, #b4befe);
            color: #1e1e2e;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #b4befe, #7aa2f7);
            transform: translateY(-2px);
        }
        .video-card {
            background-color: #313244;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            transition: transform 0.2s ease;
        }
        .video-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        }
        .video-title {
            font-size: 1.4em;
            font-weight: 700;
            color: #b4befe;
            margin-bottom: 10px;
        }
        .video-link {
            color: #7aa2f7;
            text-decoration: none;
            font-weight: 500;
        }
        .video-link:hover {
            text-decoration: underline;
        }
        .stProgress .st-bo {
            background-color: #45475a;
        }
        .stProgress .st-bo div {
            background-color: #7aa2f7;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🎥 YouTube Video Recommender")
    st.markdown("**Discover relevant YouTube videos based on captions, titles, and descriptions**", unsafe_allow_html=True)

    # Check cookies file
    if not os.path.exists(COOKIES_FILE):
        st.warning(f"Cookies file not found at {COOKIES_FILE}. Caption fetching may fail without cookies. Please provide a valid YouTube cookies file.")

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("", placeholder="Enter search query (e.g., 'how to make money')", label_visibility="collapsed")
    with col2:
        search_button = st.button("Search", use_container_width=True)

    if search_button:
        if not query.strip():
            st.error("Please enter a search query.")
            return
        
        try:
            youtube = get_authenticated_service(YOUTUBE_API_KEY)
        except Exception:
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Searching videos...")
        videos = search_videos(youtube, query)
        progress_bar.progress(30)
        
        if not videos:
            status_text.error("No videos found or API limit reached. Check your API key in .env or try 'python tutorial'.")
            progress_bar.empty()
            return
        
        status_text.text("Analyzing content...")
        relevant_videos = analyze_videos(youtube, query, videos, COOKIES_FILE)
        progress_bar.progress(100)
        
        if not relevant_videos:
            status_text.error("No relevant videos found. Ensure cookies.txt is valid or try a different query like 'python tutorial'.")
            progress_bar.empty()
            return
        
        status_text.success("Search complete!")
        progress_bar.empty()
        
        st.markdown("### Top 5 Relevant Videos")
        for i, video in enumerate(relevant_videos, 1):
            with st.container():
                st.markdown(f"<div class='video-card'>", unsafe_allow_html=True)
                st.markdown(f"<div class='video-title'>{i}. {video['title']}</div>", unsafe_allow_html=True)
                
                col_img, col_info = st.columns([1, 2])
                with col_img:
                    st.image(f"https://img.youtube.com/vi/{video['video_id']}/hqdefault.jpg", width=240)
                with col_info:
                    st.markdown(f"**Description**: {video['description'][:150]}...")
                    st.markdown(f"**Relevance Score**: {video['relevance_score']:.2f}/100")
                    st.markdown(f"**Likes**: {video['likes']:,} | **Views**: {video['views']:,}")
                    st.markdown(f"**Positive Comments**: {video['positive_ratio']:.2%}")
                    st.markdown(f"**Top Comment**: {video['top_comment'][:100]}...")
                    st.markdown(f"**Watch**: <a href='https://www.youtube.com/watch?v={video['video_id']}' class='video-link' target='_blank'>YouTube Link</a>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
