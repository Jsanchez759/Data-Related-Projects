import re
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound

def get_video_id(url):    
    pattern = r'https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None


def get_transcript(url):
    video_id = get_video_id(url)
    ytt_api = YouTubeTranscriptApi()

    try:
        transcripts = ytt_api.fetch(video_id)
        dict_transcript = transcripts.to_raw_data()

        return dict_transcript
    
    except NoTranscriptFound as e:
        return None
    except Exception as e:
        return None


def process(transcript):
    txt = ""

    for i in transcript:
        try:
            txt += f"Text: {i['text']} Start: {i['start']}\n"
        except KeyError:
            pass
    return txt
