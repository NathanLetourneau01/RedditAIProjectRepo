import os
import json
import random
import time
import subprocess
import pickle
import openai
from pathlib import Path
from vosk import Model, KaldiRecognizer
import wave
import numpy as np
from pydub import AudioSegment
from kokoro import KPipeline
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from PIL import Image, ImageDraw, ImageFont
import textwrap
import torch 

# ================= CONFIG =================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
background_videos = [r"C:\Users\tidep\Videos\output_video_only.mp4"]
output_filename = r"C:\Users\tidep\Videos\final_karaoke_video.mp4"
blender_script = r"C:\Users\tidep\karaoke_blender.py"
timestamps_file = "timestamps.json"
audio_path = "audio.mp3"
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"
CLIENT_SECRETS_FILE = "client_secrets.json"
TOKEN_FILE = "token.pickle"
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

# ================= FUNCTIONS =================
def text_to_speech(text, filename=audio_path):
    pipeline = KPipeline(lang_code='a')
    generator = pipeline(text, voice='am_adam')
    combined_audio = AudioSegment.empty()
    for gs, ps, audio_tensor in generator:
        audio_np = audio_tensor.cpu().numpy()
        audio_int16 = np.int16(audio_np * 32767)
        segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=24000,
            sample_width=2,
            channels=1
        )
        combined_audio += segment
    combined_audio.export(filename, format="mp3")
    return filename

def get_word_timestamps(audio_filename):
    if not os.path.exists(VOSK_MODEL_PATH):
        raise FileNotFoundError(f"Vosk model not found at path: {VOSK_MODEL_PATH}")
    model = Model(VOSK_MODEL_PATH)
    wav_filename = "audio.wav"
    subprocess.run([
        'ffmpeg', '-i', audio_filename, '-acodec', 'pcm_s16le',
        '-ac', '1', '-ar', '16000', wav_filename, '-y'
    ], check=True, capture_output=True, text=True)
    results = []
    with wave.open(wav_filename, "rb") as wf:
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)
        while True:
            data = wf.readframes(4000)
            if len(data) == 0: break
            if rec.AcceptWaveform(data):
                results.extend(json.loads(rec.Result()).get('result', []))
        results.extend(json.loads(rec.FinalResult()).get('result', []))
    os.remove(wav_filename)
    return results

def generate_caption(prompt):
    openai.api_key = "Blank"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

def generate_story(prompt1, partOfPrompt):
    openai.api_key = "Blank"
    prompt = f"{partOfPrompt} {prompt1} make it exactly 2100 words."
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

def generate_title(prompt):
    openai.api_key = "Blank"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

def get_authenticated_service():
    creds = None
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "rb") as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
        creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "wb") as token:
            pickle.dump(creds, token)
    return build("youtube", "v3", credentials=creds)

def upload_youtube_short(video_file, title, description="awesome", tags=None, privacy_status="public"):
    youtube = get_authenticated_service()
    request = youtube.videos().insert(
        part="snippet,status",
        body={
            "snippet": {
                "title": title,
                "description": description,
                "tags": tags,
                "categoryId": "22"
            },
            "status": {"privacyStatus": privacy_status}
        },
        media_body=MediaFileUpload(video_file, chunksize=-1, resumable=True)
    )
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"Uploading: {int(status.progress() * 100)}%")
    return response['id']

# ================= MAIN =================
def main():
    try:
        # Step 1: Story generation
        newPrompt = "create a dramatic 10-word starting story for YouTube clickbait"
        partOfPrompt = generate_caption(newPrompt)
        story = generate_story("", partOfPrompt)
        print(story)

        # Step 2: Convert story to audio
        text_to_speech(story, audio_path)

        # Step 3: Get word timestamps
        word_timings = get_word_timestamps(audio_path)
        if not word_timings: return
        with open(timestamps_file, "w") as f:
            json.dump(word_timings, f)

        # Step 4: Choose background video
        bg_video = random.choice(background_videos)

        # Step 5: Call Blender GPU rendering
        subprocess.run([
            "blender", "--background", "--python", blender_script,
            "--",
            "--audio", audio_path,
            "--timestamps", timestamps_file,
            "--background", bg_video,
            "--output", output_filename
        ], check=True)

        # Step 6: Upload to YouTube
        titlePrompt = f"create a title for this story: {story[:40]}"
        title = generate_title(titlePrompt)[:25] + "..."
        upload_youtube_short(
            output_filename,
            title=title,
            tags=["gaming", "redditstories", "storytime", "aita"]
        )

    finally:
        # Cleanup
        if os.path.exists(audio_path): os.remove(audio_path)
        if os.path.exists(timestamps_file): os.remove(timestamps_file)

if __name__ == "__main__":

    main()
