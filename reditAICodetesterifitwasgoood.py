import random
import os
import time
import json
import wave
import subprocess
import openai
from pathlib import Path
from gtts import gTTS
from tqdm import tqdm
from vosk import Model, KaldiRecognizer
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.VideoClip import TextClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip


# ======= CONFIG =======
background_videos = [r"C:\Users\tidep\subway_surfer.mp4.mp4"]
output_filename = r"C:\Users\tidep\Videos\final_karaoke_video.mp4"
story_prompt = "Write a funny Reddit-style story as if posted on r/TIFU make it around 45 seconds if read out loud and make sure to make it super exciting and spell out every acronym as words"
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"

# YouTube API config
CLIENT_SECRETS_FILE = "client_secrets.json"
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
TOKEN_FILE = "token.json"

# ======= STORY GENERATOR =======
def generate_story(prompt):
    openai.api_key = "Blank"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

# ======= TEXT TO SPEECH =======
def text_to_speech(text, filename="audio.mp3"):
    tts = gTTS(text)
    tts.save(filename)
    return filename

# ======= GET WORD TIMESTAMPS =======
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
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                part_result = json.loads(rec.Result())
                results.extend(part_result.get('result', []))
        part_result = json.loads(rec.FinalResult())
        results.extend(part_result.get('result', []))
    os.remove(wav_filename)
    return results

# ======= CREATE KARAOKE VIDEO =======
def create_karaoke_video(audio_file, story_text, word_timestamps, background_video):
    audio = AudioFileClip(audio_file)
    video_duration = audio.duration
    background = VideoFileClip(background_video).subclipped(0, video_duration)
    background = resize(background, height=1920)
    clips = [background]

    for word_info in tqdm(word_timestamps, desc="Aligning words"):
        word = word_info['word']
        start_time = word_info['start']
        end_time = word_info['end']
        duration = end_time - start_time

        highlight_clip = (TextClip(
            text=word, font=r"C:\Windows\Fonts\arialbd.ttf", font_size=60, color="white",
            stroke_color='black', stroke_width=2, duration=duration,
            method='caption', size=(800, 200))
            .with_start(start_time)
            .with_duration(duration)
            .with_position(("center", "center"))
        )
        clips.append(highlight_clip)

    final = CompositeVideoClip(clips).with_audio(audio)
    final.write_videofile(output_filename, fps=60, codec='libx264', audio_codec='aac')

# ======= YOUTUBE UPLOAD =======
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

def get_authenticated_service():
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
    creds = flow.run_local_server(port=0)
    return build('youtube', 'v3', credentials=creds)

def upload_youtube_short(video_file, title="Karaoke Short", description="Fun karaoke story!", privacy_status="private"):
    youtube = get_authenticated_service()
    request_body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": ["shorts", "karaoke", "funny"],
            "categoryId": "22"
        },
        "status": {
            "privacyStatus": privacy_status,
            "selfDeclaredMadeForKids": False
        }
    }
    media_file = MediaFileUpload(video_file)
    request = youtube.videos().insert(part="snippet,status", body=request_body, media_body=media_file)
    response = request.execute()
    print(f"✅ Uploaded as YouTube Short! Video ID: {response['id']}")
    return response['id']
def resize_for_shorts(input_file, output_file):
    import subprocess
    subprocess.run([
        "ffmpeg",
        "-i", input_file,
        "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2",
        "-c:a", "copy",
        output_file,
        "-y"  # Overwrite if exists
    ])
# ======= MAIN FLOW =======
def main():
    audio_path = "audio.mp3"
    try:
        print("Generating story...")
        story = generate_story(story_prompt)
        print("\nGenerated Story:\n", story)
        time.sleep(0.3)

        print("\nConverting story to speech...")
        text_to_speech(story, audio_path)
        time.sleep(0.3)

        print("\nAnalyzing speech for word timings...")
        word_timings = get_word_timestamps(audio_path)
        if not word_timings:
            print("Could not extract word timings. Exiting.")
            return
        time.sleep(0.3)

        print("\nCreating final video...")
        bg_video = random.choice(background_videos)
        create_karaoke_video(audio_path, story, word_timings, bg_video)
        resize_for_shorts(video_file, output_filename)
        print(f"\n✅ Done! Check your folder for '{output_filename}'")

        # ======= Upload as YouTube Short =======
        print("\nUploading video as YouTube Short...")
        upload_youtube_short(output_filename, title="Karaoke Short", description=story)

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
            print(f"\n🧹 Cleaned up temporary file")

if __name__ == "__main__":

    main()
