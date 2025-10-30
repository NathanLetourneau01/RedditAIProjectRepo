import os
import math
import datetime
import subprocess
from collections import defaultdict
import httplib2
import skia
import moderngl
from PIL import Image
from tqdm import tqdm
import random
import os
import time
import json
import wave
import subprocess
import openai
import requests
from pathlib import Path
from gtts import gTTS
from moviepy.video.VideoClip import TextClip, ColorClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from tqdm import tqdm
from vosk import Model, KaldiRecognizer
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from kokoro import KPipeline
from IPython.display import Audio, display
from pydub import AudioSegment
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import random
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image, ImageDraw, ImageFont
import textwrap
import torch.cuda
from pydub import AudioSegment
import os
from PIL import Image, ImageDraw, ImageFont
import textwrap
import subprocess
import schedule
from openai.error import APIConnectionError, ServiceUnavailableError, Timeout
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
def generate_storyLONG(prompt1, partOfPrompt):
    openai.api_key = "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"
    prompt = partOfPrompt + " make sure to make the story about that and here are the parameters "+ prompt1 + " make sure it is exactly 2500 words"
    retries = 3
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=[{"role": "user", "content": prompt}]
            )
            thing = response['choices'][0]['message']['content']
        
        except (APIConnectionError, ServiceUnavailableError, Timeout) as e:
            print(f"API error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2  # exponential backoff
        except Exception as e:
            print(f"Unexpected error: {e}")
            break
    if len(thing) < 400:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=[{"role": "user", "content": prompt}]
            )
            thing = response['choices'][0]['message']['content']
        
        except (APIConnectionError, ServiceUnavailableError, Timeout) as e:
            print(f"API error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2  # exponential backoff
        except Exception as e:
            print(f"Unexpected error: {e}")
    prompt2 = "this is the prompt " + partOfPrompt + "and this is the story you need to add on to add EXACTLY 1400 words length is crutial never simplify and always expand including at least 1 update section and a conclusion that leaves both sides hating eachother as is defined in the origional prompt." + thing[1200:] + " just add on do not rewrite or say anything else"
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=[{"role": "user", "content": prompt2}]
            )
            thing2 = response['choices'][0]['message']['content']
        
        except (APIConnectionError, ServiceUnavailableError, Timeout) as e:
            print(f"API error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2  # exponential backoff
        except Exception as e:
            print(f"Unexpected error: {e}")
            break
    thing3 = thing + thing2
    return thing3
# Step 4: Extract and print the response
def generate_captionLONG(newPrompt):
    openai.api_key = "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"
    retries = 5
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=[{"role": "user", "content": newPrompt}]
            )
            return response['choices'][0]['message']['content']
        
        except (APIConnectionError, ServiceUnavailableError, Timeout) as e:
            print(f"API error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2  # exponential backoff
        except Exception as e:
            print(f"Unexpected error: {e}")
            break
    return None
import os
import requests

from kokoro import KPipeline
from pydub import AudioSegment
import numpy as np
import torch

def text_to_speechLONG(text, filename="audio.mp3"):
    """
    Generate TTS using Kokoro and save as a single MP3 file.

    Args:
        text (str): The text to convert to speech.
        filename (str): Output MP3 file name.

    Returns:
        str: Path to the generated MP3 file.
    """
    # Initialize TTS pipeline
    pipeline = KPipeline(lang_code='a')
    
    # Run the TTS generator
    generator = pipeline(text, voice='am_adam')
    
    # Initialize empty audio segment
    combined_audio = AudioSegment.empty()
    
    # Iterate through the generated audio chunks
    for i, (gs, ps, audio_tensor) in enumerate(generator):
        # Convert tensor to numpy
        audio_np = audio_tensor.cpu().numpy()
        
        # Convert float32 [-1,1] → int16 to avoid distortion
        audio_int16 = np.int16(audio_np * 32767)
        
        # Convert numpy array to pydub AudioSegment
        segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=24000,
            sample_width=2,  # 16-bit PCM
            channels=1
        )
        
        # Append chunk to combined audio
        combined_audio += segment
    
    # Export final MP3
    combined_audio.export(filename, format="mp3")
    
    return filename
def generate_subway_thumbnailLONG(video_path, title, output_path="thumbnail.jpg", overlay_image_path=None):
    """
    Generate a YouTube thumbnail using a frame from a Subway Surfers video as background,
    with both the main title and "Reddit AITA Stories" bolded.
    Main title wraps slightly longer lines, is positioned 2% above center, 
    truncates at word boundary if over 100 characters, and aligns to the overlay left.
    """
    width, height = 1280, 720

    # Truncate title if over 100 characters without cutting mid-word
    if len(title) > 100:
        truncated = title[:100]  # get first 100 chars
        if ' ' in truncated:
            truncated = truncated.rsplit(' ', 1)[0]  # remove last partial word
        title = truncated + "..."

    # Extract a random frame from the video
    clip = VideoFileClip(video_path)
    frame_time = random.uniform(0, clip.duration)
    frame = clip.get_frame(frame_time)  # numpy array (H,W,C)
    frame_img = Image.fromarray(frame).resize((width, height))

    draw = ImageDraw.Draw(frame_img)

    overlay_box = None
    if overlay_image_path:
        try:
            overlay = Image.open(overlay_image_path).convert("RGBA")

            # Scale overlay: 75% width × 60% height
            target_w = int(width * 0.75)
            target_h = int(height * 0.6)
            overlay = overlay.resize((target_w, target_h), Image.LANCZOS)

            # Center the overlay
            pos_x = (width - target_w) // 2
            pos_y = (height - target_h) // 2
            overlay_box = (pos_x, pos_y, pos_x + target_w, pos_y + target_h)

            frame_img.paste(overlay, (pos_x, pos_y), overlay)  # keep transparency
        except Exception as e:
            print(f"⚠️ Could not add overlay image: {e}")

    # Load bold fonts
    try:
        font_main = ImageFont.truetype("arialbd.ttf", 50)  # main title bold
        font_header = ImageFont.truetype("arialbd.ttf", 50)  # header bold
    except:
        font_main = ImageFont.load_default()
        font_header = ImageFont.load_default()

    shadow_offset = 2

    # --- Helper function for wrapped text (left-aligned on overlay) ---
    def draw_multiline_textLONG(draw, text, font, max_chars_per_line, start_x, start_y, fill):
        wrapper = textwrap.TextWrapper(width=max_chars_per_line)
        lines = wrapper.wrap(text=text)
        y_offset = start_y
        for line in lines:
            # Draw shadow
            draw.text((start_x + shadow_offset, y_offset + shadow_offset), line, font=font, fill=(255,255,255))
            # Draw main text
            draw.text((start_x, y_offset), line, font=font, fill=fill)
            bbox = draw.textbbox((0,0), line, font=font)
            line_h = bbox[3] - bbox[1]
            y_offset += line_h + 10  # spacing between lines

    # --- Main Title Text (on overlay, left-aligned, auto-wrapped) ---
    main_text = f"“{title}”"
    max_chars_per_line = 34  # wrap 2 characters longer

    if overlay_box:
        start_x = overlay_box[0] + 40  # padding from left edge of overlay
        overlay_center_y = (overlay_box[1] + overlay_box[3]) // 2
        start_y = overlay_center_y - int((overlay_box[3]-overlay_box[1])*0.13)  # 13% above center
    else:
        start_x = 50  # fallback margin if no overlay
        start_y = height // 2 - int(height * 0.13)

    draw_multiline_textLONG(draw, main_text, font_main, max_chars_per_line, start_x=start_x, start_y=start_y, fill=(0,0,0))

    # --- Header Text "Reddit AITA Stories" ---
    header_text = "Reddit AITA Stories"
    bbox_header = draw.textbbox((0, 0), header_text, font=font_header)
    header_w = bbox_header[2] - bbox_header[0]
    header_h = bbox_header[3] - bbox_header[1]

    # Position: 30% from top, 35% from right
    y_header = int(height * 0.30) - header_h // 2
    x_header = width - int(width * 0.35) - header_w

    # Draw shadow and bold header text
    draw.text((x_header+shadow_offset, y_header+shadow_offset), header_text, font=font_header, fill=(255,255,255))
    draw.text((x_header, y_header), header_text, font=font_header, fill=(0,0,0))

    # Save thumbnail
    frame_img.save(output_path, "JPEG")
    
    return output_path
# 3. NEW: Function to get word-level timestamps using Vosk
def get_word_timestampsLONG(audio_filename, VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"):
    if not os.path.exists(VOSK_MODEL_PATH):
        raise FileNotFoundError(f"Vosk model not found at path: {VOSK_MODEL_PATH}")
       
    model = Model(VOSK_MODEL_PATH)
   
    # Convert audio to the format Vosk needs (16-bit PCM WAV)
    wav_filename = "audio.wav"
    subprocess.run([
        'ffmpeg', '-i', audio_filename, '-acodec', 'pcm_s16le',
        '-ac', '1', '-ar', '16000', wav_filename, '-y'
    ], check=True, capture_output=True, text=True) # Added text=True for better error reporting
    
    results = []
   
    # --- START OF THE FIX ---
    # Use a 'with' statement to automatically manage opening and closing the file.
    # The file is guaranteed to be closed once the code exits this block.
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
    # --- The file 'wf' is now automatically and safely closed ---
    # --- END OF THE FIX ---
   
    # Now that the 'with' block is finished, this delete command will work.
    os.remove(wav_filename)
   
    return results
# 4. REWRITTEN: Main video creation function with precise synchronization
from collections import defaultdict
import math

# === Step 1: GPU-accelerated text rendering ===

def create_mgl_contextLONG():
    return moderngl.create_standalone_context()

def render_word_imageLONG(word, font_size=180, stroke_width=15, font_name='Impact'): # MODIFIED: Increased font_size and stroke_width
    """
    Renders a single word to a PIL image using Skia.
    Returns a PIL.Image in RGBA mode.
    """
    font = skia.Font(skia.Typeface(font_name), font_size)
    text_width = font.measureText(word)
    width = int(text_width + stroke_width * 8)
    height = int(font_size * 3)

    # Use CPU offscreen surface to avoid GL complications
    surface = skia.Surface(width, height)
    canvas = surface.getCanvas()
    canvas.clear(skia.ColorTRANSPARENT)

    paint_stroke = skia.Paint(
        Color=skia.ColorBLACK,
        Style=skia.Paint.kStroke_Style,
        StrokeWidth=stroke_width, # MODIFIED: Uses the new stroke_width
        AntiAlias=True,
    )
    paint_fill = skia.Paint(
        Color=skia.ColorWHITE,
        Style=skia.Paint.kFill_Style,
        AntiAlias=True,
    )

    # Draw outline first, then fill
    canvas.drawString(word, stroke_width * 2, height * 0.75, font, paint_stroke)
    canvas.drawString(word, stroke_width * 2, height * 0.75, font, paint_fill)

    img = surface.makeImageSnapshot()
    img_data = img.tobytes()
    pil_img = Image.frombytes('RGBA', (width, height), img_data)

    return pil_img


def batch_render_wordsLONG(words, output_dir, font_size=120, stroke_width=6, font_name='Impact', # MODIFIED: Updated default parameters here too
                         atlas_size=4096, padding=8, video_width=1920, video_height=1080,
                         default_x=50, default_y=50):
    """
    Render words to PNGs with coordinates that match video placement.

    Parameters:
    - video_width, video_height: size of the video frame
    - default_x, default_y: starting position on video for each word
    """
    os.makedirs(output_dir, exist_ok=True)

    font = skia.Font(skia.Typeface(font_name), font_size)
    paint_stroke = skia.Paint(
        Color=skia.ColorBLACK,
        Style=skia.Paint.kStroke_Style,
        StrokeWidth=stroke_width, # MODIFIED: Uses the new stroke_width
        AntiAlias=True,
    )
    paint_fill = skia.Paint(
        Color=skia.ColorWHITE,
        Style=skia.Paint.kFill_Style,
        AntiAlias=True,
    )

    word_map = {}

    for i, word in enumerate(words):
        text_width = font.measureText(word)
        text_height = font_size

        # Make surface just big enough for this word
        width = int(text_width + stroke_width * 4)
        height = int(text_height * 1.5)
        surface = skia.Surface(width, height)
        canvas = surface.getCanvas()
        canvas.clear(skia.ColorTRANSPARENT)

        # Draw word at top-left of its own surface
        canvas.drawString(word, stroke_width * 2, int(height * 0.75), font, paint_stroke)
        canvas.drawString(word, stroke_width * 2, int(height * 0.75), font, paint_fill)

        img = surface.makeImageSnapshot()
        pil_img = Image.frombytes('RGBA', (width, height), img.tobytes())

        # Save PNG
        png_path = os.path.join(output_dir, f"word_{i:05d}.png")
        pil_img.save(png_path)

        # Record coordinates for video overlay
        word_map[word] = {
            "path": png_path,
            "x": default_x,  # Absolute horizontal position on video
            "y": default_y,  # Absolute vertical position on video
            "w": width,
            "h": height
        }

    # Save mapping
    with open(os.path.join(output_dir, "word_map.json"), "w", encoding="utf-8") as f:
        json.dump(word_map, f, ensure_ascii=False, indent=2)


# === Step 2: Generate FFmpeg filter_complex for overlays ===

def generate_filter_complexLONG(batch_timestamps, word_to_index, segment_start):
    """
    batch_timestamps: list of dicts with keys word, start, end (times absolute)
    word_to_index: dict word->index in images folder
    segment_start: float start time of this batch in seconds
    """
    # We'll build overlay chain like:
    # [0:v][1:v] overlay=enable='between(t,start,end)' [tmp0];
    # [tmp0][2:v] overlay=enable='between(t,start,end)' [tmp1];
    # ...
    filter_complex = ""
    inputs = ""
    overlay_chain = "[0:v]"
    for i, item in enumerate(batch_timestamps):
        start = max(0, item["start"] - segment_start)
        duration = item["end"] - item["start"]
        end = start + duration
        idx = word_to_index[item["word"]]
        inputs += f" -i rendered_words/word_{idx:05d}.png"
        overlay_chain_next = f"[tmp{i}]" if i < len(batch_timestamps) - 1 else ""
        overlay_filter = (
            f"{overlay_chain}[{i+1}:v] overlay=enable='between(t,{start},{end})':x=10:y=10 {overlay_chain_next};"
        )
        filter_complex += overlay_filter + "\n"
        overlay_chain = overlay_chain_next if overlay_chain_next else overlay_chain

    # Remove trailing semicolon/newline for last overlay
    filter_complex = filter_complex.strip().rstrip(";")

    return filter_complex, inputs

# === Step 3: Run FFmpeg on batch ===

def run_ffmpeg_batchLONG(input_video, output_video, filter_complex, input_images, segment_start, segment_duration):
    # Extract segment from input video
    segment_file = f"temp_segment_{segment_start:.2f}.mp4"
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", str(segment_start),
        "-t", str(segment_duration),
        "-i", input_video,
        "-c", "copy",
        segment_file
    ], check=True)

    # Run overlay + encode with NVENC
    cmd = [
        "ffmpeg", "-y",
        "-i", segment_file
    ]
    cmd.extend(input_images.split())
    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[tmp{}]".format(len(input_images.split())),  # final overlay output label?
        "-c:v", "h264_nvenc",
        "-preset", "p1",
        "-rc", "vbr",
        "-cq", "19",
        "-b:v", "0",
        output_video
    ]) #replace p1 with if it breaks llhq
    subprocess.run(cmd, check=True)
    os.remove(segment_file)

# === Step 4: Concatenate all segments ===

def concatenate_segmentsLONG(segment_data_list, output_dir=".", batch_duration=15):
    """
    Create video segments with overlays.

    segment_data_list: list of dicts like
        {"background_video": "path.mp4", "overlays": [{"image": "word_000.png", "start": 0.5, "end": 1.2}, ...]}
    """
    os.makedirs(output_dir, exist_ok=True)
    segment_files = []

    for i, segment_data in enumerate(segment_data_list):
        segment_filename = os.path.join(output_dir, f"segment_{i:03d}.mp4")

        # Check that background video exists
        bg_video = segment_data["background_video"]
        if not os.path.exists(bg_video):
            raise FileNotFoundError(f"Background video missing: {bg_video}")

        # Check that all overlay images exist
        for overlay in segment_data.get("overlays", []):
            if not os.path.exists(overlay["image"]):
                raise FileNotFoundError(f"Overlay image missing: {overlay['image']}")

        # Build ffmpeg overlay command
        inputs = ["-i", bg_video]
        filter_complex_parts = []
        for idx, overlay in enumerate(segment_data.get("overlays", [])):
            inputs += ["-i", overlay["image"]]
            filter_complex_parts.append(
                f"[0:v][{idx+1}:v] overlay=enable='between(t,{overlay['start']},{overlay['end']})':x=10:y=10 [tmp{idx}]"
            )

        filter_complex = "; ".join(filter_complex_parts)
        map_out = f"[tmp{len(segment_data.get('overlays', []))-1}]" if segment_data.get("overlays") else "[0:v]"

        cmd = [
            "ffmpeg", "-y", *inputs,
            "-filter_complex", filter_complex,
            "-map", map_out,
            "-c:v", "h264_nvenc",
            "-preset", "llhq",
            "-rc", "vbr",
            "-cq", "19",
            "-b:v", "0",
            segment_filename
        ]

        # Run ffmpeg
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to create segment {segment_filename}\n{e}")

        # Confirm the file exists
        if not os.path.exists(segment_filename):
            raise FileNotFoundError(f"Segment file was not created: {segment_filename}")

        segment_files.append(segment_filename)

    return segment_files

# === Step 5: Main pipeline ===

import os
import shutil
# ---------------- AUTHENTICATION ----------------
def get_authenticated_serviceLONG():
    creds = None

    # Load saved credentials
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "rb") as token:
            creds = pickle.load(token)

    # Refresh or request new credentials
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)  # Opens browser automatically

        with open(TOKEN_FILE, "wb") as token:
            pickle.dump(creds, token)

    # Build YouTube service
    return build('youtube', 'v3', credentials=creds)

# ---------------- UPLOAD FUNCTION ----------------
def upload_youtube_shortLONG(video_file, title="Karaoke Short", description="awesome", tags=None, privacy_status="private", thumbnail_file=None, SCOPES=None, CLIENT_SECRETS_FILE=None, TOKEN_FILE=None, publish_time=None):
    youtube = get_authenticated_serviceLONG()
    newTitle = title
    # Truncate title if necessary
    if len(newTitle) > 25:
        truncated = newTitle[:25]
        if ' ' in truncated:
            truncated = truncated.rsplit(' ', 1)[0]
        newTitle = truncated + "..."

    # Prepare media for resumable upload (50MB chunks)
    media = MediaFileUpload(video_file, chunksize=50*1024*1024, resumable=True)

    request = youtube.videos().insert(
        part="snippet,status",
        body={
            "snippet": {
                "title": "[FULL STORY] " + newTitle + " - BEST REDDIT STORIES",
                "description": description,
                "tags": tags,
                "categoryId": "24",  # Entertainment
            },
            "status": {
                "privacyStatus": privacy_status,
                "publishAt": publish_time,
                "selfDeclaredMadeForKids": False
            }
        },
        media_body=media
    )

    print("🚀 Uploading video to YouTube Shorts...")
    response = None
    while response is None:
        try:
            status, response = request.next_chunk()
            if status:
                print(f"Uploading... {int(status.progress() * 100)}%")
        except Exception as e:
            print(f"⚠️ Error during upload, retrying in 5s: {e}")
            time.sleep(5)

    video_id = response['id']
    print(f"\n✅ Upload complete! Video ID: {video_id}")
    thumbnail_file = generate_subway_thumbnailLONG(video_file,"[FULL STORY] " + title, overlay_image_path="C:\\Users\\tidep\\redditAI\\ThumbnailTemplate.webp")
    # Upload thumbnail if provided
    if thumbnail_file:
        try:
            youtube.thumbnails().set(
                videoId=video_id,
                media_body=MediaFileUpload(thumbnail_file, mimetype='image/jpeg')
            ).execute()
            print("✅ Thumbnail uploaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to upload thumbnail: {e}")

    return video_id

# ====== MODIFY MAIN() TO CALL IT AT THE END ======
def main(SCOPES, CLIENT_SECRETS_FILE, TOKEN_FILE, publish_time):
    # Define file paths
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print("GPU is available!")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device index: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("GPU is not available. Using CPU.")
    number = random.randint(1, 7)
    if(number == 1):
        background_videos =r"C:\Users\tidep\redditAI\BackgroundVideosFolder\output_video_only.mp4"
    elif(number == 2):
        background_videos =r"C:\Users\tidep\redditAI\BackgroundVideosFolder\GTADriving1.mp4"
    elif(number == 3):
        background_videos =r"C:\Users\tidep\redditAI\BackgroundVideosFolder\Minecraft Parkour 17 Minutes No Copyright Gameplay ｜ 64 [R5lxDaRiYXI].webm"
    elif(number == 4):
        background_videos =r"C:\Users\tidep\redditAI\BackgroundVideosFolder\Minecraft Parkour Gameplay No Copyright - Free To Use (4K) [qOFX00DjjxM].webm"
    elif(number == 5):
        background_videos =r"C:\Users\tidep\redditAI\BackgroundVideosFolder\Minecraft Parkour Gameplay No Copyright (FREE TO USE) [YW7NV8J8oxI].webm"
    elif(number == 6):
        background_videos =r"C:\Users\tidep\redditAI\BackgroundVideosFolder\Minecraft Parkour Gameplay No Copyright [u7kdVe8q5zs].mp4"
    elif(number == 7):
        background_videos =r"C:\Users\tidep\redditAI\BackgroundVideosFolder\Subway Surfers Gameplay No Copyright (4K) [pIkYLpWdZXM].webm"
    output_filename = r"C:\Users\tidep\Videos\final_karaoke_video.mp4"
    story_prompt = '''
    Before writing, silently create an outline in your reasoning (do not output it) that ensures the story will add up to at exactly 2500 words MAKE SURE IT IS 2500 words always expand always never shy away from longer
    Be maximally detailed, exhaustive, and expansive. Do not summarize. Expand on every point with examples, explanations, and counterarguments. Length is critical.
    Introduction and setup (around 450–500 words)
    First conflict (around 450–500 words)
    Escalation and fallout (around 450–500 words)
    leave it on a cliffhanger
    Step 2: Writing the Story
    begin immediately with the first sentence introducing who is involved and their ages. (Example: “I (24) live with my girlfriend (25) and her mother (48) in a small apartment.”)
    Write in a natural, human Reddit voice. Use plain, everyday words. Never sound formal, exaggerated, or corny.
    Do not use any acronyms (example: write “Am I the A-hole” instead of “AITA,” write “boyfriend” instead of “BF”).
    Do not use asterisks, emojis, or internet shorthand.
    Make it dramatic, emotional, and frustrating, with real-life conflicts such as:
    Family drama (in-laws, siblings, parents interfering)
    Relationship issues (cheating, dishonesty, living arrangements, finances)
    Friends or coworkers betraying trust
    The conflict must escalate intensely, with realistic dialogue and reactions.
    Include at least 1 “Update:” sections where the narrator adds new developments that make the situation worse
    End with both sides angry at each other, and the narrator asking clearly: “Am I the asshole?”
    Step 3: Length Control
    After writing, double-check the story is exactly 2500 words.
    If it is shorter, add realistic detail or dialogue until it is exactly 2500. and make sure the grammar is simple it cant be hard to understand or complex.
    Do not include the word count in your output and please start the story with the first prompt i gave you then go into everything else'''
        # IMPORTANT: Update this path to where you extracted the Vosk model
    VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"
    #if for some reason you still cant get it to make it to the 2500 word count you can run the story through it again asking it to expand on this story then add the responce to your story you already have
    # ===============================

    # 1. Dummy story generator (no changes)
    story_text_file = "story.txt"
    audio_path = "audio.mp3"
    video_only_output = "video_with_subtitles.mp4" # Temporary video file
    
    try:
        # Step 1: Generate Story
        print("💡 Generating story prompt...")
        # (This is the completed prompt from your code)
        newPrompt = """create a starting story for a reddit AITA story it has to be very dramatic to immediately hook anyone who sees it in their youtube feed and it needs to be straight to the point. keep it to about 11 words no more dont include anything that isnt needed here is an example "AITA for fighting my brother when my girlfriend cheated with him?" or "AITA for moving out of my house when my mom hit my sister?" make it creative and SO dramatic and out of proportion"""
        
        story_title = generate_captionLONG(newPrompt)
        print(f"Generated Title: {story_title}")

        print("✍️ Generating full story...")
        full_story = story_title + generate_storyLONG(story_prompt, story_title) + " was the OP in the wrong? comment down below and make sure to like and subscribe"
        
        # Save story to a text file for reference
        with open(story_text_file, "w", encoding="utf-8") as f:
            f.write(f"Title: {story_title}\n\n")
            f.write(full_story)
        print(f"Story saved to {story_text_file}")

        # Step 2: Convert Text to Speech
        print("🎙️ Generating TTS audio...")
        text_to_speechLONG(full_story, audio_path)
        print(f"Audio saved to {audio_path}")

        # Step 3: Get Word-Level Timestamps
        print("🎤 Analyzing audio for word timestamps...")
        timestamps = get_word_timestampsLONG(audio_path)
        if not timestamps:
            raise ValueError("Failed to get word timestamps. Cannot proceed.")
        print(f"Found timestamps for {len(timestamps)} words.")
        title_word_count = len(story_title.split())
        title_timestamps = timestamps[:title_word_count]
        title_duration = max(item["end"] for item in title_timestamps)
        title_image_path = generate_subway_thumbnailShort(background_videos,"[FULL STORY] " + story_title, overlay_image_path="C:\\Users\\tidep\\redditAI\\ThumbnailTemplate.webp")
        print("🎤 Analyzing audio for word timestamps...")
        if not timestamps:
            raise ValueError("Failed to get word timestamps. Cannot proceed.")
        print(f"Found timestamps for {len(timestamps)} words.")
        # Step 4: Create the video with subtitles (but no new audio yet)
        print("🎬 Processing video to add subtitles...")
        process_videoLONG(background_videos, timestamps, video_only_output, batch_duration=5, story_title=story_title, title_duration=title_duration, title_image_path=title_image_path)
        print(f"Temporary video with subtitles created at {video_only_output}")

        # =================================================================
        # ✨ NEW: Step 5: Merge Video with TTS Audio
        # =================================================================
        print("🔊 Merging final video with TTS audio...")
        ffmpeg_command = [
            "ffmpeg",
            "-y",                   # Overwrite output file if it exists
            "-i", video_only_output, # Input 0: The video with subtitles
            "-i", audio_path,       # Input 1: The TTS audio
            "-map", "0:v",          # Select the video stream from input 0
            "-map", "1:a",          # Select the audio stream from input 1
            "-c:v", "copy",         # Copy the video stream without re-encoding
            "-shortest",            # Finish encoding when the shortest input ends
            output_filename         # The final output file path
        ]
        subprocess.run(ffmpeg_command, check=True)
        print(f"✅ Final video with audio saved to {output_filename}")

        # Step 6: Upload to YouTube
        print("⬆️ Uploading to YouTube...")
        # Ensure description is not None
        description_text = f"{story_title}\n\n#redditstories #aita #reddit #redditstorytime #askreddit #bestredditstories #amitheasshole #redditaita #redditdrama #relationshipadvice #entitledparents #tifu #prorevenge #nuclearrevenge #maliciouscompliance #aitaaita"
        tags_list = [
            "reddit",
            "aita",
            "redditstories",
            "redditstorytime",
            "askreddit",
            "bestredditstories",
            "amitheasshole",
            "redditaita",
            "redditdrama",
            "relationshipadvice",
            "entitledparents",
            "tifu",
            "prorevenge",
            "nuclearrevenge",
            "maliciouscompliance",
            "aitaaita",
            "redditupdates",
            "trueaita",
            "redditcompilation",
            "storytime",
            "redditconfessions",
            "familydrama",
            "toxicrelationships",
            "workplaceaita",
            "friendshipdrama",
            "crazyredditstories",
            "aitaop",
            "aitaopupdate",
            "redditjudgment",
            "funnyreddit",
            "topredditposts",
            "relationshipstories",
            "aitaentitled",
            "redditmoraldilemmas"
        ]

        upload_youtube_shortLONG(
            video_file=output_filename,
            title=story_title,
            description=description_text,
            tags=tags_list, SCOPES=SCOPES, CLIENT_SECRETS_FILE=CLIENT_SECRETS_FILE, TOKEN_FILE=TOKEN_FILE, publish_time=publish_time)

    except Exception as e:
        print(f"❌ An error occurred during the process: {e}")
    
    finally:
        # Step 7: Clean up temporary files
        print("🧹 Cleaning up temporary files...")
        if os.path.exists(video_only_output):
            os.remove(video_only_output)
        # You can add other temp files to clean up here, like audio.mp3
        # if os.path.exists(audio_path):
        #     os.remove(audio_path)
        print("✨ Process complete.")

# This ensures the main function runs when you execute the scrip

    # 1. Dummy story generator (no changes)
def generate_storyShort(prompt1, partOfPrompt):
    openai.api_key = "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"
    prompt = partOfPrompt + " make sure to make the story about that and here are the parameters "+ prompt1 + " make sure it is exactly 350 words"
    retries = 3
    thing = ""
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=[{"role": "user", "content": prompt}]
            )
            thing = response['choices'][0]['message']['content']
        
        except (APIConnectionError, ServiceUnavailableError, Timeout) as e:
            print(f"API error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2  # exponential backoff
        except Exception as e:
            print(f"Unexpected error: {e}")
            break
    if len(thing) < 400:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=[{"role": "user", "content": prompt}]
            )
            thing = response['choices'][0]['message']['content']
        
        except (APIConnectionError, ServiceUnavailableError, Timeout) as e:
            print(f"API error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2  # exponential backoff
        except Exception as e:
            print(f"Unexpected error: {e}")
    prompt2 = "this is the prompt " + partOfPrompt + "and this is the story you need to add on to add 100 words include 1 update section and a conclusion that leaves both sides hating eachother as is defined in the origional prompt." + thing + " just add on do not rewrite or say anything else"
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=[{"role": "user", "content": prompt2}]
            )
            thing2 = response['choices'][0]['message']['content']
        
        except (APIConnectionError, ServiceUnavailableError, Timeout) as e:
            print(f"API error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2  # exponential backoff
        except Exception as e:
            print(f"Unexpected error: {e}")
            break
    thing3 = thing + thing2
    return thing3
from PIL import Image, ImageDraw, ImageFont
import textwrap

def generate_subway_thumbnailShort(
    video_path,
    title,
    output_path="thumbnail.jpg",
    overlay_image_path="C:\\Users\\tidep\\redditAI\\ThumbnailTemplate.webp"
):
    """
    Generate a YouTube thumbnail using the overlay image as the exact canvas size.
    """
    # Load overlay first
    overlay = Image.open(overlay_image_path).convert("RGBA")
    width, height = overlay.size  # shrink canvas to overlay size

    # Use overlay as base canvas
    frame_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    frame_img.paste(overlay, (0, 0), overlay)

    draw = ImageDraw.Draw(frame_img)

    # Truncate title if over 100 characters without cutting mid-word
    if len(title) > 100:
        truncated = title[:100]
        if ' ' in truncated:
            truncated = truncated.rsplit(' ', 1)[0]
        title = truncated + "..."
    else:
        truncated = title

    # Load bold fonts
    try:
        font_main = ImageFont.truetype("arialbd.ttf", 50)
        font_header = ImageFont.truetype("arialbd.ttf", 50)
    except:
        font_main = ImageFont.load_default()
        font_header = ImageFont.load_default()

    shadow_offset = 2

    # --- Helper: Wrapped text ---
    def draw_multiline_textShort(draw, text, font, max_chars_per_line, start_x, start_y, fill):
        wrapper = textwrap.TextWrapper(width=max_chars_per_line)
        lines = wrapper.wrap(text=text)
        y_offset = start_y
        for line in lines:
            draw.text((start_x + shadow_offset, y_offset + shadow_offset),
                      line, font=font, fill=(255, 255, 255))
            draw.text((start_x, y_offset), line, font=font, fill=fill)
            bbox = draw.textbbox((0, 0), line, font=font)
            line_h = bbox[3] - bbox[1]
            y_offset += line_h + 10

    # --- Main Title ---
    main_text = f"“{title}”"
    max_chars_per_line = 34
    start_x = 40
    start_y = height // 2 - int(height * 0.13)

    draw_multiline_textShort(draw, main_text, font_main, max_chars_per_line,
                        start_x, start_y, (0, 0, 0))

    # --- Header Text ---
    header_text = "Reddit AITA Stories"
    bbox_header = draw.textbbox((0, 0), header_text, font=font_header)
    header_w = bbox_header[2] - bbox_header[0]
    header_h = bbox_header[3] - bbox_header[1]

    y_header = int(height * 0.30) - header_h // 2
    x_header = width - int(width * 0.35) - header_w

    draw.text((x_header+shadow_offset, y_header+shadow_offset),
              header_text, font=font_header, fill=(255,255,255))
    draw.text((x_header, y_header),
              header_text, font=font_header, fill=(0,0,0))

    # --- Save final image ---
    frame_img = frame_img.convert("RGB")  # flatten alpha for JPEG
    frame_img.save(output_path, "JPEG")

    print(f"✅ Subway Surfers thumbnail saved at {output_path}")
    return output_path
def convert_to_short_formatShort(input_file, output_file, duration=59):
    """
    Converts a video into vertical 9:16 format for YouTube Shorts
    and trims it to under 60 seconds.
    """
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", input_file,
            "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,"
                   "pad=1080:1920:(ow-iw)/2:(oh-ih)/2",
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "fast",
            "-c:a", "aac",
            "-b:a", "128k",
            output_file
        ]
        subprocess.run(cmd, check=True)
        print(f"✅ Converted to Shorts format: {output_file}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed to convert video to short format: {e}")
# Step 4: Extract and print the response
def generate_captionShort(newPrompt):
    openai.api_key = "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"
    retries = 5
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=[{"role": "user", "content": newPrompt}]
            )
            return response['choices'][0]['message']['content']
        
        except (APIConnectionError, ServiceUnavailableError, Timeout) as e:
            print(f"API error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2  # exponential backoff
        except Exception as e:
            print(f"Unexpected error: {e}")
            break
    return None
import os
import requests

from kokoro import KPipeline
from pydub import AudioSegment
import numpy as np
import torch

def text_to_speechShort(text, filename="audio.mp3"):
    """
    Generate TTS using Kokoro and save as a single MP3 file.

    Args:
        text (str): The text to convert to speech.
        filename (str): Output MP3 file name.

    Returns:
        str: Path to the generated MP3 file.
    """
    # Initialize TTS pipeline
    pipeline = KPipeline(lang_code='a')
    
    # Run the TTS generator
    generator = pipeline(text, voice='am_adam')
    
    # Initialize empty audio segment
    combined_audio = AudioSegment.empty()
    
    # Iterate through the generated audio chunks
    for i, (gs, ps, audio_tensor) in enumerate(generator):
        # Convert tensor to numpy
        audio_np = audio_tensor.cpu().numpy()
        
        # Convert float32 [-1,1] → int16 to avoid distortion
        audio_int16 = np.int16(audio_np * 32767)
        
        # Convert numpy array to pydub AudioSegment
        segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=24000,
            sample_width=2,  # 16-bit PCM
            channels=1
        )
        
        # Append chunk to combined audio
        combined_audio += segment
    
    # Export final MP3
    combined_audio.export(filename, format="mp3")
    
    return filename

def get_word_timestampsShort(audio_filename, VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"):
    """
    Analyzes a WAV audio file and returns word timestamps using Vosk.
    This version uses a 'with' statement to guarantee the file is closed.
    """
    if not os.path.exists(VOSK_MODEL_PATH):
        raise FileNotFoundError(f"Vosk model not found at path: {VOSK_MODEL_PATH}")
       
    model = Model(VOSK_MODEL_PATH)
   
    # Convert audio to the format Vosk needs (16-bit PCM WAV)
    wav_filename = "audio.wav"
    subprocess.run([
        'ffmpeg', '-i', audio_filename, '-acodec', 'pcm_s16le',
        '-ac', '1', '-ar', '16000', wav_filename, '-y'
    ], check=True, capture_output=True, text=True) # Added text=True for better error reporting
    
    results = []
   
    # --- START OF THE FIX ---
    # Use a 'with' statement to automatically manage opening and closing the file.
    # The file is guaranteed to be closed once the code exits this block.
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
    # --- The file 'wf' is now automatically and safely closed ---
    # --- END OF THE FIX ---
   
    # Now that the 'with' block is finished, this delete command will work.
    os.remove(wav_filename)
   
    return results
# 4. REWRITTEN: Main video creation function with precise synchronization
from collections import defaultdict
import math

# === Step 1: GPU-accelerated text rendering ===

def create_mgl_contextShort():
    return moderngl.create_standalone_context()

def render_word_imageShort(word, font_size=180, stroke_width=15, font_name='Impact'): # MODIFIED: Increased font_size and stroke_width
    """
    Renders a single word to a PIL image using Skia.
    Returns a PIL.Image in RGBA mode.
    """
    font = skia.Font(skia.Typeface(font_name), font_size)
    text_width = font.measureText(word)
    width = int(text_width + stroke_width * 8)
    height = int(font_size * 3)

    # Use CPU offscreen surface to avoid GL complications
    surface = skia.Surface(width, height)
    canvas = surface.getCanvas()
    canvas.clear(skia.ColorTRANSPARENT)

    paint_stroke = skia.Paint(
        Color=skia.ColorBLACK,
        Style=skia.Paint.kStroke_Style,
        StrokeWidth=stroke_width, # MODIFIED: Uses the new stroke_width
        AntiAlias=True,
    )
    paint_fill = skia.Paint(
        Color=skia.ColorWHITE,
        Style=skia.Paint.kFill_Style,
        AntiAlias=True,
    )

    # Draw outline first, then fill
    canvas.drawString(word, stroke_width * 2, height * 0.75, font, paint_stroke)
    canvas.drawString(word, stroke_width * 2, height * 0.75, font, paint_fill)

    img = surface.makeImageSnapshot()
    img_data = img.tobytes()
    pil_img = Image.frombytes('RGBA', (width, height), img_data)

    return pil_img


def batch_render_wordsShort(words, output_dir, font_size=120, stroke_width=6, font_name='Impact', # MODIFIED: Updated default parameters here too
                         atlas_size=4096, padding=8, video_width=1920, video_height=1080,
                         default_x=50, default_y=50):
    """
    Render words to PNGs with coordinates that match video placement.

    Parameters:
    - video_width, video_height: size of the video frame
    - default_x, default_y: starting position on video for each word
    """
    os.makedirs(output_dir, exist_ok=True)

    font = skia.Font(skia.Typeface(font_name), font_size)
    paint_stroke = skia.Paint(
        Color=skia.ColorBLACK,
        Style=skia.Paint.kStroke_Style,
        StrokeWidth=stroke_width, # MODIFIED: Uses the new stroke_width
        AntiAlias=True,
    )
    paint_fill = skia.Paint(
        Color=skia.ColorWHITE,
        Style=skia.Paint.kFill_Style,
        AntiAlias=True,
    )

    word_map = {}

    for i, word in enumerate(words):
        text_width = font.measureText(word)
        text_height = font_size

        # Make surface just big enough for this word
        width = int(text_width + stroke_width * 4)
        height = int(text_height * 1.5)
        surface = skia.Surface(width, height)
        canvas = surface.getCanvas()
        canvas.clear(skia.ColorTRANSPARENT)

        # Draw word at top-left of its own surface
        canvas.drawString(word, stroke_width * 2, int(height * 0.75), font, paint_stroke)
        canvas.drawString(word, stroke_width * 2, int(height * 0.75), font, paint_fill)

        img = surface.makeImageSnapshot()
        pil_img = Image.frombytes('RGBA', (width, height), img.tobytes())

        # Save PNG
        png_path = os.path.join(output_dir, f"word_{i:05d}.png")
        pil_img.save(png_path)

        # Record coordinates for video overlay
        word_map[word] = {
            "path": png_path,
            "x": default_x,  # Absolute horizontal position on video
            "y": default_y,  # Absolute vertical position on video
            "w": width,
            "h": height
        }

    # Save mapping
    with open(os.path.join(output_dir, "word_map.json"), "w", encoding="utf-8") as f:
        json.dump(word_map, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(words)} word images and word_map.json in {output_dir}")

# === Step 2: Generate FFmpeg filter_complex for overlays ===

def generate_filter_complexShort(batch_timestamps, word_to_index, segment_start):
    """
    batch_timestamps: list of dicts with keys word, start, end (times absolute)
    word_to_index: dict word->index in images folder
    segment_start: float start time of this batch in seconds
    """
    # We'll build overlay chain like:
    # [0:v][1:v] overlay=enable='between(t,start,end)' [tmp0];
    # [tmp0][2:v] overlay=enable='between(t,start,end)' [tmp1];
    # ...
    filter_complex = ""
    inputs = ""
    overlay_chain = "[0:v]"
    for i, item in enumerate(batch_timestamps):
        start = max(0, item["start"] - segment_start)
        duration = item["end"] - item["start"]
        end = start + duration
        idx = word_to_index[item["word"]]
        inputs += f" -i rendered_words/word_{idx:05d}.png"
        overlay_chain_next = f"[tmp{i}]" if i < len(batch_timestamps) - 1 else ""
        overlay_filter = (
            f"{overlay_chain}[{i+1}:v] overlay=enable='between(t,{start},{end})':x=10:y=10 {overlay_chain_next};"
        )
        filter_complex += overlay_filter + "\n"
        overlay_chain = overlay_chain_next if overlay_chain_next else overlay_chain

    # Remove trailing semicolon/newline for last overlay
    filter_complex = filter_complex.strip().rstrip(";")

    return filter_complex, inputs

# === Step 3: Run FFmpeg on batch ===

def run_ffmpeg_batchShort(input_video, output_video, filter_complex, input_images, segment_start, segment_duration):
    # Extract segment from input video
    segment_file = f"temp_segment_{segment_start:.2f}.mp4"
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", str(segment_start),
        "-t", str(segment_duration),
        "-i", input_video,
        "-c", "copy",
        segment_file
    ], check=True)

    # Run overlay + encode with NVENC
    cmd = [
        "ffmpeg", "-y",
        "-i", segment_file
    ]
    cmd.extend(input_images.split())
    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[tmp{}]".format(len(input_images.split())),  # final overlay output label?
        "-c:v", "h264_nvenc",
        "-preset", "p1",
        "-rc", "vbr",
        "-cq", "19",
        "-b:v", "0",
        output_video
    ]) #replace p1 with if it breaks llhq
    print("Running FFmpeg overlay + encode...")
    subprocess.run(cmd, check=True)
    os.remove(segment_file)

# === Step 4: Concatenate all segments ===

def concatenate_segmentsShort(segment_data_list, output_dir=".", batch_duration=5):
    """
    Create video segments with overlays.

    segment_data_list: list of dicts like
        {"background_video": "path.mp4", "overlays": [{"image": "word_000.png", "start": 0.5, "end": 1.2}, ...]}
    """
    os.makedirs(output_dir, exist_ok=True)
    segment_files = []

    for i, segment_data in enumerate(segment_data_list):
        segment_filename = os.path.join(output_dir, f"segment_{i:03d}.mp4")
        print("Creating segment:", segment_filename)

        # Check that background video exists
        bg_video = segment_data["background_video"]
        if not os.path.exists(bg_video):
            raise FileNotFoundError(f"Background video missing: {bg_video}")

        # Check that all overlay images exist
        for overlay in segment_data.get("overlays", []):
            if not os.path.exists(overlay["image"]):
                raise FileNotFoundError(f"Overlay image missing: {overlay['image']}")

        # Build ffmpeg overlay command
        inputs = ["-i", bg_video]
        filter_complex_parts = []
        for idx, overlay in enumerate(segment_data.get("overlays", [])):
            inputs += ["-i", overlay["image"]]
            filter_complex_parts.append(
                f"[0:v][{idx+1}:v] overlay=enable='between(t,{overlay['start']},{overlay['end']})':x=10:y=10 [tmp{idx}]"
            )

        filter_complex = "; ".join(filter_complex_parts)
        map_out = f"[tmp{len(segment_data.get('overlays', []))-1}]" if segment_data.get("overlays") else "[0:v]"

        cmd = [
            "ffmpeg", "-y", *inputs,
            "-filter_complex", filter_complex,
            "-map", map_out,
            "-c:v", "h264_nvenc",
            "-preset", "llhq",
            "-rc", "vbr",
            "-cq", "19",
            "-b:v", "0",
            segment_filename
        ]

        # Run ffmpeg
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to create segment {segment_filename}\n{e}")

        # Confirm the file exists
        if not os.path.exists(segment_filename):
            raise FileNotFoundError(f"Segment file was not created: {segment_filename}")

        segment_files.append(segment_filename)

    return segment_files

# === Step 5: Main pipeline ===

import os
import shutil
import subprocess

def process_videoLONG(input_video, timestamps, output_file, batch_duration=5, story_title=None, title_duration=0, title_image_path=None):
    """
    Main pipeline for creating a karaoke-style video:
      - Renders word images
      - Splits video into batches
      - Overlays words on each segment
      - Concatenates segments into final video
    """
    # Step 0: Prepare unique words
    unique_words = sorted({item["word"] for item in timestamps})
    word_to_index = {w: i for i, w in enumerate(unique_words)}

    # Step 1: Render all unique words
    batch_render_wordsShort(unique_words, "rendered_words")
    title_image_path = "title_image.png"
    # Step 2: Split timestamps into batches
    video_duration = max(item["end"] for item in timestamps)
    batches = []
    current_start = 0
    while current_start < video_duration:
        current_end = min(current_start + batch_duration, video_duration)
        batch_ts = [
            item for item in timestamps
            if item["start"] < current_end and item["end"] > current_start
        ]
        batches.append((current_start, current_end - current_start, batch_ts))
        current_start = current_end
    if story_title and title_duration > 0 and batches:
        title_image_path = generate_subway_thumbnailShort(
            input_video,
            "[FULL STORY] " + story_title,
            overlay_image_path="C:\\Users\\tidep\\redditAI\\ThumbnailTemplate.webp"
        )
        # Prepend thumbnail overlay to first batch
        first_batch = batches[0]
        title_overlay = {"image": title_image_path, "start": 0, "end": title_duration}

        # put thumbnail overlay *after* the words so it hides them
        batches[0] = (first_batch[0], first_batch[1], first_batch[2] + [title_overlay])
    # Step 3: Temporary folder for segment files
    segment_dir = "temp_segments"
    os.makedirs(segment_dir, exist_ok=True)
    segment_files = []

    # Step 3a: Process each batch
    for idx, (seg_start, seg_dur, batch_ts) in enumerate(batches):
        segment_path = os.path.join(segment_dir, f"segment_{idx:03d}.mp4")
        segment_files.append(segment_path)

        if not batch_ts:
            # No overlays, just copy the segment
            subprocess.run([
                "ffmpeg", "-y",
                "-ss", str(seg_start),
                "-t", str(seg_dur),
                "-i", input_video,
                "-c", "copy",
                segment_path
            ], check=True)
            continue

        # Collect overlay images and ensure they exist
        input_images = []
        existing_batch_ts = []
        for item in batch_ts:
            if "word" in item:
                idx_word = word_to_index[item["word"]]
                path = f"rendered_words/word_{idx_word:05d}.png"
            elif "image" in item:
                path = item["image"]
            else:
                continue
            if os.path.exists(path):
                input_images.append(path)
                existing_batch_ts.append(item)
            else:
                print(f"Warning: missing image {path}, skipping.")

        if not existing_batch_ts:
            # No overlays after filtering, just copy
            subprocess.run([
                "ffmpeg", "-y",
                "-ss", str(seg_start),
                "-t", str(seg_dur),
                "-i", input_video,
                "-c", "copy",
                segment_path
            ], check=True)
            continue

        # Step 3b: Build filter_complex dynamically
        last_label = "[0:v]"
        filter_lines = []

        for i, (img_path, item) in enumerate(zip(input_images, existing_batch_ts)):
            if "word" in item:
                start = max(0, item["start"] - seg_start)
                end = max(0.01, item["end"] - seg_start)
            else:  # image overlay (like title)
                start = 0
                end = seg_dur  # show for the whole batch

            next_label = f"[tmp{i}]"
            filter_lines.append(
                f"{last_label}[{i+1}:v] overlay=enable='between(t,{start},{end})':x=(main_w-overlay_w)/2:y=(main_h-overlay_h)/2 {next_label}"
            )
            last_label = next_label

        filter_complex = ";\n".join(filter_lines)

        # Step 3c: Build ffmpeg command
        cmd = ["ffmpeg", "-y", "-ss", str(seg_start), "-t", str(seg_dur), "-i", input_video]
        for img in input_images:
            cmd.extend(["-i", img])

        cmd += [
            "-filter_complex", filter_complex,
            "-map", last_label,  # final overlay
            "-map", "0:a?",      # copy audio if exists
            "-c:v", "h264_nvenc",
            "-preset", "llhq",
            "-rc", "vbr",
            "-cq", "19",
            "-b:v", "0",
            segment_path
        ]

        subprocess.run([str(x) for x in cmd], check=True)

    # Step 4: Concatenate all segments
    print("Concatenating segments...")
    segments_txt = os.path.join(segment_dir, "segments.txt")
    with open(segments_txt, "w", encoding="utf-8") as f:
        for seg in segment_files:
            # Write absolute path to avoid folder issues
            f.write(f"file '{os.path.abspath(seg)}'\n")

    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", segments_txt,
        "-c", "copy", output_file
    ], check=True)

    print(f"✅ Output saved to {output_file}")

    # Cleanup temporary segment files
    for f in segment_files + [segments_txt]:
        if os.path.exists(f):
            os.remove(f)

    # Remove temporary folder
    shutil.rmtree(segment_dir)



# ---------------- AUTHENTICATION ----------------
def get_authenticated_serviceShort():
    creds = None

    # Load saved credentials
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "rb") as token:
            creds = pickle.load(token)

    # Refresh or request new credentials
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)  # Opens browser automatically

        with open(TOKEN_FILE, "wb") as token:
            pickle.dump(creds, token)

    # Build YouTube service
    return build('youtube', 'v3', credentials=creds)

# ---------------- UPLOAD FUNCTION ----------------
def upload_youtube_shortShort(video_file, title="Karaoke Short", description="awesome", tags=None, privacy_status="private", thumbnail_file=None, SCOPES=None, CLIENT_SECRETS_FILE=None, TOKEN_FILE=None, publish_time=None):
    youtube = get_authenticated_serviceShort()
    newTitle = title
    # Truncate title if necessary
    if len(newTitle) > 40:
        truncated = newTitle[:40]
        if ' ' in truncated:
            truncated = truncated.rsplit(' ', 1)[0]
        newTitle = truncated + "..."

    # Prepare media for resumable upload (50MB chunks)
    media = MediaFileUpload(video_file, chunksize=50*1024*1024, resumable=True)

    request = youtube.videos().insert(
        part="snippet,status",
        body={
            "snippet": {
                "title": newTitle,
                "description": description,
                "tags": tags,
                "categoryId": "24",  # Entertainment
            },
            "status": {
                "privacyStatus": privacy_status,
                "publishAt": publish_time,
                "selfDeclaredMadeForKids": False
            }
        },
        media_body=media
    )

    print("🚀 Uploading video to YouTube Shorts...")
    response = None
    while response is None:
        try:
            status, response = request.next_chunk()
            if status:
                print(f"Uploading... {int(status.progress() * 100)}%")
        except Exception as e:
            print(f"⚠️ Error during upload, retrying in 5s: {e}")
            time.sleep(5)

    video_id = response['id']
    print(f"\n✅ Upload complete! Video ID: {video_id}")
    # Upload thumbnail if provided
    if thumbnail_file:
        try:
            youtube.thumbnails().set(
                videoId=video_id,
                media_body=MediaFileUpload(thumbnail_file, mimetype='image/jpeg')
            ).execute()
            print("✅ Thumbnail uploaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to upload thumbnail: {e}")

    return video_id

# ====== MODIFY MAIN() TO CALL IT AT THE END ======
def ShortsMain(SCOPES, CLIENT_SECRETS_FILE, TOKEN_FILE, publish_time):
    # Define file paths
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print("GPU is available!")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device index: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("GPU is not available. Using CPU.")
    background_videos =r"C:\Users\tidep\redditAI\BackgroundVideosFolder\GTAVIDEOFORSHORTS - Made with Clipchamp.mp4"
    output_filename = r"C:\Users\tidep\Videos\final_karaoke_video.mp4"

    story_prompt = '''
    make sure the story is about 425 words and is divided into 3 parts DO NOT LABEL ANY OF THE PARTS
    Introduction and setup (around 100 words)
    First conflict (around 150–200 words)
    Escalation and fallout (around 200 words)
    leave it on a cliffhanger
    Step 2: Writing the Story
    begin immediately with the first sentence introducing who is involved and their ages. (Example: “I (24) live with my girlfriend (25) and her mother (48) in a small apartment.”)
    Write in a natural, human Reddit voice. Use plain, everyday words. Never sound formal, exaggerated, or corny.
    Do not use any acronyms (example: write “Am I the A-hole” instead of “AITA,” write “boyfriend” instead of “BF”).
    Do not use asterisks, emojis, or internet shorthand.
    Make it dramatic, emotional, and frustrating, with real-life conflicts such as:
    Family drama (in-laws, siblings, parents interfering)
    Relationship issues (cheating, dishonesty, living arrangements, finances)
    Friends or coworkers betraying trust
    The conflict must escalate intensely, with realistic dialogue and reactions.
    Include at least 1 “Update:” sections where the narrator adds new developments that make the situation worse
    End with both sides angry at each other, and the narrator asking clearly: “Am I the asshole?”
    Step 3: Length Control
    After writing, double-check the story is exactly 425 words.
    If it is shorter, add realistic detail or dialogue until it is exactly 425. and make sure the grammar is simple it cant be hard to understand or complex.
    Do not include the word count in your output and please start the story with the first prompt i gave you then go into everything else'''
        # IMPORTANT: Update this path to where you extracted the Vosk model
    VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"
    #if for some reason you still cant get it to make it to the 2500 word count you can run the story through it again asking it to expand on this story then add the responce to your story you already have
    # ===============================
    story_text_file = "story.txt"
    audio_path = "audio.mp3"
    video_only_output = "video_with_subtitles.mp4" # Temporary video file
    
    try:
        # Step 1: Generate Story
        print("💡 Generating story prompt...")
        # (This is the completed prompt from your code)
        newPrompt = """create a starting story for a reddit AITA story it has to be very dramatic to immediately hook anyone who sees it in their youtube feed and it needs to be straight to the point make no more than 11 words. dont include anything that isnt needed here is an example "AITA for fighting my brother when my girlfriend cheated with him?" or "AITA for moving out of my house when my mom hit my sister?" make it creative and SO dramatic and out of proportion make sure its not too wordy just hit straight for dramatic and unrealistic"""
        
        story_title = generate_captionShort(newPrompt)
        print(f"Generated Title: {story_title}")

        print("✍️ Generating full story...")
        full_story = story_title + " " + generate_storyShort(story_prompt, story_title)
        # Save story to a text file for reference
        with open(story_text_file, "w", encoding="utf-8") as f:
            f.write(f"Title: {story_title}\n\n")
            f.write(full_story)
        print(f"Story saved to {story_text_file}")

        # Step 2: Convert Text to Speech
        print("🎙️ Generating TTS audio...")
        text_to_speechShort(full_story, audio_path)
        print(f"Audio saved to {audio_path}")
        timestamps = get_word_timestampsShort(audio_path)
        # Step 3: Get Word-Level Timestamps
        title_word_count = len(story_title.split())
        title_timestamps = timestamps[:title_word_count]
        title_duration = max(item["end"] for item in title_timestamps)
        title_image_path = generate_subway_thumbnailShort(background_videos,"[FULL STORY] " + story_title, overlay_image_path="C:\\Users\\tidep\\redditAI\\ThumbnailTemplate.webp")
        print("🎤 Analyzing audio for word timestamps...")
        if not timestamps:
            raise ValueError("Failed to get word timestamps. Cannot proceed.")
        print(f"Found timestamps for {len(timestamps)} words.")

        # Step 4: Create the video with subtitles (but no new audio yet)
        print("🎬 Processing video to add subtitles...")
        process_videoLONG(background_videos, timestamps, video_only_output, batch_duration=5, story_title=story_title, title_duration=title_duration, title_image_path=title_image_path)
        print(f"Temporary video with subtitles created at {video_only_output}")

        # =================================================================
        # ✨ NEW: Step 5: Merge Video with TTS Audio
        # =================================================================
        print("🔊 Merging final video with TTS audio...")
        ffmpeg_command = [
            "ffmpeg",
            "-y",                   # Overwrite output file if it exists
            "-i", video_only_output, # Input 0: The video with subtitles
            "-i", audio_path,       # Input 1: The TTS audio
            "-map", "0:v",          # Select the video stream from input 0
            "-map", "1:a",          # Select the audio stream from input 1
            "-c:v", "copy",         # Copy the video stream without re-encoding
            "-shortest",            # Finish encoding when the shortest input ends
            output_filename         # The final output file path
        ]
        subprocess.run(ffmpeg_command, check=True)
        print(f"✅ Final video with audio saved to {output_filename}")
        video_duration = max(item["end"] for item in timestamps)
        # Step 6: Upload to YouTube
        print("⬆️ Uploading to YouTube...")
        # Ensure description is not None
        shorts_output = r"C:\Users\tidep\Videos\final_youtube_short.mp4"
        convert_to_short_formatShort(output_filename, shorts_output, video_duration)

        description_text = f"{story_title}\n\n#redditstories #aita #reddit #shorts"
        tags_list = ["reddit", "aita", "redditstories", "askreddit", "storytime"]
        
        upload_youtube_shortShort(
            video_file=output_filename,
            title=story_title,
            description=description_text,
            tags=tags_list, SCOPES=SCOPES, CLIENT_SECRETS_FILE=CLIENT_SECRETS_FILE, TOKEN_FILE=TOKEN_FILE, publish_time=publish_time)

    except Exception as e:
        print(f"❌ An error occurred during the process: {e}")
    
    finally:
        # Step 7: Clean up temporary files
        print("🧹 Cleaning up temporary files...")
        if os.path.exists(video_only_output):
            os.remove(video_only_output)
        # You can add other temp files to clean up here, like audio.mp3
        # if os.path.exists(audio_path):
        #     os.remove(audio_path)
        print("✨ Process complete.")
if __name__ == "__main__":
    while True:
        today = datetime.datetime.now().date()
        print(today)
        try:
                #FOR REDDIT AITA STORIES
            SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
            CLIENT_SECRETS_FILE = "C:\\Users\\tidep\\redditAI\\Client_secrets_File\\RedditAI2.json"
            TOKEN_FILE = "C:\\Users\\tidep\\redditAI\\Pickle_jar\\redditAI.pickle"

                # Best upload times (hours and minutes)
            upload_times = [(11, 0), (15, 0), (19, 0)]  # 11:00, 15:00, 19:00
            upload_timesShorts = [
                                (7, 0),   # 7:00 AM
                                (9, 30),  # 9:30 AM
                                (11, 0),  # 11:00 AM
                                (12, 30), # 12:30 PM
                                (14, 30), # 2:30 PM
                                (16, 0),  # 4:00 PM
                                (18, 0),  # 6:00 PM
                                (20, 0),  # 8:00 PM
                                (22, 0)   # 10:00 PM
                            ]
                # Start from today
            current_day = today + datetime.timedelta(days=1)
            print(current_day)
            for i in range(2):
                for hour, minute in upload_times:
                    publish_datetime = datetime.datetime.combine(current_day,datetime.time(hour, minute, tzinfo=datetime.timezone.utc))
                    print(publish_datetime)
                    publish_time = publish_datetime.isoformat()
                    print(publish_time)
                        # Call your main functions
                    main(SCOPES, CLIENT_SECRETS_FILE, TOKEN_FILE, publish_time)
                for hour, minute in upload_timesShorts:
                    publish_datetime = datetime.datetime.combine(current_day,datetime.time(hour, minute, tzinfo=datetime.timezone.utc))
                    publish_time = publish_datetime.isoformat()
                    ShortsMain(SCOPES, CLIENT_SECRETS_FILE, TOKEN_FILE, publish_time)
                    time.sleep(180)  # Sleep for 3 minutes
                        

                    # Move to the next day
                current_day += datetime.timedelta(days=1)
                #FOR SADDECREPITGRANDMA
            SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
            CLIENT_SECRETS_FILE = "C:\\Users\\tidep\\redditAI\\Client_secrets_File\\SaddecrepitGrandma.json"
            TOKEN_FILE = "C:\\Users\\tidep\\redditAI\\Pickle_jar\\Saddecrepitgrandma.pickle"
            upload_times = [(11, 0), (15, 0), (19, 0)]  # 11:00, 15:00, 19:00
            upload_timesShorts = [
                                (7, 0),   # 7:00 AM
                                (9, 30),  # 9:30 AM
                                (11, 0),  # 11:00 AM
                                (12, 30), # 12:30 PM
                                (14, 30), # 2:30 PM
                                (16, 0),  # 4:00 PM
                                (18, 0),  # 6:00 PM
                                (20, 0),  # 8:00 PM
                                (22, 0)   # 10:00 PM
                            ]
                # Start from today
            
            current_day2 = today = datetime.datetime.now().date() + datetime.timedelta(days=1)
            for i in range(2):
                for hour, minute in upload_times:
                    publish_datetime = datetime.datetime.combine(current_day2, datetime.time(hour, minute, tzinfo=datetime.timezone.utc))
                    publish_time = publish_datetime.isoformat()

                    # Call your main functions
                main(SCOPES, CLIENT_SECRETS_FILE, TOKEN_FILE, publish_time)
                for hour, minute in upload_timesShorts:
                    publish_datetime = datetime.datetime.combine(current_day2, datetime.time(hour, minute, tzinfo=datetime.timezone.utc))
                    publish_time = publish_datetime.isoformat()
                    ShortsMain(SCOPES, CLIENT_SECRETS_FILE, TOKEN_FILE, publish_time)
                    time.sleep(180)  # Sleep for 3 minutes
                current_day2 += datetime.timedelta(days=1)
        except Exception as e:
            print("Shit broke")
            time.sleep(50000)