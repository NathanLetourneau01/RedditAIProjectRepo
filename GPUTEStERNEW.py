import os
import math
import subprocess
from collections import defaultdict

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
# ============ CONFIG ============
# Ensure this path is correct
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print("GPU is available!")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device index: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("GPU is not available. Using CPU.")
background_videos =r"C:\Users\tidep\Videos\output_video_only.mp4" #C:\Users\tidep\Downloads\subway_surfer.mp4, C:\Users\tidep\subway_surfer.mp4.mp4
output_filename = r"C:\Users\tidep\Videos\final_karaoke_video.mp4"
story_prompt = '''
Before writing, silently create an outline in your reasoning (do not output it) that ensures the story will add up to at exactly 1900 words MAKE SURE IT IS 1900 words always expand always never shy away from longer
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
After writing, double-check the story is exactly 1900 words.
If it is shorter, add realistic detail or dialogue until it is exactly 1900. and make sure the grammar is simple it cant be hard to understand or complex.
Do not include the word count in your output'''
    # IMPORTANT: Update this path to where you extracted the Vosk model
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"
#if for some reason you still cant get it to make it to the 2500 word count you can run the story through it again asking it to expand on this story then add the responce to your story you already have
# ===============================

# 1. Dummy story generator (no changes)
def generate_story(prompt1, partOfPrompt):
    openai.api_key = "Blank"
    prompt = partOfPrompt + " make sure to make the story about that and here are the parameters "+ prompt1 + " make sure it is exactly 2100 words"
    retries = 3
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
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
                model="gpt-3.5-turbo",
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
                model="gpt-3.5-turbo",
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
def generate_caption(newPrompt):
    openai.api_key = "Blank"
    retries = 5
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
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

def text_to_speech(text, filename="audio.mp3"):
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
def generate_subway_thumbnail(video_path, title, output_path="thumbnail.jpg", overlay_image_path=None):
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
    def draw_multiline_text(draw, text, font, max_chars_per_line, start_x, start_y, fill):
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

    draw_multiline_text(draw, main_text, font_main, max_chars_per_line, start_x=start_x, start_y=start_y, fill=(0,0,0))

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
    print(f"✅ Subway Surfers thumbnail saved at {output_path}")
    return output_path
# 3. NEW: Function to get word-level timestamps using Vosk
def get_word_timestamps(audio_filename):
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

def create_mgl_context():
    return moderngl.create_standalone_context()

def render_word_image(word, font_size=100, stroke_width=6, font_name='Arial Bold'):
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
        StrokeWidth=stroke_width,
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


def batch_render_words(words, output_dir, font_size=100, stroke_width=6, font_name='Arial Bold',
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
        StrokeWidth=stroke_width,
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

def generate_filter_complex(batch_timestamps, word_to_index, segment_start):
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

def run_ffmpeg_batch(input_video, output_video, filter_complex, input_images, segment_start, segment_duration):
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

def concatenate_segments(segment_data_list, output_dir=".", batch_duration=5):
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

def process_video(input_video, timestamps, output_file, batch_duration=5):
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
    batch_render_words(unique_words, "rendered_words")

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
            idx_word = word_to_index[item["word"]]
            path = f"rendered_words/word_{idx_word:05d}.png"
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
            start = max(0, item["start"] - seg_start)
            end = max(0.01, item["end"] - seg_start)
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

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
CLIENT_SECRETS_FILE = "client_secret.json"
TOKEN_FILE = "token.pickle"

def get_authenticated_service():
    creds = None

    # Load saved credentials if they exist
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "rb") as token:
            creds = pickle.load(token)

    # If no valid credentials, refresh silently or authenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())  # 🔹 silently refresh
        else:
            # First time setup: use console flow instead of browser
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
            creds = flow.run_console()  # 🔹 prints URL + prompt for code
            print("Paste the code from the browser into the terminal to complete authentication")

        # Save credentials for next run
        with open(TOKEN_FILE, "wb") as token:
            pickle.dump(creds, token)

    return build("youtube", "v3", credentials=creds)

def upload_youtube_short(video_file, title="Karaoke Short", description="awesome", tags=None, privacy_status="public"):
    youtube = get_authenticated_service()
    thumbnail_file = generate_subway_thumbnail("C:\\Users\\tidep\\Videos\\output_video_only.mp4","[FULL STORY] " + title, overlay_image_path="C:\\Users\\tidep\\Downloads\\IMG_2900.webp")
    print("Finished thumbnail part")
    if len(title) > 25:
            truncated = title[:25]  # get first 25 chars
            if ' ' in truncated:
                truncated = truncated.rsplit(' ', 1)[0]  # remove last partial word
            title = truncated + "..."
    request = youtube.videos().insert(
        part="snippet,status",
        body={
            "snippet": {
                "title": "[FULL STORY] " + title + "-BEST REDDIT STORIES",
                "description": description,
                "tags": tags,
                "categoryId": "24",  # Entertainment
            },
            "status": {
                "privacyStatus": privacy_status,
                "selfDeclaredMadeForKids": False
            },
        },
        media_body=MediaFileUpload(video_file, chunksize=-1, resumable=True)
    )
    print("it uploaded i think")
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"Uploading to YouTube Shorts... {int(status.progress() * 100)}%")
    video_id = response['id']
    print(f"\n✅ Upload Complete! Video ID: {response['id']}")
    youtube = get_authenticated_service()
    youtube.thumbnails().set(
    videoId=video_id,
    media_body=MediaFileUpload(thumbnail_file, mimetype='image/jpeg')).execute()
    return video_id

# ====== MODIFY MAIN() TO CALL IT AT THE END ======
def main():
    audio_path = "audio.mp3"
   
    try:
        # Step 1: Generate Story
        print("generating prompt")
        newPrompt = "create a starting story for a reddit AITA story it has to be very dramatic to immediately hook anyone who sees it in their youtube feed and it needs to be straight to the point. dont include anything that isnt needed here is an example \" AITA for fighting my brother when my girlfriend cheated with him?\" or \" AITA for moving out of my house when my mom hit my sister?\" make it creative and SO dramatic and out of proportion it has to be crazy. and it cannot be inappropriate it must work with youtube's guidelines and make this starting story around 10 words immediately throwing you into it think clickbait"
        partOfPrompt = generate_caption(newPrompt)
        print(partOfPrompt + " LALALALALALLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLALALALAL")
        print("Generating story...")
        story = generate_story(story_prompt, partOfPrompt)
        print("\nGenerated Story:\n", story)
        time.sleep(0.3)
       
        # Step 2: Convert to Speech
        print("\nConverting story to speech...")
        text_to_speech(story, audio_path)
        time.sleep(0.3)

        # Step 3: NEW - Get Word Timestamps
        print("\nAnalyzing speech for word timings...")
        word_timings = get_word_timestamps(audio_path)
        if not word_timings:
            print("Could not extract word timings. Exiting.")
            return
        time.sleep(0.3)

        # Step 4: Create Final Video
        print("\nCreating final video...")
        #bg_video = random.choice(background_videos)
        process_video(background_videos, word_timings, output_filename, batch_duration=5)
        print(f"\n✅ Done! Check your folder for '{output_filename}'")
        # Step 5: Upload to YouTube Shorts
        print("\nUploading video to YouTube Shorts...")
        upload_youtube_short(
            output_filename,
            title= partOfPrompt,
            description="An awesome reddit story I found",
            tags=["gaming", "letsplay", "videogames", "gamers", "gameplay", 
            "xbox", "playstation", "nintendo", "switch", "gamingcommunity", "gaminglife", "gamingchannel", 
            "gamingcontent",
            "redditstories", "reddit", "askreddit", "redditreadings", "storytime", "redditstory", 
            "nuclearrevenge", "prorevenge", "aita", "tifu", "entitledparents"
            ],
            privacy_status="public"
        )

    finally:
        # Clean up the generated audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
            print(f"\n🧹 Cleaned up temporary file")

# Run the main function
if __name__ == "__main__":
    main()
# === Step 1: GPU-accelerated text rendering ===



