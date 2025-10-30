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
background_videos = [r"C:\Users\tidep\Videos\output_video_only.mp4"] #C:\Users\tidep\Downloads\subway_surfer.mp4, C:\Users\tidep\subway_surfer.mp4.mp4
output_filename = r"C:\Users\tidep\Videos\final_karaoke_video.mp4"
story_prompt = '''
Before writing, silently create an outline in your reasoning (do not output it) that ensures the story will add up to at exactly 2100 words.
Plan how many words to allocate to each section:
Be maximally detailed, exhaustive, and expansive. Do not summarize. Expand on every point with examples, explanations, and counterarguments. Length is critical.
Introduction and setup (around 350–400 words)
First conflict (around 550–600 words)
Escalation and fallout (around 550–600 words)
Update sections (around 450–500 words total, split across at least two updates)
Final reflection and “Am I the asshole” ending make sure this section does not end in a make up both sides must be angry at eachother (around 150–200 words)
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
After writing, double-check the story is exactly 2100 words.
If it is shorter, add realistic detail or dialogue until it is exactly 2100. and make sure the grammar is simple it cant be hard to understand or complex.
Do not include the word count in your output—just'''
    # IMPORTANT: Update this path to where you extracted the Vosk model
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"

# ===============================

# 1. Dummy story generator (no changes)
def generate_story(prompt1, partOfPrompt):
    openai.api_key = "Blank"
    prompt = partOfPrompt + " make sure to make the story about that and here are the parameters "+ prompt1 + " make sure it is exactly 2100 words"
# Step 3: Send it to ChatGPT
    response = openai.ChatCompletion.create(
    model="gpt-4o",  # or "gpt-4" if you have access
    messages=[
        {"role": "user", "content": prompt }])
    max_tokens= 40000
# Step 4: Extract and print the response
    output = response['choices'][0]['message']['content']
    return(output)

def generate_title(titlePrompt):
    openai.api_key = "Blank"
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",  # or "gpt-4" if you have access
    messages=[
        {"role": "user", "content": titlePrompt}])
    output = response['choices'][0]['message']['content']
    return(output)
def generate_caption(newPrompt):
    openai.api_key = "Blank"
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",  # or "gpt-4" if you have access
    messages=[
        {"role": "user", "content": newPrompt}])
    output = response['choices'][0]['message']['content']
    return(output)
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
def create_karaoke_video(audio_file, word_timestamps, background_video, output_filename):
    """
    Creates a GPU-accelerated karaoke-style video in Blender.
    - Shows each word at its start/end time.
    - Overlays words on a background video.
    - Adds audio from audio_file.
    """
    import bpy
    import os

    # Clear existing scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    # Set render engine to GPU Cycles
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.cycles.feature_set = 'SUPPORTED'
    scene.cycles.samples = 128  # Adjust for quality/speed

    # Set FPS and frame range
    fps = 24
    scene.render.fps = fps
    video_duration = 0
    if word_timestamps:
        video_duration = max([w['end'] for w in word_timestamps])
    scene.frame_start = 1
    scene.frame_end = int(video_duration * fps) + 5

    # Add background video in Video Sequence Editor
    scene.sequence_editor_create()
    scene.sequence_editor.sequences.new_movie(
        name="Background",
        filepath=background_video,
        channel=1,
        frame_start=1
    )

    # Add audio as sound strip
    scene.sequence_editor.sequences.new_sound(
        name="Audio",
        filepath=audio_file,
        channel=2,
        frame_start=1
    )

    # Add text objects for each word
    for word_info in word_timestamps:
        word = word_info['word']
        start_frame = int(word_info['start'] * fps)
        end_frame = int(word_info['end'] * fps)

        # Create a text object
        bpy.ops.object.text_add(location=(0, 0, 0))
        text_obj = bpy.context.object
        text_obj.data.body = word
        text_obj.data.align_x = 'CENTER'
        text_obj.data.align_y = 'CENTER'
        text_obj.scale = (0.5, 0.5, 0.5)  # adjust text size

        # Keyframe visibility for karaoke effect
        text_obj.hide_viewport = True
        text_obj.hide_render = True
        text_obj.keyframe_insert(data_path="hide_render", frame=start_frame-1)
        text_obj.hide_render = False
        text_obj.keyframe_insert(data_path="hide_render", frame=start_frame)
        text_obj.hide_render = True
        text_obj.keyframe_insert(data_path="hide_render", frame=end_frame)

    # Set output settings
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.audio_codec = 'AAC'
    scene.render.filepath = os.path.abspath(output_filename)

    # Render video
    bpy.ops.render.render(animation=True)
CLIENT_SECRETS_FILE = "client_secrets.json"  # Place your file here
TOKEN_FILE = "token.pickle"
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
# 5. REWRITTEN: Main control flow to include the new timestamp step
def get_authenticated_service():
    creds = None
    # Load saved credentials
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "rb") as token:
            creds = pickle.load(token)
    # If no valid credentials, let user log in
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
        creds = flow.run_local_server(port=0)  # <- changed here
        # Save the credentials for next time
        with open(TOKEN_FILE, "wb") as token:
            pickle.dump(creds, token)
    return build("youtube", "v3", credentials=creds)

def upload_youtube_short(video_file, title="Karaoke Short", description="awesome", tags=None, privacy_status="public"):
    youtube = get_authenticated_service()
    thumbnail_file = generate_subway_thumbnail("C:\\Users\\tidep\\Videos\\output_video_only.mp4", title, overlay_image_path="C:\\Users\\tidep\\Downloads\\NewThumbnail.webp")
    request = youtube.videos().insert(
        part="snippet,status",
        body={
            "snippet": {
                "title": title,
                "description": description,
                "tags": tags,
                "categoryId": "22",  # People & Blogs
                "thumbnail_file": thumbnail_file
            },
            "status": {
                "privacyStatus": privacy_status,
                "selfDeclaredMadeForKids": False
            },
        },
        media_body=MediaFileUpload(video_file, chunksize=-1, resumable=True)
    )

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
blender_script = r"C:\Users\tidep\karaoke_blender.py"
# ====== MODIFY MAIN() TO CALL IT AT THE END ======
def main():
    try:
        # Step 1: Generate Story
        print("Generating story prompt...")
        newPrompt = "create a dramatic 10-word starting story for YouTube clickbait"
        partOfPrompt = generate_caption(newPrompt)
        print("Generating story...")
        story = generate_story(story_prompt, partOfPrompt)
        print("\nGenerated Story:\n", story)
        time.sleep(0.3)

        # Step 2: Convert Story to Speech
        print("\nConverting story to speech...")
        text_to_speech(story, audio_path)
        time.sleep(0.3)

        # Step 3: Get Word Timestamps
        print("\nAnalyzing speech for word timings...")
        word_timings = get_word_timestamps(audio_path)
        if not word_timings:
            print("Could not extract word timings. Exiting.")
            return

        # Save timestamps to JSON for Blender
        with open(timestamps_file, "w") as f:
            json.dump(word_timings, f)

        # Step 4: Select background video
        bg_video = random.choice(background_videos)

        # Step 5: Call Blender for GPU Karaoke Video
        print("\nRendering karaoke video in Blender...")
        subprocess.run([
            "blender", "--background", "--python", blender_script,
            "--",
            "--audio", audio_path,
            "--timestamps", timestamps_file,
            "--background", bg_video,
            "--output", output_filename
        ], check=True)

        print(f"\n✅ Done! Video saved at '{output_filename}'")

        # Step 6: Generate title and upload to YouTube Shorts
        titlePrompt = f"create a title for this story: {story[:40]}"
        thinggy = generate_title(titlePrompt)
        truncated_title = thinggy[:25] + "..." if len(thinggy) > 25 else thinggy

        print("\nUploading video to YouTube Shorts...")
        upload_youtube_short(
            output_filename,
            title="[FULL STORY] " + truncated_title + "-BEST REDDIT STORY",
            description="An awesome reddit story I found",
            tags=["gaming", "redditstories", "storytime", "aita"],
            privacy_status="public"
        )

    finally:
        # Cleanup temporary files
        if os.path.exists(audio_path):
            os.remove(audio_path)

# ============================ RUN SCRIPT =============================

if __name__ == "__main__":

    main()
