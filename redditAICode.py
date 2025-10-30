import random
import os
import time
import json
import wave
import subprocess
from gtts import gTTS
from moviepy.video.VideoClip import TextClip, ColorClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from tqdm import tqdm
from vosk import Model, KaldiRecognizer

# ============ CONFIG ============
# Ensure this path is correct
background_videos = [r"C:\Users\tidep\subway_surfer.mp4.mp4"]
output_filename = r"C:\Users\tidep\Videos\final_karaoke_video.mp4"
story_prompt = "Write a funny Reddit-style story as if posted on r/TIFU"
# IMPORTANT: Update this path to where you extracted the Vosk model
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"

# ===============================

# 1. Dummy story generator (no changes)
def generate_story(prompt):
    return ("TIFU by trying to impress my crush by doing a backflip. "
            "Instead, I fell flat on my face and now I have a huge bruise. "
            "At least she noticed me!")

# 2. Text to speech (no changes)
def text_to_speech(text, filename="audio.mp3"):
    tts = gTTS(text)
    tts.save(filename)
    return filename

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
def create_karaoke_video(audio_file, story_text, word_timestamps, background_video):
    """
    Creates a karaoke-style video with word-by-word highlighting.
    """
    audio = AudioFileClip(audio_file)
    video_duration = audio.duration
    background = VideoFileClip(background_video).subclipped(0, video_duration)#.resize((1280, 720))

    clips = [background]
    
    # Create a text clip for each word with precise timing
    for i, word_info in enumerate(tqdm(word_timestamps, desc="Aligning words")):
        word = word_info['word']
        start_time = word_info['start']
        end_time = word_info['end']
        duration = end_time - start_time
        
        # Create a text clip that highlights the current word

        highlight_clip = (TextClip(text=word, font=r"C:\Windows\Fonts\arialbd.ttf", font_size=60, color="white",
                                  stroke_color='black', stroke_width=2, duration=duration,
                                  method='caption', size=(800, 200))
                          .with_start(start_time)
                          .with_duration(duration)
                          .with_position(("center", "center")))

        clips.append(highlight_clip)

    final = CompositeVideoClip(clips)
    final = final.with_audio(audio)
    final.write_videofile(output_filename, fps=24, codec='libx264', audio_codec='aac')

# 5. REWRITTEN: Main control flow to include the new timestamp step
def main():
    audio_path = "audio.mp3"
    
    try:
        # Step 1: Generate Story
        print("Generating story...")
        story = generate_story(story_prompt)
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
        bg_video = random.choice(background_videos)
        create_karaoke_video(audio_path, story, word_timings, bg_video)

        print(f"\n✅ Done! Check your folder for '{output_filename}'")

    finally:
        # Clean up the generated audio file
        if os.path.exists(audio_path):
            
            os.remove(audio_path)
            print(f"\n🧹 Cleaned up temporary file")

# Run the main function
if __name__ == "__main__":
    main()
