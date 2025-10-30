import json
import random
import subprocess
import wave
from pathlib import Path
from typing import Any, Dict, List

from gtts import gTTS
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.VideoClip import TextClip
from tqdm import tqdm
from vosk import KaldiRecognizer, Model

# ============ MODERN CONFIG ============
# Use pathlib.Path for robust, cross-platform path handling.
# IMPORTANT: Update these paths to match your system.

# --- Paths ---
# Use a list of Path objects for background videos.
BACKGROUND_VIDEOS: List[Path] = [
    Path(r"C:\Users\tidep\subway_surfer.mp4.mp4")
]
# Path for the final output video.
OUTPUT_FILENAME: Path = Path(r"C:\Users\tidep\Videos\final_karaoke_video.mp4")
# Path to the Vosk speech recognition model folder.
VOSK_MODEL_PATH: Path = Path("vosk-model-small-en-us-0.15")
# Path to a TrueType Font file (.ttf) for the text.
FONT_PATH: Path = Path(r"C:\Windows\Fonts\arialbd.ttf")

# --- Content ---
STORY_PROMPT: str = "Write a funny Reddit-style story as if posted on r/TIFU"
# =======================================

WordTimestamp = Dict[str, Any]

def generate_story(prompt: str) -> str:
    """
    Generates a story based on a prompt.

    NOTE: This is a placeholder. Replace with a real language model call if desired.
    """
    print(f"1. Generating story for prompt: '{prompt}'")
    return (
        "TIFU by trying to impress my crush by doing a backflip. "
        "Instead, I fell flat on my face and now I have a huge bruise. "
        "At least she noticed me!"
    )

def text_to_speech(text: str, output_path: Path) -> bool:
    """
    Converts text into an MP3 audio file using Google Text-to-Speech.

    Args:
        text: The text to convert.
        output_path: The file path to save the MP3 audio to.

    Returns:
        True if successful, False otherwise.
    """
    print(f"2. Converting text to speech -> {output_path.name}")
    try:
        tts = gTTS(text=text, lang='en')
        tts.save(str(output_path))
        return True
    except Exception as e:
        print(f"❌ Error during text-to-speech conversion: {e}")
        return False

def get_word_timestamps(audio_path: Path) -> List[WordTimestamp]:
    """
    Analyzes an audio file using Vosk to get word-level timestamps.

    Args:
        audio_path: Path to the input audio file (e.g., MP3).

    Returns:
        A list of dictionaries with word, start, and end times.
    """
    if not VOSK_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Vosk model not found at path: {VOSK_MODEL_PATH}\n"
            "Please download the model and update the VOSK_MODEL_PATH variable."
        )

    model = Model(str(VOSK_MODEL_PATH))
    wav_path = audio_path.with_suffix(".wav")

    print("   - Converting audio to WAV for analysis...")
    try:
        subprocess.run(
            [
                'ffmpeg', '-i', str(audio_path), '-acodec', 'pcm_s16le',
                '-ac', '1', '-ar', '16000', str(wav_path), '-y'
            ],
            check=True, capture_output=True, text=True,
        )
    except FileNotFoundError:
        raise FileNotFoundError("❌ `ffmpeg` not found. Please ensure ffmpeg is installed and in your system's PATH.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"❌ ffmpeg failed to convert audio.\nffmpeg stderr: {e.stderr}")

    results: List[WordTimestamp] = []
    try:
        with wave.open(str(wav_path), "rb") as wf:
            rec = KaldiRecognizer(model, wf.getframerate())
            rec.SetWords(True)

            # Modern walrus operator (:=) simplifies the loop
            while data := wf.readframes(4000):
                if rec.AcceptWaveform(data):
                    part_result = json.loads(rec.Result())
                    results.extend(part_result.get('result', []))

            part_result = json.loads(rec.FinalResult())
            results.extend(part_result.get('result', []))
    finally:
        wav_path.unlink(missing_ok=True)  # Safely delete the temp WAV file

    return results

def align_words_with_timestamps(original_text: str, vosk_timestamps: List[WordTimestamp]) -> List[WordTimestamp]:
    """
    Aligns the original script's words with Vosk's timestamps to correct transcription errors.
    """
    original_words = original_text.split()

    if len(original_words) != len(vosk_timestamps):
        print("\n⚠️ WARNING: Word count mismatch between original text and transcription!")
        print(f"   - Original script words: {len(original_words)}")
        print(f"   - Vosk detected words:   {len(vosk_timestamps)}")
        print("   - Video will use the shorter word count to avoid errors.")

    limit = min(len(original_words), len(vosk_timestamps))
    
    return [
        {
            'word': original_words[i],
            'start': vosk_timestamps[i]['start'],
            'end': vosk_timestamps[i]['end'],
            'conf': vosk_timestamps[i].get('conf', 1.0)
        }
        for i in range(limit)
    ]

def create_karaoke_video(audio_path: Path, word_timestamps: List[WordTimestamp], bg_video_path: Path) -> None:
    """
    Creates a karaoke-style video by overlaying timed text on a background video.
    """
    if not FONT_PATH.exists():
        raise FileNotFoundError(f"❌ Font file not found at: {FONT_PATH}")

    audio_clip = AudioFileClip(str(audio_path))
    video_duration = audio_clip.duration
    
    print("   - Loading and preparing background video...")
    background_clip = VideoFileClip(str(bg_video_path)).subclip(0, video_duration)
    video_size = background_clip.size

    text_clips: List[TextClip] = []
    for word_info in tqdm(word_timestamps, desc="   - Aligning words to video"):
        duration = word_info['end'] - word_info['start']
        
        text_clip = (
            TextClip(
                txt=word_info['word'],
                font=str(FONT_PATH),
                fontsize=70,
                color="yellow",
                stroke_color='black',
                stroke_width=2,
            )
            .set_start(word_info['start'])
            .set_duration(duration)
            .set_position(("center", "center"))
        )
        text_clips.append(text_clip)

    print("   - Assembling final video clip...")
    final_clip = CompositeVideoClip([background_clip] + text_clips, size=video_size)
    final_clip = final_clip.set_audio(audio_clip)
    
    print(f"   - Writing video file to '{OUTPUT_FILENAME}'...")
    final_clip.write_videofile(
        str(OUTPUT_FILENAME), fps=24, codec='libx264', audio_codec='aac'
    )

def main() -> None:
    """Main function to run the video creation pipeline."""
    temp_audio_path = Path("temp_audio.mp3")
    
    try:
        # Step 1: Generate Story
        story = generate_story(STORY_PROMPT)

        # Step 2: Text to Speech
        if not text_to_speech(story, temp_audio_path):
            return  # Exit if TTS failed

        # Step 3: Get Word Timestamps
        print("3. Analyzing speech for word timings...")
        word_timings_from_vosk = get_word_timestamps(temp_audio_path)
        if not word_timings_from_vosk:
            print("❌ ERROR: Could not extract word timings. Exiting.")
            return

        # Step 4: Align Words
        print("4. Correcting transcription and aligning words...")
        corrected_word_timings = align_words_with_timestamps(story, word_timings_from_vosk)

        # Step 5: Create Video
        print("5. Creating final video...")
        bg_video = random.choice(BACKGROUND_VIDEOS)
        create_karaoke_video(temp_audio_path, corrected_word_timings, bg_video)

        print(f"\n✅ Done! Video saved as '{OUTPUT_FILENAME}'")

    except (FileNotFoundError, RuntimeError, Exception) as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        # Modern cleanup using pathlib
        if temp_audio_path.exists():
            temp_audio_path.unlink()
            print(f"\n🧹 Cleaned up temporary file: {temp_audio_path.name}")

if __name__ == "__main__":
    main()
