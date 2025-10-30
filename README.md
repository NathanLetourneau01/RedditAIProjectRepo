# Automatic YouTube Video Generator / Uploader

## Overview
This project automatically generates and uploads Reddit-style story videos (AITA, TIFU, horror, true crime) to YouTube. It supports both long-form and short-form videos.

## Technologies Used
- Python, PyTorch, Vosk, MoviePy, ffmpeg, Blender
- APIs: ChatGPT, YouTube, Google
- TTS/STT Tools: gTTS, Coqui, Kokero
- Docker, JSON & Pickle handling, GPU acceleration

## Features
- Generates stories and titles automatically via ChatGPT API
- Converts text to speech and syncs with subtitles
- Creates video, adds text overlays, and generates thumbnails
- Uploads and schedules videos to YouTube, rotating across multiple accounts
- Supports both long-form and YouTube Shorts formats

## Video Examples
- Long Form: [YouTube Link](https://www.youtube.com/watch?v=weFIous3Rs4)
- Short Form: [YouTube Link](https://www.youtube.com/shorts/BjTcZOoVzY8)
- **Channel Example:** [Reddit AITA Stories](https://www.youtube.com/@RedditAITAStories12/shorts)
## Skills Gained
- Python automation and subprocess handling
- Video processing and TTS/STT integration
- API integration and workflow optimization
  
## In-Depth Description

### FullVideoUploader.py
**Language:** Python  

Developed an algorithm that generates Reddit AITA-style stories and uploads them automatically to your YouTube channel.  

**Description:**  
- Uses Python libraries such as PyTorch, Vosk, MoviePy, ffmpeg, Blender, and Docker.  
- Generates stories like AITA, TIFU, horror, and true crime using ChatGPT API.  
- Automatically generates a title and prepends it to the story so it reads aloud first.  
- Converts the story to audio via a TTS bot.  
- Syncs the text with the audio using Vosk.  
- Creates video frames with text overlays aligned to audio timestamps.  
- Combines video and audio, uploads to YouTube with a description, title, and tags.  
- Generates a thumbnail using a random video frame with a base image and formatted title.  

---

### Youtube short uploader.py
**Language:** Python  

Generates and uploads Reddit AITA-style short videos automatically, while ensuring compliance with YouTube Shorts regulations.  

**Description:**  
- Adjusts story and video length to fit YouTube Shorts regulations.  
- Formats videos to a 9:16 aspect ratio, reformatting the background video and adjusting text/title placement.  
- Maintains the same workflow as long-form videos but optimized for short-form content.  

---

### YoutubeUploader/Generator.py 
**Language:** Python  

This script combines both the short-form and long-form video generators into a single automated workflow.  

**Description:**  
- Automatically uploads videos and schedules them for publication.  
- Supports multiple YouTube accounts and cycles through them after each defined upload batch (e.g., 12 shorts/day, 3 long-form videos/day).  
- Uses `.pickle` and `.json` files to manage state between uploads.  
- Designed to run indefinitely, continuously generating and uploading content.  
