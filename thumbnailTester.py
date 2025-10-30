from PIL import Image, ImageDraw, ImageFont
import random
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image, ImageDraw, ImageFont
import textwrap

def generate_subway_thumbnail(video_path, title, output_path="thumbnail.jpg", overlay_image_path="C:\\Users\\tidep\\Downloads\\NewThumbnail.webp"):
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

generate_subway_thumbnail("C:\\Users\\tidep\\redditAI\\SubwaySurfers15min.mp4",title = "Generate a YouTube thumbnail using a frame from a Subway Surfers video as background,with both the main title and \"Reddit AITA Stories\" bolded.Main title wraps slightly longer lines, is positioned 2% above center, and truncates with '...' if over 100 characters.", overlay_image_path="C:\\Users\\tidep\\Downloads\\NewThumbnail.webp")
