from flask import Flask, render_template, request, send_from_directory
from yt_dlp import YoutubeDL
from faster_whisper import WhisperModel
import os
from google import genai
import time
from gtts import gTTS
import subprocess

app = Flask(__name__)
DOWNLOAD_DIR = "downloads"
AUDIO_TTS_DIR = "tts_audio"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(AUDIO_TTS_DIR, exist_ok=True)

# -----------------------------
# 1) Tải audio từ YouTube
# -----------------------------
def download_audio(url, output_dir=DOWNLOAD_DIR):
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "noplaylist": True,
        "outtmpl": f"{output_dir}/audio.%(ext)s",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ]
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return f"{output_dir}/audio.mp3"

# -----------------------------
# 2) Chuyển audio → text (Whisper local)
# -----------------------------
def transcribe_audio(audio_path):
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, language="vi", beam_size=5)
    transcript = "\n".join([s.text for s in segments])
    return transcript

# -----------------------------
# 3) Tóm tắt bằng Gemini
# -----------------------------
def summarize_text(text, prompt="Tóm tắt nội dung sau bằng tiếng Việt, ngắn gọn:"):
    client = genai.Client(api_key="AIzaSyCdv1NlvEth4fY-IVGvCzuQX4HUS0CyMC0")
    retries = 5
    delay = 5

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"{prompt}\n\n{text}"
            )
            return response.text
        except Exception as e:
            print(f"Lỗi attempt {attempt+1}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise

# -----------------------------
# 4) Chuyển tóm tắt → audio
# -----------------------------
def text_to_speech(summary_text, voice="female"):
    """
    voice: 'female' or 'male'
    gTTS mặc định là giọng nữ, để giọng nam cần pyttsx3 hoặc TTS khác
    Ở đây dùng gTTS giọng nữ làm chuẩn, sau đó convert sang .ogg để browser Chrome play
    """
    tts_file_mp3 = os.path.join(AUDIO_TTS_DIR, "summary.mp3")
    tts_file_ogg = os.path.join(AUDIO_TTS_DIR, "summary.ogg")

    tts = gTTS(text=summary_text, lang="vi", tld="com")  # tld="com" dùng giọng nữ chuẩn
    tts.save(tts_file_mp3)

    # Convert mp3 → ogg (browser friendly)
    subprocess.run(["ffmpeg", "-y", "-i", tts_file_mp3, tts_file_ogg], check=True)
    return "summary.ogg"

@app.route("/", methods=["GET", "POST"])
def index():
    transcript = None
    summary = None
    audio_file = None

    if request.method == "POST":
        url = request.form.get("link_youtube").strip()
        voice = request.form.get("voice", "female")

        try:
            audio_file = download_audio(url)
            transcript = transcribe_audio(audio_file)
            summary = summarize_text(transcript)
            audio_file = text_to_speech(summary, voice)

        except Exception as e:
            transcript = f"❌ Lỗi: {e}"
            summary = None
            audio_file = None

    return render_template("index.html",  
        transcript=transcript,
        summary=summary,
        audio_file=audio_file
    )

@app.route("/tts_audio/<filename>")
def serve_tts(filename):
    return send_from_directory(AUDIO_TTS_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
