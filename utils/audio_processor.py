import os
import subprocess
import glob

import yt_dlp

DOWNLOAD_DIR = 'downloades'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def _run_ffmpeg(args: list[str]) -> None:
    subprocess.run(
        ["ffmpeg", "-y", *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info).replace(".webm", ".wav").replace(".m4a", ".wav")
    return filename


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV using ffmpeg, without importing pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    _run_ffmpeg([
        "-i", input_path,
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        output_path,
    ])
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 2) -> list:
    """Split a WAV file into shorter segments using ffmpeg so STT doesn't see a giant clip."""
    chunk_seconds = max(1, chunk_minutes) * 60
    base_dir = os.path.dirname(wav_path) or '.'
    base_name = os.path.splitext(os.path.basename(wav_path))[0]
    chunk_prefix = os.path.join(base_dir, f"{base_name}_chunk_")

    _run_ffmpeg([
        "-i", wav_path,
        "-f", "segment",
        "-segment_time", str(chunk_seconds),
        "-reset_timestamps", "1",
        "-c:a", "pcm_s16le",
        f"{chunk_prefix}%03d.wav",
    ])

    chunks = sorted(glob.glob(f"{chunk_prefix}*.wav"))
    return chunks

def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks

