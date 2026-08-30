import gc
import os

# Applied these before importing native ML libraries to avoid excessive MKL memory use.
os.environ["MKL_DISABLE_FAST_MM"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from faster_whisper import WhisperModel

_model = None


def load_model():

    global _model

    if _model is None:
        print("Loading Whisper model ...")
        _model = WhisperModel(
            "tiny", device="cpu", compute_type="int8", cpu_threads=1, num_workers=1
        )
        print("Whisper model loaded.")
    return _model


def _whisper_language_code(language: str) -> str | None:
    normalized = (language or "").strip().lower()
    language_map = {
        "english": "en",
        "en": "en",
        "hinglish": None,
    }
    return language_map.get(normalized, None)


def transcribe_chunk_whisper(chunk_path: str, language: str = "english", is_cancelled=None) -> str:

    model = load_model()
    language_code = _whisper_language_code(language)

    segments, info = model.transcribe(
        audio=chunk_path,
        beam_size=1,
        vad_filter=True,
        chunk_length=30,
        without_timestamps=True,
        condition_on_previous_text=False,
        language=language_code,
    )


    text_parts = []
    for segment in segments:
        if is_cancelled:
            is_cancelled()
        text_parts.append(segment.text)

    return " ".join(text_parts).strip()


def transcribe_all(chunks: list, language: str = "english", is_cancelled=None) -> str:

    full_transcript = ""

    print("Using Whisper for transcription.")

    try:
        for i, chunk in enumerate(chunks):
            if is_cancelled:
                is_cancelled()

            print(f"Transcribing chunk {i + 1}/{len(chunks)}...")

            text = transcribe_chunk_whisper(
                chunk, language=language, is_cancelled=is_cancelled
            )
            full_transcript += text + " "
    finally:
        unload_model()

    print("Transcription complete.")

    return full_transcript.strip()


def unload_model() -> None:
    global _model
    _model = None
    gc.collect()
