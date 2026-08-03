from faster_whisper import WhisperModel
import os

_model = None


def load_model():

    global _model

    if _model is None:
        print("Loading Whisper model ...")
        _model = WhisperModel("tiny", device="cpu", compute_type="int8")
        print("Whisper model loaded.")
    return _model


def _whisper_language_code(language: str) -> str | None:
    normalized = (language or "").strip().lower()
    language_map = {
        "english": "en",
        "en": "en",
        "hindi": "hi",
        "hi": "hi",
        "hinglish": None,
    }
    return language_map.get(normalized, None)


def transcribe_chunk_whisper(chunk_path: str, language: str = "english") -> str:

    model = load_model()
    language_code = _whisper_language_code(language)

    try:
        segments, info = model.transcribe(
            audio=chunk_path,
            beam_size=1,
            vad_filter=True,
            chunk_length=30,
            without_timestamps=True,
            condition_on_previous_text=False,
            language=language_code,
        )
    except TypeError as exc:
        if "audio" not in str(exc):
            raise
        segments, info = model.transcribe(
            chunk_path,
            beam_size=1,
            vad_filter=True,
            chunk_length=30,
            without_timestamps=True,
            condition_on_previous_text=False,
            language=language_code,
        )

    text = " ".join([segment.text for segment in segments])

    return text.strip()


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """Use the same local Whisper model for English and Hindi audio."""
    return transcribe_chunk_whisper(chunk_path, language=language)


def transcribe_all(chunks: list, language: str = "english") -> str:

    full_transcript = ""

    print("Using Whisper for transcription.")

    for i, chunk in enumerate(chunks):

        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")

        text = transcribe_chunk(chunk, language=language)

        full_transcript += text + " "

    print("Transcription complete.")

    return full_transcript.strip()