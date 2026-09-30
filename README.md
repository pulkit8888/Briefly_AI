# Briefly AI

Briefly AI accepts a YouTube URL or a local audio/video file path, transcribes it locally with Whisper, creates a meeting report with Mistral, and provides a transcript-grounded chat interface.

## Features

- Download audio from a YouTube URL or convert a local media file with FFmpeg.
- Transcribe English, Hindi, or Hinglish audio with local `faster-whisper`.
- Generate a title, summary, action items, key decisions, and open questions.
- Ask questions about the transcript through a ChromaDB-backed RAG chat.
- Use Mistral Small as the primary LLM and Mistral Large as the fallback model.
- Cancel an active analysis when the Streamlit page is refreshed.

## How it works

```text
YouTube URL or local file path
            |
            v
utils/audio_processor.py
  - YouTube: yt-dlp downloads audio as WAV
  - Local file: FFmpeg converts to 16 kHz mono WAV
  - FFmpeg splits audio into two-minute WAV chunks
            |
            v
core/transcriber.py
  - faster-whisper transcribes chunks sequentially
  - VAD removes silence
  - chunk text is joined into one transcript
            |
            +-------------------------------+
            |                               |
            v                               v
core/summarizer.py + core/extractor.py   core/vector_store.py
  - title and map-reduce summary           - split transcript into chunks
  - actions, decisions, questions          - create sentence embeddings
  - Mistral Small -> Mistral Large          - persist them in ChromaDB
            |                               |
            +---------------+---------------+
                            v
                     core/rag_engine.py
              retrieve the 4 closest chunks
              and answer with Mistral
```

## Project structure

```text
Ai_assis/
├── app.py                    # Streamlit UI and web-pipeline orchestration
├── main.py                   # Command-line pipeline and chat loop
├── requirements.txt          # Python dependencies
├── .env                      # Your Mistral API key (create locally; not committed)
├── .vscode/
│   └── settings.json          # Selects the project .venv for VS Code/Pylance
├── core/
│   ├── llm.py                # Mistral primary/fallback configuration
│   ├── run_control.py        # Refresh-aware cancellation token
│   ├── transcriber.py        # Local Whisper transcription
│   ├── summarizer.py         # Title generation and map-reduce summary
│   ├── extractor.py          # Actions, decisions, and questions
│   ├── vector_store.py       # ChromaDB embeddings and retrieval
│   └── rag_engine.py         # Transcript question-answering chain
└── utils/
    └── audio_processor.py    # YouTube download, conversion, and chunking
```

Runtime folders, ignored by Git:

- `downloads/` stores downloaded, converted, and chunked audio.
- `vector_db/` stores the persisted ChromaDB collection.

## Requirements

- Python 3.10 or newer.
- FFmpeg available on your system `PATH`.
- A Mistral API key.

On Windows, install FFmpeg with:

```powershell
winget install --id=Gyan.FFmpeg -e
```

Verify it with:

```powershell
ffmpeg -version
```

## Setup

From the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a `.env` file:

```env
MISTRAL_API_KEY=your_mistral_api_key

# Optional model overrides
MISTRAL_MODEL=mistral-small-latest
MISTRAL_FALLBACK_MODEL=mistral-large-latest
```

`MISTRAL_MODEL` is used first. If that call raises an error, LangChain retries the request using `MISTRAL_FALLBACK_MODEL`. Both models use the same Mistral API key; no OpenAI key is required.

## Run the app

Start the Streamlit app with the project environment:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Then open `http://localhost:8501`.

1. Enter a YouTube URL or a local file path.
2. Choose `english` or `hinglish`.
3. Select **Analyse**.
4. Review the title, summary, transcript, actions, decisions, and questions.
5. Use **Chat with your Meeting** to ask about the transcript.

The web UI currently accepts a local file path; it does not include a browser file-upload control.

### VS Code / Pylance

The repository includes `.vscode/settings.json`, which points VS Code to `.venv`. If Pylance still marks installed packages as missing, reload the VS Code window or run **Python: Select Interpreter** and select:

```text
.venv\Scripts\python.exe
```

## CLI usage

```powershell
.\.venv\Scripts\python.exe main.py
```

The CLI asks for a source and language, prints the report, then starts an interactive chat. Type `exit`, `quit`, or `q` to leave the chat.

## Detailed execution flow

### 1. Input and audio preparation

`app.py` is the web entry point. On **Analyse**, it calls the functions below directly in sequence. `main.py` performs the same sequence for the command line.

`utils/audio_processor.py` detects whether the source starts with `http://` or `https://`:

- `download_youtube_audio()` uses `yt-dlp` to download YouTube audio as WAV.
- `convert_to_wav()` uses FFmpeg to convert a local media path to mono, 16 kHz PCM WAV.
- `chunk_audio()` uses FFmpeg to split the WAV into two-minute files.
- `process_input()` returns the sorted list of generated chunk paths.

### 2. Transcription

`core/transcriber.py` lazily loads Whisper Tiny once, then sends each chunk to `faster-whisper` with:

- CPU + `int8` inference
- one CPU worker
- voice activity detection (`vad_filter=True`)
- 30-second internal Whisper windows
- no timestamps in the returned text

`transcribe_all()` joins the text of every audio chunk into one transcript string.

### 3. LLM reporting

`core/llm.py` provides the LLM used everywhere in the app. It creates:

1. Mistral Small (`MISTRAL_MODEL`, default `mistral-small-latest`)
2. Mistral Large fallback (`MISTRAL_FALLBACK_MODEL`, default `mistral-large-latest`)

`core/summarizer.py` generates the title from the first 2,000 characters, then produces the summary with map-reduce processing:

- split transcript: 3,000 characters with 200-character overlap
- map: summarize each chunk
- reduce: combine chunk summaries into bullet points

`core/extractor.py` makes three Mistral calls to extract action items, key decisions, and open questions.

### 4. RAG chat

`core/vector_store.py` splits the transcript into 500-character chunks with 50-character overlap. It creates embeddings with `all-MiniLM-L6-v2` and stores them in the local ChromaDB collection `meeting_transcript`.

`core/rag_engine.py` retrieves the four most similar chunks for a question, adds them to the prompt as context, and asks the Mistral LLM to answer only from that context.

The vector database is persistent and shared across runs. It deduplicates chunks by content hash, so reset `vector_db/` if you need a completely fresh knowledge base.

## Refresh cancellation

`core/run_control.py` allows one active Streamlit analysis at a time. When the page is refreshed, `app.py` signals the previous run to stop.

- FFmpeg conversion and chunking are stopped by terminating their running process.
- YouTube download checks the signal through its progress hook.
- Whisper checks between generated transcript segments and between audio chunks.
- The app checks the signal before beginning each later report/RAG stage.

An already-running remote Mistral request cannot be interrupted by this application; the pipeline stops as soon as that request returns. This is why cancellation may take a short time during an API call.

## Memory settings

`core/transcriber.py` and `core/vector_store.py` set the following before loading native ML libraries:

```text
MKL_DISABLE_FAST_MM=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

Whisper also uses one worker and embedding requests use a batch size of four. These limits reduce memory spikes and help avoid `mkl_malloc: failed to allocate memory` on CPU-only machines. Restart Streamlit after changing code or environment settings so these take effect.

## Supported languages

| Input | Whisper language value | Availability |
|---|---|---|
| `english` | `en` | Web UI and CLI |
| `hinglish` | automatic detection | Web UI and CLI |
| `hindi` | `hi` | CLI/API function only; not in the current web dropdown |

## Key modules

| File | Key functions | Purpose |
|---|---|---|
| `app.py` | `update_step()` | Streamlit UI, progress state, cancellation, report rendering, and chat UI |
| `main.py` | `run_pipeline()` | CLI orchestration and chat loop |
| `core/llm.py` | `get_llm()` | Mistral primary and fallback runnable |
| `core/run_control.py` | `start_run()`, `cancel_active_run()` | Refresh-aware cancellation state |
| `utils/audio_processor.py` | `process_input()` | Media download/conversion/chunking |
| `core/transcriber.py` | `transcribe_all()` | Local Whisper transcription |
| `core/summarizer.py` | `generate_title()`, `summarize()` | Title and summary generation |
| `core/extractor.py` | `extract_action_items()` | Meeting insight extraction |
| `core/vector_store.py` | `build_vector_store()` | Embeddings and ChromaDB persistence |
| `core/rag_engine.py` | `build_rag_chain()`, `ask_question()` | Retrieval-augmented chat |
