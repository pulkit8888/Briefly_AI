# 🎬 Briefly AI — AI Video & Meeting Assistant

> **Transcribe · Summarise · Chat with your meetings**
>
> An end-to-end AI pipeline that takes any YouTube URL or local video/audio file, transcribes it with Whisper, extracts structured insights with Mistral LLM, and lets you have a RAG-powered conversation with the full transcript — all through a sleek Streamlit UI or a lightweight CLI.

---

## 📋 Table of Contents

- [What It Does](#-what-it-does)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Running the App](#-running-the-app)
- [CLI Usage](#-cli-usage)
- [Pipeline Walkthrough](#-pipeline-walkthrough)
- [Module Reference](#-module-reference)
- [Supported Languages](#-supported-languages)
- [Environment Variables](#-environment-variables)
- [Known Limitations & Tips](#-known-limitations--tips)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ What It Does

Briefly AI turns any video or audio source into a complete intelligence report in minutes:

| Feature | Description |
|---|---|
| 🔊 **Audio Extraction** | Downloads audio from YouTube URLs via `yt-dlp` or converts local files via `ffmpeg` |
| 📝 **Speech-to-Text** | Runs local Whisper (`faster-whisper`) — no external STT API needed |
| 🏷️ **Title Generation** | Auto-generates a professional meeting title using Mistral LLM |
| 📋 **Summarisation** | Map-Reduce summarisation pipeline for long transcripts |
| ✅ **Action Items** | Extracts tasks with owner and deadline from the transcript |
| 🔑 **Key Decisions** | Lists every decision made during the session |
| ❓ **Open Questions** | Surfaces unresolved questions / follow-up items |
| 🧠 **RAG Chat** | Chat with the full transcript using a Chroma + Mistral RAG pipeline |
| 🖥️ **Streamlit UI** | Beautiful dark-mode web interface with live pipeline status |
| 💻 **CLI Mode** | Headless terminal interface for scripting and automation |

---

## 🏗️ Architecture

```
Input (YouTube URL / Local File)
          │
          ▼
┌─────────────────────┐
│   Audio Processor   │  yt-dlp  +  ffmpeg
│  (utils/audio_      │  ───────────────────
│   processor.py)     │  Download → Convert WAV → Chunk (2-min segments)
└─────────┬───────────┘
          │  audio chunks []
          ▼
┌─────────────────────┐
│    Transcriber      │  faster-whisper (local, CPU/GPU)
│  (core/transcriber  │  ───────────────────────────────
│       .py)          │  Chunk-by-chunk STT → full transcript string
└─────────┬───────────┘
          │  transcript (str)
          ├──────────────────────────────────────────────┐
          ▼                                              ▼
┌──────────────────────┐                    ┌────────────────────────┐
│  Summariser +        │  Mistral LLM       │   Vector Store Builder │
│  Extractor           │  (LangChain LCEL)  │   (core/vector_store   │
│  (core/summarizer.py │  ────────────────  │        .py)            │
│   core/extractor.py) │  Title             │  HuggingFace Embeddings│
│                      │  Summary           │  all-MiniLM-L6-v2      │
│                      │  Action Items      │  + ChromaDB            │
│                      │  Key Decisions     └─────────┬──────────────┘
│                      │  Open Questions              │  vector store
└──────────────────────┘                    ┌─────────▼──────────────┐
                                            │    RAG Engine          │
                                            │  (core/rag_engine.py)  │
                                            │  Retriever (k=4)       │
                                            │  + Mistral LLM         │
                                            │  + LCEL Chain          │
                                            └────────────────────────┘
                                                       │
                                            ┌──────────▼─────────────┐
                                            │   Streamlit UI / CLI   │
                                            │  Interactive Q&A Chat  │
                                            └────────────────────────┘
```

---

## 📁 Project Structure

```
AI-Video-Assistant--main/
│
├── app.py                  # Streamlit web application (main UI)
├── main.py                 # CLI entry point
├── requirements.txt        # All Python dependencies
├── .env                    # API keys (not committed — you create this)
├── .gitignore
│
├── core/                   # Core AI pipeline modules
│   ├── transcriber.py      # Whisper speech-to-text
│   ├── summarizer.py       # LLM-based summarisation + title generation
│   ├── extractor.py        # Action items / decisions / questions extraction
│   ├── rag_engine.py       # RAG chain (retriever + Mistral LLM)
│   └── vector_store.py     # ChromaDB vector store builder & loader
│
└── utils/                  # Utility helpers
    └── audio_processor.py  # YouTube download, format conversion, chunking
```

> **Auto-created at runtime** (gitignored):
> - `downloads/` — downloaded/converted WAV files
> - `vector_db/` — persisted ChromaDB embeddings

---

## 🛠️ Tech Stack

### AI / ML
| Library | Role |
|---|---|
| `faster-whisper` | Local speech-to-text (Whisper tiny model, CPU, int8) |
| `openai-whisper` | Whisper model definitions |
| `torch` + `torchaudio` | PyTorch backend for Whisper |
| `mistralai` + `langchain-mistralai` | LLM for summarisation, extraction & RAG |
| `sentence-transformers` | HuggingFace embeddings (`all-MiniLM-L6-v2`) |
| `langchain-huggingface` | LangChain ↔ HuggingFace bridge |

### Orchestration
| Library | Role |
|---|---|
| `langchain` | LLM orchestration framework |
| `langchain-core` | LCEL (LangChain Expression Language) chains |
| `langchain-community` | Community integrations |
| `langchain-text-splitters` | Recursive character text splitting |
| `langchain-chroma` | ChromaDB ↔ LangChain bridge |

### Vector Store
| Library | Role |
|---|---|
| `chromadb` | Local persistent vector database |
| `tiktoken` | Token counting for text splitting |

### Audio / Video
| Tool | Role |
|---|---|
| `yt-dlp` | YouTube audio download |
| `ffmpeg` *(system binary)* | Audio format conversion & chunking |
| `ffmpeg-python` | Optional Python FFmpeg bindings |

### UI & Utilities
| Library | Role |
|---|---|
| `streamlit` | Web UI framework |
| `streamlit-extras` | Additional Streamlit widgets |
| `watchdog` | Hot-reload file watcher for Streamlit |
| `reportlab` + `fpdf2` | PDF export utilities |
| `deep-translator` | Hindi → English translation (Google backend) |
| `python-dotenv` | `.env` file loading |
| `numpy`, `tqdm`, `requests` | General utilities |

---

## ✅ Prerequisites

Before you start, ensure the following are installed on your system:

### 1. Python 3.10+
```bash
python --version   # should be 3.10 or higher
```

### 2. FFmpeg (system binary — required)
FFmpeg must be accessible from your `PATH`.

**Windows:**
```powershell
# Option A — via winget
winget install --id=Gyan.FFmpeg -e

# Option B — via Chocolatey
choco install ffmpeg

# Option C — manual: download from https://ffmpeg.org/download.html
# and add the /bin folder to your PATH environment variable
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg       # Debian/Ubuntu
sudo dnf install ffmpeg       # Fedora
```

Verify:
```bash
ffmpeg -version
```

### 3. Mistral API Key
The app uses [Mistral AI](https://mistral.ai/) as its LLM backend. Sign up and grab a free API key at:
👉 **https://console.mistral.ai/**

---

## 🚀 Installation

### Step 1 — Clone the repository
```bash
git clone https://github.com/your-username/AI-Video-Assistant.git
cd AI-Video-Assistant--main
```

### Step 2 — Create a virtual environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

> ⚠️ **PyTorch note:** If you have a CUDA-capable GPU and want faster transcription,
> install the GPU version of PyTorch first before running the command above:
> ```bash
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```
> Then edit `core/vector_store.py` and `core/transcriber.py` — change `"cpu"` → `"cuda"`.

---

## ⚙️ Configuration

Create a `.env` file in the project root:

```env
# .env
MISTRAL_API_KEY=your_mistral_api_key_here
```

That's the only required configuration. The Whisper model (`tiny`) and HuggingFace embedding model (`all-MiniLM-L6-v2`) are downloaded automatically on first run.

---

## ▶️ Running the App

### Web UI (Streamlit) — Recommended

```bash
streamlit run app.py
```

The app will open automatically at **http://localhost:8501**

**Usage:**
1. Paste a **YouTube URL** or a **local file path** (MP4, MP3, WAV, etc.) in the sidebar
2. Select the **language** (`english` or `hinglish`)
3. Click **⚡ Analyse**
4. Watch the live pipeline status bars update in the sidebar
5. When complete, view:
   - Session Title
   - Summary
   - Full Transcript (expandable)
   - Action Items · Key Decisions · Open Questions
6. Use the **💬 Chat** section to ask questions about the meeting

---

## 💻 CLI Usage

For headless / scripted usage:

```bash
python main.py
```

**Interactive prompts:**
```
Enter YouTube URL or local file path: https://youtube.com/watch?v=dQw4w9WgXcQ
Language (english/hinglish): english
```

**Output:**
```
============================================================
📌 Title:  Product Roadmap Q3 Discussion
📋 Summary:
  • Team agreed on three priority features for Q3 ...
  • Budget allocation was discussed ...
✅ Action Items:
  1. John to finalize design mockups by Friday ...
🔑 Key Decisions:
  1. Launch date moved to September 15 ...
❓ Open Questions:
  1. Who owns the customer success handoff? ...
============================================================
💬 Chat with your meeting (type 'exit' to quit)

You: What was decided about the launch date?
🤖 Assistant: The team decided to move the launch date to September 15 ...

```

---

## 🔄 Pipeline Walkthrough

```
Step 1: Audio Processing
  └─ YouTube URL  → yt-dlp downloads best audio → converted to 16kHz mono WAV
  └─ Local file   → ffmpeg converts to 16kHz mono WAV
  └─ WAV file is split into 2-minute chunks (for memory-efficient transcription)

Step 2: Transcription
  └─ faster-whisper (tiny model) runs on each chunk sequentially
  └─ VAD (Voice Activity Detection) filter removes silence
  └─ All chunk texts are concatenated into one full transcript string

Step 3: Title Generation
  └─ First 2000 characters of transcript → Mistral LLM → short title (≤8 words)

Step 4: Summarisation (Map-Reduce)
  └─ Transcript split into 3000-char chunks with 200-char overlap
  └─ Each chunk summarised independently (map phase)
  └─ All summaries combined → final bullet-point summary (reduce phase)

Step 5: Extraction
  └─ Action Items  — task, owner, deadline extracted per item
  └─ Key Decisions — numbered list of decisions
  └─ Open Questions — unresolved or follow-up items

Step 6: RAG Engine Build
  └─ Transcript split into 500-char chunks (50-char overlap)
  └─ Each chunk embedded with all-MiniLM-L6-v2 (HuggingFace)
  └─ Embeddings stored in ChromaDB (persisted to vector_db/)
  └─ LCEL chain: question → retriever (k=4) → Mistral LLM → answer
```

---

## 📦 Module Reference

### `utils/audio_processor.py`
| Function | Description |
|---|---|
| `process_input(source)` | Main entry — detects URL vs local file, returns list of WAV chunk paths |
| `download_youtube_audio(url)` | Downloads best audio from YouTube using `yt-dlp`, converts to WAV |
| `convert_to_wav(input_path)` | Converts any audio/video to 16kHz mono WAV via `ffmpeg` |
| `chunk_audio(wav_path, chunk_minutes=2)` | Splits a WAV file into fixed-length segments using `ffmpeg` |

### `core/transcriber.py`
| Function | Description |
|---|---|
| `load_model()` | Lazily loads the `faster-whisper` `tiny` model (cached globally) |
| `transcribe_chunk(chunk_path, language)` | Transcribes a single audio chunk |
| `transcribe_all(chunks, language)` | Iterates all chunks and returns the concatenated transcript |

### `core/summarizer.py`
| Function | Description |
|---|---|
| `summarize(transcript)` | Map-Reduce summarisation via Mistral LLM |
| `generate_title(transcript)` | Generates a short meeting title from the first 2000 chars |
| `split_transcript(transcript)` | Splits transcript into 3000-char chunks for summarisation |

### `core/extractor.py`
| Function | Description |
|---|---|
| `extract_action_items(transcript)` | Extracts tasks with owner & deadline |
| `extract_key_decisions(transcript)` | Extracts key decisions as a numbered list |
| `extract_questions(transcript)` | Extracts open/unresolved questions |

### `core/vector_store.py`
| Function | Description |
|---|---|
| `build_vector_store(transcript)` | Splits, embeds, and stores transcript in ChromaDB |
| `load_vector_store()` | Loads an existing ChromaDB collection from disk |
| `get_retriever(vector_store, k=4)` | Returns a similarity-search retriever (top-k) |

### `core/rag_engine.py`
| Function | Description |
|---|---|
| `build_rag_chain(transcript)` | Builds the full LCEL RAG chain from transcript |
| `load_rag_chain()` | Loads a previously persisted RAG chain |
| `ask_question(rag_chain, question)` | Invokes the chain and returns the LLM answer |

---

## 🌐 Supported Languages

| Input | Whisper Code | Notes |
|---|---|---|
| `english` | `en` | Full support |
| `hinglish` | `None` (auto-detect) | Whisper auto-detects Hindi/English mix |
| `hindi` | `hi` | Supported via language map |

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `MISTRAL_API_KEY` | ✅ Yes | API key from [console.mistral.ai](https://console.mistral.ai/) |

---

## ⚠️ Known Limitations & Tips

- **Whisper `tiny` model** is fast but less accurate than larger models. For production use, change `"tiny"` to `"base"`, `"small"`, or `"medium"` in `core/transcriber.py`.
- **CPU-only by default.** For GPU acceleration, change `device="cpu"` to `device="cuda"` in both `core/transcriber.py` and `core/vector_store.py`.
- **Long videos** (>1 hour) will take longer to transcribe. The 2-minute chunking strategy helps with memory but not wall-clock time.
- **ChromaDB persistence:** The vector store is saved in `vector_db/`. Delete this folder if you want to reset embeddings between sessions.
- **Mistral API rate limits:** Free-tier keys may hit rate limits on very long transcripts. The map-reduce summarisation already helps mitigate this.
- **FFmpeg must be in PATH.** If you see `FileNotFoundError: ffmpeg`, ensure the `ffmpeg` binary is accessible globally.

---


<div align="center">

**Built with ❤️ using Whisper · LangChain · Mistral AI · ChromaDB · Streamlit**

</div>
