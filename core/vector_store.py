import os

# Limit the embedding model's MKL/OpenMP worker allocation on CPU.
os.environ["MKL_DISABLE_FAST_MM"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

try:
    from langchain_chroma import Chroma
except ModuleNotFoundError:
    from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import hashlib
import json

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"

def get_embedding_model():
    """ for GPU inference, change 'cpu' to 'cuda'"""
    return HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 1},
    )

def build_vector_store(transcript : str)->Chroma:
    print("Building vector Store")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    chunks = splitter.split_text(transcript)

    def _text_hash(text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    # prepare documents with a stable content hash for deduplication
    docs = [
        Document(page_content=chunk, metadata = {'chunk_index' : i, 'text_hash': _text_hash(chunk)})
        for i,chunk in enumerate(chunks)
    ]

    hashes_file = os.path.join(CHROMA_DIR, "hashes.json")

    # If a persisted vector DB exists, load it and only add new documents
    if os.path.exists(CHROMA_DIR) and any(os.scandir(CHROMA_DIR)):
        vector_store = load_vector_store()

        # load known hashes
        try:
            with open(hashes_file, 'r', encoding='utf-8') as f:
                known_hashes = set(json.load(f))
        except Exception:
            known_hashes = set()

        new_docs = [d for d in docs if d.metadata.get('text_hash') not in known_hashes]

        if new_docs:
            vector_store.add_documents(new_docs)
            try:
                vector_store.persist()
            except Exception:
                pass

            # update persisted hash list
            known_hashes.update([d.metadata.get('text_hash') for d in new_docs])
            os.makedirs(CHROMA_DIR, exist_ok=True)
            with open(hashes_file, 'w', encoding='utf-8') as f:
                json.dump(list(known_hashes), f)

        return vector_store

    # No persisted DB — create fresh one and save hashes
    embedding_model = get_embedding_model()
    vector_store = Chroma.from_documents(
        documents= docs,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )

    # persist hashes for future dedup checks
    os.makedirs(CHROMA_DIR, exist_ok=True)
    try:
        with open(hashes_file, 'w', encoding='utf-8') as f:
            json.dump([d.metadata.get('text_hash') for d in docs], f)
    except Exception:
        pass

    return vector_store



def load_vector_store() ->Chroma:
    embedding_model = get_embedding_model()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function= embedding_model,
        persist_directory=CHROMA_DIR
    )

    return vector_store

def get_retriever(vector_store : Chroma, k :int = 4):
    return vector_store.as_retriever(
        search_type = 'similarity',
        search_kwargs = {"k":k}
    )

