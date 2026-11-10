# ScamSagot Streamlit Interface

This app is a Streamlit front end for the RAG behavior in `../ScamSagot_Final_RAG_Pipeline_and_Chunking_Experiment.ipynb`.

It preserves the notebook's source-aware chunking, `all-MiniLM-L6-v2` embeddings, dense FAISS `IndexFlatIP` retrieval over normalized embeddings, and Ollama `/api/chat` answer generation. It does not implement BM25, RRF, Chroma, or hybrid retrieval.

## Run locally

Open PowerShell in this folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

In the app, upload the consolidated ScamSagot corpus PDF, choose a chunking configuration, then enter the Ollama URL reachable from the computer running Streamlit and the exact model identifier installed in Ollama. Click **Check connection** before asking questions. The app does not assume a URL or model name.

The Sources panel displays the actual passages retrieved for each question, including rank, chunk ID, FAISS score, source title, and available combined-corpus/original page metadata. Source titles and page details depend on the corpus markers and text that can be extracted.

The app expects the consolidated corpus to contain the seven source markers defined in `rag_engine.py`. Scanned PDFs without extractable text need OCR before use. Answers are informational and are not legal advice.
