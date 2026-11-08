# RAG-test
Testing Retrieval-Augmented Generation with a database of recipes

The current example stores dataset chunks in the `dataset_chunks` Qdrant
collection. The older `recipes` collection is not modified or migrated.
If the dataset, chunking settings, or embedding model change, rebuild the
`dataset_chunks` collection before rerunning; existing points are not removed
automatically.

Provider adapters are available for Anthropic, Ollama and OpenAI. Ollama
requires a running server and a locally installed model; OpenAI requires an API
key. The current example in `main.py` still uses Anthropic.

A UTF-8 text loader and text chunker are available in
`src/processing/text_loader.py` and `src/processing/text_chunker.py`; they are
not yet connected to the example pipeline or a web interface.
