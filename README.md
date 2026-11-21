# RAG-test

RAG-test aims to provide a local, single-user web application for asking
questions about a document. The user will upload one TXT, Markdown or
text-based PDF document, choose models and settings, build an index, and
receive answers with supporting excerpts.

**This describes the target architecture.** Some components already exist,
but the web UI, document-scoped retrieval and parts of the workflow are still
planned. An Ollama embedding adapter exists but is not yet selected by the
application.

The existing dataset example can be run from the repository root with
`python -m src.main`. It uses Anthropic and SentenceTransformers, accesses the
dataset, and may start Qdrant via Docker Compose. Direct execution with
`python src/main.py` is not supported.

For the agreed target behavior in common collection-handling scenarios, see
[Collection handling](COLLECTION_HANDLING.md). It is not fully implemented yet.

## Architecture

```text
                  User
                   |
                   v
        Streamlit web UI [planned]
   upload | model selection | settings | question
                   |
                   v
     +-------- ingestion --------+
     |                           |
     v                           |
Document bytes                   |
     |                           |
     v                           |
Document loader                  |
  TXT / Markdown / PDF           |
     |                           |
     v                           |
SourceRecord + source metadata   |
     |                           |
     v                           |
Text chunker                     |
     |                           |
     v                           |
ChunkPayload + source metadata   |
     |                           |
     v                           |
Embedding provider               |
     |                           |
     v                           |
Qdrant: document-scoped index <--+
                   ^
                   |
     +-------- question answering --------+
     |                                    |
     v                                    |
Question -> embedding provider -> Qdrant search
                                     |
                                     v
                           Relevant document chunks
                                     |
                                     v
                        Prompt + generation provider
                                     |
                                     v
                     Answer + supporting excerpts
                                     |
                                     v
                               Streamlit UI
```

The target default is Ollama for both generation and embeddings, so local
operation does not require an API key. Anthropic and OpenAI generation, and
SentenceTransformers embeddings, are alternative providers. Qdrant stores
the embeddings.

## Workflow, step by step

Module paths below are relative to `src/`. Planned integrations and behavior
are marked explicitly; existing modules are building blocks, not yet a
complete application.

1. **Prepare services and models.** The user starts Qdrant and Ollama and
   installs the chosen Ollama models before running the application. The
   planned application will check service readiness and report missing models;
   it will not silently start services or download models. The existing
   `storage/qdrant_service.py` provides Qdrant readiness checks, but its
   current startup behavior does not yet match this target.

2. **Choose a document and settings.** The planned Streamlit UI will accept
   one TXT, Markdown or text-based PDF document and allow the user to select
   generation and embedding models. Basic and advanced settings will include
   chunking, retrieval and generation options. The UI will validate settings
   before indexing or answering.

3. **Load document records.** `loaders/document_loader.py` selects a loader
   using the filename extension. `loaders/text_loader.py` decodes UTF-8 TXT
   and Markdown; `loaders/pdf_loader.py` extracts text page by page.
   `models/record_models.py` defines `SourceRecord`: text plus source metadata,
   including page numbers for PDFs.

4. **Create chunks.** `chunkers/text_chunker.py` splits each text record into
   bounded pieces and preserves its metadata. `models/chunk_models.py` defines
   `ChunkPayload`, the text and metadata passed to storage. The existing
   chunker prefers whitespace boundaries; configurable overlap and any
   structure-aware Markdown splitting would require further work.

5. **Embed and index.** `ingestion/document_ingestion.py` connects the
   document loader and chunker to `ingestion/ingestion_service.py`, which
   sends chunks to a storage callback in batches. `ingestion/document_indexing.py`
   validates and loads the document before creating a new collection, then
   embeds and stores chunks in batches. Invalid or empty documents create no
   collection; an embedding or Qdrant failure after creation can leave a
   partial collection. It marks the index ready only after completed writes,
   but cannot resume a failed build. Document records and chunks for each record
   are still held in memory. The caller must provide the actual embedding
   provider and model identity.
   `storage/chunk_service.py` embeds and stores chunks in Qdrant, waiting for
   writes to complete; `utils/id_utils.py` derives identifiers from chunk
   payloads. `storage/collection_service.py` creates a collection
   or checks that an existing collection uses an unnamed vector with the
   requested dimension and cosine distance.
   `embeddings/embedding_provider.py` defines an embedding interface, and
   `embeddings/sentence_transformer_provider.py` adapts the existing model
   loaded by `embeddings/embedding_loader.py`.
   `embeddings/ollama_embedding_provider.py` provides Ollama embeddings, but
   runtime provider selection remains planned. A helper can determine vector
   dimensions by embedding a probe; matching dimensions alone do not guarantee
   that different models produce compatible vectors.

6. **Keep the index compatible with its settings.** The planned application
   will associate an index with its document, embedding model and indexing
   settings. `models/index_models.py` defines an `IndexSettings` data
   structure for the document's SHA-256 fingerprint and filename, embedding
   provider and model, and maximum chunk size. `utils/id_utils.py` provides
   a helper to fingerprint the original document bytes, and
   `build_index_settings` combines it with the other settings.
   `IndexBuildManifest` describes a document ID, build ID, collection name and
   index settings. The document builder requires caller-supplied document and
   build IDs and saves a manifest in Qdrant metadata. Callers must ensure build
   IDs are unique; the manifest does not establish ownership or enable resume.
   Lower-level collection creation can still omit a manifest.
   `models/registry_models.py` defines document and build records for a future
   registry. `storage/registry_service.py` can initialize its SQLite schema at
   a caller-supplied path. `register_document` saves a document ID, and
   `register_build` saves a build's manifest for a registered document.
   `get_build` reads a registered build by ID and validates its stored data.
   These operations do not inspect Qdrant or establish collection ownership.
   Indexing integration and recovery remain planned. Registry writers must
   enable SQLite foreign keys on each connection.
   `storage/collection_service.py` can create a new collection with those
   settings and an initial `building` status in its metadata and read settings
   back; this requires Qdrant collection metadata support. Helpers can read
   the status, reject indexes that are not ready, and mark a building index
   ready after the caller completes ingestion. Collections created before the
   status was added may not have it and are not treated as ready.
   `compare_index_settings` reports which settings differ;
   `compare_collection_index_settings` reads stored settings before comparing
   them. `ensure_collection_index_settings` raises an
   `IndexSettingsMismatchError` when settings differ, identifying the
   collection and fields for a future UI. Missing or invalid metadata raises
   `IndexSettingsMetadataError`, identifying the collection and reason.
   `ensure_document_index_compatible` checks readiness, settings and vector
   configuration without changing a collection. `retrieve_document_chunks`
   uses those checks before searching; the older `retrieve_chunks` and
   `rag/rag_service.py` do not yet use them. The caller must supply settings
   matching the actual embedder and prevent concurrent collection changes:
   checks are not atomic, and a model name alone does not detect changed model
   weights. Marking an index ready relies on completing all writes successfully.
   Changing document content, embedding model or chunking settings will
   require an explicit rebuild before querying. Retrieval and generation
   setting changes will not require re-indexing. To rebuild, a caller can use
   `build_document_index` with a new collection name and switch only after it
   succeeds; the old collection is not modified or deleted. Failed builds can
   leave partial collections, and a failed final status update can leave their
   status unknown. Ownership, cleanup, UI confirmation and application
   integration are still planned.

7. **Ask a question.** The planned UI will submit the question to
   `rag/rag_service.py`. The question is embedded using the index's embedding
   model, and `storage/chunk_service.py` searches Qdrant for relevant chunks.
   The checked document retrieval helper expects a collection dedicated to
   that document; it does not filter points within a shared collection. The
   planned application must select and protect the correct collection.

8. **Generate and display the answer.** `rag/prompt_utils.py` builds a prompt
   from the question and retrieved text. `providers/ollama_provider.py`,
   `anthropic_provider.py` and `openai_provider.py` provide generation
   adapters; the Ollama and OpenAI adapters use `providers/http_helper.py`.
   The planned UI will display the answer and supporting excerpts with source
   references. If no relevant context is found, the application should say
   so rather than generate an unsupported answer.

## Scope and limitations

- The first planned UI supports one active document at a time. Saved document
  libraries and durable conversation history are later work.
- PDF support is for extractable text only: there is no OCR. The existing PDF
  loader rejects encrypted PDFs and PDFs with no extractable text; image-only
  pages yield no text.
- Existing storage does not automatically remove old points when document
  content or indexing settings change. Safe rebuild and index ownership
  handling are part of the planned workflow.
- `loaders/dataset_loader.py` and `chunkers/dataset_chunker.py` support
  structured dataset records as an alternative to documents.
  `utils/dataset_utils.py` and `utils/text_utils.py` format dataset values.
  The old DataFrame-based `ingestion/dataset_ingestion.py` is obsolete.
