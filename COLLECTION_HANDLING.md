# Collection handling

**Agreed target behavior: not fully implemented yet.**

A collection is a document's searchable index in Qdrant. Rebuilding creates
a separate replacement collection, not changes to the one currently in use.

## Use cases

| Use case | Expected behavior |
| --- | --- |
| **First build** | Validate the document, then embed and save directly to Qdrant in batches, without a separate embedding file or accumulating all embeddings in memory. Allow questions only after completion and verification. |
| **Upload an unsupported, invalid or empty document** | Reject it before creating a collection and explain the problem. |
| **Reopen an unchanged document** | Reuse its saved index when settings match, without rebuilding or re-embedding. |
| **Change answer or search settings** | Reuse the index. Generation model, prompt, result count and score threshold changes do not require re-indexing. |
| **Change the document, embedding model or chunking** | Require an explicit build or selection of a compatible saved index. Reuse embeddings only for matching input and model settings; renaming alone needs no re-embedding. |
| **Rebuild an existing index** | Keep the working collection available with its original document and settings, clearly identified. Never treat it as representing the pending changes. |
| **Rebuild succeeds** | Automatically use the verified replacement, retain the previous good version for rollback, and safely remove older versions. |
| **Build fails or the app stops** | Preserve completed batches; never answer from the partial replacement. Offer resume or explicit discard after restart. Resume verifies saved points and skips completed work. The working index stays intact. |
| **Start another build with one unfinished** | Allow one unfinished replacement per document. Resume matching work, or explicitly discard it before starting different work. |
| **Roll back** | Use the retained previous good collection and its settings, without re-embedding. |
| **Delete a document or collection** | Show affected data and request confirmation. Check ownership and references; protect unrelated collections and data still in use. |
| **Resources run out or cleanup fails** | Explain the problem; preserve protected versions. Report pending cleanup and block further replacements if older collections cannot be safely removed. |
| **Request a backup** | Save a Qdrant snapshot, the document and its application bookkeeping. Backups happen only on explicit request. |
| **Lose a collection** | Restore a usable backup or reuse surviving compatible vectors; otherwise re-embed. Disk-loss protection requires a backup on a separate disk or system. |

## How many collections are kept?

Normally, **the current and one previous successful collection per document**.
A rebuild temporarily adds a third:

```text
First build:    A
Second build:  A + B       (use B)
Third build:   B + C       (use C; delete A after safe checks)
```

Older versions are removed only after the replacement is verified and in use,
and no references or running requests still need them. Failed builds never
remove either good version. Clearing the UI or chat does not delete saved
indexes.
