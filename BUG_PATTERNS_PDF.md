# pdf2txt Bug Patterns

Bug catalog for escaped bugs. Each entry links to the prevention principle it violated
(see [FGT.md](FGT.md) Prevention Principles).

## Bug Catalog

### BUG-001: page_needs_ocr() heuristic fails on embedded fonts

| Field | Value |
|-------|-------|
| **Symptom** | PDFs with embedded fonts report text layers but contain no extractable text. Heuristic says "no OCR needed" but output is empty. |
| **Root Cause** | pymupdf's `get_text("blocks")` gives false positives for pages with embedded fonts — reports text blocks exist but extracted text is empty or gibberish |
| **Fix** | Removed heuristic after 7 iterations. Simplified to: always OCR images, extract text from text layers. Simpler approach is more robust. |
| **Principle Violated** | #4 (Invariants Must Be Tested) — heuristic was never validated against ground truth corpus |
| **Prevention Test** | QUAL-001: extraction of known scanned PDF produces non-empty, readable text |
| **Siblings** | Any PDF classification heuristic relying on pymupdf text block detection |
| **Data Revalidation** | N/A — no stored data, output regenerated on each run |

**Pattern:** Heuristic complexity spiral. The function was modified 7 times across 12
commits (CP-005). Each fix addressed one edge case but introduced others. The simple
approach (OCR all images unconditionally) was more robust than the clever one. See
FGT.md Heuristic Complexity Review trigger.

---

### BUG-002: Feature vector dimension change breaks classifier

| Field | Value |
|-------|-------|
| **Symptom** | AdaptiveLearner crashes with dimension mismatch when loading saved model after feature vector changed from 12 to 14 elements |
| **Root Cause** | `ImageFeature.to_vector()` was updated to include `log_area` and `log_area * is_body` interaction features, but existing database records stored old 12-element vectors |
| **Fix** | Added DB migration to recompute vectors. Added backward-compatible vector handling. |
| **Principle Violated** | #6 (Verify Before Trust) — assumed stored data matched current schema without validation |
| **Prevention Test** | SCHEMA-001: `ImageFeature.to_vector()` returns exactly 14 elements. INV-003: feature vector dimension matches expected constant. |
| **Siblings** | Any ML feature pipeline with persistent state — schema changes need migration plans |
| **Data Revalidation** | Yes — database records reprocessed with new 14-element vectors |

**Pattern:** Schema evolution without migration. When a data structure changes,
all stored instances of that structure must be migrated or the code must handle
both old and new formats. This is CP-001 (Stale Data After Logic Fix).

---

### BUG-003: GPU VRAM exhaustion during OCR

| Field | Value |
|-------|-------|
| **Symptom** | Process killed or hangs when OCR models exceed available GPU VRAM on systems with <8GB |
| **Root Cause** | No check of available GPU memory before loading Surya/PaddleOCR models. Large PDFs with many images could exhaust VRAM mid-batch. |
| **Fix** | Added `--cpu` flag for forced CPU mode. Added VRAM-based batch size auto-tuning. Added `clear_gpu_memory()` calls between batches. |
| **Principle Violated** | #8 (Defenses Must Be Operational) — no resource guard on GPU allocation |
| **Prevention Test** | BOUND-001: OCR gracefully falls back to CPU when GPU unavailable or VRAM insufficient |
| **Siblings** | Any GPU-dependent pipeline without resource guards |
| **Data Revalidation** | N/A — process crash prevented output generation |

**Pattern:** Resource-dependent operations without guards. The defense (GPU memory
management) didn't exist — it wasn't hollow, it was absent. Any pipeline using
GPU resources must check availability before loading and have a CPU fallback path.
