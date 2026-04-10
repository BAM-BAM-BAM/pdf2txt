# pdf2txt Code Patterns

Project-specific implementation patterns for document text extraction.
Supplements the generic methodology in [FGT.md](FGT.md).

This file does NOT contain: Generic FGT methodology (see `FGT.md`),
domain rules (see `FGT_DOMAIN_PDF.md`), review triggers (see `REVIEWS_PDF.md`).

---

## Single Source of Truth (Registry)

### Supported Format Registry

The canonical set of supported extensions lives in one place:

```python
# pdf2txt.py (top-level constant)
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.rtf', '.odt'}
```

**Rules:**
- All format discovery (`find_documents`) filters against this set
- All dispatch logic (`extract_text`) routes based on this set
- Adding a new format means: add to this set + add extractor + add tests
- No hardcoded extension checks elsewhere in the codebase

**Enforcement:** `INV-001` test verifies that every extension in
`SUPPORTED_EXTENSIONS` has a handler in `extract_text()`.

---

## Format Dispatch Pattern

Route by file extension at two chokepoints. All extractors return the
same type (`list[str]`), keeping the downstream pipeline format-agnostic.

```python
def extract_text(path: Path, **ocr_params) -> list[str]:
    """Dispatch extraction by file extension. Returns list of text sections."""
    ext = path.suffix.lower()
    if ext == '.pdf':
        return extract_text_from_pdf(path, **ocr_params)
    elif ext == '.docx':
        return extract_text_from_docx(path)
    elif ext in ('.doc', '.rtf', '.odt'):
        return extract_text_via_libreoffice(path)
    else:
        raise ValueError(f"Unsupported format: {ext}")
```

**Key decisions:**
- Single dispatch function, no class hierarchy (complexity proportional to problem)
- OCR params pass through harmlessly to non-PDF extractors
- LibreOffice is optional (only needed for legacy formats)
- ~95% of the pipeline (quality scoring, parallel processing, HUD) is format-agnostic

---

## Error Collection Pattern

Process all documents and collect results rather than aborting on first failure.
Users see ALL issues at once instead of fix-one-rerun cycles.

```python
def process_batch(documents: list[Path]) -> tuple[list[Result], list[Error]]:
    successes, errors = [], []
    for doc in documents:
        try:
            result = process_document(doc)
            successes.append(result)
        except Exception as e:
            errors.append((doc, e))
    return successes, errors
```

**When to apply:** Any pipeline where items are independent and partial
results are useful. pdf2txt processes each document independently -- one
corrupted file shouldn't block the rest.

---

## OCR Fallback Pattern

OCR is inherently fragile. Multiple engines with graceful degradation:

```
Primary: Surya (GPU-accelerated, highest accuracy)
  -> Fallback: PaddleOCR (alternative GPU engine)
       -> Fallback: CPU mode (--cpu flag)
            -> Fallback: Skip OCR (text-only extraction)
```

**GPU memory management:**
- Check available VRAM before loading models
- Auto-tune batch size based on available memory
- `--force-ocr` flag: OCR all pages regardless of text detection
- `--cpu` flag: force CPU mode when GPU is unavailable or OOM
- Always clear GPU memory after OCR batch completes

**Lesson learned:** The `page_needs_ocr()` heuristic was modified 7 times
across 12 commits before being simplified. Simple approach (OCR images,
extract text from text layers) beats clever detection. See FGT.md
Heuristic Complexity Review trigger.

---

## Testing Patterns

### Golden File Tests

Save known-good extraction output as fixtures:

```
tests/
  fixtures/
    sample.pdf          # Known input
    sample_expected.md  # Expected output
```

Test that `extract_text(sample.pdf)` produces output matching the fixture.
When extraction logic changes, update fixtures deliberately (not silently).

### Three-Tier Strategy

1. **Unit tests:** Individual extractor functions with fixture files
2. **Integration tests:** Full `process_document()` pipeline end-to-end
3. **Quality tests (QUAL-*):** Output values are correct, not just non-empty

### Proactive Detection (PRO-*)

```python
def test_pro_no_hardcoded_extensions():
    """No extension strings outside SUPPORTED_EXTENSIONS."""
    # Scan source for '.pdf', '.docx' etc. outside the constant definition
```

---

## Module Size Enforcement

The codebase is currently a single 3,496-line file (`pdf2txt.py`).
This is a known technical debt item (see BACKLOG.md).

**Target structure after modularization:**

```
pdf2txt/
  __init__.py
  cli.py           # Argument parsing
  extractors/      # One per format
  ocr/             # OCR engine wrappers
  learning/        # Adaptive ML system
  quality.py       # Text quality scoring
  hud.py           # Curses-based progress display
  models.py        # Data models (QualityMetrics, ImageFeature)
```

**MODULE_LIMITS** (for PRO-* enforcement):

| Directory | Max Lines | Current |
|-----------|-----------|---------|
| `./` | 3,500 | 3,496 (monolith) |

Tighten limits to 500 per module after modularization.
