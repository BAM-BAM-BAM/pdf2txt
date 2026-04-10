# FGT Domain Knowledge: Document Processing

This file contains domain-specific knowledge for document text extraction.
It supplements the generic FGT methodology defined in [FGT.md](FGT.md).

---

## Terminology

| Term | Definition |
|------|-----------|
| OCR | Optical Character Recognition -- converting images of text to actual text |
| pymupdf | Python library (fitz) for PDF manipulation and text extraction |
| Surya | GPU-accelerated OCR engine for high-accuracy text recognition |
| PaddleOCR | Alternative OCR engine with good multi-language support |
| VRAM | Video RAM -- GPU memory used by OCR models |
| Feature vector | Numeric representation of image characteristics for ML-based OCR decisions (14 elements) |
| Quality score | 0.0-1.0 metric combining word ratio, content score, and gibberish penalty |
| Adaptive learning | ML system that learns which images are worth OCR-ing based on past results |
| HUD | Heads-up display -- curses-based progress interface during batch processing |
| LibreOffice headless | Server-mode LibreOffice used to convert DOC/RTF/ODT to DOCX for extraction |

---

## Supported Formats

| Format | Library | Notes |
|--------|---------|-------|
| PDF | pymupdf + OCR engines | Native text + OCR for scanned pages |
| DOCX | python-docx | Paragraphs, headings, tables, page breaks |
| DOC | LibreOffice headless | Converts to DOCX, then extracts |
| RTF | LibreOffice headless | Converts to DOCX, then extracts |
| ODT | LibreOffice headless | Converts to DOCX, then extracts |

---

## Invariants

| ID | Rule | Constraint |
|----|------|-----------|
| INV-001 | Supported extensions complete | Every extension in `SUPPORTED_EXTENSIONS` must have a handler in `extract_text()` |
| INV-002 | Quality score range | `QualityMetrics.total_score` must be in [0.0, 1.0] |
| INV-003 | Feature vector dimension | `ImageFeature.to_vector()` must return exactly 14 elements |
| INV-004 | Output file naming | Output .md file must have same stem as input document |
| INV-005 | Section list non-empty | `extract_text()` must return at least one section (possibly empty string) for any supported format |

---

## Domain Constants

| Constant | Value | Source |
|----------|-------|--------|
| SUPPORTED_EXTENSIONS | {.pdf, .docx, .doc, .rtf, .odt} | Application design |
| Feature vector dimensions | 14 | `ImageFeature.to_vector()` definition |
| Quality score whitespace threshold | 95% | `is_mostly_white` (>95% pixels above 240) |
| Brightness contrast threshold | std > 30 | `has_contrast` field in ImageFeature |

---

## Canonical Data Formats

Every data type with multiple possible representations must have ONE canonical
storage format. Validate at ingestion. Convert only at display.

| Data Type | Canonical Format | Example | Validation Rule |
|-----------|-----------------|---------|-----------------|
| File path | `pathlib.Path` | `Path("/docs/file.pdf")` | `isinstance(x, Path)` |
| Extracted text | `list[str]` | `["Page 1 text", "Page 2 text"]` | All extractors return `list[str]` |
| Quality score | float 0.0-1.0 | `0.85` | `0.0 <= x <= 1.0` |
| Feature vector | `list[float]` len=14 | `[0.5, 0.3, ...]` | `len(v) == 14` |
| File extension | lowercase with dot | `".pdf"` | `ext == ext.lower() and ext.startswith(".")` |

---

## Domain Expert Perspectives

| Perspective | Role | Focus Areas | Key Questions |
|-------------|------|-------------|---------------|
| **Document Analyst** | Content extraction | Text fidelity, structure | "Is all text captured? Is structure preserved?" |
| **OCR Specialist** | Image-to-text | Recognition accuracy | "Are scanned pages handled? What about fonts?" |
| **Data Engineer** | Pipeline design | Performance, errors | "Does it scale? How are failures handled?" |

---

## Domain-Specific Rules

### Text Extraction (All Formats)
- Preserve reading order (columns, tables)
- Convert headings to markdown `#` syntax
- Convert tables to markdown table format
- Output .md file alongside each source document

### PDF-Specific
- Handle embedded fonts correctly
- OCR fallback for scanned/image-based pages
- Adaptive learning for OCR optimization

### DOCX-Specific
- Split on page breaks when present
- Preserve heading hierarchy
- Extract table content as markdown tables

### Non-DOCX Legacy Formats (.doc, .rtf, .odt)
- Require LibreOffice for conversion
- Convert to DOCX first, then extract
- LibreOffice not safely concurrent -- be cautious with parallelism

### Error Handling
- Corrupted files should fail gracefully
- Missing dependencies (LibreOffice) need clear error messages
- Password-protected files need clear error messages
- Large files need progress indication
- Unsupported formats raise ValueError

---

## Domain-Specific Bug Patterns

| Pattern | Format | Symptom | Prevention |
|---------|--------|---------|------------|
| Embedded fonts fool text detection | PDF | pymupdf reports text blocks but extracted text is empty | Always verify extracted text is non-empty; don't trust `get_text("blocks")` alone |
| Empty DOCX extraction | DOCX | No text in scanned DOCX | Expected -- DOCX OCR not supported |
| LibreOffice lock | DOC/RTF/ODT | Concurrent conversion fails | Limit parallelism for these formats |
| GPU VRAM exhaustion | PDF (OCR) | Process killed or hangs during OCR | Check VRAM before loading models; provide CPU fallback |
| Feature vector mismatch | PDF (ML) | Classifier crashes on dimension mismatch | Maintain backward compat or trigger migration when vector dims change |

---

## Pitfalls

- **Embedded fonts fool text detection**: pymupdf's `get_text("blocks")` may report text blocks for pages with embedded fonts that contain no extractable text. Always verify extracted text is non-empty.
- **get_images() vs get_text("blocks")**: pymupdf's `get_text("blocks")` doesn't reliably surface all images. Use `get_images()` as the authoritative image list.
- **GPU VRAM unpredictable**: OCR model memory usage varies per page complexity. Always check available VRAM before loading models. Provide CPU fallback.
- **LibreOffice concurrency**: LibreOffice headless mode is not safely concurrent. Limit parallelism for .doc/.rtf/.odt conversions.
- **Transformers version compatibility**: transformers 5.0.0 is incompatible with surya-ocr. Pin to 4.57.x.
- **WSL GPU zombies**: WSL may hold zombie GPU processes after crashes. Requires full WSL restart to clear.
- **When in doubt, OCR**: False negatives (missing text) cost more than false positives (wasted compute). Prefer OCR-ing unnecessarily over missing text.

---

## Integration with Generic FGT

1. Use Software Architect perspective for code structure
2. Use QA Engineer perspective for edge cases (malformed files, missing deps)
3. Add domain-specific invariants here as discovered
