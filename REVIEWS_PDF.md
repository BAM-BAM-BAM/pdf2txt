# pdf2txt Review Triggers

When to pause and think before committing. The goal is to turn review insights
into automated tests -- if you find yourself checking the same thing twice,
write a test instead.

This file does NOT contain: Generic FGT methodology (see `FGT.md`),
domain rules (see `FGT_DOMAIN_PDF.md`), code patterns (see `PATTERNS_PDF.md`).

---

## Review Types

| Trigger | Ask | Automate As |
|---------|-----|-------------|
| Changed extraction logic | Is reading order preserved? Are headings converted? Tables formatted? | QUAL-\*, XVAL-\* |
| Changed OCR pipeline | GPU memory managed? CPU fallback works? Quality scoring applied? | BOUND-\*, INT-\* |
| Added new format support | Dispatcher updated? Tests added? Extension in SUPPORTED_EXTENSIONS? LibreOffice dependency documented? | INV-\*, CONTRACT-\* |
| Changed output format | Markdown well-formed? Source path correct? Page/section labels correct? | SCHEMA-\*, QUAL-\* |
| Changed ML/adaptive logic | Feature vector dimensions consistent? DB migration needed? Classifier retrained? | SCHEMA-\*, INV-\* |
| Changed config/CLI args | Do all consumers handle the new flags? Backward compatible? | BOUND-\* |
| Changed data interpretation | Is existing data contaminated by old logic? (Bug Abstraction step 6) | PRO-\*, INT-\* |
| End of feature | Does changing input actually change output? Are outputs correct? | SENS-\*, QUAL-\*, INT-\* |
| Heuristic modified 3+ times | Is complexity justified? Different approach simpler? Revert to cruder-but-robust? | Architecture review (FGT.md) |

---

## Pre-Implementation

- Does similar functionality already exist? (Don't duplicate)
- What modules does this change affect? (Identify contracts)
- What domain rules apply? (Check `FGT_DOMAIN_PDF.md`)

## Post-Implementation

- Build passes
- All tests pass (`python3 -m pytest tests/ -v`)
- No stub code (empty returns, unused params, TODO comments)

---

## PDF-Specific Reviews

### Text Extraction Review

**Trigger:** After modifying any extraction logic (PDF, DOCX, or legacy format).

Checklist:
- [ ] Reading order preserved (multi-column layouts)
- [ ] Headings converted to markdown `#` syntax
- [ ] Tables formatted as markdown tables
- [ ] Empty pages handled (not omitted, not duplicated)
- [ ] Page breaks respected in multi-page documents
- [ ] Unicode characters preserved correctly
- [ ] Output .md file has correct filename (same stem as input)

### OCR Pipeline Review

**Trigger:** After modifying OCR logic, GPU memory handling, or model loading.

Checklist:
- [ ] GPU VRAM checked before loading models
- [ ] `--cpu` flag forces CPU mode correctly
- [ ] `--force-ocr` flag OCRs all pages
- [ ] Batch size auto-tuned for available memory
- [ ] GPU memory cleared after OCR completes
- [ ] Graceful fallback when GPU unavailable
- [ ] Quality scoring applied to OCR output

### Format Support Review

**Trigger:** After adding a new document format.

Checklist:
- [ ] Extension added to `SUPPORTED_EXTENSIONS`
- [ ] Dispatcher (`extract_text()`) routes to new handler
- [ ] Extractor returns `list[str]` (consistent with all others)
- [ ] Tests added for new format (at least: non-empty extraction, empty file, malformed file)
- [ ] LibreOffice dependency documented if needed
- [ ] `--formats` flag updated if applicable

### Output Quality Review

**Trigger:** End of feature or after significant extraction changes.

Checklist:
- [ ] Markdown output is well-formed (renders correctly)
- [ ] Source file path included in output metadata
- [ ] Page/section labels are accurate
- [ ] No gibberish text from failed OCR (quality score check)
- [ ] Extracted text matches visual content of source document
