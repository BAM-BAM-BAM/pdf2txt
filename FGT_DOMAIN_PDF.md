# FGT Domain Knowledge: Document Processing

This file contains domain-specific knowledge for document text extraction.
It supplements the generic FGT methodology defined in [FGT.md](FGT.md).

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
- LibreOffice not safely concurrent - be cautious with parallelism

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
| Empty DOCX extraction | DOCX | No text in scanned DOCX | Expected - DOCX OCR not supported |
| LibreOffice lock | DOC/RTF/ODT | Concurrent conversion fails | Limit parallelism for these formats |

---

## Integration with Generic FGT

1. Use Software Architect perspective for code structure
2. Use QA Engineer perspective for edge cases (malformed files, missing deps)
3. Add domain-specific invariants here as discovered
