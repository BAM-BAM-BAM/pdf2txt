# FGT Development Log: pdf2txt

| Date | Task ID | Summary | Lessons |
|------|---------|---------|---------|
| 2026-04-09 | DOC-001 | Add DOCX/DOC/RTF/ODT support | Thin dispatch at two chokepoints (file discovery + extraction) keeps ~95% of code untouched. Format-agnostic pipeline design pays off. |

---

## Entries

### 2026-04-09: Multi-format document support (DOC-001)

**Change**: Extended pdf2txt to process DOCX, DOC, RTF, and ODT files alongside PDFs.

**Approach**: Added `extract_text()` dispatcher that routes by file extension. DOCX uses python-docx directly; DOC/RTF/ODT convert via LibreOffice headless to DOCX first. All extractors return `list[str]` (sections of text) - the existing pipeline (quality scoring, parallel processing, HUD, improve mode) works unchanged.

**Key decisions**:
- Single dispatch function, no class hierarchy - keeps complexity proportional to the problem
- OCR params pass through harmlessly to non-PDF extractors (no conditional CLI logic needed)
- LibreOffice is optional - only required for legacy .doc/.rtf/.odt formats
- DOCX headings convert to markdown `#` syntax, tables to markdown tables

**Tests**: 34 new tests covering find_documents, DOCX extraction, dispatcher, create_markdown, process_document. All 56 tests pass (22 existing + 34 new).

**Lessons**:
- The existing pipeline was already ~95% format-agnostic - only `find_pdfs()` and `extract_text_from_pdf()` were truly PDF-specific
- Adding format support at the extraction layer means all downstream features (parallel, HUD, quality scoring, improve mode) work automatically
