# FGT Development Log: pdf2txt

| Date | Task ID | Summary | Lessons |
|------|---------|---------|---------|
| 2026-01-26 | PDF-001 | Initial PDF extraction with parallel processing | Format-agnostic pipeline design with ProcessPoolExecutor. Use spawn context to avoid fork/curses issues. |
| 2026-01-26 | PDF-002 | Add PaddleOCR and Surya OCR engines | Multiple OCR backends with fallback. GPU memory is a hard constraint -- always plan for CPU fallback. |
| 2026-01-27 | PDF-003 | OCR detection heuristic (page_needs_ocr) | Heuristic failed on embedded fonts after 7 iterations. Eventually removed. Heuristic complexity spiral (CP-005). |
| 2026-01-27 | PDF-004 | Adaptive ML learning system for OCR decisions | 12 commits of ML evolution (LogisticRegression -> DecisionTree, feature vectors, Thompson sampling). Over-engineered -- start simple. |
| 2026-01-28 | PDF-005 | Feature vector dimension change (12 to 14) | Broke classifier DB records. Schema changes need migration plans. Stale data after logic fix (CP-001). |
| 2026-01-28 | PDF-006 | GPU VRAM management (--force-ocr, --cpu) | Added resource guards and CPU fallback. Defenses must exist, not just be operational (Principle 8). |
| 2026-04-09 | PDF-007 | FGT files added retroactively (commit 23/24) | Cargo-culted from VE project. PATTERNS/REVIEWS contained wrong-domain content. FGT_LOG empty. Apply methodology from day 1 (CP-006). |
| 2026-04-09 | DOC-001 | Add DOCX/DOC/RTF/ODT support | Thin dispatch at two chokepoints keeps ~95% of code untouched. Format-agnostic pipeline design pays off. |

---

## Entries

### 2026-01-26: Initial PDF extraction (PDF-001)

**Change**: Built core extraction pipeline with parallel processing via ProcessPoolExecutor.

**Lessons**:
- Use `multiprocessing.get_context("spawn")` to avoid terminal/curses issues with fork
- Design the pipeline to be format-agnostic from the start -- it paid off when adding DOCX support later

### 2026-01-26: OCR engine integration (PDF-002)

**Change**: Added Surya (GPU) and PaddleOCR as OCR backends.

**Lessons**:
- GPU memory is a hard constraint, not a soft one. Systems with <8GB VRAM will OOM on large PDFs
- Always provide a CPU fallback path for GPU-dependent operations
- Pin dependency versions (transformers 5.0.0 broke surya-ocr compatibility)

### 2026-01-27: page_needs_ocr() heuristic spiral (PDF-003)

**Change**: Attempted to build a smart heuristic to decide which pages need OCR. Modified 7 times across 12 commits.

**Lessons**:
- pymupdf's `get_text("blocks")` reports false positives for embedded fonts
- The heuristic was eventually removed in favor of a simpler approach (OCR all images, extract text from text layers)
- This is CP-005 (Heuristic Complexity Spiral) -- when a heuristic has been modified 3+ times, trigger architecture review
- Simple and robust beats clever and fragile

### 2026-01-27: Adaptive ML system (PDF-004)

**Change**: Built feature extraction, Bayesian learning, K-means clustering, Thompson sampling, and skip validation for OCR decisions.

**Lessons**:
- 12 commits of incremental ML complexity that could have been avoided with a simpler approach
- LogisticRegression was replaced by DecisionTreeClassifier for better accuracy
- Never add Bayesian exploration or Thompson sampling unless you have quantitative evidence the simpler approach is insufficient
- The ML pipeline was the most complex part of the codebase but provided marginal improvement

### 2026-01-28: Feature vector dimension change (PDF-005)

**Change**: Updated ImageFeature.to_vector() from 12 to 14 elements (added log_area and interaction term). Broke existing classifier database records.

**Lessons**:
- Schema changes need migration plans before implementation
- Stored data must be validated against current schema (Principle 6: Verify Before Trust)
- This is CP-001 (Stale Data After Logic Fix) -- fix the code AND the data

### 2026-01-28: GPU VRAM management (PDF-006)

**Change**: Added --force-ocr, --cpu flags. Implemented VRAM-based batch size auto-tuning and memory cleanup.

**Lessons**:
- Resource-dependent operations need guards (check availability before loading)
- WSL may hold zombie GPU processes after crashes -- requires full WSL restart
- A defense that doesn't exist is worse than a hollow defense (Principle 8 extension)

### 2026-04-09: FGT retrofit and cargo-cult detection (PDF-007)

**Change**: FGT methodology files were added in commit 23 of 24. PATTERNS_PDF.md and REVIEWS_PDF.md were copied from the VE project without adaptation.

**Lessons**:
- Cargo-culting FGT files provides false confidence -- the wrong-domain content is never consulted
- FGT_LOG remained empty for all 24 commits, meaning the self-improvement loop never activated
- Retroactive methodology adoption is significantly less effective than proactive adoption
- This experience led to Principle 10 (Scaffold Before Building) and the cargo-cult warning in CLAUDE_MD_TEMPLATE

### 2026-04-09: Multi-format document support (DOC-001)

**Change**: Extended pdf2txt to process DOCX, DOC, RTF, and ODT files alongside PDFs.

**Approach**: Added `extract_text()` dispatcher that routes by file extension. DOCX uses python-docx directly; DOC/RTF/ODT convert via LibreOffice headless to DOCX first. All extractors return `list[str]` (sections of text) -- the existing pipeline (quality scoring, parallel processing, HUD, improve mode) works unchanged.

**Key decisions**:
- Single dispatch function, no class hierarchy -- keeps complexity proportional to the problem
- OCR params pass through harmlessly to non-PDF extractors (no conditional CLI logic needed)
- LibreOffice is optional -- only required for legacy .doc/.rtf/.odt formats
- DOCX headings convert to markdown `#` syntax, tables to markdown tables

**Tests**: 34 new tests covering find_documents, DOCX extraction, dispatcher, create_markdown, process_document. All 56 tests pass (22 existing + 34 new).

**Lessons**:
- The existing pipeline was already ~95% format-agnostic -- only `find_pdfs()` and `extract_text_from_pdf()` were truly PDF-specific
- Adding format support at the extraction layer means all downstream features (parallel, HUD, quality scoring, improve mode) work automatically
