# Learnings from pdf2txt for the Event Scraper/Planner App

> **STATUS: INTEGRATED** -- Key lessons from this document have been integrated into
> `FGT_LOG.md` (entries PDF-001 through PDF-007), `BUG_PATTERNS_PDF.md` (BUG-001 through
> BUG-003), and `FGT_DOMAIN_PDF.md` (pitfalls, invariants, domain constants). This file
> is retained as historical reference. New lessons should go directly into the FGT files.

## Purpose

This document captures hard-won lessons from 24 commits of iterative development on a PDF text extraction tool (`pdf2txt`). It's written for a new Claude Code instance building a social event scraping and aggregation app, so you can avoid the same evolutionary pain and get things right from the start.

---

## 1. Architecture: Don't Build a Monolith

### What happened in pdf2txt
The entire project ended up as a **single 3,496-line Python file** (`pdf2txt.py`) containing 30+ classes and functions covering: data models, OCR engines, adaptive ML learning, a curses-based HUD, file I/O, multiprocessing workers, argument parsing, and more. This happened incrementally — each feature seemed small enough to add inline, until it wasn't.

### What to do differently
Start with **module separation from commit #1**:

```
event_app/
├── scrapers/          # One module per source (instagram.py, pub_websites.py, etc.)
│   ├── base.py        # Abstract scraper interface
│   ├── instagram.py
│   └── pub_sites.py
├── models/            # Data models (Event, Venue, Source)
│   ├── event.py
│   └── venue.py
├── aggregator/        # Deduplication, merging, conflict resolution
├── storage/           # Database layer
├── display/           # UI/output formatting
├── cli.py             # Argument parsing only
└── main.py            # Orchestration only
```

**Key principle**: Each scraper WILL break independently (websites change, APIs change, rate limits change). If they're in separate modules, you fix one without touching others.

---

## 2. The Scraper Fragility Problem (Lessons from OCR Fragility)

### What happened in pdf2txt
OCR was inherently fragile — different PDF types needed different handling:
- Scanned docs needed full-page OCR
- Digital PDFs with images needed "hybrid" mode
- Some PDFs had embedded fonts that fooled detection heuristics
- GPU memory could OOM unpredictably

This led to **7 commits fixing/refactoring OCR handling** before it stabilized, including removing a `page_needs_ocr()` function that never worked reliably, simplifying to a brute-force approach, then re-adding hybrid mode more carefully.

### What to do differently for scrapers
Web scrapers are even more fragile than OCR. Plan for failure from day one:

1. **Each scraper returns a Result type**, not raw data:
   ```python
   @dataclass
   class ScrapeResult:
       events: list[Event]
       errors: list[ScrapeError]
       source_health: SourceHealth  # healthy, degraded, broken
       scrape_timestamp: datetime
   ```

2. **Scraper health tracking** — don't just log errors, track patterns:
   - Success rate over last N runs
   - Auto-disable scrapers that fail >3 consecutive times
   - Alert when a previously-healthy scraper starts degrading

3. **Don't try to be clever with detection heuristics early**. In pdf2txt, `page_needs_ocr()` was a clever heuristic that was eventually removed in favor of a simpler approach. Start with the simplest extraction logic per source, then optimize.

4. **Pin your selectors/parsing logic as "scraper configs"** that are easy to update when a website changes its HTML structure, rather than hardcoding CSS selectors deep in logic.

---

## 3. FGT Methodology: Apply It From Day Zero, Properly

### What happened in pdf2txt
FGT docs were added in **commit 23 of 24** — essentially retroactively. Worse:
- `FGT_LOG.md` is completely empty (no lessons were ever recorded)
- `PATTERNS_PDF.md` contains React Flow / TypeScript patterns from a *different* project (a financial visualization tool), not PDF/Python patterns
- `REVIEWS_PDF.md` references VE (another project's) review types, config schemas, and React components
- `FGT_DOMAIN_PDF.md` has only skeleton content ("Add patterns as discovered during development")

The FGT framework was **cargo-culted from another project** without being adapted to this one's actual needs.

### What to do differently

**Create domain-specific FGT files from scratch for the event app:**

#### FGT_DOMAIN_EVENTS.md should cover:
| Perspective | Focus for Event App |
|-------------|-------------------|
| **Event Data Analyst** | Is event data complete? (title, date, time, location, description) Are dates parsed correctly across formats? |
| **Scraper Engineer** | Are we respecting rate limits? Handling auth? Dealing with pagination? What happens when HTML changes? |
| **UX Designer** | Is deduplication transparent? Can users see source attribution? Is the calendar view usable? |
| **Data Engineer** | How do we handle time zones? Recurring events? Events that span multiple days? |

#### PATTERNS_EVENTS.md should cover (in Python, not TypeScript):
- Scraper base class pattern
- Event normalization pattern (different sources use different date/time formats)
- Deduplication strategy (fuzzy matching on title + venue + date)
- Rate limiting pattern
- Retry with backoff pattern

#### REVIEWS_EVENTS.md should include:
- **New Scraper Review**: Does it handle empty results? Pagination? Rate limiting? Auth expiry?
- **Event Normalization Review**: Are dates parsed to UTC? Are venues geocoded consistently? Are descriptions sanitized?
- **Aggregation Review**: Are duplicates detected? Are conflicts resolved? Is source attribution preserved?

#### FGT_LOG.md — Actually use it!
After every bug fix, record: what broke, why FGT didn't catch it, what invariant/test was added.

---

## 4. Testing: Write Tests Alongside Features, Not After

### What happened in pdf2txt
Tests (`test_adaptive_learner.py`, 22 tests) were added only after the adaptive learning system was fully built. By then:
- The feature vector size had already changed once (12 → 14 elements), breaking assumptions
- Database migrations were needed for schema changes
- The ML model had already been swapped (LogisticRegression → DecisionTreeClassifier)

The test file imports directly from the monolithic `pdf2txt.py`, making it hard to test components in isolation.

### What to do differently
1. **Write scraper tests first** — even before the scraper works:
   - Save sample HTML/JSON responses as fixtures
   - Test that your parser extracts the correct events from known HTML
   - When the website changes, update the fixture and fix the parser
   
2. **Test the aggregation layer independently** from scrapers:
   - Feed it synthetic events with known duplicates
   - Verify deduplication, merge logic, conflict resolution
   
3. **Integration test each scraper against live sources sparingly** — not in CI, but as a manual smoke test with `--live` flag

4. **Golden file tests for event normalization** — save expected output for known inputs

---

## 5. Dependency Management: Pin Everything Early

### What happened in pdf2txt
- `transformers` 5.0.0 broke `surya-ocr` — had to pin to 4.57.x
- PaddleOCR and Surya have conflicting VRAM requirements
- GPU memory management required manual `clear_gpu_memory()` calls
- WSL zombie GPU processes after crashes required full WSL restarts

### What to do differently
1. **Use a `pyproject.toml` or `requirements.txt` with pinned versions from day one**
2. **Instagram scraping** will likely need `instaloader` or similar — these break with Instagram API changes frequently. Pin the version and have a fallback strategy.
3. **Use `requests` + `beautifulsoup4` for pub websites** — stable and well-tested
4. **Consider `playwright`/`selenium` only if sites require JS rendering**, and isolate browser-dependent scrapers from simple HTTP ones
5. **If using any ML for event deduplication** (e.g., sentence-transformers for fuzzy matching), pin those models and test in CI

---

## 6. The Heuristic → ML Evolution (Avoid It)

### What happened in pdf2txt
The project went through this evolution:
1. Simple heuristic: `page_needs_ocr()` — didn't work well
2. Removed heuristic, brute force: OCR everything
3. Hybrid approach: OCR only images, extract text from text layers
4. Adaptive ML system: Feature extraction → Bayesian learning → K-means clustering → Thompson sampling → Skip validation
5. Swapped ML model: LogisticRegression → DecisionTreeClassifier
6. Added quality scoring, corpus mismatch detection, exploration rates...

This was **12 commits of incremental ML complexity** that could have been avoided with a simpler approach upfront.

### What to do differently for event dedup/matching
- **Start with exact matching** (same title + same date + same venue name = duplicate)
- **Then add fuzzy matching** (Levenshtein distance on titles, ±1 hour on times)
- **Only add ML if the simple approach demonstrably fails** on real data
- **Never add Bayesian exploration, Thompson sampling, or adaptive thresholds** unless you have quantitative evidence the simpler approach is insufficient

---

## 7. Database Schema: Design for Migration From Day One

### What happened in pdf2txt
The database schema evolved through multiple migrations:
- Added `quality_score`, `quality_word_count`, `previous_quality_score`, `quality_delta`, `extraction_mode` columns
- Added indexes on `ocr_performed` and composite indexes for performance
- Migration logic had to be hand-coded with `PRAGMA table_info` checks
- Tests had to be updated when the schema changed

### What to do differently
1. **Use a proper migration tool** (Alembic for SQLAlchemy, or even just numbered SQL migration files)
2. **Design the event schema with all likely fields from the start**:
   ```python
   @dataclass
   class Event:
       id: str                    # Generated UUID
       title: str
       description: str | None
       start_time: datetime       # UTC
       end_time: datetime | None  # UTC
       venue_name: str | None
       venue_address: str | None
       venue_lat: float | None
       venue_lon: float | None
       source_url: str
       source_type: str           # "instagram", "website", "manual"
       source_id: str | None      # Platform-specific ID
       image_url: str | None
       tags: list[str]
       price: str | None          # Free-text ("$10", "Free", "£5-£15")
       recurring: bool
       recurrence_rule: str | None  # iCal RRULE format
       scraped_at: datetime
       confidence: float          # 0-1, how confident we are in the extraction
       raw_data: str | None       # JSON blob of original scraped data
   ```
3. **Store raw scraped data alongside normalized data** — you'll want to re-parse when your normalization improves, without re-scraping

---

## 8. FGT Bidirectional Updates

### What should go back into the core FGT.md
Based on pdf2txt's experience, these additions would improve FGT for any project:

1. **New Pillar Candidate — "Scaffold Before Building"**:
   > Set up module structure, FGT domain files, test framework, and CI before writing the first feature. Retroactive adoption of methodology is significantly less effective than proactive adoption.

2. **Anti-pattern to add to FGT.md**: "Cargo-culting FGT files"
   > Copying FGT domain/pattern/review files from another project without adapting them to the current domain provides false confidence. Each project needs domain-specific content written from scratch, informed by the methodology's structure but populated with domain-relevant details.

3. **FGT_LOG is critical infrastructure, not optional**:
   > An empty FGT_LOG means the self-improvement loop (Pillar 3) never activated. Enforce FGT_LOG entries as part of the commit process for any bug fix.

4. **New review trigger for FGT.md**: "Heuristic Complexity Review"
   > When a heuristic or detection algorithm has been modified 3+ times, trigger an Architecture Review to evaluate whether:
   > - The approach should be simplified
   > - The problem needs a fundamentally different solution
   > - The complexity is justified by measurable improvement

### What pdf2txt's FGT_DOMAIN_PDF.md should be updated with
Based on what was actually learned (but never recorded):

- **Bug Pattern**: `page_needs_ocr()` heuristics fail on PDFs with embedded fonts that report text layers but contain no extractable text
- **Bug Pattern**: PyMuPDF's `get_text("blocks")` doesn't reliably surface all images; `get_images()` must be checked separately
- **Bug Pattern**: GPU VRAM usage by OCR models is unpredictable; always check available memory before loading models
- **Invariant**: Feature vectors must maintain backward compatibility OR trigger retraining when dimensions change
- **Domain Rule**: When in doubt about whether to OCR, prefer doing it (false negatives cost more than false positives in text extraction)

---

## 9. Process Improvement Recommendations for the Event App

### Pre-Implementation Checklist (before writing any code)
- [ ] Module structure created with `__init__.py` files
- [ ] `FGT_DOMAIN_EVENTS.md` written with event-specific perspectives and rules
- [ ] `PATTERNS_EVENTS.md` written with Python scraper patterns
- [ ] `REVIEWS_EVENTS.md` written with scraper, normalization, and aggregation review checklists
- [ ] Database schema designed with migration tooling
- [ ] `requirements.txt` / `pyproject.toml` with pinned dependencies
- [ ] Test directory with at least one fixture per scraper source
- [ ] CI pipeline configured (even if it just runs `pytest`)
- [ ] `FGT_LOG.md` created and referenced in CLAUDE.md with "log every bug fix" instruction

### During Development
- Write scraper → write test for scraper → commit together
- Every bug fix includes: fix + test + FGT_LOG entry
- Review scrapers from Data Engineer perspective (rate limits, pagination, error handling) not just "does it work"
- Keep each scraper under 200 lines; if it's growing, the HTML parsing needs to be extracted

### Key Invariants to Establish Early
- Every event must have a title and start_time (reject otherwise)
- All times stored as UTC with timezone info preserved in metadata
- Deduplication must be idempotent (running it twice produces the same result)
- No scraper should make more than N requests per minute (configurable per source)
- Raw scraped data is always preserved alongside normalized data

---

## 10. Summary: The Top 5 Things That Would Have Saved the Most Time

1. **Module separation from day one** — the monolithic file made every change riskier and harder to test
2. **FGT applied proactively, not retroactively** — the methodology docs were added last, and were copied from another project without adaptation
3. **Simple solutions before clever ones** — the heuristic → ML pipeline added enormous complexity that may not have been needed
4. **Tests alongside features** — testing was added after major refactors, missing opportunities to catch regressions earlier  
5. **Pin dependencies immediately** — a breaking `transformers` upgrade could have been avoided with version pinning

---

*Generated from analysis of pdf2txt project (24 commits, 3,496 LOC) on 2026-04-05*
