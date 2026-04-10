# Backlog

Items from expert reviews, user feedback, and identified improvements.
Every item must be tracked here until resolved.

**FGT Principle 9: Findings Must Be Tracked.**
If it's not in this file, it doesn't exist.

## Open

### Architecture
- [ ] Further split pdf2txt.py (2,278 lines) into extractors.py + ocr.py + cli.py -- source: PATTERNS_PDF.md target structure

### Documentation
- [ ] Expand FGT_DOMAIN_PDF.md with additional bug patterns as discovered -- source: FGT methodology

## Resolved

### Architecture (2026-04-10)
- [x] Modularize pdf2txt.py into separate modules -- split to pdf2txt_models.py (175), pdf2txt_quality.py (130), pdf2txt_learning.py (904), pdf2txt_hud.py (274), pdf2txt.py (2,278)
- [x] Extract AdaptiveLearner into its own module -- pdf2txt_learning.py
- [x] Extract TextQualityScorer to quality.py -- pdf2txt_quality.py
- [x] Extract HUD/curses display to hud.py -- pdf2txt_hud.py

### Testing (2026-04-10)
- [x] Add QUAL-001 test for extraction output quality -- test_fgt_categories.py
- [x] Add INV-001 test: all extensions have handlers -- test_fgt_categories.py
- [x] Add INV-003 test: feature vector dimension = 14 -- test_fgt_categories.py
- [x] Add BOUND-001 through BOUND-004 tests -- test_fgt_categories.py
- [x] Add PRO-001 test for module size enforcement -- test_fgt_categories.py
- [x] Add PRO-002 test for hardcoded extension detection -- test_fgt_categories.py
- [x] Rename existing 56 tests to FGT category prefixes -- test_adaptive_learner.py, test_document_formats.py

### Infrastructure (2026-04-10)
- [x] Add GitHub Actions CI workflow (test + lint + FGT validation) -- .github/workflows/verify.yml
- [x] Configure Claude Code hooks for FGT enforcement -- scripts/fgt_stop_check.sh
- [x] Set up branch protection on main (test + fgt-validation required) -- via gh api
