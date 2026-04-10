# Backlog

Items from expert reviews, user feedback, and identified improvements.
Every item must be tracked here until resolved.

**FGT Principle 9: Findings Must Be Tracked.**
If it's not in this file, it doesn't exist.

## Open

### Architecture
- [ ] Modularize pdf2txt.py (3,496 lines) into separate modules: cli, extractors/, ocr/, learning/, quality, hud, models -- source: CP-002 (Monolith Accumulation), LEARNINGS_FOR_EVENT_APP.md
- [ ] Extract AdaptiveLearner into its own module -- source: architecture review
- [ ] Extract TextQualityScorer to quality.py -- source: architecture review
- [ ] Extract HUD/curses display to hud.py -- source: architecture review

### Testing
- [ ] Add QUAL-* tests for extraction output quality (known PDF -> expected text) -- source: CP-008
- [ ] Add INV-001 test: every SUPPORTED_EXTENSIONS entry has a handler in extract_text() -- source: FGT methodology
- [ ] Add INV-003 test: ImageFeature.to_vector() returns exactly 14 elements -- source: BUG-002
- [ ] Add BOUND-001 test: OCR gracefully falls back to CPU when GPU unavailable -- source: BUG-003
- [ ] Add PRO-* test for module size enforcement (flag files >500 lines after modularization) -- source: CP-002
- [ ] Rename existing tests to use FGT category prefixes (INV-/QUAL-/BOUND-/etc.) -- source: FGT methodology

### Documentation
- [ ] Expand FGT_DOMAIN_PDF.md with additional bug patterns as discovered -- source: FGT methodology

### Infrastructure
- [ ] Add GitHub Actions CI workflow -- source: CP-012
- [ ] Configure Claude Code hooks for FGT enforcement -- source: CP-012
- [ ] Set up branch protection on main -- source: CP-012

## Resolved
