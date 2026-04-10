# pdf2txt - Project Instructions

## FGT Domain Files (MANDATORY)

FGT domain files (`PATTERNS_PDF.md`, `REVIEWS_PDF.md`, `FGT_DOMAIN_PDF.md`,
`BUG_PATTERNS_PDF.md`) must be **written from scratch** for this project's domain.
Copying from another project is an anti-pattern -- it provides false confidence and the
wrong domain content will never be consulted.

## Pre-Commit Verification (MANDATORY)

Before EVERY commit, run:

```bash
python3 -m pytest tests/ -v    # ALL tests must pass
python3 -c "import pdf2txt"    # Verify import succeeds
```

NEVER trust stale test results from previous sessions or compacted context. Re-run every time.

## Bug Abstraction Protocol (MANDATORY for every bug fix)

When fixing ANY bug, you MUST complete ALL steps before committing:

1. **Fix the instance** -- resolve the specific reported problem
2. **Abstract to class** -- identify the bug CATEGORY using the principles in `BUG_PATTERNS_PDF.md`
3. **Add BUG-XXX entry** -- document in `BUG_PATTERNS_PDF.md` with: Symptom, Root Cause, Fix, Principle Violated
4. **Add prevention test** -- write a test that catches the CLASS of error, not just this instance (see "Which test to write" below)
5. **Search for siblings** -- grep the codebase for the same pattern in other files
6. **Revalidate existing data** -- if the fix changes ANY logic that determines how data is classified, filtered, displayed, or acted upon, re-run the fixed logic on all existing data
7. **Persist the pattern** -- update FGT_LOG.md and memory/docs for future sessions

## Which Test to Write

| You want to ensure... | Write a... | Prefix |
|----------------------|-----------|--------|
| A domain rule always holds (e.g., quality score in [0,1]) | Invariant test | `INV-*` |
| Module A's reference to Module B stays correct | Contract test | `CONTRACT-*` |
| A structural anti-pattern doesn't exist in the codebase | Proactive test | `PRO-*` |
| A displayed value matches the authoritative computation | Cross-validation test | `XVAL-*` |
| An edge case is handled (empty file, corrupted PDF, no GPU) | Boundary test | `BOUND-*` |
| A structure has required fields/headers/shape | Schema test | `SCHEMA-*` |
| An end-to-end workflow produces expected results | Integration test | `INT-*` |
| The actual output values users see are correct | Output quality test | `QUAL-*` |

## Satellite State Prevention (MANDATORY)

Any value computed independently in multiple places WILL drift. Before writing ANY
calculation, ask:

1. Does a single source of truth already compute this? -> USE it.
2. Is there a named constant/config for this? -> REFERENCE it.
3. Am I duplicating logic from another module? -> IMPORT it.

## Session Protocol

At session end or before context compaction, update:

- `BUG_PATTERNS_PDF.md` -- if any bugs were fixed
- `FGT_LOG.md` -- if any significant changes were made
- `BACKLOG.md` -- if any new items were identified

At session start:
- Read `BACKLOG.md` to understand what's outstanding
- Read `BUG_PATTERNS_PDF.md` before acting
- Run tests to verify current state

## Domain Expert Perspectives

When reviewing changes, consider:

| Expert | Key Questions |
|--------|---------------|
| **Software Architect** | Is this properly abstracted? Separation of concerns? |
| **QA Engineer** | What edge cases exist? Malformed PDFs? Empty files? |
| **Document Analyst** | Is text fidelity preserved? Reading order correct? |
| **Data Engineer** | Does it scale? How are large files handled? |

## Key Files

| File | Purpose |
|------|---------|
| `FGT.md` | Core methodology (symlink to canonical) |
| `FGT_DOMAIN_PDF.md` | PDF-specific domain knowledge |
| `PATTERNS_PDF.md` | Code patterns for this project |
| `REVIEWS_PDF.md` | Review checklists |
| `BUG_PATTERNS_PDF.md` | Bug catalog with prevention principles |
| `BACKLOG.md` | Outstanding work items (Principle 9) |
| `FGT_LOG.md` | Development history and lessons |
| `CLAUDE_WEB_PDF.md` | For Claude web project uploads |

## Prevention Principles (quick index -- definitions in FGT.md)

1. **Explicit Contracts** -- untested module dependencies
2. **Single Source of Truth** -- duplicated definitions that drifted
3. **Existence Implies Usage** -- dead code or unused parameters
4. **Invariants Must Be Tested** -- rule that should have been a test
5. **Structure Implies Content** -- empty/hollow output fields
6. **Verify Before Trust** -- trusted stale data without re-checking
7. **Proactive Detection** -- bug class that could have been caught by structural scan
8. **Defenses Must Be Operational** -- validation/filter/gate operating on null/empty = no defense
9. **Findings Must Be Tracked** -- review findings not in BACKLOG.md/BUG_PATTERNS_PDF.md don't exist
10. **Scaffold Before Building** -- module boundaries, config, tests, and domain FGT files before first feature

## Quick Reference
```
Bug Response:
1. FIX         -> Resolve the specific problem
2. ABSTRACT    -> Which principle was violated? What CLASS of bug?
3. DOCUMENT    -> Add BUG-XXX to BUG_PATTERNS_PDF.md
4. PREVENT     -> Add test that catches the CLASS
5. SEARCH      -> Grep for siblings in CODE
6. REVALIDATE  -> Check if existing DATA is contaminated
7. PERSIST     -> Update FGT_LOG.md and memory/docs
```
