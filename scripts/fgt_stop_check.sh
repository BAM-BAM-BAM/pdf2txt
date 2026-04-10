#!/bin/bash
# FGT Stop Hook: Warn if commits made without updating FGT files
# Used by Claude Code Stop hook to enforce FGT methodology

COMMIT_THRESHOLD=3

# Count commits since last FGT_LOG.md or BUG_PATTERNS_PDF.md change
FGT_LOG_LAST=$(git log -1 --format="%H" -- FGT_LOG.md 2>/dev/null)
BUG_PATTERNS_LAST=$(git log -1 --format="%H" -- BUG_PATTERNS_PDF.md 2>/dev/null)

# Get the most recent of the two
if [ -n "$FGT_LOG_LAST" ] && [ -n "$BUG_PATTERNS_LAST" ]; then
    LAST_FGT_COMMIT=$(git log -1 --format="%H" -- FGT_LOG.md BUG_PATTERNS_PDF.md 2>/dev/null)
elif [ -n "$FGT_LOG_LAST" ]; then
    LAST_FGT_COMMIT="$FGT_LOG_LAST"
elif [ -n "$BUG_PATTERNS_LAST" ]; then
    LAST_FGT_COMMIT="$BUG_PATTERNS_LAST"
else
    echo "WARNING: Neither FGT_LOG.md nor BUG_PATTERNS_PDF.md has been committed yet."
    echo "Consider updating FGT files before ending this session."
    exit 0
fi

# Count commits since last FGT file update
COMMITS_SINCE=$(git rev-list --count "$LAST_FGT_COMMIT"..HEAD 2>/dev/null || echo 0)

if [ "$COMMITS_SINCE" -ge "$COMMIT_THRESHOLD" ]; then
    echo "WARNING: $COMMITS_SINCE commits since last FGT_LOG.md or BUG_PATTERNS_PDF.md update."
    echo "Consider updating FGT files before ending this session."
    echo "  - FGT_LOG.md: Record lessons from significant changes"
    echo "  - BUG_PATTERNS_PDF.md: Document any bugs fixed"
    echo "  - BACKLOG.md: Track any new items identified"
fi
