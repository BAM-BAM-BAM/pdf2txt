#!/home/jprior/claude/pdf2txt/.venv/bin/python
"""Extract text from PDFs and create corresponding markdown files."""

__version__ = "1.0.0"

import argparse
import curses
import math
import multiprocessing
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

# Use spawn to avoid terminal/curses issues with fork
_mp_context = multiprocessing.get_context("spawn")


@dataclass
class QualityMetrics:
    """Quality assessment metrics for extracted text."""
    real_word_ratio: float = 0.0
    content_score: float = 0.0
    gibberish_penalty: float = 0.0
    punctuation_score: float = 0.0
    total_score: float = 0.0
    word_count: int = 0


class TextQualityScorer:
    """Score text quality for comparison between extractions."""

    # Top 500 common English words (condensed set)
    COMMON_WORDS: frozenset = frozenset([
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
        "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
        "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
        "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
        "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
        "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
        "is", "are", "was", "were", "been", "has", "had", "did", "does", "done",
        "said", "made", "found", "used", "called", "may", "each", "own", "should", "here",
        "where", "more", "very", "through", "long", "little", "own", "while", "still", "find",
        "part", "being", "much", "too", "many", "those", "such", "before", "same", "right",
        "mean", "different", "move", "between", "must", "need", "might", "try", "world", "again",
        "place", "great", "show", "every", "last", "never", "old", "under", "keep", "let",
        "begin", "seem", "help", "always", "home", "both", "around", "off", "end", "against",
        "high", "few", "important", "until", "next", "without", "public", "another", "read", "number",
        "word", "page", "chapter", "section", "figure", "table", "document", "file", "data", "information",
        "system", "process", "method", "result", "analysis", "example", "following", "based", "using", "include",
        "however", "therefore", "although", "within", "during", "since", "provide", "according", "available", "report",
        "form", "service", "case", "study", "research", "development", "program", "company", "business", "market",
        "product", "customer", "management", "project", "support", "review", "application", "user", "group", "level",
        "value", "change", "control", "test", "performance", "quality", "standard", "policy", "issue", "problem",
    ])

    # Precompiled patterns for gibberish detection
    REPEATED_CHARS = re.compile(r'(.)\1{3,}')
    CONSONANT_CLUSTER = re.compile(r'[bcdfghjklmnpqrstvwxz]{5,}', re.IGNORECASE)
    NO_VOWELS = re.compile(r'\b[bcdfghjklmnpqrstvwxz]{5,}\b', re.IGNORECASE)
    ENCODING_ARTIFACTS = re.compile(r'â€|Ã©|Ã¨|Ã |ï»¿|\ufffd|\\x[0-9a-f]{2}', re.IGNORECASE)
    NON_PRINTABLE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')
    WORD_PATTERN = re.compile(r'\b[a-zA-Z]+\b')
    SENTENCE_END = re.compile(r'[.!?]')

    def score(self, text: str) -> QualityMetrics:
        """Score text quality and return detailed metrics."""
        if not text or not text.strip():
            return QualityMetrics()

        words = self.WORD_PATTERN.findall(text.lower())
        word_count = len(words)

        if word_count == 0:
            return QualityMetrics(word_count=0)

        # Real word ratio (weight: 0.35)
        common_count = sum(1 for w in words if w in self.COMMON_WORDS)
        real_word_ratio = common_count / word_count

        # Content score - log-scaled word count (weight: 0.30)
        content_score = min(1.0, math.log10(word_count + 1) / 4.0)

        # Gibberish penalty (weight: -0.20)
        gibberish_count = 0
        gibberish_count += len(self.REPEATED_CHARS.findall(text))
        gibberish_count += len(self.CONSONANT_CLUSTER.findall(text))
        gibberish_count += len(self.NO_VOWELS.findall(text))
        gibberish_count += len(self.ENCODING_ARTIFACTS.findall(text))
        gibberish_count += len(self.NON_PRINTABLE.findall(text))
        gibberish_penalty = min(1.0, gibberish_count / max(word_count / 10, 1))

        # Punctuation score - sentence structure (weight: 0.15)
        sentence_ends = len(self.SENTENCE_END.findall(text))
        expected_sentences = word_count / 15  # Average sentence ~15 words
        if expected_sentences > 0:
            punctuation_score = min(1.0, sentence_ends / expected_sentences)
        else:
            punctuation_score = 0.0

        # Calculate total weighted score
        total_score = (
            real_word_ratio * 0.35 +
            content_score * 0.30 -
            gibberish_penalty * 0.20 +
            punctuation_score * 0.15
        )
        total_score = max(0.0, min(1.0, total_score))

        return QualityMetrics(
            real_word_ratio=real_word_ratio,
            content_score=content_score,
            gibberish_penalty=gibberish_penalty,
            punctuation_score=punctuation_score,
            total_score=total_score,
            word_count=word_count
        )

    def compare(self, existing: str, new: str) -> tuple[bool, QualityMetrics, QualityMetrics]:
        """Compare existing and new text, return (is_new_better, old_metrics, new_metrics)."""
        old_metrics = self.score(existing)
        new_metrics = self.score(new)
        is_better = new_metrics.total_score > old_metrics.total_score
        return is_better, old_metrics, new_metrics


def strip_markdown_metadata(md_text: str) -> str:
    """Remove title, source line, and page markers from markdown for fair comparison."""
    lines = md_text.split('\n')
    content_lines = []
    for line in lines:
        # Skip: # Title, > Source:, ---, *Page N*
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        if stripped.startswith('>'):
            continue
        if stripped == '---':
            continue
        if stripped.startswith('*Page') and stripped.endswith('*'):
            continue
        content_lines.append(line)
    return '\n'.join(content_lines)


# Global scorer instance
_quality_scorer = TextQualityScorer()


@dataclass
class ProcessingStats:
    """Track processing statistics."""
    total_files: int = 0
    processed_files: int = 0
    skipped_files: int = 0
    failed_files: int = 0
    improved_files: int = 0
    kept_existing: int = 0
    total_bytes: int = 0
    processed_bytes: int = 0
    md_bytes: int = 0
    total_pages: int = 0
    processed_pages: int = 0
    ocr_pages: int = 0
    ocr_chars: int = 0
    current_file: str = ""
    current_file_pages: int = 0
    current_page: int = 0
    current_status: str = ""
    start_time: float = field(default_factory=time.time)
    log_messages: list = field(default_factory=list)

    def elapsed(self) -> float:
        return time.time() - self.start_time

    def files_per_min(self) -> float:
        elapsed = self.elapsed()
        if elapsed > 0:
            return (self.processed_files / elapsed) * 60
        return 0

    def mb_per_min(self) -> float:
        elapsed = self.elapsed()
        if elapsed > 0:
            mb = self.processed_bytes / (1024 * 1024)
            return (mb / elapsed) * 60
        return 0

    def log(self, msg: str):
        self.log_messages.append(msg)
        if len(self.log_messages) > 100:
            self.log_messages.pop(0)


@dataclass
class FileResult:
    """Pickle-safe result object returned by worker processes."""
    pdf_path: Path
    success: bool
    message: str
    improve_detail: str | None = None
    processed_bytes: int = 0
    md_bytes: int = 0
    pages_processed: int = 0
    ocr_pages: int = 0
    ocr_chars: int = 0
    was_improved: bool = False
    was_kept: bool = False
    was_skipped: bool = False
    was_failed: bool = False


class RetroHUD:
    """80's style terminal HUD using curses."""

    def __init__(self, stats: ProcessingStats):
        self.stats = stats
        self.stdscr = None

    def __enter__(self):
        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        curses.curs_set(0)
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_GREEN, -1)   # Title
            curses.init_pair(2, curses.COLOR_CYAN, -1)    # Labels
            curses.init_pair(3, curses.COLOR_YELLOW, -1)  # Values
            curses.init_pair(4, curses.COLOR_RED, -1)     # Errors
            curses.init_pair(5, curses.COLOR_MAGENTA, -1) # Progress bar
        self.stdscr.nodelay(True)
        return self

    def __exit__(self, *args):
        curses.nocbreak()
        curses.echo()
        curses.curs_set(1)
        curses.endwin()

    def draw_box(self, y: int, x: int, h: int, w: int, title: str = ""):
        """Draw a retro-style box."""
        # Corners and edges
        self.stdscr.addch(y, x, curses.ACS_ULCORNER)
        self.stdscr.addch(y, x + w - 1, curses.ACS_URCORNER)
        self.stdscr.addch(y + h - 1, x, curses.ACS_LLCORNER)
        self.stdscr.addch(y + h - 1, x + w - 1, curses.ACS_LRCORNER)

        for i in range(1, w - 1):
            self.stdscr.addch(y, x + i, curses.ACS_HLINE)
            self.stdscr.addch(y + h - 1, x + i, curses.ACS_HLINE)

        for i in range(1, h - 1):
            self.stdscr.addch(y + i, x, curses.ACS_VLINE)
            self.stdscr.addch(y + i, x + w - 1, curses.ACS_VLINE)

        if title:
            title_str = f"[ {title} ]"
            self.stdscr.addstr(y, x + 2, title_str, curses.color_pair(1) | curses.A_BOLD)

    def draw_progress_bar(self, y: int, x: int, width: int, progress: float, label: str = ""):
        """Draw a retro progress bar."""
        bar_width = width - len(label) - 10
        filled = int(bar_width * progress)
        empty = bar_width - filled

        self.stdscr.addstr(y, x, label, curses.color_pair(2))
        self.stdscr.addstr(y, x + len(label), " [", curses.color_pair(5))
        self.stdscr.addstr(y, x + len(label) + 2, "█" * filled, curses.color_pair(5) | curses.A_BOLD)
        self.stdscr.addstr(y, x + len(label) + 2 + filled, "░" * empty, curses.color_pair(5))
        self.stdscr.addstr(y, x + len(label) + 2 + bar_width, "]", curses.color_pair(5))
        pct_str = f" {progress * 100:5.1f}%"
        self.stdscr.addstr(y, x + len(label) + 3 + bar_width, pct_str, curses.color_pair(3))

    def draw_stat(self, y: int, x: int, label: str, value: str):
        """Draw a labeled stat."""
        self.stdscr.addstr(y, x, label, curses.color_pair(2))
        self.stdscr.addstr(y, x + len(label), value, curses.color_pair(3) | curses.A_BOLD)

    def truncate_path(self, path: str, max_len: int) -> str:
        """Truncate path to fit display."""
        if len(path) <= max_len:
            return path
        return "..." + path[-(max_len - 3):]

    def refresh(self):
        """Refresh the HUD display."""
        try:
            self.stdscr.erase()  # erase() doesn't flash like clear()
            height, width = self.stdscr.getmaxyx()
            width = min(width, 100)  # Cap width

            # Title banner
            banner = [
                "╔═══════════════════════════════════════════════════════════╗",
                "║   ██████╗ ██████╗ ███████╗██████╗ ████████╗██╗  ██╗████████╗ ║",
                "║   ██╔══██╗██╔══██╗██╔════╝╚════██╗╚══██╔══╝╚██╗██╔╝╚══██╔══╝ ║",
                "║   ██████╔╝██║  ██║█████╗   █████╔╝   ██║    ╚███╔╝    ██║    ║",
                "║   ██╔═══╝ ██║  ██║██╔══╝  ██╔═══╝    ██║    ██╔██╗    ██║    ║",
                "║   ██║     ██████╔╝██║     ███████╗   ██║   ██╔╝ ██╗   ██║    ║",
                "║   ╚═╝     ╚═════╝ ╚═╝     ╚══════╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝    ║",
                "╚═══════════════════════════════════════════════════════════╝",
            ]

            # Simpler banner if terminal is narrow
            if width < 70:
                banner = [
                    "┌─────────────────────────┐",
                    "│     P D F 2 T X T       │",
                    "│      v" + __version__.center(17) + "│",
                    "└─────────────────────────┘",
                ]

            for i, line in enumerate(banner):
                if i < height - 1:
                    self.stdscr.addstr(i, 0, line[:width-1], curses.color_pair(1) | curses.A_BOLD)

            y_offset = len(banner) + 1

            # Main stats box
            box_height = 13
            if y_offset + box_height < height:
                self.draw_box(y_offset, 0, box_height, min(width - 1, 78), "PROCESSING STATUS")

                # Current file
                y = y_offset + 1
                current = self.truncate_path(self.stats.current_file, 55)
                self.draw_stat(y, 2, "FILE: ", current if current else "(idle)")

                # File progress
                y += 2
                file_progress = self.stats.processed_files / max(self.stats.total_files, 1)
                self.draw_progress_bar(y, 2, 74, file_progress, "FILES  ")

                # Page progress (current file)
                y += 1
                page_progress = self.stats.current_page / max(self.stats.current_file_pages, 1)
                self.draw_progress_bar(y, 2, 74, page_progress, "PAGES  ")

                # Stats row 1
                y += 2
                elapsed = self.stats.elapsed()
                self.draw_stat(y, 2, "ELAPSED: ", f"{elapsed:6.1f}s")
                self.draw_stat(y, 22, "FILES: ", f"{self.stats.processed_files}/{self.stats.total_files}")
                self.draw_stat(y, 42, "RATE: ", f"{self.stats.files_per_min():5.1f} files/min")

                # Stats row 2
                y += 1
                mb_processed = self.stats.processed_bytes / (1024 * 1024)
                mb_total = self.stats.total_bytes / (1024 * 1024)
                self.draw_stat(y, 2, "PDF IN: ", f"{mb_processed:6.2f}/{mb_total:.2f} MB")
                self.draw_stat(y, 32, "THROUGHPUT: ", f"{self.stats.mb_per_min():6.2f} MB/min")

                # Stats row 3 - Output and compression
                y += 1
                md_mb = self.stats.md_bytes / (1024 * 1024)
                if self.stats.processed_bytes > 0:
                    ratio = (self.stats.md_bytes / self.stats.processed_bytes) * 100
                    self.draw_stat(y, 2, "MD OUT: ", f"{md_mb:6.2f} MB")
                    self.draw_stat(y, 26, "RATIO: ", f"{ratio:5.1f}% of input")
                else:
                    self.draw_stat(y, 2, "MD OUT: ", f"{md_mb:6.2f} MB")

                # Stats row 4 - OCR stats
                y += 2
                self.draw_stat(y, 2, "OCR PAGES: ", f"{self.stats.ocr_pages:5d}")
                self.draw_stat(y, 22, "OCR CHARS: ", f"{self.stats.ocr_chars:,}")
                status = self.stats.current_status[:30] if self.stats.current_status else "Ready"
                self.draw_stat(y, 48, "STATUS: ", status)

            # Results box
            y_offset += box_height + 1
            results_height = 6
            if y_offset + results_height < height:
                self.draw_box(y_offset, 0, results_height, min(width - 1, 78), "RESULTS")
                y = y_offset + 1
                self.draw_stat(y, 2, "PROCESSED: ", f"{self.stats.processed_files:4d}")
                self.draw_stat(y, 22, "SKIPPED: ", f"{self.stats.skipped_files:4d}")
                self.draw_stat(y, 42, "FAILED: ", f"{self.stats.failed_files:4d}")
                if self.stats.failed_files > 0:
                    self.stdscr.addstr(y, 50, f"{self.stats.failed_files:4d}", curses.color_pair(4) | curses.A_BOLD)

                y += 1
                if self.stats.improved_files > 0 or self.stats.kept_existing > 0:
                    self.draw_stat(y, 2, "IMPROVED: ", f"{self.stats.improved_files:4d}")
                    self.draw_stat(y, 22, "KEPT: ", f"{self.stats.kept_existing:4d}")

                y += 2
                total_pages = self.stats.processed_pages
                self.draw_stat(y, 2, "TOTAL PAGES: ", f"{total_pages:,}")

            # Log box
            y_offset += results_height + 1
            log_height = max(5, height - y_offset - 1)
            if y_offset + log_height < height and log_height > 2:
                self.draw_box(y_offset, 0, log_height, min(width - 1, 78), "ACTIVITY LOG")

                # Show recent log messages
                visible_logs = log_height - 2
                recent_logs = self.stats.log_messages[-visible_logs:]
                for i, msg in enumerate(recent_logs):
                    if y_offset + 1 + i < height - 1:
                        truncated = msg[:74] if len(msg) > 74 else msg
                        color = curses.color_pair(4) if "FAIL" in msg or "ERROR" in msg else curses.color_pair(2)
                        self.stdscr.addstr(y_offset + 1 + i, 2, truncated, color)

            self.stdscr.refresh()
        except curses.error:
            pass  # Ignore curses errors from terminal resize


def convert_windows_path(path_str: str) -> Path:
    """Convert Windows-style path to WSL path if needed."""
    windows_pattern = r'^([A-Za-z]):[/\\](.*)$'
    match = re.match(windows_pattern, path_str)

    if match:
        drive_letter = match.group(1).lower()
        rest_of_path = match.group(2)
        rest_of_path = rest_of_path.replace('\\', '/')
        return Path(f'/mnt/{drive_letter}/{rest_of_path}')

    return Path(path_str)


def find_pdfs(directory: Path, recursive: bool = False, quiet: bool = False) -> list[Path]:
    """Find all PDF files in directory (case-insensitive)."""
    if not quiet:
        mode = "recursively " if recursive else ""
        print(f"Searching {mode}for PDFs in: {directory}", flush=True)

    pdfs = []
    search = directory.rglob if recursive else directory.glob
    for pattern in ['*.pdf', '*.PDF', '*.Pdf']:
        pdfs.extend(search(pattern))

    result = list(set(pdfs))
    if not quiet:
        print(f"Found {len(result)} PDF file(s)", flush=True)
    return result


def check_tesseract_available() -> bool:
    """Check if Tesseract OCR is available on the system."""
    import shutil
    return shutil.which("tesseract") is not None


class SuppressOutputFD:
    """Context manager to suppress stdout/stderr at the OS file descriptor level.

    This catches output from native code (like PyMuPDF/Tesseract) that bypasses
    Python's sys.stdout/stderr. Curses still works because it opens /dev/tty directly.
    """

    def __init__(self, suppress: bool = True):
        self.suppress = suppress
        self._stdout_fd = None
        self._stderr_fd = None
        self._devnull_fd = None

    def __enter__(self):
        if self.suppress:
            # Save copies of the original file descriptors
            self._stdout_fd = os.dup(1)
            self._stderr_fd = os.dup(2)
            # Open /dev/null
            self._devnull_fd = os.open(os.devnull, os.O_WRONLY)
            # Redirect stdout and stderr to /dev/null
            os.dup2(self._devnull_fd, 1)
            os.dup2(self._devnull_fd, 2)
        return self

    def __exit__(self, *args):
        if self.suppress:
            # Restore original file descriptors
            os.dup2(self._stdout_fd, 1)
            os.dup2(self._stderr_fd, 2)
            # Close the saved copies
            os.close(self._stdout_fd)
            os.close(self._stderr_fd)
            os.close(self._devnull_fd)


def page_has_images(page) -> bool:
    """Check if a page contains images (potential OCR candidates)."""
    image_list = page.get_images(full=False)
    return len(image_list) > 0


def extract_text_from_pdf(
    pdf_path: Path,
    use_ocr: bool = True,
    stats: ProcessingStats | None = None,
    hud: RetroHUD | None = None
) -> list[str]:
    """Extract text from PDF, returning list of page contents."""
    import pymupdf

    pages = []
    # Suppress output when HUD is active to prevent terminal corruption
    with SuppressOutputFD(suppress=(hud is not None)):
        doc = pymupdf.open(pdf_path)
    ocr_available = check_tesseract_available() if use_ocr else False
    total_pages = len(doc)

    if stats:
        stats.current_file_pages = total_pages
        stats.total_pages += total_pages

    try:
        for page_num, page in enumerate(doc, start=1):
            if stats:
                stats.current_page = page_num
                stats.current_status = f"Reading page {page_num}/{total_pages}"
                if hud:
                    hud.refresh()

            with SuppressOutputFD(suppress=(hud is not None)):
                text = page.get_text().strip()

            if use_ocr and ocr_available and page_has_images(page):
                try:
                    if stats:
                        stats.current_status = f"OCR page {page_num}/{total_pages}..."
                        if hud:
                            hud.refresh()

                    with SuppressOutputFD(suppress=(hud is not None)):
                        tp = page.get_textpage_ocr(full=True, language="eng")
                        ocr_text = page.get_text(textpage=tp).strip()
                    orig_len = len(text)
                    ocr_len = len(ocr_text)

                    if ocr_len > orig_len:
                        text = ocr_text
                        if stats:
                            stats.ocr_pages += 1
                            stats.ocr_chars += ocr_len
                            stats.log(f"  OCR p{page_num}: +{ocr_len:,} chars")
                    else:
                        if stats:
                            stats.log(f"  OCR p{page_num}: kept original ({orig_len:,} chars)")
                except Exception as e:
                    if stats:
                        stats.log(f"  OCR p{page_num}: FAILED - {e}")

            pages.append(text)
            if stats:
                stats.processed_pages += 1

    finally:
        doc.close()
    return pages


def create_markdown(pdf_path: Path, pages: list[str]) -> str:
    """Create markdown content from extracted pages."""
    lines = [
        f"# {pdf_path.stem}",
        "",
        f"> Source: {pdf_path}",
        "",
        "---",
        "",
    ]

    for i, page_text in enumerate(pages, start=1):
        if i > 1:
            lines.extend(["", "---", f"*Page {i}*", ""])
        lines.append(page_text)

    return '\n'.join(lines)


def process_pdf(
    pdf_path: Path,
    overwrite: bool,
    dry_run: bool,
    use_ocr: bool = True,
    improve: bool = False,
    stats: ProcessingStats | None = None,
    hud: RetroHUD | None = None
) -> tuple[bool, str, str | None]:
    """Process a single PDF file.

    Returns: (success, message, improve_detail)
    - improve_detail is set when improve mode makes a decision
    """
    md_path = pdf_path.with_suffix('.md')

    # Improve mode: always extract and compare
    if improve and md_path.exists():
        if dry_run:
            return True, f"Would compare: {md_path.name}", None

        try:
            # Extract new version
            pages = extract_text_from_pdf(pdf_path, use_ocr=use_ocr, stats=stats, hud=hud)
            new_markdown = create_markdown(pdf_path, pages)
            new_text = '\n'.join(pages)

            # Read and strip existing
            existing_markdown = md_path.read_text(encoding='utf-8')
            existing_text = strip_markdown_metadata(existing_markdown)

            # Compare quality
            is_better, old_metrics, new_metrics = _quality_scorer.compare(existing_text, new_text)

            if is_better:
                md_path.write_text(new_markdown, encoding='utf-8')
                if stats:
                    stats.md_bytes += len(new_markdown.encode('utf-8'))
                detail = f"Improved: {old_metrics.total_score:.2f} → {new_metrics.total_score:.2f}"
                return True, f"Improved: {md_path.name}", detail
            else:
                # Count existing file size for kept files
                if stats:
                    stats.md_bytes += md_path.stat().st_size
                detail = f"Kept existing: {old_metrics.total_score:.2f} > {new_metrics.total_score:.2f}"
                return False, f"Kept (better quality): {pdf_path.name}", detail

        except Exception as e:
            return False, f"FAILED: {pdf_path.name} - {e}", None

    # Standard mode: skip existing unless overwrite
    if md_path.exists() and not overwrite:
        return False, f"Skipped (exists): {pdf_path.name}", None

    if dry_run:
        action = "Would overwrite" if md_path.exists() else "Would create"
        return True, f"{action}: {md_path.name}", None

    try:
        pages = extract_text_from_pdf(pdf_path, use_ocr=use_ocr, stats=stats, hud=hud)
        markdown_content = create_markdown(pdf_path, pages)
        md_path.write_text(markdown_content, encoding='utf-8')
        if stats:
            stats.md_bytes += len(markdown_content.encode('utf-8'))
        return True, f"Created: {md_path.name}", None
    except Exception as e:
        return False, f"FAILED: {pdf_path.name} - {e}", None


def _worker_init_suppress_output():
    """Initializer for worker processes to suppress stdout/stderr (for HUD mode).

    Redirects at the OS file descriptor level to catch native code output.
    """
    import os
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)  # redirect stdout
    os.dup2(devnull_fd, 2)  # redirect stderr
    os.close(devnull_fd)


def process_pdf_worker(
    pdf_path: Path,
    overwrite: bool,
    dry_run: bool,
    use_ocr: bool,
    improve: bool
) -> FileResult:
    """Process a single PDF in a separate process. Returns FileResult for aggregation."""
    import pymupdf

    md_path = pdf_path.with_suffix('.md')
    file_size = pdf_path.stat().st_size

    result = FileResult(
        pdf_path=pdf_path,
        success=False,
        message="",
        processed_bytes=file_size
    )

    def extract_pages() -> tuple[list[str], int, int]:
        """Extract text from all pages, returning (pages, ocr_pages, ocr_chars)."""
        doc = pymupdf.open(pdf_path)
        ocr_available = check_tesseract_available() if use_ocr else False
        pages = []
        ocr_pages_count = 0
        ocr_chars_count = 0

        try:
            for page in doc:
                text = page.get_text().strip()

                if use_ocr and ocr_available and page_has_images(page):
                    try:
                        tp = page.get_textpage_ocr(full=True, language="eng")
                        ocr_text = page.get_text(textpage=tp).strip()
                        if len(ocr_text) > len(text):
                            text = ocr_text
                            ocr_pages_count += 1
                            ocr_chars_count += len(ocr_text)
                    except Exception:
                        pass  # Keep original text

                pages.append(text)
        finally:
            doc.close()

        return pages, ocr_pages_count, ocr_chars_count

    # Improve mode: always extract and compare
    if improve and md_path.exists():
        if dry_run:
            result.success = True
            result.message = f"Would compare: {md_path.name}"
            return result

        try:
            pages, ocr_pages_count, ocr_chars_count = extract_pages()
            new_markdown = create_markdown(pdf_path, pages)
            new_text = '\n'.join(pages)

            existing_markdown = md_path.read_text(encoding='utf-8')
            existing_text = strip_markdown_metadata(existing_markdown)

            is_better, old_metrics, new_metrics = _quality_scorer.compare(existing_text, new_text)

            result.pages_processed = len(pages)
            result.ocr_pages = ocr_pages_count
            result.ocr_chars = ocr_chars_count

            if is_better:
                md_path.write_text(new_markdown, encoding='utf-8')
                result.md_bytes = len(new_markdown.encode('utf-8'))
                result.success = True
                result.was_improved = True
                result.message = f"Improved: {md_path.name}"
                result.improve_detail = f"Improved: {old_metrics.total_score:.2f} → {new_metrics.total_score:.2f}"
            else:
                result.md_bytes = md_path.stat().st_size
                result.success = False
                result.was_kept = True
                result.message = f"Kept (better quality): {pdf_path.name}"
                result.improve_detail = f"Kept existing: {old_metrics.total_score:.2f} > {new_metrics.total_score:.2f}"

            return result

        except Exception as e:
            result.was_failed = True
            result.message = f"FAILED: {pdf_path.name} - {e}"
            return result

    # Standard mode: skip existing unless overwrite
    if md_path.exists() and not overwrite:
        result.was_skipped = True
        result.message = f"Skipped (exists): {pdf_path.name}"
        result.processed_bytes = 0  # Didn't process
        return result

    if dry_run:
        action = "Would overwrite" if md_path.exists() else "Would create"
        result.success = True
        result.message = f"{action}: {md_path.name}"
        return result

    try:
        pages, ocr_pages_count, ocr_chars_count = extract_pages()
        markdown_content = create_markdown(pdf_path, pages)
        md_path.write_text(markdown_content, encoding='utf-8')

        result.md_bytes = len(markdown_content.encode('utf-8'))
        result.pages_processed = len(pages)
        result.ocr_pages = ocr_pages_count
        result.ocr_chars = ocr_chars_count
        result.success = True
        result.message = f"Created: {md_path.name}"
        return result

    except Exception as e:
        result.was_failed = True
        result.message = f"FAILED: {pdf_path.name} - {e}"
        return result


def aggregate_result(stats: ProcessingStats, result: FileResult, improve_mode: bool):
    """Aggregate a worker result into processing stats."""
    stats.md_bytes += result.md_bytes
    stats.processed_pages += result.pages_processed
    stats.ocr_pages += result.ocr_pages
    stats.ocr_chars += result.ocr_chars

    if result.was_failed:
        stats.failed_files += 1
    elif result.was_skipped:
        stats.skipped_files += 1
    elif result.was_improved:
        stats.improved_files += 1
        stats.processed_files += 1
        stats.processed_bytes += result.processed_bytes
    elif result.was_kept:
        stats.kept_existing += 1
        stats.processed_files += 1
        stats.processed_bytes += result.processed_bytes
    elif result.success:
        stats.processed_files += 1
        stats.processed_bytes += result.processed_bytes


def run_simple_parallel(pdfs: list[Path], args, use_ocr: bool, max_workers: int) -> int:
    """Run processing with simple text output using parallel workers."""
    stats = ProcessingStats()
    stats.total_files = len(pdfs)
    stats.total_bytes = sum(p.stat().st_size for p in pdfs)
    improve_mode = getattr(args, 'improve', False)

    if args.verbose or args.dry_run:
        if args.dry_run:
            print("(Dry run - no files will be created)")
        if improve_mode:
            print("(Improve mode - comparing quality)")
        print(f"Using {max_workers} parallel workers")
        print()

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=_mp_context) as executor:
        futures = {
            executor.submit(
                process_pdf_worker,
                pdf_path,
                args.overwrite,
                args.dry_run,
                use_ocr,
                improve_mode
            ): pdf_path for pdf_path in sorted(pdfs)
        }

        for future in as_completed(futures):
            result = future.result()
            aggregate_result(stats, result, improve_mode)

            if args.verbose:
                print(f"  {result.message}")
                if result.improve_detail:
                    print(f"     {result.improve_detail}")

    elapsed = stats.elapsed()

    if not args.quiet:
        print()
        if improve_mode:
            print(f"Summary: {stats.processed_files} processed, {stats.improved_files} improved, {stats.kept_existing} kept existing, {stats.failed_files} failed")
        else:
            print(f"Summary: {stats.processed_files} processed, {stats.skipped_files} skipped, {stats.failed_files} failed")

        if args.verbose and stats.processed_files > 0 and elapsed > 0:
            total_mb = stats.processed_bytes / (1024 * 1024)
            md_mb = stats.md_bytes / (1024 * 1024)
            files_per_min = (stats.processed_files / elapsed) * 60
            mb_per_min = (total_mb / elapsed) * 60
            ratio = (stats.md_bytes / stats.processed_bytes * 100) if stats.processed_bytes > 0 else 0

            print()
            print(f"Stats:")
            print(f"  Time elapsed:    {elapsed:.1f}s")
            print(f"  Workers used:    {max_workers}")
            print(f"  PDF input:       {total_mb:.2f} MB ({stats.processed_pages} pages)")
            print(f"  MD output:       {md_mb:.2f} MB ({ratio:.1f}% of input)")
            print(f"  Processing rate: {files_per_min:.1f} files/min, {mb_per_min:.2f} MB/min")
            print(f"  OCR stats:       {stats.ocr_pages} pages, {stats.ocr_chars:,} chars extracted")

    return 1 if stats.failed_files > 0 else 0


def run_with_hud_parallel(pdfs: list[Path], args, use_ocr: bool, max_workers: int) -> int:
    """Run processing with the retro HUD using parallel workers."""
    stats = ProcessingStats()
    stats.total_files = len(pdfs)
    stats.total_bytes = sum(p.stat().st_size for p in pdfs)
    improve_mode = getattr(args, 'improve', False)

    # Create executor BEFORE curses to avoid any startup output interfering
    # Use initializer to suppress output in worker processes
    executor = ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=_mp_context,
        initializer=_worker_init_suppress_output
    )

    # Submit all tasks before entering curses
    futures = {
        executor.submit(
            process_pdf_worker,
            pdf_path,
            args.overwrite,
            args.dry_run,
            use_ocr,
            improve_mode
        ): pdf_path for pdf_path in sorted(pdfs)
    }

    try:
        with RetroHUD(stats) as hud:
            stats.current_status = f"{max_workers} workers processing..."
            stats.log(f"Parallel mode: {max_workers} workers")
            hud.refresh()

            active_count = len(futures)

            for future in as_completed(futures):
                result = future.result()
                active_count -= 1

                aggregate_result(stats, result, improve_mode)

                stats.current_file = str(result.pdf_path)
                stats.current_status = f"Workers: {active_count} active | {stats.processed_files + stats.skipped_files + stats.failed_files}/{stats.total_files} done"
                stats.log(f"  → {result.message}")
                if result.improve_detail:
                    stats.log(f"     {result.improve_detail}")

                hud.refresh()

            # Final display
            stats.current_file = ""
            stats.current_status = "COMPLETE"
            stats.log("")
            stats.log("═" * 40)
            stats.log(f" FINISHED - {stats.processed_files} files processed")
            stats.log(f" Workers used: {max_workers}")
            if improve_mode:
                stats.log(f" Improved: {stats.improved_files} | Kept: {stats.kept_existing}")
            stats.log(f" Time: {stats.elapsed():.1f}s | OCR pages: {stats.ocr_pages}")
            stats.log("═" * 40)
            hud.refresh()

            # Wait for keypress
            hud.stdscr.nodelay(False)
            hud.stdscr.addstr(hud.stdscr.getmaxyx()[0] - 1, 2, "Press any key to exit...",
                             curses.color_pair(3) | curses.A_BLINK)
            hud.stdscr.refresh()
            hud.stdscr.getch()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    return 1 if stats.failed_files > 0 else 0


def run_with_hud(pdfs: list[Path], args, use_ocr: bool) -> int:
    """Run processing with the retro HUD."""
    stats = ProcessingStats()
    stats.total_files = len(pdfs)
    stats.total_bytes = sum(p.stat().st_size for p in pdfs)
    improve_mode = getattr(args, 'improve', False)

    with RetroHUD(stats) as hud:
        hud.refresh()

        for pdf_path in sorted(pdfs):
            stats.current_file = str(pdf_path)
            stats.current_page = 0
            stats.current_file_pages = 0
            file_size = pdf_path.stat().st_size
            stats.log(f"Processing: {pdf_path.name}")
            hud.refresh()

            success, message, improve_detail = process_pdf(
                pdf_path,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                use_ocr=use_ocr,
                improve=improve_mode,
                stats=stats,
                hud=hud
            )

            stats.log(f"  → {message}")
            if improve_detail:
                stats.log(f"     {improve_detail}")

            if improve_mode and "Improved:" in message:
                stats.improved_files += 1
                stats.processed_files += 1
                stats.processed_bytes += file_size
            elif improve_mode and "Kept" in message:
                stats.kept_existing += 1
                stats.processed_files += 1  # Count for throughput (we did extract)
                stats.processed_bytes += file_size
            elif success:
                stats.processed_files += 1
                stats.processed_bytes += file_size
            elif "Skipped" in message:
                stats.skipped_files += 1
            else:
                stats.failed_files += 1

            hud.refresh()

        # Final display
        stats.current_file = ""
        stats.current_status = "COMPLETE"
        stats.log("")
        stats.log("═" * 40)
        stats.log(f" FINISHED - {stats.processed_files} files processed")
        if improve_mode:
            stats.log(f" Improved: {stats.improved_files} | Kept: {stats.kept_existing}")
        stats.log(f" Time: {stats.elapsed():.1f}s | OCR pages: {stats.ocr_pages}")
        stats.log("═" * 40)
        hud.refresh()

        # Wait for keypress
        hud.stdscr.nodelay(False)
        hud.stdscr.addstr(hud.stdscr.getmaxyx()[0] - 1, 2, "Press any key to exit...",
                         curses.color_pair(3) | curses.A_BLINK)
        hud.stdscr.refresh()
        hud.stdscr.getch()

    return 1 if stats.failed_files > 0 else 0


def run_simple(pdfs: list[Path], args, use_ocr: bool) -> int:
    """Run processing with simple text output."""
    processed = 0
    skipped = 0
    failed = 0
    improved = 0
    kept_existing = 0
    total_bytes = 0
    md_bytes = 0
    total_pages = 0
    ocr_pages = 0
    ocr_chars = 0
    start_time = time.time()
    improve_mode = getattr(args, 'improve', False)

    if args.verbose or args.dry_run:
        if args.dry_run:
            print("(Dry run - no files will be created)")
        if improve_mode:
            print("(Improve mode - comparing quality)")
        print()

    for pdf_path in sorted(pdfs):
        if args.verbose:
            print(f"Processing: {pdf_path}")

        file_size = pdf_path.stat().st_size
        md_path = pdf_path.with_suffix('.md')

        # Improve mode: extract and compare
        if improve_mode and md_path.exists():
            if args.dry_run:
                print(f"  Would compare: {md_path}")
                processed += 1
                total_bytes += file_size
                continue

            try:
                import pymupdf
                doc = pymupdf.open(pdf_path)
                ocr_available = check_tesseract_available() if use_ocr else False
                pages = []
                doc_pages = len(doc)

                if args.verbose:
                    print(f"  {doc_pages} page(s)")

                for page_num, page in enumerate(doc, start=1):
                    text = page.get_text().strip()

                    if use_ocr and ocr_available and page_has_images(page):
                        try:
                            if args.verbose:
                                print(f"  Page {page_num}/{doc_pages}: OCR...", end="", flush=True)
                            tp = page.get_textpage_ocr(full=True, language="eng")
                            ocr_text = page.get_text(textpage=tp).strip()
                            orig_len = len(text)
                            ocr_len = len(ocr_text)
                            if ocr_len > orig_len:
                                text = ocr_text
                                ocr_pages += 1
                                ocr_chars += ocr_len
                                if args.verbose:
                                    print(f" +{ocr_len:,} chars")
                            else:
                                if args.verbose:
                                    print(f" kept original ({orig_len:,} chars)")
                        except Exception as e:
                            if args.verbose:
                                print(f" failed - {e}")

                    pages.append(text)
                    total_pages += 1

                doc.close()

                new_markdown = create_markdown(pdf_path, pages)
                new_text = '\n'.join(pages)

                existing_markdown = md_path.read_text(encoding='utf-8')
                existing_text = strip_markdown_metadata(existing_markdown)

                is_better, old_metrics, new_metrics = _quality_scorer.compare(existing_text, new_text)

                # Always count as processed for throughput (we did extract the PDF)
                processed += 1
                total_bytes += file_size

                if is_better:
                    md_path.write_text(new_markdown, encoding='utf-8')
                    md_bytes += len(new_markdown.encode('utf-8'))
                    improved += 1
                    if args.verbose:
                        print(f"  Improved: {old_metrics.total_score:.2f} → {new_metrics.total_score:.2f}")
                    elif not args.quiet:
                        print(f"  Improved: {pdf_path.name} ({old_metrics.total_score:.2f} → {new_metrics.total_score:.2f})")
                else:
                    md_bytes += md_path.stat().st_size  # Count existing file
                    kept_existing += 1
                    if args.verbose:
                        print(f"  Kept existing: {old_metrics.total_score:.2f} > {new_metrics.total_score:.2f}")

            except Exception as e:
                failed += 1
                print(f"  FAILED: {e}", file=sys.stderr)

            continue

        # Standard mode: skip existing unless overwrite
        if md_path.exists() and not args.overwrite:
            if args.verbose:
                print(f"  Skipped (exists)")
            skipped += 1
            continue

        if args.dry_run:
            action = "Would overwrite" if md_path.exists() else "Would create"
            print(f"  {action}: {md_path}")
            processed += 1
            total_bytes += file_size
            continue

        try:
            import pymupdf
            doc = pymupdf.open(pdf_path)
            ocr_available = check_tesseract_available() if use_ocr else False
            pages = []
            doc_pages = len(doc)

            if args.verbose:
                print(f"  {doc_pages} page(s)")

            for page_num, page in enumerate(doc, start=1):
                text = page.get_text().strip()

                if use_ocr and ocr_available and page_has_images(page):
                    try:
                        if args.verbose:
                            print(f"  Page {page_num}/{doc_pages}: OCR...", end="", flush=True)
                        tp = page.get_textpage_ocr(full=True, language="eng")
                        ocr_text = page.get_text(textpage=tp).strip()
                        orig_len = len(text)
                        ocr_len = len(ocr_text)
                        if ocr_len > orig_len:
                            text = ocr_text
                            ocr_pages += 1
                            ocr_chars += ocr_len
                            if args.verbose:
                                print(f" +{ocr_len:,} chars")
                        else:
                            if args.verbose:
                                print(f" kept original ({orig_len:,} chars)")
                    except Exception as e:
                        if args.verbose:
                            print(f" failed - {e}")

                pages.append(text)
                total_pages += 1

            doc.close()

            markdown_content = create_markdown(pdf_path, pages)
            md_path.write_text(markdown_content, encoding='utf-8')
            md_bytes += len(markdown_content.encode('utf-8'))

            if args.verbose:
                print(f"  Created: {md_path}")

            processed += 1
            total_bytes += file_size

        except Exception as e:
            failed += 1
            print(f"  FAILED: {e}", file=sys.stderr)

    elapsed = time.time() - start_time

    if not args.quiet:
        print()
        if improve_mode:
            print(f"Summary: {processed} processed, {improved} improved, {kept_existing} kept existing, {failed} failed")
        else:
            print(f"Summary: {processed} processed, {skipped} skipped, {failed} failed")

        if args.verbose and (processed > 0 or improved > 0) and elapsed > 0:
            total_mb = total_bytes / (1024 * 1024)
            md_mb = md_bytes / (1024 * 1024)
            files_per_min = (processed / elapsed) * 60 if processed > 0 else 0
            mb_per_min = (total_mb / elapsed) * 60
            ratio = (md_bytes / total_bytes * 100) if total_bytes > 0 else 0

            print()
            print(f"Stats:")
            print(f"  Time elapsed:    {elapsed:.1f}s")
            print(f"  PDF input:       {total_mb:.2f} MB ({total_pages} pages)")
            print(f"  MD output:       {md_mb:.2f} MB ({ratio:.1f}% of input)")
            print(f"  Processing rate: {files_per_min:.1f} files/min, {mb_per_min:.2f} MB/min")
            print(f"  OCR stats:       {ocr_pages} pages, {ocr_chars:,} chars extracted")

    return 1 if failed > 0 else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract text from PDFs and create markdown files alongside each PDF."
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "directory",
        help="Directory to search for PDFs (supports Windows paths in WSL)"
    )

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed progress"
    )
    verbosity.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress all output except errors"
    )

    parser.add_argument(
        "--hud",
        action="store_true",
        help="Show retro 80's style HUD (implies verbose)"
    )
    parser.add_argument(
        "-n", "--dry-run",
        action="store_true",
        help="List PDFs that would be processed without creating files"
    )
    parser.add_argument(
        "-f", "--force", "--overwrite",
        action="store_true",
        dest="overwrite",
        help="Overwrite existing .md files (default: skip)"
    )
    parser.add_argument(
        "--improve",
        action="store_true",
        help="Re-extract and only overwrite if new extraction is better quality"
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Search for PDFs recursively in subdirectories"
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR for image-based pages (OCR requires Tesseract)"
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=None,
        metavar="N",
        help="Number of parallel workers (default: CPU count, use 1 for sequential)"
    )

    args = parser.parse_args()

    # Convert and validate path
    directory = convert_windows_path(args.directory)

    if not directory.exists():
        print(f"Error: Directory does not exist: {directory}", file=sys.stderr)
        return 1

    if not directory.is_dir():
        print(f"Error: Path is not a directory: {directory}", file=sys.stderr)
        return 1

    # Check OCR availability
    use_ocr = not args.no_ocr
    if use_ocr and not check_tesseract_available():
        if not args.quiet and not args.hud:
            print("Warning: Tesseract not found. OCR disabled.", file=sys.stderr)
            print("Install with: apt install tesseract-ocr", file=sys.stderr)
            print()
        use_ocr = False

    # Find PDFs
    pdfs = find_pdfs(directory, recursive=args.recursive, quiet=args.quiet)

    if not pdfs:
        return 0

    # Determine worker count
    max_workers = args.jobs if args.jobs is not None else (os.cpu_count() or 4)
    max_workers = max(1, max_workers)  # Ensure at least 1

    # Run with HUD or simple mode, sequential or parallel
    if max_workers == 1:
        # Sequential processing (original behavior)
        if args.hud and not args.dry_run:
            return run_with_hud(pdfs, args, use_ocr)
        else:
            return run_simple(pdfs, args, use_ocr)
    else:
        # Parallel processing
        if args.hud and not args.dry_run:
            return run_with_hud_parallel(pdfs, args, use_ocr, max_workers)
        else:
            return run_simple_parallel(pdfs, args, use_ocr, max_workers)


if __name__ == "__main__":
    sys.exit(main())
