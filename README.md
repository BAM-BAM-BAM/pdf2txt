# pdf2txt

Extract text from PDFs and create markdown files with OCR support and parallel processing.

## Features

- **Text Extraction**: Converts PDF files to markdown format
- **OCR Support**: Uses Tesseract to extract text from image-based pages
- **Parallel Processing**: Utilizes all CPU cores for faster batch processing
- **Quality Scoring**: `--improve` mode re-extracts and only overwrites if quality is better
- **Retro HUD**: Optional 80's style terminal interface with progress tracking
- **WSL Support**: Handles Windows paths transparently

## Installation

```bash
# Clone the repository
git clone https://github.com/BAM-BAM-BAM/pdf2txt.git
cd pdf2txt

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install Tesseract for OCR support
sudo apt install tesseract-ocr
```

## Usage

```bash
# Basic usage - process all PDFs in a directory
./pdf2txt.py /path/to/pdfs

# Recursive search with verbose output
./pdf2txt.py -r -v /path/to/pdfs

# Use the retro HUD display
./pdf2txt.py -r --hud /path/to/pdfs

# Control parallel workers (default: CPU count)
./pdf2txt.py -r -j 8 /path/to/pdfs    # Use 8 workers
./pdf2txt.py -r -j 1 /path/to/pdfs    # Sequential processing

# Re-extract and improve existing files
./pdf2txt.py -r --improve /path/to/pdfs

# Dry run - show what would be processed
./pdf2txt.py -r -n /path/to/pdfs

# Force overwrite existing markdown files
./pdf2txt.py -r -f /path/to/pdfs

# Disable OCR
./pdf2txt.py -r --no-ocr /path/to/pdfs
```

## CLI Options

| Option | Description |
|--------|-------------|
| `-r, --recursive` | Search subdirectories for PDFs |
| `-v, --verbose` | Show detailed progress |
| `-q, --quiet` | Suppress all output except errors |
| `--hud` | Show retro 80's style HUD |
| `-j N, --jobs N` | Number of parallel workers (default: CPU count) |
| `-n, --dry-run` | List files without processing |
| `-f, --force` | Overwrite existing markdown files |
| `--improve` | Re-extract and only overwrite if better quality |
| `--no-ocr` | Disable OCR for image-based pages |

## Output Format

Creates a `.md` file alongside each PDF with:
- Title from filename
- Source path reference
- Page separators with page numbers
- Extracted text content

## Requirements

- Python 3.10+
- PyMuPDF
- Tesseract (optional, for OCR)

## License

MIT
