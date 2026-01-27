#!/usr/bin/env python3
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


@dataclass
class ImageFeature:
    """Features extracted from an image region for learning."""
    # Geometric features
    width: int
    height: int
    area: int
    aspect_ratio: float

    # Position features (normalized to page dimensions)
    page_y_center: float  # 0-1, top to bottom
    region: str  # "header", "body", "footer", "margin"

    # Context features
    surrounding_text_density: float  # chars per 100px around image
    has_nearby_caption: bool

    # Visual features
    brightness_mean: float  # 0-255
    brightness_std: float  # contrast indicator
    is_mostly_white: bool  # >95% pixels above 240
    has_contrast: bool  # std > 30

    def to_vector(self) -> list[float]:
        """Convert to numeric vector for clustering."""
        return [
            self.width / 1000,  # Normalize to ~0-2 range
            self.height / 1000,
            self.area / 1_000_000,
            self.aspect_ratio,
            self.page_y_center,
            {"header": 0.0, "body": 0.5, "footer": 1.0, "margin": 0.25}.get(self.region, 0.5),
            self.surrounding_text_density / 100,
            1.0 if self.has_nearby_caption else 0.0,
            self.brightness_mean / 255,
            self.brightness_std / 128,
            1.0 if self.is_mostly_white else 0.0,
            1.0 if self.has_contrast else 0.0,
        ]

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "width": self.width,
            "height": self.height,
            "area": self.area,
            "aspect_ratio": self.aspect_ratio,
            "page_y_center": self.page_y_center,
            "region": self.region,
            "surrounding_text_density": self.surrounding_text_density,
            "has_nearby_caption": self.has_nearby_caption,
            "brightness_mean": self.brightness_mean,
            "brightness_std": self.brightness_std,
            "is_mostly_white": self.is_mostly_white,
            "has_contrast": self.has_contrast,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ImageFeature":
        """Create from dictionary."""
        return cls(**d)


@dataclass
class OCROutcome:
    """Record of an OCR attempt with features and results."""
    timestamp: float
    pdf_path: str
    page_num: int
    image_index: int
    features: ImageFeature
    ocr_performed: bool
    text_length: int
    word_count: int
    is_useful: bool  # text_length > 10 and word_count >= 2
    cluster_id: int = -1  # Assigned cluster (-1 = not yet clustered)


@dataclass
class ClusterStats:
    """Aggregated statistics for a feature cluster."""
    cluster_id: int
    sample_count: int
    useful_count: int
    # Beta distribution parameters for Bayesian updating
    alpha: float  # Prior + successes
    beta: float  # Prior + failures
    # Decay-weighted recent stats
    recent_useful_rate: float
    last_updated: float

    def usefulness_probability(self) -> float:
        """Expected probability that OCR will be useful for this cluster."""
        return self.alpha / (self.alpha + self.beta)

    def confidence(self) -> float:
        """Confidence in the estimate (0-1 based on sample count)."""
        return min(1.0, self.sample_count / 50)  # Full confidence at 50 samples

    def thompson_sample(self) -> float:
        """Thompson sampling: draw from Beta distribution for exploration."""
        import random
        return random.betavariate(self.alpha, self.beta)


class AdaptiveLearner:
    """Adaptive OCR learning system using feature-based clustering.

    Learns which image types are worth OCR'ing based on:
    - Image features (size, position, brightness, etc.)
    - K-means clustering to group similar images
    - Bayesian updating with Beta distribution per cluster
    - Thompson sampling for exploration/exploitation balance
    """

    DEFAULT_DB_PATH = Path.home() / ".pdf2txt" / "learning.db"
    NUM_CLUSTERS = 12
    MIN_SAMPLES_FOR_PREDICTION = 20
    MIN_SAMPLES_PER_CLUSTER = 10
    EXPLORATION_RATE = 0.10  # Always OCR 10% for learning
    SKIP_CONFIDENCE_THRESHOLD = 0.60
    SKIP_USEFULNESS_THRESHOLD = 0.10

    def __init__(self, db_path: Path | None = None, enabled: bool = True):
        self.db_path = db_path or self.DEFAULT_DB_PATH
        self.enabled = enabled
        self._conn = None
        self._cluster_centers: list[list[float]] | None = None
        self._cluster_stats: dict[int, ClusterStats] = {}
        self._stats = {
            "images_seen": 0,
            "images_skipped": 0,
            "images_ocrd": 0,
            "ocr_useful": 0,       # OCR'd and found useful text
            "ocr_empty": 0,        # OCR'd but no useful text (wasted effort)
            "exploration_ocrs": 0,
            "exploration_useful": 0,  # Exploration found useful text (would've been bad skip)
            "exploration_empty": 0,   # Exploration found nothing (confirms skip OK)
        }

        if self.enabled:
            self._init_db()
            self._load_cluster_stats()

    def _init_db(self):
        """Initialize SQLite database with required tables."""
        import sqlite3

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS ocr_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                pdf_path TEXT NOT NULL,
                page_num INTEGER NOT NULL,
                image_index INTEGER NOT NULL,
                -- Features (stored individually for querying)
                width INTEGER,
                height INTEGER,
                area INTEGER,
                aspect_ratio REAL,
                page_y_center REAL,
                region TEXT,
                surrounding_text_density REAL,
                has_nearby_caption INTEGER,
                brightness_mean REAL,
                brightness_std REAL,
                is_mostly_white INTEGER,
                has_contrast INTEGER,
                -- Outcomes
                ocr_performed INTEGER NOT NULL,
                text_length INTEGER,
                word_count INTEGER,
                is_useful INTEGER,
                cluster_id INTEGER DEFAULT -1,
                -- Index for cleanup
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_ocr_outcomes_timestamp ON ocr_outcomes(timestamp);
            CREATE INDEX IF NOT EXISTS idx_ocr_outcomes_cluster ON ocr_outcomes(cluster_id);

            CREATE TABLE IF NOT EXISTS cluster_stats (
                cluster_id INTEGER PRIMARY KEY,
                sample_count INTEGER NOT NULL DEFAULT 0,
                useful_count INTEGER NOT NULL DEFAULT 0,
                alpha REAL NOT NULL DEFAULT 1.0,
                beta REAL NOT NULL DEFAULT 1.0,
                recent_useful_rate REAL NOT NULL DEFAULT 0.5,
                last_updated REAL NOT NULL,
                center_vector TEXT  -- JSON array of cluster center
            );

            CREATE TABLE IF NOT EXISTS learning_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            -- Track processed files by content hash to avoid reprocessing
            CREATE TABLE IF NOT EXISTS processed_files (
                file_hash TEXT PRIMARY KEY,
                pdf_path TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                page_count INTEGER NOT NULL,
                image_count INTEGER NOT NULL,
                processed_at REAL NOT NULL,
                last_seen_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_processed_files_path ON processed_files(pdf_path);
        """)
        self._conn.commit()

    def _load_cluster_stats(self):
        """Load cluster statistics from database."""
        if not self._conn:
            return

        cursor = self._conn.execute("SELECT * FROM cluster_stats")
        for row in cursor:
            self._cluster_stats[row["cluster_id"]] = ClusterStats(
                cluster_id=row["cluster_id"],
                sample_count=row["sample_count"],
                useful_count=row["useful_count"],
                alpha=row["alpha"],
                beta=row["beta"],
                recent_useful_rate=row["recent_useful_rate"],
                last_updated=row["last_updated"],
            )

        # Load cluster centers if available
        import json
        cursor = self._conn.execute(
            "SELECT value FROM learning_meta WHERE key = 'cluster_centers'"
        )
        row = cursor.fetchone()
        if row:
            self._cluster_centers = json.loads(row["value"])

    def _save_cluster_stats(self, stats: ClusterStats):
        """Save cluster statistics to database."""
        if not self._conn:
            return

        self._conn.execute("""
            INSERT OR REPLACE INTO cluster_stats
            (cluster_id, sample_count, useful_count, alpha, beta, recent_useful_rate, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            stats.cluster_id,
            stats.sample_count,
            stats.useful_count,
            stats.alpha,
            stats.beta,
            stats.recent_useful_rate,
            stats.last_updated,
        ))
        self._conn.commit()

    def _find_nearest_cluster(self, feature_vector: list[float]) -> int:
        """Find the nearest cluster for a feature vector."""
        if not self._cluster_centers:
            return -1

        min_dist = float("inf")
        nearest = -1
        for i, center in enumerate(self._cluster_centers):
            dist = sum((a - b) ** 2 for a, b in zip(feature_vector, center))
            if dist < min_dist:
                min_dist = dist
                nearest = i
        return nearest

    def should_ocr(self, features: ImageFeature) -> tuple[bool, str, bool]:
        """Decide whether to OCR this image.

        Returns:
            Tuple of (should_ocr, reason, is_exploration)
        """
        if not self.enabled:
            return True, "learning disabled", False

        self._stats["images_seen"] += 1
        feature_vector = features.to_vector()

        # Exploration: always OCR some images to keep learning
        import random
        if random.random() < self.EXPLORATION_RATE:
            self._stats["exploration_ocrs"] += 1
            return True, "exploration", True

        # Not enough data yet - use heuristics
        total_samples = sum(s.sample_count for s in self._cluster_stats.values())
        if total_samples < self.MIN_SAMPLES_FOR_PREDICTION:
            should, reason = self._heuristic_decision(features)
            return should, reason, False

        # Find cluster
        cluster_id = self._find_nearest_cluster(feature_vector)
        if cluster_id < 0 or cluster_id not in self._cluster_stats:
            should, reason = self._heuristic_decision(features)
            return should, reason, False

        stats = self._cluster_stats[cluster_id]
        if stats.sample_count < self.MIN_SAMPLES_PER_CLUSTER:
            return True, f"cluster {cluster_id} needs more samples", False

        # Use Thompson sampling for exploration/exploitation
        sampled_usefulness = stats.thompson_sample()
        confidence = stats.confidence()

        # Only skip if confident AND predicted usefulness is very low
        if confidence > self.SKIP_CONFIDENCE_THRESHOLD and sampled_usefulness < self.SKIP_USEFULNESS_THRESHOLD:
            self._stats["images_skipped"] += 1
            return False, f"cluster {cluster_id}: {sampled_usefulness:.1%} useful (conf: {confidence:.1%})", False

        self._stats["images_ocrd"] += 1
        return True, f"cluster {cluster_id}: {sampled_usefulness:.1%} useful", False

    def _heuristic_decision(self, features: ImageFeature) -> tuple[bool, str]:
        """Fallback heuristics when not enough learning data."""
        # Skip tiny images (likely icons/bullets)
        if features.area < 400:  # 20x20
            self._stats["images_skipped"] += 1
            return False, "heuristic: tiny image"

        # Skip mostly-white images with no contrast (likely blank/whitespace)
        if features.is_mostly_white and not features.has_contrast:
            self._stats["images_skipped"] += 1
            return False, "heuristic: blank/white"

        # Skip margin decorations (narrow aspect ratio in margins)
        if features.region == "margin" and (features.aspect_ratio > 5 or features.aspect_ratio < 0.2):
            self._stats["images_skipped"] += 1
            return False, "heuristic: margin decoration"

        self._stats["images_ocrd"] += 1
        return True, "heuristic: worth trying"

    def record_outcome(
        self,
        features: ImageFeature,
        pdf_path: str,
        page_num: int,
        image_index: int,
        ocr_performed: bool,
        text: str,
        is_exploration: bool = False,
    ):
        """Record the outcome of an OCR decision."""
        if not self.enabled or not self._conn:
            return

        text_length = len(text) if text else 0
        word_count = len(text.split()) if text else 0
        is_useful = text_length > 10 and word_count >= 2

        # Track accuracy metrics
        if ocr_performed:
            if is_useful:
                self._stats["ocr_useful"] += 1
            else:
                self._stats["ocr_empty"] += 1

            # Track exploration accuracy separately
            if is_exploration:
                if is_useful:
                    self._stats["exploration_useful"] += 1  # Would've been bad to skip
                else:
                    self._stats["exploration_empty"] += 1   # Confirms skipping is OK

        feature_vector = features.to_vector()
        cluster_id = self._find_nearest_cluster(feature_vector) if self._cluster_centers else -1

        # Insert outcome record
        self._conn.execute("""
            INSERT INTO ocr_outcomes (
                timestamp, pdf_path, page_num, image_index,
                width, height, area, aspect_ratio, page_y_center, region,
                surrounding_text_density, has_nearby_caption,
                brightness_mean, brightness_std, is_mostly_white, has_contrast,
                ocr_performed, text_length, word_count, is_useful, cluster_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            time.time(), pdf_path, page_num, image_index,
            features.width, features.height, features.area, features.aspect_ratio,
            features.page_y_center, features.region,
            features.surrounding_text_density, int(features.has_nearby_caption),
            features.brightness_mean, features.brightness_std,
            int(features.is_mostly_white), int(features.has_contrast),
            int(ocr_performed), text_length, word_count, int(is_useful), cluster_id,
        ))
        self._conn.commit()

        # Update cluster stats if we have clusters and OCR was performed
        if ocr_performed and cluster_id >= 0:
            self._update_cluster_stats(cluster_id, is_useful)

    def _update_cluster_stats(self, cluster_id: int, is_useful: bool):
        """Update statistics for a cluster with Bayesian updating."""
        if cluster_id not in self._cluster_stats:
            self._cluster_stats[cluster_id] = ClusterStats(
                cluster_id=cluster_id,
                sample_count=0,
                useful_count=0,
                alpha=1.0,  # Prior: assume 50% useful
                beta=1.0,
                recent_useful_rate=0.5,
                last_updated=time.time(),
            )

        stats = self._cluster_stats[cluster_id]
        stats.sample_count += 1
        if is_useful:
            stats.useful_count += 1
            stats.alpha += 1
        else:
            stats.beta += 1

        # Exponential decay for recent rate (weight recent more heavily)
        decay = 0.95
        stats.recent_useful_rate = decay * stats.recent_useful_rate + (1 - decay) * (1.0 if is_useful else 0.0)
        stats.last_updated = time.time()

        self._save_cluster_stats(stats)

    def recluster(self, force: bool = False):
        """Re-run K-means clustering on all outcomes.

        Called periodically to update cluster assignments.
        """
        if not self.enabled or not self._conn:
            return

        # Get all feature vectors
        cursor = self._conn.execute("""
            SELECT id, width, height, area, aspect_ratio, page_y_center, region,
                   surrounding_text_density, has_nearby_caption,
                   brightness_mean, brightness_std, is_mostly_white, has_contrast,
                   is_useful
            FROM ocr_outcomes
            WHERE ocr_performed = 1
            ORDER BY timestamp DESC
            LIMIT ?
        """, (self.MAX_RECORDS,))

        rows = cursor.fetchall()
        if len(rows) < self.NUM_CLUSTERS * 2:
            return  # Not enough data for meaningful clustering

        # Build feature matrix
        import json
        vectors = []
        ids = []
        outcomes = []
        for row in rows:
            features = ImageFeature(
                width=row["width"],
                height=row["height"],
                area=row["area"],
                aspect_ratio=row["aspect_ratio"],
                page_y_center=row["page_y_center"],
                region=row["region"],
                surrounding_text_density=row["surrounding_text_density"],
                has_nearby_caption=bool(row["has_nearby_caption"]),
                brightness_mean=row["brightness_mean"],
                brightness_std=row["brightness_std"],
                is_mostly_white=bool(row["is_mostly_white"]),
                has_contrast=bool(row["has_contrast"]),
            )
            vectors.append(features.to_vector())
            ids.append(row["id"])
            outcomes.append(bool(row["is_useful"]))

        # Simple K-means implementation (avoid sklearn dependency)
        centers, assignments = self._kmeans(vectors, self.NUM_CLUSTERS)

        # Update cluster assignments in database
        for record_id, cluster_id in zip(ids, assignments):
            self._conn.execute(
                "UPDATE ocr_outcomes SET cluster_id = ? WHERE id = ?",
                (cluster_id, record_id)
            )

        # Rebuild cluster stats from assignments
        self._cluster_stats.clear()
        for cluster_id, is_useful in zip(assignments, outcomes):
            if cluster_id not in self._cluster_stats:
                self._cluster_stats[cluster_id] = ClusterStats(
                    cluster_id=cluster_id,
                    sample_count=0,
                    useful_count=0,
                    alpha=1.0,
                    beta=1.0,
                    recent_useful_rate=0.5,
                    last_updated=time.time(),
                )
            stats = self._cluster_stats[cluster_id]
            stats.sample_count += 1
            if is_useful:
                stats.useful_count += 1
                stats.alpha += 1
            else:
                stats.beta += 1

        # Save cluster centers
        self._cluster_centers = centers
        self._conn.execute(
            "INSERT OR REPLACE INTO learning_meta (key, value) VALUES (?, ?)",
            ("cluster_centers", json.dumps(centers))
        )

        # Save all cluster stats
        for stats in self._cluster_stats.values():
            stats.recent_useful_rate = stats.useful_count / max(stats.sample_count, 1)
            self._save_cluster_stats(stats)

        self._conn.commit()

    def _kmeans(self, vectors: list[list[float]], k: int, max_iter: int = 100) -> tuple[list[list[float]], list[int]]:
        """Simple K-means clustering implementation."""
        import random

        if not vectors:
            return [], []

        n_features = len(vectors[0])

        # Initialize centers randomly from data points
        centers = random.sample(vectors, min(k, len(vectors)))
        while len(centers) < k:
            centers.append([random.random() for _ in range(n_features)])

        assignments = [-1] * len(vectors)

        for _ in range(max_iter):
            # Assign points to nearest center
            new_assignments = []
            for vec in vectors:
                min_dist = float("inf")
                nearest = 0
                for i, center in enumerate(centers):
                    dist = sum((a - b) ** 2 for a, b in zip(vec, center))
                    if dist < min_dist:
                        min_dist = dist
                        nearest = i
                new_assignments.append(nearest)

            # Check for convergence
            if new_assignments == assignments:
                break
            assignments = new_assignments

            # Update centers
            new_centers = [[0.0] * n_features for _ in range(k)]
            counts = [0] * k
            for vec, cluster_id in zip(vectors, assignments):
                for j, val in enumerate(vec):
                    new_centers[cluster_id][j] += val
                counts[cluster_id] += 1

            for i in range(k):
                if counts[i] > 0:
                    new_centers[i] = [v / counts[i] for v in new_centers[i]]
                else:
                    new_centers[i] = centers[i]  # Keep old center if empty
            centers = new_centers

        return centers, assignments

    @staticmethod
    def compute_file_hash(pdf_path: Path) -> str:
        """Compute MD5 hash of a PDF file for deduplication."""
        import hashlib
        hasher = hashlib.md5()
        with open(pdf_path, "rb") as f:
            # Read in chunks for memory efficiency
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def is_file_processed(self, pdf_path: Path) -> bool:
        """Check if a file has already been processed (by content hash)."""
        if not self.enabled or not self._conn:
            return False

        file_hash = self.compute_file_hash(pdf_path)
        cursor = self._conn.execute(
            "SELECT file_hash FROM processed_files WHERE file_hash = ?",
            (file_hash,)
        )
        exists = cursor.fetchone() is not None

        if exists:
            # Update last_seen_at timestamp
            self._conn.execute(
                "UPDATE processed_files SET last_seen_at = ?, pdf_path = ? WHERE file_hash = ?",
                (time.time(), str(pdf_path), file_hash)
            )
            self._conn.commit()

        return exists

    def record_file_processed(
        self,
        pdf_path: Path,
        page_count: int,
        image_count: int,
    ):
        """Record that a file has been processed."""
        if not self.enabled or not self._conn:
            return

        file_hash = self.compute_file_hash(pdf_path)
        file_size = pdf_path.stat().st_size
        now = time.time()

        self._conn.execute("""
            INSERT OR REPLACE INTO processed_files
            (file_hash, pdf_path, file_size, page_count, image_count, processed_at, last_seen_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (file_hash, str(pdf_path), file_size, page_count, image_count, now, now))
        self._conn.commit()

    def get_stats(self) -> dict:
        """Get learning statistics for display."""
        if not self.enabled or not self._conn:
            return {"enabled": False}

        cursor = self._conn.execute("SELECT COUNT(*) as total FROM ocr_outcomes")
        total_records = cursor.fetchone()["total"]

        cursor = self._conn.execute(
            "SELECT COUNT(*) as useful FROM ocr_outcomes WHERE is_useful = 1 AND ocr_performed = 1"
        )
        useful_records = cursor.fetchone()["useful"]

        cursor = self._conn.execute(
            "SELECT COUNT(*) as ocrd FROM ocr_outcomes WHERE ocr_performed = 1"
        )
        ocrd_records = cursor.fetchone()["ocrd"]

        cluster_info = []
        for stats in sorted(self._cluster_stats.values(), key=lambda s: s.cluster_id):
            cluster_info.append({
                "id": stats.cluster_id,
                "samples": stats.sample_count,
                "useful_rate": stats.usefulness_probability(),
                "confidence": stats.confidence(),
            })

        # Get processed files stats
        cursor = self._conn.execute("SELECT COUNT(*) as count FROM processed_files")
        processed_files_count = cursor.fetchone()["count"]

        cursor = self._conn.execute(
            "SELECT SUM(page_count) as pages, SUM(image_count) as images FROM processed_files"
        )
        row = cursor.fetchone()
        total_pages = row["pages"] or 0
        total_images = row["images"] or 0

        return {
            "enabled": True,
            "db_path": str(self.db_path),
            "total_records": total_records,
            "ocrd_records": ocrd_records,
            "useful_records": useful_records,
            "overall_useful_rate": useful_records / max(ocrd_records, 1),
            "num_clusters": len(self._cluster_stats),
            "clusters": cluster_info,
            "session_stats": self._stats.copy(),
            "processed_files": processed_files_count,
            "total_pages_processed": total_pages,
            "total_images_seen": total_images,
        }

    def reset(self):
        """Reset the learning database."""
        if self._conn:
            self._conn.close()
            self._conn = None

        if self.db_path.exists():
            self.db_path.unlink()

        self._cluster_centers = None
        self._cluster_stats.clear()
        self._stats = {
            "images_seen": 0,
            "images_skipped": 0,
            "images_ocrd": 0,
            "ocr_useful": 0,
            "ocr_empty": 0,
            "exploration_ocrs": 0,
            "exploration_useful": 0,
            "exploration_empty": 0,
        }

        if self.enabled:
            self._init_db()

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


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
    log_messages: list = field(default_factory=list)


class RetroHUD:
    """80's style terminal HUD using curses."""

    def __init__(self, stats: ProcessingStats, learner: "AdaptiveLearner | None" = None):
        self.stats = stats
        self.learner = learner
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
        self.print_final_summary()

    def print_final_summary(self):
        """Print a final summary after exiting curses mode."""
        s = self.stats
        elapsed = s.elapsed()
        mb_processed = s.processed_bytes / (1024 * 1024)
        mb_total = s.total_bytes / (1024 * 1024)
        md_mb = s.md_bytes / (1024 * 1024)
        ratio = (s.md_bytes / s.processed_bytes * 100) if s.processed_bytes > 0 else 0
        files_per_min = (s.processed_files / elapsed * 60) if elapsed > 0 else 0
        mb_per_min = (mb_processed / elapsed * 60) if elapsed > 0 else 0

        print()
        print("═" * 60)
        print("  PDF2TXT - FINAL RESULTS")
        print("═" * 60)
        print(f"  Files:     {s.processed_files:,} processed, {s.skipped_files:,} skipped, {s.failed_files:,} failed")
        if s.improved_files > 0 or s.kept_existing > 0:
            print(f"  Quality:   {s.improved_files:,} improved, {s.kept_existing:,} kept existing")
        print(f"  Pages:     {s.processed_pages:,} total, {s.ocr_pages:,} OCR'd ({s.ocr_chars:,} chars)")
        print(f"  Data:      {mb_processed:.2f} MB in → {md_mb:.2f} MB out ({ratio:.1f}%)")
        print(f"  Time:      {elapsed:.1f}s ({files_per_min:.1f} files/min, {mb_per_min:.2f} MB/min)")
        print("═" * 60)
        print()

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

            # Title banner (all lines 65 chars)
            banner = [
                "╔═══════════════════════════════════════════════════════════════╗",
                "║   ██████╗ ██████╗ ███████╗██████╗ ████████╗██╗  ██╗████████╗  ║",
                "║   ██╔══██╗██╔══██╗██╔════╝╚════██╗╚══██╔══╝╚██╗██╔╝╚══██╔══╝  ║",
                "║   ██████╔╝██║  ██║█████╗   █████╔╝   ██║    ╚███╔╝    ██║     ║",
                "║   ██╔═══╝ ██║  ██║██╔══╝  ██╔═══╝    ██║    ██╔██╗    ██║     ║",
                "║   ██║     ██████╔╝██║     ███████╗   ██║   ██╔╝ ██╗   ██║     ║",
                "║   ╚═╝     ╚═════╝ ╚═╝     ╚══════╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝     ║",
                "╚═══════════════════════════════════════════════════════════════╝",
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

            # Results box (taller if learning enabled)
            y_offset += box_height + 1
            has_learning = self.learner and self.learner.enabled
            results_height = 10 if has_learning else 6
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

                y += 1
                total_pages = self.stats.processed_pages
                self.draw_stat(y, 2, "TOTAL PAGES: ", f"{total_pages:,}")

                # Learning stats
                if has_learning:
                    y += 1
                    self.stdscr.addstr(y, 2, "─" * 30, curses.color_pair(2))
                    self.stdscr.addstr(y, 34, " LEARNING ", curses.color_pair(1) | curses.A_BOLD)
                    self.stdscr.addstr(y, 44, "─" * 30, curses.color_pair(2))

                    y += 1
                    ls = self.learner._stats
                    self.draw_stat(y, 2, "IMAGES: ", f"{ls['images_seen']:4d}")
                    self.draw_stat(y, 18, "OCR'd: ", f"{ls['images_ocrd']:4d}")
                    self.draw_stat(y, 34, "SKIPPED: ", f"{ls['images_skipped']:4d}")

                    # OCR efficiency: what % of OCRs found useful text
                    total_ocrd = ls['ocr_useful'] + ls['ocr_empty']
                    if total_ocrd > 0:
                        ocr_eff = ls['ocr_useful'] / total_ocrd * 100
                        self.draw_stat(y, 52, "OCR EFF: ", f"{ocr_eff:4.1f}%")

                    # Second row: exploration accuracy
                    y += 1
                    exp_total = ls['exploration_useful'] + ls['exploration_empty']
                    if exp_total > 0:
                        # Miss rate: exploration found useful text we would've skipped
                        miss_rate = ls['exploration_useful'] / exp_total * 100
                        self.draw_stat(y, 2, "EXPLORE: ", f"{exp_total:4d}")
                        self.draw_stat(y, 18, "WOULD MISS: ", f"{ls['exploration_useful']:3d}")
                        # Color code: green if low miss rate, red if high
                        miss_color = curses.color_pair(4) if miss_rate > 20 else curses.color_pair(1)
                        self.stdscr.addstr(y, 48, f"MISS RATE: {miss_rate:4.1f}%", miss_color | curses.A_BOLD)

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


def check_paddleocr_available() -> bool:
    """Check if PaddleOCR is available."""
    try:
        import paddleocr
        return True
    except ImportError:
        return False


def check_surya_available() -> bool:
    """Check if Surya OCR is available."""
    try:
        # Suppress Surya's "Checking connectivity" message during import
        with SuppressOutputFD(suppress=True):
            import surya
        return True
    except ImportError:
        return False


def resolve_ocr_engine(requested: str, use_ocr: bool = True) -> tuple[bool, str, list[str]]:
    """Resolve which OCR engine to use with fallback chain.

    Args:
        requested: Requested OCR engine ("surya", "paddle", "tesseract", "none")
        use_ocr: Whether OCR is enabled

    Returns:
        Tuple of (ocr_available, actual_engine, log_messages)
    """
    log_msgs = []

    if not use_ocr or requested == "none":
        return False, "none", ["OCR: disabled"]

    # Try requested engine first, then fallbacks
    fallback_order = {
        "surya": ["surya", "paddle", "tesseract"],
        "paddle": ["paddle", "tesseract"],
        "tesseract": ["tesseract"],
    }

    engines = fallback_order.get(requested, ["tesseract"])
    checks = {
        "surya": check_surya_available,
        "paddle": check_paddleocr_available,
        "tesseract": check_tesseract_available,
    }

    for engine in engines:
        if checks[engine]():
            if engine != requested:
                log_msgs.append(f"OCR: {requested} unavailable, using {engine}")
            else:
                log_msgs.append(f"OCR: using {engine}")
            return True, engine, log_msgs

    log_msgs.append(f"OCR: no engines available ({'/'.join(engines)})")
    return False, "none", log_msgs


def get_gpu_info() -> dict:
    """Get GPU information for debugging."""
    info = {
        'cuda_available': False,
        'device_count': 0,
        'devices': [],
        'error': None,
    }

    # Try torch first for CUDA info
    torch_available = False
    try:
        import torch
        torch_available = True
        info['cuda_available'] = torch.cuda.is_available()
        if info['cuda_available']:
            info['device_count'] = torch.cuda.device_count()
            for i in range(info['device_count']):
                device_info = {
                    'index': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory_total': torch.cuda.get_device_properties(i).total_memory,
                    'memory_allocated': torch.cuda.memory_allocated(i),
                    'memory_reserved': torch.cuda.memory_reserved(i),
                }
                info['devices'].append(device_info)
    except ImportError:
        pass
    except Exception as e:
        info['error'] = str(e)

    # Always try nvidia-smi for detailed/accurate info (works even without torch)
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.free,memory.used,memory.total,'
             'utilization.gpu,temperature.gpu,power.draw,display_active',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            # If we didn't get device info from torch, create entries from nvidia-smi
            if not info['devices']:
                info['cuda_available'] = True
                info['device_count'] = len(lines)
                for i, line in enumerate(lines):
                    info['devices'].append({'index': i})

            # Update each device with nvidia-smi data
            for i, line in enumerate(lines):
                if i < len(info['devices']):
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 8:
                        dev = info['devices'][i]
                        dev['name'] = parts[0]
                        dev['memory_free_mb'] = int(parts[1])
                        dev['memory_used_mb'] = int(parts[2])
                        dev['memory_total_mb'] = int(parts[3])
                        dev['gpu_util'] = parts[4]
                        dev['temperature'] = parts[5]
                        dev['power_draw'] = parts[6]
                        dev['display_active'] = parts[7]
            info['error'] = None  # Clear any torch error if nvidia-smi worked
    except FileNotFoundError:
        if not torch_available:
            info['error'] = 'torch not installed and nvidia-smi not found'
    except Exception as e:
        if not info['devices']:
            info['error'] = str(e)

    return info


def print_gpu_debug_info():
    """Print GPU debugging information."""
    info = get_gpu_info()

    print("[DEBUG] GPU Information:")
    if info['error']:
        print(f"  Error: {info['error']}")
    else:
        print(f"  CUDA available: {info['cuda_available']}")
        if info['cuda_available']:
            print(f"  Device count: {info['device_count']}")
            for dev in info['devices']:
                print(f"  Device {dev['index']}: {dev['name']}")
                if 'memory_total_mb' in dev:
                    used = dev['memory_used_mb']
                    total = dev['memory_total_mb']
                    free = dev['memory_free_mb']
                    pct = (used / total * 100) if total > 0 else 0
                    print(f"    Memory: {used:,} MB used / {total:,} MB total ({pct:.1f}% used)")
                    print(f"    Free: {free:,} MB")
                    # Display and power info
                    if 'display_active' in dev:
                        display_status = dev['display_active']
                        if display_status.lower() == 'enabled':
                            print(f"    Display: Active (using VRAM for framebuffer)")
                        else:
                            print(f"    Display: {display_status}")
                    if 'gpu_util' in dev:
                        print(f"    GPU Load: {dev['gpu_util']}%  |  Temp: {dev['temperature']}°C  |  Power: {dev['power_draw']}W")
                else:
                    total_mb = dev['memory_total'] / (1024 * 1024)
                    alloc_mb = dev['memory_allocated'] / (1024 * 1024)
                    reserved_mb = dev['memory_reserved'] / (1024 * 1024)
                    print(f"    Total: {total_mb:,.0f} MB")
                    print(f"    PyTorch allocated: {alloc_mb:,.0f} MB")
                    print(f"    PyTorch reserved: {reserved_mb:,.0f} MB")
        else:
            # Check if CUDA_VISIBLE_DEVICES is set (--cpu mode)
            cuda_env = os.environ.get('CUDA_VISIBLE_DEVICES', None)
            if cuda_env == '':
                print("  (CUDA disabled via --cpu flag)")
            else:
                print("  (No CUDA devices found)")
    print()


def clear_gpu_memory():
    """Clear GPU memory and attempt to kill zombie CUDA processes."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass

    # Try to identify and kill zombie GPU processes (Linux/WSL only)
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            import os
            import signal
            current_pid = os.getpid()
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    try:
                        pid = int(line.strip())
                        # Don't kill ourselves or parent processes
                        if pid != current_pid and pid != os.getppid():
                            # Check if process exists but is zombie/orphaned
                            try:
                                os.kill(pid, 0)  # Check if exists
                            except ProcessLookupError:
                                # Process doesn't exist but holding GPU memory - can't do much in userspace
                                pass
                    except (ValueError, ProcessLookupError, PermissionError):
                        pass
    except Exception:
        pass


# Global OCR instances (lazy-loaded)
_paddle_ocr_instance = None
_surya_ocr_instance = None


def get_paddle_ocr():
    """Get or create the global PaddleOCR instance."""
    global _paddle_ocr_instance
    if _paddle_ocr_instance is None:
        from paddleocr import PaddleOCR
        # Suppress PaddleOCR's verbose logging via Python logging module
        import logging
        logging.getLogger('ppocr').setLevel(logging.ERROR)
        logging.getLogger('paddle').setLevel(logging.ERROR)
        logging.getLogger('paddlenlp').setLevel(logging.ERROR)
        logging.getLogger('paddlex').setLevel(logging.ERROR)
        # Suppress stdout/stderr during initialization (native code spam)
        with SuppressOutputFD(suppress=True):
            # Note: PaddleOCR v3+ uses use_textline_orientation instead of use_angle_cls
            _paddle_ocr_instance = PaddleOCR(use_textline_orientation=True, lang='en')
    return _paddle_ocr_instance


def configure_surya_batch_sizes():
    """Configure Surya batch sizes based on available VRAM.

    Surya defaults:
    - RECOGNITION_BATCH_SIZE=512 (~40-50MB per item = ~20GB VRAM)
    - DETECTOR_BATCH_SIZE=default (~280MB per item)

    These must be set BEFORE importing surya modules.
    """
    # Skip if already configured or running on CPU
    if os.environ.get('RECOGNITION_BATCH_SIZE') or os.environ.get('CUDA_VISIBLE_DEVICES') == '':
        return

    # Get free VRAM
    gpu_info = get_gpu_info()
    if not gpu_info['cuda_available'] or not gpu_info['devices']:
        return

    dev = gpu_info['devices'][0]
    free_mb = dev.get('memory_free_mb', 0)

    if free_mb <= 0:
        return

    # Reserve ~4GB for models, use remaining for batching
    # Recognition: ~50MB per batch item
    # Detection: ~280MB per batch item
    available_for_batching = max(0, free_mb - 4000)  # Reserve 4GB for models

    # Calculate safe batch sizes
    rec_batch = max(4, min(64, available_for_batching // 50))
    det_batch = max(1, min(8, available_for_batching // 280))

    os.environ['RECOGNITION_BATCH_SIZE'] = str(rec_batch)
    os.environ['DETECTOR_BATCH_SIZE'] = str(det_batch)


def get_surya_ocr():
    """Get or create the global Surya OCR models."""
    global _surya_ocr_instance
    if _surya_ocr_instance is None:
        # Clear any leftover GPU memory before loading heavy models
        clear_gpu_memory()
        # Configure batch sizes based on available VRAM (must be before import)
        configure_surya_batch_sizes()
        from surya.recognition import RecognitionPredictor, FoundationPredictor
        from surya.detection import DetectionPredictor
        detection = DetectionPredictor()
        foundation = FoundationPredictor()
        recognition = RecognitionPredictor(foundation)
        _surya_ocr_instance = {
            'recognition': recognition,
            'detection': detection,
        }
    return _surya_ocr_instance


def ocr_page_with_paddle(page, dpi: int = 300, debug: bool = False) -> str:
    """Extract text from a PDF page using PaddleOCR."""
    import io
    import numpy as np
    from PIL import Image

    # Render page to image
    pix = page.get_pixmap(dpi=dpi)
    img_data = pix.tobytes("png")

    # Convert PNG bytes to numpy array (PaddleOCR needs numpy array, not raw bytes)
    img = Image.open(io.BytesIO(img_data))
    # Ensure RGB mode (PaddleOCR doesn't handle RGBA well)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img_array = np.array(img)

    # Run PaddleOCR with output suppression (native code spams binary data)
    ocr = get_paddle_ocr()
    with SuppressOutputFD(suppress=True):
        result = ocr.ocr(img_array)

    # Extract text from results - handle multiple API versions
    if not result:
        return ""

    lines = []

    # New PaddleOCR API (v3+) returns dict with 'rec_texts'
    if isinstance(result, dict):
        if debug:
            import sys
            print(f"[DEBUG] PaddleOCR result type: dict, keys: {list(result.keys())}", file=sys.stderr)
        if 'rec_texts' in result:
            lines = [str(t) for t in result['rec_texts'] if t]
        elif 'data' in result and isinstance(result['data'], dict):
            # Another possible format
            if 'rec_texts' in result['data']:
                lines = [str(t) for t in result['data']['rec_texts'] if t]
    # Old API returns list of lists: [[[bbox, (text, conf)], ...]]
    elif isinstance(result, list):
        if debug:
            import sys
            print(f"[DEBUG] PaddleOCR result type: list, len: {len(result)}, first item type: {type(result[0]) if result else None}", file=sys.stderr)
        if not result[0]:
            return ""
        for line in result[0]:
            if line and len(line) >= 2:
                text = line[1][0] if isinstance(line[1], (list, tuple)) else line[1]
                if text:
                    lines.append(str(text))
    else:
        if debug:
            import sys
            print(f"[DEBUG] PaddleOCR result type: {type(result)}", file=sys.stderr)

    return '\n'.join(lines)


def ocr_page_with_surya(page, dpi: int = 300) -> str:
    """Extract text from a PDF page using Surya OCR."""
    import io
    from PIL import Image

    # Render page to image
    pix = page.get_pixmap(dpi=dpi)
    img_data = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_data))

    # Get OCR models
    models = get_surya_ocr()

    # Run recognition with detection (new Surya API)
    rec_results = models['recognition']([img], det_predictor=models['detection'])

    # Extract text from results
    if not rec_results or not rec_results[0]:
        return ""

    lines = []
    for text_line in rec_results[0].text_lines:
        if text_line.text:
            lines.append(text_line.text)

    return '\n'.join(lines)


def extract_image_features(
    page,
    bbox: tuple[float, float, float, float],
    img,
    text_blocks: list | None = None,
) -> ImageFeature:
    """Extract features from an image region for learning.

    Args:
        page: PyMuPDF page object
        bbox: (x0, y0, x1, y1) bounding box in page coordinates
        img: PIL Image of the region
        text_blocks: List of text blocks from page.get_text("blocks") for context

    Returns:
        ImageFeature with all extracted characteristics
    """
    import numpy as np

    x0, y0, x1, y1 = bbox
    page_rect = page.rect
    page_height = page_rect.height
    page_width = page_rect.width

    # Geometric features
    width = int(x1 - x0)
    height = int(y1 - y0)
    area = width * height
    aspect_ratio = width / max(height, 1)

    # Position features
    y_center = (y0 + y1) / 2
    page_y_center = y_center / max(page_height, 1)

    # Determine region (header/body/footer/margin)
    x_center = (x0 + x1) / 2
    if page_y_center < 0.12:
        region = "header"
    elif page_y_center > 0.88:
        region = "footer"
    elif x_center < page_width * 0.1 or x_center > page_width * 0.9:
        region = "margin"
    else:
        region = "body"

    # Context features - surrounding text density
    surrounding_text_density = 0.0
    has_nearby_caption = False

    if text_blocks:
        # Count text chars within 50 page units of this image
        search_margin = 50
        text_chars_nearby = 0
        for block in text_blocks:
            bx0, by0, bx1, by1, content, _, block_type = block
            if block_type != 0:  # Skip non-text blocks
                continue
            # Check if block is near the image
            if (bx0 < x1 + search_margin and bx1 > x0 - search_margin and
                by0 < y1 + search_margin and by1 > y0 - search_margin):
                text_chars_nearby += len(str(content))
                # Check for caption-like text below image
                if by0 > y1 and by0 < y1 + 30 and len(str(content)) < 200:
                    content_lower = str(content).lower()
                    if any(kw in content_lower for kw in ["figure", "fig.", "image", "photo", "chart", "table"]):
                        has_nearby_caption = True
        # Normalize: chars per 100 pixels of search area
        search_area = (x1 - x0 + 2 * search_margin) * (y1 - y0 + 2 * search_margin)
        surrounding_text_density = (text_chars_nearby / max(search_area, 1)) * 10000

    # Visual features from the image
    if img.mode != 'L':
        gray = img.convert('L')
    else:
        gray = img
    pixels = np.array(gray)

    brightness_mean = float(np.mean(pixels))
    brightness_std = float(np.std(pixels))
    is_mostly_white = float(np.mean(pixels > 240)) > 0.95
    has_contrast = brightness_std > 30

    return ImageFeature(
        width=width,
        height=height,
        area=area,
        aspect_ratio=aspect_ratio,
        page_y_center=page_y_center,
        region=region,
        surrounding_text_density=surrounding_text_density,
        has_nearby_caption=has_nearby_caption,
        brightness_mean=brightness_mean,
        brightness_std=brightness_std,
        is_mostly_white=is_mostly_white,
        has_contrast=has_contrast,
    )


def ocr_image_region(
    page,
    bbox,
    ocr_engine: str = "surya",
    dpi: int = 300,
    learner: AdaptiveLearner | None = None,
    pdf_path: str = "",
    page_num: int = 0,
    image_index: int = 0,
    text_blocks: list | None = None,
) -> tuple[str, bool]:
    """OCR a specific region of a page.

    Args:
        page: PyMuPDF page object
        bbox: Tuple of (x0, y0, x1, y1) defining the region
        ocr_engine: "surya", "paddle", or "tesseract"
        dpi: Resolution for rendering
        learner: Optional AdaptiveLearner for skip decisions
        pdf_path: PDF file path for logging
        page_num: Page number for logging
        image_index: Image index on page for logging
        text_blocks: Text blocks for context analysis

    Returns:
        Tuple of (extracted_text, was_ocr_performed)
    """
    import io
    from PIL import Image

    # Create a clip rect for the region
    import pymupdf
    clip = pymupdf.Rect(bbox[0], bbox[1], bbox[2], bbox[3])

    # Render just this region at the specified DPI
    mat = pymupdf.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, clip=clip)
    img_data = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_data))

    # Skip tiny images (likely decorative)
    if img.width < 20 or img.height < 10:
        return "", False

    # Extract features for learning
    features = None
    is_exploration = False
    if learner:
        features = extract_image_features(page, bbox, img, text_blocks)
        do_ocr, reason, is_exploration = learner.should_ocr(features)
        if not do_ocr:
            # Record that we skipped this image (no OCR, no text)
            learner.record_outcome(features, pdf_path, page_num, image_index, False, "")
            return "", False

    # Perform OCR
    text = ""
    if ocr_engine == "surya":
        models = get_surya_ocr()
        rec_results = models['recognition']([img], det_predictor=models['detection'])
        if rec_results and rec_results[0]:
            lines = [tl.text for tl in rec_results[0].text_lines if tl.text]
            text = '\n'.join(lines)

    elif ocr_engine == "paddle":
        import numpy as np
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img_array = np.array(img)
        ocr = get_paddle_ocr()
        with SuppressOutputFD(suppress=True):
            result = ocr.ocr(img_array)
        if result:
            lines = []
            if isinstance(result, dict) and 'rec_texts' in result:
                lines = [str(t) for t in result['rec_texts'] if t]
            elif isinstance(result, list) and result[0]:
                for line in result[0]:
                    if line and len(line) >= 2:
                        line_text = line[1][0] if isinstance(line[1], (list, tuple)) else line[1]
                        if line_text:
                            lines.append(str(line_text))
            text = '\n'.join(lines)

    else:  # tesseract
        tp = page.get_textpage_ocr(full=True, language="eng", clip=clip)
        text = page.get_text(textpage=tp, clip=clip).strip()

    # Record outcome for learning
    if learner and features:
        learner.record_outcome(features, pdf_path, page_num, image_index, True, text, is_exploration)

    return text, True


def extract_page_hybrid(
    page,
    ocr_engine: str = "surya",
    dpi: int = 300,
    debug: bool = False,
    learner: AdaptiveLearner | None = None,
    pdf_path: str = "",
    page_num: int = 0,
) -> tuple[str, int, int, int]:
    """Extract text from page using hybrid approach: text extraction + OCR for images.

    Args:
        page: PyMuPDF page object
        ocr_engine: OCR engine to use for image regions
        dpi: Resolution for rendering images
        debug: Print debug info
        learner: Optional AdaptiveLearner for skip decisions
        pdf_path: PDF file path for logging
        page_num: Page number for logging

    Returns:
        Tuple of (extracted_text, ocr_regions_count, ocr_chars_count, skipped_count)
    """
    import pymupdf

    # Get all blocks with positions
    # Block format: (x0, y0, x1, y1, text_or_img, block_no, block_type)
    # block_type: 0=text, 1=image
    blocks = page.get_text("blocks")

    # Separate text and image blocks
    content_blocks = []  # (y0, text, is_ocr)

    ocr_regions = 0
    ocr_chars = 0
    skipped_regions = 0
    image_index = 0

    # Track which areas we've processed as image blocks
    processed_rects = []

    for block in blocks:
        x0, y0, x1, y1, content, block_no, block_type = block

        if block_type == 0:
            # Text block - use as-is
            text = content.strip()
            if text:
                content_blocks.append((y0, text, False))
        else:
            # Image block - OCR this region (learner may skip)
            processed_rects.append(pymupdf.Rect(x0, y0, x1, y1))
            try:
                img_text, was_ocrd = ocr_image_region(
                    page,
                    (x0, y0, x1, y1),
                    ocr_engine,
                    dpi,
                    learner=learner,
                    pdf_path=pdf_path,
                    page_num=page_num,
                    image_index=image_index,
                    text_blocks=blocks,
                )
                image_index += 1

                if not was_ocrd:
                    skipped_regions += 1
                    if debug:
                        print(f"    [DEBUG] Skipped image block at y={y0:.0f} (learning)")
                elif img_text and img_text.strip():
                    content_blocks.append((y0, img_text.strip(), True))
                    ocr_regions += 1
                    ocr_chars += len(img_text)
                    if debug:
                        print(f"    [DEBUG] OCR'd image block at y={y0:.0f}: {len(img_text)} chars")
            except Exception as e:
                if debug:
                    print(f"    [DEBUG] Failed to OCR image at y={y0:.0f}: {e}")

    # Also check for images via get_images() that weren't detected as blocks
    # This catches images that PyMuPDF doesn't return as block_type=1
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        try:
            # Get bounding box(es) for this image on the page
            img_rects = page.get_image_rects(xref)
            for rect in img_rects:
                # Skip if we already processed this area as an image block
                already_processed = any(
                    rect.intersects(pr) and rect.get_area() > 0 and
                    abs(rect.get_area() - pr.get_area()) / max(rect.get_area(), 1) < 0.5
                    for pr in processed_rects
                )
                if already_processed:
                    continue

                x0, y0, x1, y1 = rect
                processed_rects.append(rect)

                img_text, was_ocrd = ocr_image_region(
                    page,
                    (x0, y0, x1, y1),
                    ocr_engine,
                    dpi,
                    learner=learner,
                    pdf_path=pdf_path,
                    page_num=page_num,
                    image_index=image_index,
                    text_blocks=blocks,
                )
                image_index += 1

                if not was_ocrd:
                    skipped_regions += 1
                    if debug:
                        print(f"    [DEBUG] Skipped image (xref={xref}) at y={y0:.0f} (learning)")
                elif img_text and img_text.strip():
                    content_blocks.append((y0, img_text.strip(), True))
                    ocr_regions += 1
                    ocr_chars += len(img_text)
                    if debug:
                        print(f"    [DEBUG] OCR'd image (xref={xref}) at y={y0:.0f}: {len(img_text)} chars")
        except Exception as e:
            if debug:
                print(f"    [DEBUG] Failed to process image xref={xref}: {e}")

    # Sort by y-position (reading order: top to bottom)
    content_blocks.sort(key=lambda x: x[0])

    # Combine all text
    final_text = '\n\n'.join(block[1] for block in content_blocks)

    return final_text, ocr_regions, ocr_chars, skipped_regions


def extract_page_text(
    page,
    ocr_engine: str,
    force_ocr: bool = False,
    suppress_output: bool = False,
    learner: AdaptiveLearner | None = None,
    pdf_path: str = "",
    page_num: int = 0,
) -> tuple[str, bool, int, str]:
    """Extract text from a single page, using OCR as appropriate.

    Strategy:
    - force_ocr: Full page OCR (re-OCR everything, ignore extracted text)
    - has_images: Hybrid mode (extract text + OCR each image, merge by position)
    - otherwise: Just extract text (fast, no OCR needed)

    Args:
        page: PyMuPDF page object
        ocr_engine: OCR engine to use ("surya", "paddle", "tesseract", "none")
        force_ocr: Force full-page OCR regardless of content
        suppress_output: Suppress stdout/stderr during processing
        learner: Optional AdaptiveLearner for image skip decisions
        pdf_path: PDF file path for learning
        page_num: Page number for learning

    Returns:
        Tuple of (text, used_ocr, ocr_chars, log_message)
    """
    if ocr_engine == "none" and not force_ocr:
        with SuppressOutputFD(suppress=suppress_output):
            text = page.get_text().strip()
        return text, False, 0, ""

    has_images = len(page.get_images(full=False)) > 0

    try:
        if force_ocr:
            # Full page OCR: user wants everything re-OCR'd
            with SuppressOutputFD(suppress=suppress_output):
                if ocr_engine == "surya":
                    ocr_text = ocr_page_with_surya(page)
                elif ocr_engine == "paddle":
                    ocr_text = ocr_page_with_paddle(page)
                else:  # tesseract
                    tp = page.get_textpage_ocr(full=True, language="eng")
                    ocr_text = page.get_text(textpage=tp).strip()
            return ocr_text, True, len(ocr_text), f"OCR +{len(ocr_text):,} chars ({ocr_engine})"

        elif has_images:
            # Hybrid: always extract text + OCR every image, merge by position
            with SuppressOutputFD(suppress=suppress_output):
                hybrid_text, ocr_regions, ocr_chars, skipped = extract_page_hybrid(
                    page,
                    ocr_engine=ocr_engine,
                    learner=learner,
                    pdf_path=pdf_path,
                    page_num=page_num,
                )
            if ocr_regions > 0 or skipped > 0:
                msg_parts = []
                if ocr_regions > 0:
                    msg_parts.append(f"+{ocr_regions} imgs (+{ocr_chars:,} chars)")
                if skipped > 0:
                    msg_parts.append(f"skipped {skipped}")
                return hybrid_text, ocr_regions > 0, ocr_chars, ", ".join(msg_parts)
            else:
                return hybrid_text, False, 0, ""

        else:
            # No images, just extract text
            with SuppressOutputFD(suppress=suppress_output):
                text = page.get_text().strip()
            return text, False, 0, ""

    except Exception as e:
        # Fallback to basic extraction on error
        with SuppressOutputFD(suppress=suppress_output):
            text = page.get_text().strip()
        return text, False, 0, f"FAILED - {e}"


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


def extract_text_from_pdf(
    pdf_path: Path,
    use_ocr: bool = True,
    ocr_engine: str = "paddle",
    force_ocr: bool = False,
    stats: ProcessingStats | None = None,
    hud: RetroHUD | None = None,
    learner: AdaptiveLearner | None = None,
) -> list[str]:
    """Extract text from PDF, returning list of page contents."""
    import pymupdf

    suppress = hud is not None

    with SuppressOutputFD(suppress=suppress):
        doc = pymupdf.open(pdf_path)

    # Resolve OCR engine with fallbacks
    ocr_available, active_engine, _ = resolve_ocr_engine(ocr_engine, use_ocr)

    total_pages = len(doc)
    if stats:
        stats.current_file_pages = total_pages
        stats.total_pages += total_pages

    pages = []
    try:
        for page_num, page in enumerate(doc, start=1):
            if stats:
                stats.current_page = page_num
                stats.current_status = f"Reading page {page_num}/{total_pages}"
                if hud:
                    hud.refresh()

            # Extract page text (with OCR if available and needed)
            engine = active_engine if ocr_available else "none"
            text, used_ocr, ocr_chars, log_msg = extract_page_text(
                page,
                engine,
                force_ocr,
                suppress_output=suppress,
                learner=learner,
                pdf_path=str(pdf_path),
                page_num=page_num,
            )

            if stats:
                if used_ocr:
                    stats.ocr_pages += 1
                    stats.ocr_chars += ocr_chars
                if log_msg:
                    stats.log(f"  p{page_num}: {log_msg}")
                stats.processed_pages += 1

            pages.append(text)
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
    ocr_engine: str = "paddle",
    force_ocr: bool = False,
    improve: bool = False,
    stats: ProcessingStats | None = None,
    hud: RetroHUD | None = None,
    learner: AdaptiveLearner | None = None,
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
            pages = extract_text_from_pdf(
                pdf_path, use_ocr=use_ocr, ocr_engine=ocr_engine,
                force_ocr=force_ocr, stats=stats, hud=hud, learner=learner
            )
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
        pages = extract_text_from_pdf(
            pdf_path, use_ocr=use_ocr, ocr_engine=ocr_engine,
            force_ocr=force_ocr, stats=stats, hud=hud, learner=learner
        )
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
    ocr_engine: str,
    force_ocr: bool,
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

    log_msgs = []

    def extract_pages() -> tuple[list[str], int, int]:
        """Extract text from all pages, returning (pages, ocr_pages, ocr_chars)."""
        doc = pymupdf.open(pdf_path)

        # Resolve OCR engine with fallbacks
        ocr_available, active_engine, engine_logs = resolve_ocr_engine(ocr_engine, use_ocr)
        log_msgs.extend(engine_logs)

        pages = []
        ocr_pages_count = 0
        ocr_chars_count = 0
        total_pages = len(doc)

        try:
            for page_num, page in enumerate(doc, start=1):
                engine = active_engine if ocr_available else "none"
                text, used_ocr, ocr_chars, log_msg = extract_page_text(
                    page, engine, force_ocr, suppress_output=False
                )

                if used_ocr:
                    ocr_pages_count += 1
                    ocr_chars_count += ocr_chars
                if log_msg:
                    log_msgs.append(f"  p{page_num}/{total_pages}: {log_msg}")

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
            result.log_messages = log_msgs

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
            result.log_messages = log_msgs
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
        result.log_messages = log_msgs
        result.success = True
        result.message = f"Created: {md_path.name}"
        return result

    except Exception as e:
        result.was_failed = True
        result.message = f"FAILED: {pdf_path.name} - {e}"
        result.log_messages = log_msgs
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


def run_simple_parallel(pdfs: list[Path], args, use_ocr: bool, ocr_engine: str, force_ocr: bool, max_workers: int) -> int:
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
        print(f"OCR engine: {ocr_engine}")
        print()

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=_mp_context) as executor:
        futures = {
            executor.submit(
                process_pdf_worker,
                pdf_path,
                args.overwrite,
                args.dry_run,
                use_ocr,
                ocr_engine,
                force_ocr,
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
            if getattr(args, 'debug', False) and result.log_messages:
                for msg in result.log_messages:
                    print(f"    [DEBUG] {msg}")

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


def run_with_hud_parallel(pdfs: list[Path], args, use_ocr: bool, ocr_engine: str, force_ocr: bool, max_workers: int) -> int:
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
            ocr_engine,
            force_ocr,
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


def run_with_hud(
    pdfs: list[Path],
    args,
    use_ocr: bool,
    ocr_engine: str,
    force_ocr: bool,
    learner: AdaptiveLearner | None = None,
) -> int:
    """Run processing with the retro HUD."""
    stats = ProcessingStats()
    stats.total_files = len(pdfs)
    stats.total_bytes = sum(p.stat().st_size for p in pdfs)
    improve_mode = getattr(args, 'improve', False)

    with RetroHUD(stats, learner=learner) as hud:
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
                ocr_engine=ocr_engine,
                force_ocr=force_ocr,
                improve=improve_mode,
                stats=stats,
                hud=hud,
                learner=learner,
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
        # Show learning stats if enabled
        if learner and learner.enabled:
            learn_stats = learner._stats
            if learn_stats["images_seen"] > 0:
                stats.log(f" Learning: {learn_stats['images_ocrd']} OCR'd, {learn_stats['images_skipped']} skipped")
        stats.log("═" * 40)
        hud.refresh()

        # Wait for keypress
        hud.stdscr.nodelay(False)
        hud.stdscr.addstr(hud.stdscr.getmaxyx()[0] - 1, 2, "Press any key to exit...",
                         curses.color_pair(3) | curses.A_BLINK)
        hud.stdscr.refresh()
        hud.stdscr.getch()

    return 1 if stats.failed_files > 0 else 0


def run_simple(
    pdfs: list[Path],
    args,
    use_ocr: bool,
    ocr_engine: str,
    force_ocr: bool,
    learner: AdaptiveLearner | None = None,
) -> int:
    """Run processing with simple text output."""
    stats = ProcessingStats()
    stats.total_files = len(pdfs)
    stats.total_bytes = sum(p.stat().st_size for p in pdfs)
    improve_mode = getattr(args, 'improve', False)

    if args.verbose or args.dry_run:
        if args.dry_run:
            print("(Dry run - no files will be created)")
        if improve_mode:
            print("(Improve mode - comparing quality)")
        if learner and learner.enabled:
            print("(Adaptive learning enabled)")
        print(f"OCR engine: {ocr_engine}")
        print()

    for pdf_path in sorted(pdfs):
        if args.verbose:
            print(f"Processing: {pdf_path}")

        file_size = pdf_path.stat().st_size

        success, message, improve_detail = process_pdf(
            pdf_path,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            use_ocr=use_ocr,
            ocr_engine=ocr_engine,
            force_ocr=force_ocr,
            improve=improve_mode,
            stats=stats,
            learner=learner,
        )

        if args.verbose:
            print(f"  {message}")
            if improve_detail:
                print(f"     {improve_detail}")
        if getattr(args, 'debug', False) and stats.log_messages:
            for msg in stats.log_messages:
                print(f"    [DEBUG] {msg}")
            stats.log_messages.clear()

        if improve_mode and "Improved:" in message:
            stats.improved_files += 1
            stats.processed_files += 1
            stats.processed_bytes += file_size
        elif improve_mode and "Kept" in message:
            stats.kept_existing += 1
            stats.processed_files += 1
            stats.processed_bytes += file_size
        elif success:
            stats.processed_files += 1
            stats.processed_bytes += file_size
        elif "Skipped" in message:
            stats.skipped_files += 1
        else:
            stats.failed_files += 1

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
            print(f"  PDF input:       {total_mb:.2f} MB ({stats.processed_pages} pages)")
            print(f"  MD output:       {md_mb:.2f} MB ({ratio:.1f}% of input)")
            print(f"  Processing rate: {files_per_min:.1f} files/min, {mb_per_min:.2f} MB/min")
            print(f"  OCR stats:       {stats.ocr_pages} pages, {stats.ocr_chars:,} chars extracted")

        # Print learning stats if enabled
        if learner and learner.enabled:
            ls = learner._stats
            if ls["images_seen"] > 0:
                print()
                print(f"Learning:")
                print(f"  Images seen:     {ls['images_seen']:,}")
                print(f"  Images OCR'd:    {ls['images_ocrd']:,} ({ls['ocr_useful']} useful, {ls['ocr_empty']} empty)")
                print(f"  Images skipped:  {ls['images_skipped']:,}")

                # OCR efficiency
                total_ocrd = ls['ocr_useful'] + ls['ocr_empty']
                if total_ocrd > 0:
                    ocr_eff = ls['ocr_useful'] / total_ocrd * 100
                    print(f"  OCR efficiency:  {ocr_eff:.1f}% found useful text")

                # Exploration accuracy
                exp_total = ls['exploration_useful'] + ls['exploration_empty']
                if exp_total > 0:
                    miss_rate = ls['exploration_useful'] / exp_total * 100
                    print(f"  Exploration:     {exp_total} samples, {miss_rate:.1f}% would be missed if skipped")

    return 1 if stats.failed_files > 0 else 0


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
        help="Disable OCR for image-based pages"
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Force OCR on all pages, even those with extractable text"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode for OCR (slower but avoids GPU memory issues)"
    )
    parser.add_argument(
        "--ocr-engine",
        choices=["paddle", "surya", "tesseract"],
        default="surya",
        help="OCR engine to use (default: surya). Note: PaddleOCR v3.3 has a known bug, use surya or tesseract instead."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show detailed OCR debug information"
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=None,
        metavar="N",
        help="Number of parallel workers (default: CPU count - 1, use 1 for sequential)"
    )

    # Adaptive learning options
    parser.add_argument(
        "--learn",
        action="store_true",
        help="Enable adaptive OCR learning (learns which images are worth OCR'ing)"
    )
    parser.add_argument(
        "--learn-db",
        type=str,
        metavar="PATH",
        help=f"Custom path for learning database (default: ~/.pdf2txt/learning.db)"
    )
    parser.add_argument(
        "--learn-stats",
        action="store_true",
        help="Print learning statistics and exit"
    )
    parser.add_argument(
        "--learn-reset",
        action="store_true",
        help="Reset the learning database and exit"
    )
    parser.add_argument(
        "--learn-recluster",
        action="store_true",
        help="Re-run clustering on collected data and exit"
    )

    args = parser.parse_args()

    # Force CPU mode if requested (must be set before any CUDA imports)
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # Handle learning database commands early
    learn_db_path = Path(args.learn_db) if args.learn_db else None

    # --learn-stats without --learn: show stats and exit
    # --learn-stats with --learn: process files, stats shown at end
    if args.learn_stats and not args.learn:
        learner = AdaptiveLearner(db_path=learn_db_path, enabled=True)
        stats = learner.get_stats()
        if not stats["enabled"]:
            print("Learning database not found or empty.")
            return 0

        print("═" * 60)
        print("  PDF2TXT - ADAPTIVE LEARNING STATISTICS")
        print("═" * 60)
        print(f"  Database: {stats['db_path']}")
        print(f"  Processed files: {stats['processed_files']:,}")
        print(f"  Total pages: {stats['total_pages_processed']:,}")
        print(f"  Total images: {stats['total_images_seen']:,}")
        print()
        print(f"  OCR outcomes: {stats['total_records']:,} records")
        print(f"  OCR'd images: {stats['ocrd_records']:,}")
        print(f"  Useful results: {stats['useful_records']:,} ({stats['overall_useful_rate']:.1%})")
        print()
        if stats['clusters']:
            print(f"  Clusters: {stats['num_clusters']}")
            for c in stats['clusters']:
                conf_bar = "█" * int(c['confidence'] * 10)
                print(f"    #{c['id']:2d}: {c['samples']:5d} samples, "
                      f"{c['useful_rate']:.1%} useful, conf [{conf_bar:<10}]")
        else:
            print("  No clusters formed yet (need more data)")
        print("═" * 60)
        learner.close()
        return 0

    if args.learn_reset:
        learner = AdaptiveLearner(db_path=learn_db_path, enabled=True)
        db_path = learner.db_path
        learner.reset()
        print(f"Learning database reset: {db_path}")
        learner.close()
        return 0

    if args.learn_recluster:
        learner = AdaptiveLearner(db_path=learn_db_path, enabled=True)
        print("Re-clustering OCR outcomes...")
        learner.recluster(force=True)
        stats = learner.get_stats()
        print(f"Formed {stats['num_clusters']} clusters from {stats['ocrd_records']} records")
        learner.close()
        return 0

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
    ocr_engine = args.ocr_engine
    force_ocr = getattr(args, 'force_ocr', False)

    if args.debug:
        print_gpu_debug_info()
        print("[DEBUG] OCR engine availability:")
        print(f"  Surya: {check_surya_available()}")
        print(f"  PaddleOCR: {check_paddleocr_available()}")
        print(f"  Tesseract: {check_tesseract_available()}")
        print(f"  Requested: {args.ocr_engine}")
        # Pre-configure and show Surya batch sizes if using surya
        if args.ocr_engine == "surya" and check_surya_available():
            configure_surya_batch_sizes()
            rec_batch = os.environ.get('RECOGNITION_BATCH_SIZE', '512 (default)')
            det_batch = os.environ.get('DETECTOR_BATCH_SIZE', 'default')
            print(f"  Surya batch sizes: recognition={rec_batch}, detection={det_batch}")
        print()

    if use_ocr:
        # Check if requested engine is available, with fallback
        if ocr_engine == "surya" and not check_surya_available():
            if not args.quiet:
                print("Warning: Surya not available, trying PaddleOCR...", file=sys.stderr)
            ocr_engine = "paddle"

        if ocr_engine == "paddle" and not check_paddleocr_available():
            if not args.quiet:
                print("Warning: PaddleOCR not available, trying Tesseract...", file=sys.stderr)
            ocr_engine = "tesseract"

        if ocr_engine == "tesseract" and not check_tesseract_available():
            if not args.quiet:
                print("Warning: No OCR engine available. OCR disabled.", file=sys.stderr)
                print("Install with: pip install paddleocr paddlepaddle", file=sys.stderr)
            use_ocr = False

    # Find PDFs
    pdfs = find_pdfs(directory, recursive=args.recursive, quiet=args.quiet)

    if not pdfs:
        return 0

    # Determine worker count
    # OCR is memory-intensive (each worker loads 1-2GB+ ML models), so limit parallelism
    # Non-OCR extraction is lightweight and can use more workers
    cpu_count = os.cpu_count() or 4
    if use_ocr:
        # OCR is memory-intensive; Surya loads large GPU models per worker
        if ocr_engine == "surya":
            # Surya requires ~4GB VRAM per worker; use single worker to avoid OOM
            default_workers = 1
        else:
            # PaddleOCR/Tesseract: max 2 workers to avoid memory exhaustion
            default_workers = min(2, cpu_count - 1)
    else:
        # Text-only extraction can safely use more parallelism
        default_workers = max(cpu_count - 1, 1)
    max_workers = args.jobs if args.jobs is not None else default_workers
    max_workers = max(1, max_workers)  # Ensure at least 1

    # Create learner if learning is enabled
    # Note: Learning only works in sequential mode (max_workers=1) because
    # the SQLite database doesn't work well across process boundaries
    learner = None
    if args.learn:
        if max_workers > 1:
            if not args.quiet:
                print("Note: Adaptive learning requires sequential mode (-j 1). Disabling parallelism.", file=sys.stderr)
            max_workers = 1
        learner = AdaptiveLearner(db_path=learn_db_path, enabled=True)
        if not args.quiet:
            stats = learner.get_stats()
            print(f"Adaptive learning enabled (db: {learner.db_path})")
            if stats['processed_files'] > 0:
                print(f"  History: {stats['processed_files']} files, {stats['total_images_seen']} images, {stats['overall_useful_rate']:.1%} useful")
            print()

    try:
        # Run with HUD or simple mode, sequential or parallel
        if max_workers == 1:
            # Sequential processing (original behavior)
            if args.hud and not args.dry_run:
                result = run_with_hud(pdfs, args, use_ocr, ocr_engine, force_ocr, learner=learner)
            else:
                result = run_simple(pdfs, args, use_ocr, ocr_engine, force_ocr, learner=learner)
        else:
            # Parallel processing (no learner support)
            if args.hud and not args.dry_run:
                result = run_with_hud_parallel(pdfs, args, use_ocr, ocr_engine, force_ocr, max_workers)
            else:
                result = run_simple_parallel(pdfs, args, use_ocr, ocr_engine, force_ocr, max_workers)

        # Show detailed stats if --learn-stats was requested with --learn
        if args.learn_stats and learner:
            stats = learner.get_stats()
            print()
            print("═" * 60)
            print("  PDF2TXT - ADAPTIVE LEARNING STATISTICS")
            print("═" * 60)
            print(f"  Database: {stats['db_path']}")
            print(f"  Processed files: {stats['processed_files']:,}")
            print(f"  Total pages: {stats['total_pages_processed']:,}")
            print(f"  Total images: {stats['total_images_seen']:,}")
            print()
            print(f"  OCR outcomes: {stats['total_records']:,} records")
            print(f"  OCR'd images: {stats['ocrd_records']:,}")
            print(f"  Useful results: {stats['useful_records']:,} ({stats['overall_useful_rate']:.1%})")
            print()
            if stats['clusters']:
                print(f"  Clusters: {stats['num_clusters']}")
                for c in stats['clusters']:
                    conf_bar = "█" * int(c['confidence'] * 10)
                    print(f"    #{c['id']:2d}: {c['samples']:5d} samples, "
                          f"{c['useful_rate']:.1%} useful, conf [{conf_bar:<10}]")
            else:
                print("  No clusters formed yet (need more data)")
            print("═" * 60)

        return result
    finally:
        if learner:
            learner.close()


if __name__ == "__main__":
    sys.exit(main())
