# -*- coding: utf-8 -*-
"""
Gradio Web Interface for Spatial Understanding Object Tracking
Based on Google Gemini's Spatial Understanding capabilities.
"""

import os
import json
import re
import shutil
import subprocess
import tempfile
from bisect import bisect_left, bisect_right
from collections import Counter
import cv2
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw, ImageFont, ImageColor
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

CONFIG_PATH = Path(__file__).parent / "config.json"
DEFAULT_CONFIG = {
    "robot_tracking_mode": "auto",
    "local_llm_url": "http://127.0.0.1:1234/v1/chat/completions",
}


def _load_app_config() -> dict:
    """Load non-sensitive app settings from config.json."""
    if not CONFIG_PATH.exists():
        return dict(DEFAULT_CONFIG)
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as fh:
            loaded = json.load(fh)
    except Exception as e:
        print(f"Failed to read {CONFIG_PATH.name}: {e}. Using defaults.")
        return dict(DEFAULT_CONFIG)
    if not isinstance(loaded, dict):
        print(f"{CONFIG_PATH.name} must contain a JSON object. Using defaults.")
        return dict(DEFAULT_CONFIG)
    config = dict(DEFAULT_CONFIG)
    config.update(loaded)
    return config


APP_CONFIG = _load_app_config()

ROBOT_TRACKING_MODE = str(APP_CONFIG.get("robot_tracking_mode", "auto")).strip().lower()
VALID_ROBOT_TRACKING_MODES = {"auto", "manual", "manual-limited"}
if ROBOT_TRACKING_MODE not in VALID_ROBOT_TRACKING_MODES:
    print(f"Unknown ROBOT_TRACKING_MODE={ROBOT_TRACKING_MODE!r}; defaulting to 'auto'")
    ROBOT_TRACKING_MODE = "auto"
MANUAL_ROBOT_TRACKING = ROBOT_TRACKING_MODE in {"manual", "manual-limited"}
MANUAL_LIMITED_ROBOT_TRACKING = ROBOT_TRACKING_MODE == "manual-limited"

# YOLO person segmentation model (always loaded - used to exclude humans from bumper color detection)
YOLO_PERSON_MODEL = None
YOLO_PERSON_MODEL_PATH = Path(__file__).parent / "yolo26s-seg.pt"
try:
    from ultralytics import YOLO as _YOLO_CLS
    if YOLO_PERSON_MODEL_PATH.exists():
        YOLO_PERSON_MODEL = _YOLO_CLS(str(YOLO_PERSON_MODEL_PATH))
        print(f"YOLO person model loaded from {YOLO_PERSON_MODEL_PATH}")
    else:
        print(f"YOLO person model not found at {YOLO_PERSON_MODEL_PATH} - person detection disabled")
except ImportError:
    print("ultralytics not installed - person detection disabled")
except Exception as e:
    print(f"Error loading YOLO person model: {e}")

# SAM 3 predictor for ball detection (text-prompted semantic segmentation)
SAM3_PREDICTOR = None
SAM3_MODEL_PATH = Path(__file__).parent / "sam3.1_multiplex.pt"
try:
    from ultralytics.models.sam import SAM3SemanticPredictor
    sam3_overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        model=str(SAM3_MODEL_PATH),
        half=True,  # Use FP16 for faster inference
        save=False,  # We handle drawing ourselves
    )
    SAM3_PREDICTOR = SAM3SemanticPredictor(overrides=sam3_overrides)
    print(f"SAM 3 predictor initialized successfully (model: {SAM3_MODEL_PATH})")
except ImportError:
    print("SAM 3 not available (ultralytics version may not support it) - using HSV ball detection")
except Exception as e:
    print(f"SAM 3 initialization failed: {e} - using HSV ball detection")

# LMStudio configuration for local LLM team number detection
LMSTUDIO_URL = str(APP_CONFIG.get("local_llm_url", DEFAULT_CONFIG["local_llm_url"])).strip()
LMSTUDIO_ENABLED = True  # Set to False to disable LMStudio queries

import requests
import base64
try:
    import pytesseract
    from pytesseract import TesseractNotFoundError
except Exception:
    pytesseract = None
    TesseractNotFoundError = RuntimeError

if pytesseract is not None:
    try:
        detected_tesseract = shutil.which("tesseract")
        if not detected_tesseract:
            candidate_paths = [
                Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
                Path("C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"),
            ]
            detected_tesseract = next((str(path) for path in candidate_paths if path.exists()), "")
        if detected_tesseract:
            pytesseract.pytesseract.tesseract_cmd = str(detected_tesseract)
            print(f"Tesseract executable configured: {detected_tesseract}")
        else:
            print("Tesseract executable not found on PATH or in standard install locations")
    except Exception as e:
        print(f"Failed to configure Tesseract executable: {e}")

YOUTUBE_URL_PATTERN = re.compile(r"(?i)^(https?://)?(www\.)?(youtube\.com|youtu\.be)/")
VIDEO_SOURCE_EMPTY_STATUS = "*Upload a file or paste a YouTube link to begin.*"
FIELD_CALIBRATION_CACHE_PATH = Path(__file__).parent / "field_calibration_cache.json"
VIDEO_FILE_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
DEFAULT_PAGE_TITLE = "Robot Scouter"
MANUAL_PREVIEW_TARGET_WIDTH = 1280
MANUAL_PREVIEW_TARGET_HEIGHT = 720
YOUTUBE_DOWNLOAD_DIR_PREFIX = "youtube_match_"
COMPOSITE_REFERENCE_SIZE = (1920, 1080)
COMPOSITE_REFERENCE_CROP_RECTS = {
    "center": (1, 0, 1919, 709),
    "blue": (1, 739, 941, 1078),
    "red": (979, 739, 1919, 1078),
}
PAGE_TITLE_SYNC_HTML = f"""
<script>
(() => {{
  const defaultTitle = {json.dumps(DEFAULT_PAGE_TITLE)};
  let lastTitle = "";
  function syncPageTitle() {{
    const root = document.querySelector("#page-title-state");
    const input = root ? root.querySelector("input, textarea") : null;
    const nextTitle = (input && input.value ? input.value.trim() : "") || defaultTitle;
    if (nextTitle !== lastTitle) {{
      document.title = nextTitle;
      lastTitle = nextTitle;
    }}
  }}
  if (document.readyState === "loading") {{
    document.addEventListener("DOMContentLoaded", syncPageTitle);
  }} else {{
    syncPageTitle();
  }}
  setInterval(syncPageTitle, 300);
}})();
</script>
"""

TRACKING_FPS_OPTIONS = (10, 20, 30)
DEFAULT_TRACKING_FPS = 30
BALL_TRACKER_BASELINE_FPS = 30.0


def _clean_text(value: str) -> str:
    """Collapse whitespace and trim user-facing text."""
    return re.sub(r"\s+", " ", str(value or "").strip())


def _normalize_tracking_fps(value, default: int = DEFAULT_TRACKING_FPS) -> int:
    """Clamp external FPS inputs to the UI-supported tracking rates."""
    try:
        requested = int(round(float(value)))
    except (TypeError, ValueError):
        requested = int(default)
    return min(TRACKING_FPS_OPTIONS, key=lambda option: abs(option - requested))


def _compute_sampling_stride(source_fps: float, sample_fps: float) -> float:
    """Return the source-frame stride needed to approximate a target sampling rate."""
    source_fps = max(1.0, float(source_fps or DEFAULT_TRACKING_FPS))
    sample_fps = max(1.0, float(sample_fps or source_fps))
    return max(1.0, source_fps / sample_fps)


def _consume_frame_schedule(frame_index: int, next_frame_index: float, frame_stride: float) -> tuple:
    """
    Decide whether the current source frame should be sampled for a target rate.

    This keeps 20 FPS accurate on 30 FPS inputs instead of rounding to every
    frame or every other frame.
    """
    current_frame = float(frame_index)
    if current_frame + 1e-6 < next_frame_index:
        return False, next_frame_index

    while current_frame + 1e-6 >= next_frame_index:
        next_frame_index += frame_stride
    return True, next_frame_index


def _normalize_regional_name(regional_name: str) -> str:
    """Build a stable cache key for regional / event names."""
    cleaned = _clean_text(regional_name).lower()
    return re.sub(r"[^a-z0-9]+", " ", cleaned).strip()


def _ensure_ffmpeg_executable() -> str:
    """Return an FFmpeg executable path when available."""
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is None:
        try:
            import static_ffmpeg
            static_ffmpeg.add_paths()
            ffmpeg_exe = shutil.which("ffmpeg")
        except ImportError:
            ffmpeg_exe = None
    return ffmpeg_exe


def _build_composite_crop_layout(source_width: int, source_height: int) -> dict:
    """Scale the known composite-camera crop layout to the source video size."""
    source_width = int(source_width or 0)
    source_height = int(source_height or 0)
    if source_width <= 0 or source_height <= 0:
        raise gr.Error("Could not determine composite video dimensions.")

    ref_width, ref_height = COMPOSITE_REFERENCE_SIZE
    scale_x = float(source_width) / float(ref_width)
    scale_y = float(source_height) / float(ref_height)
    max_x1 = max(0, source_width - 1)
    max_y1 = max(0, source_height - 1)

    layout = {}
    for name, (x1, y1, x2, y2) in COMPOSITE_REFERENCE_CROP_RECTS.items():
        scaled_x1 = max(0, min(max_x1, int(round(x1 * scale_x))))
        scaled_y1 = max(0, min(max_y1, int(round(y1 * scale_y))))
        scaled_x2 = max(scaled_x1 + 1, min(source_width, int(round(x2 * scale_x))))
        scaled_y2 = max(scaled_y1 + 1, min(source_height, int(round(y2 * scale_y))))
        crop_width = max(1, scaled_x2 - scaled_x1)
        crop_height = max(1, scaled_y2 - scaled_y1)
        layout[name] = {
            "rect": (scaled_x1, scaled_y1, scaled_x2, scaled_y2),
            "size": (crop_width, crop_height),
            "filter": f"crop={crop_width}:{crop_height}:{scaled_x1}:{scaled_y1}",
        }
    return layout


INLINE_ALLIANCE_TEAMS_PATTERN = re.compile(
    r"(?i)\b(red|blue)\s*\(teams?\s*([^)]+)\)"
)
FLEXIBLE_ALLIANCE_TEAMS_PATTERN = re.compile(
    r"(?is)\b(red|blue)\b(.*?)(?=\b(?:red|blue)\b|https?://|\buploaded by\b|\Z)"
)
MATCH_TEXT_HINT_PATTERN = re.compile(
    r"(?i)\b("
    r"qual(?:ification)?|practice|playoff|quarterfinal|semifinal|final|"
    r"tiebreaker|match"
    r")\b"
)


def _blank_match_metadata() -> dict:
    """Create the shared match-metadata shape used by uploads and YouTube."""
    return {
        "match_label": "",
        "match_title": "",
        "regional_name": "",
        "blue_robots": ["", "", ""],
        "red_robots": ["", "", ""],
    }


def _merge_match_metadata(existing: dict, incoming: dict) -> dict:
    """Merge parsed metadata without overwriting already-found values."""
    merged = dict(existing or _blank_match_metadata())
    incoming = dict(incoming or {})

    for key in ("match_label", "match_title", "regional_name"):
        if not _clean_text(merged.get(key, "")):
            merged[key] = _clean_text(incoming.get(key, ""))

    for alliance_key in ("blue_robots", "red_robots"):
        current_values = list(merged.get(alliance_key) or [])
        while len(current_values) < 3:
            current_values.append("")
        incoming_values = list(incoming.get(alliance_key) or [])
        while len(incoming_values) < 3:
            incoming_values.append("")

        current_filled = sum(1 for value in current_values[:3] if _clean_text(value))
        incoming_filled = sum(1 for value in incoming_values[:3] if _clean_text(value))

        if incoming_filled > current_filled:
            merged[alliance_key] = [_clean_text(value) for value in incoming_values[:3]]
            continue

        for idx, value in enumerate(incoming_values[:3]):
            cleaned = _clean_text(value)
            if cleaned and not _clean_text(current_values[idx]):
                current_values[idx] = cleaned
        merged[alliance_key] = current_values[:3]

    return merged


def _looks_like_match_text(text: str) -> bool:
    """Heuristically decide whether a metadata string describes an FRC match."""
    cleaned = _clean_text(text)
    if not cleaned:
        return False
    lowered = cleaned.lower()
    if lowered.startswith("http://") or lowered.startswith("https://"):
        return False
    return (
        " - " in cleaned
        or bool(MATCH_TEXT_HINT_PATTERN.search(cleaned))
        or bool(INLINE_ALLIANCE_TEAMS_PATTERN.search(cleaned))
    )


def _parse_match_title_parts(title: str) -> tuple:
    """Split a match title into the match label, full title, and regional / event name."""
    full_title = _clean_text(title)
    if not full_title:
        return "", "", ""

    alliance_match = INLINE_ALLIANCE_TEAMS_PATTERN.search(full_title)
    if alliance_match and alliance_match.start() == 0:
        return "", "", ""

    trimmed_title = full_title[:alliance_match.start()] if alliance_match else full_title
    trimmed_title = re.sub(r"\s*[-|:]+\s*$", "", trimmed_title).strip()
    base_title = _clean_text(trimmed_title or full_title)

    if " - " in base_title:
        match_label, regional_name = base_title.split(" - ", 1)
        return _clean_text(match_label), base_title, _clean_text(regional_name)
    return base_title, base_title, ""


def _parse_alliance_teams_from_text(text: str) -> dict:
    """Extract blue / red alliance team numbers from any metadata text blob."""
    teams_by_alliance = {"blue": ["", "", ""], "red": ["", "", ""]}
    cleaned_text = str(text or "")

    for match in INLINE_ALLIANCE_TEAMS_PATTERN.finditer(cleaned_text):
        alliance = str(match.group(1)).strip().lower()
        team_blob = str(match.group(2)).strip()
        team_numbers = re.findall(r"\d+", team_blob)
        if team_numbers:
            padded = (team_numbers[:3] + ["", "", ""])[:3]
            teams_by_alliance[alliance] = padded

    # Fall back to a more forgiving parser for MP4 metadata variants where
    # the alliance text is flattened or scores are mixed into the same tag.
    for match in FLEXIBLE_ALLIANCE_TEAMS_PATTERN.finditer(cleaned_text):
        alliance = str(match.group(1)).strip().lower()
        if any(_clean_text(value) for value in teams_by_alliance.get(alliance, [])):
            continue

        section = _clean_text(match.group(2))
        if not section or "team" not in section.lower():
            continue

        teams_match = re.search(
            r"(?is)\bteams?\b\s*[:=-]?\s*\(?\s*([0-9,\s/]+)",
            section,
        )
        if teams_match:
            team_numbers = re.findall(r"\d+", teams_match.group(1))
        else:
            team_numbers = []

        if len(team_numbers) >= 3:
            teams_by_alliance[alliance] = (team_numbers[:3] + ["", "", ""])[:3]
    return teams_by_alliance


def _parse_match_metadata_from_texts(*texts: str) -> dict:
    """Parse the useful scouting metadata from one or more title/description strings."""
    metadata = _blank_match_metadata()
    for text in texts:
        cleaned = _clean_text(text)
        if not cleaned:
            continue

        parsed = _blank_match_metadata()
        if _looks_like_match_text(cleaned):
            match_label, match_title, regional_name = _parse_match_title_parts(cleaned)
            parsed.update({
                "match_label": match_label,
                "match_title": match_title,
                "regional_name": regional_name,
            })
        parsed.update(_parse_alliance_teams_from_text(cleaned))
        metadata = _merge_match_metadata(metadata, parsed)
    return metadata


def _parse_youtube_match_metadata(title: str, description: str) -> dict:
    """Parse the useful scouting metadata from a YouTube title / description."""
    return _parse_match_metadata_from_texts(title, description)


def _ensure_ffprobe_executable() -> str:
    """Return an FFprobe executable path when available."""
    ffprobe_exe = shutil.which("ffprobe")
    if ffprobe_exe:
        return ffprobe_exe

    ffmpeg_exe = _ensure_ffmpeg_executable()
    if ffmpeg_exe:
        ffmpeg_path = Path(ffmpeg_exe)
        candidate_names = ["ffprobe.exe", "ffprobe"] if ffmpeg_path.suffix.lower() == ".exe" else ["ffprobe"]
        for candidate_name in candidate_names:
            candidate = ffmpeg_path.with_name(candidate_name)
            if candidate.exists():
                return str(candidate)

    try:
        import static_ffmpeg
        static_ffmpeg.add_paths()
        return shutil.which("ffprobe") or ""
    except ImportError:
        return ""


def _normalize_metadata_tag_name(tag_name: str) -> str:
    """Normalize an ffprobe tag name for fuzzy matching."""
    return re.sub(r"[^a-z0-9]+", "", str(tag_name or "").strip().lower())


def _categorize_metadata_tag(tag_name: str) -> str:
    """Classify ffprobe tags into title, description, or other buckets."""
    normalized = _normalize_metadata_tag_name(tag_name)
    if not normalized:
        return "other"
    if normalized == "title" or normalized.endswith("title"):
        return "title"
    if any(token in normalized for token in ("description", "comment", "synopsis", "caption", "summary")):
        return "description"
    return "other"


def _extract_uploaded_video_match_metadata(video_path: str) -> dict:
    """Read embedded video metadata and parse any regional/team information it contains."""
    metadata = _blank_match_metadata()
    if not video_path:
        return metadata

    title_candidate_texts = []
    description_candidate_texts = []
    other_candidate_texts = []
    ffprobe_exe = _ensure_ffprobe_executable()
    if ffprobe_exe:
        try:
            result = subprocess.run(
                [
                    ffprobe_exe,
                    "-v", "quiet",
                    "-print_format", "json",
                    "-show_format",
                    "-show_streams",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if result.returncode == 0 and result.stdout.strip():
                probe_data = json.loads(result.stdout)
                format_tags = probe_data.get("format", {}).get("tags")
                if not isinstance(format_tags, dict):
                    format_tags = {}

                # Prefer the long-form description fields first because MatchLIVE MP4s
                # place alliance team numbers there, while title only contains the match/event.
                preferred_probe_texts = []
                for key in ("description", "synopsis", "comment", "title"):
                    for tag_name, tag_value in format_tags.items():
                        if _normalize_metadata_tag_name(tag_name) == key and _clean_text(tag_value):
                            preferred_probe_texts.append(tag_value)

                if preferred_probe_texts:
                    metadata = _merge_match_metadata(
                        metadata,
                        _parse_match_metadata_from_texts(*preferred_probe_texts),
                    )

                tag_dicts = []
                if format_tags:
                    tag_dicts.append(format_tags)
                for stream in probe_data.get("streams") or []:
                    stream_tags = stream.get("tags")
                    if isinstance(stream_tags, dict):
                        tag_dicts.append(stream_tags)

                for tag_dict in tag_dicts:
                    for tag_name, value in tag_dict.items():
                        cleaned_value = _clean_text(value)
                        if not cleaned_value:
                            continue
                        bucket = _categorize_metadata_tag(tag_name)
                        if bucket == "title":
                            title_candidate_texts.append(cleaned_value)
                        elif bucket == "description":
                            description_candidate_texts.append(cleaned_value)
                        else:
                            other_candidate_texts.append(cleaned_value)
        except Exception as exc:
            print(f"[Video Metadata] Failed to read metadata from {video_path}: {exc}")

    try:
        other_candidate_texts.append(Path(video_path).stem)
    except Exception:
        pass

    deduped_texts = []
    seen_texts = set()
    candidate_texts = title_candidate_texts + description_candidate_texts + other_candidate_texts
    for text in candidate_texts:
        cleaned = _clean_text(text)
        if cleaned and cleaned not in seen_texts:
            seen_texts.add(cleaned)
            deduped_texts.append(cleaned)

    if deduped_texts:
        metadata = _merge_match_metadata(
            metadata,
            _parse_match_metadata_from_texts(*deduped_texts),
        )
    return metadata


def _resolve_downloaded_video_path(info: dict, download_dir: Path, ydl=None) -> str:
    """Locate the final downloaded video file from a yt-dlp run."""
    candidates = []
    if isinstance(info, dict):
        for item in info.get("requested_downloads") or []:
            if isinstance(item, dict):
                candidates.append(item.get("filepath"))
        candidates.append(info.get("_filename"))
        if ydl is not None:
            try:
                candidates.append(ydl.prepare_filename(info))
            except Exception:
                pass

    for candidate in candidates:
        if candidate:
            path = Path(candidate)
            if path.exists() and path.suffix.lower() in VIDEO_FILE_EXTENSIONS:
                return str(path)

    for path in sorted(download_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if path.is_file() and path.suffix.lower() in VIDEO_FILE_EXTENSIONS:
            return str(path)

    return ""


def _get_managed_youtube_download_dir(video_path: str = None) -> Path:
    """Return the managed YouTube download directory for a path, if applicable."""
    if not video_path:
        return None
    try:
        path = Path(video_path).resolve()
    except Exception:
        return None
    parent = path.parent
    if parent.name.startswith(YOUTUBE_DOWNLOAD_DIR_PREFIX):
        return parent
    return None


def _cleanup_managed_youtube_dir(path: Path) -> bool:
    """Delete a managed YouTube download directory if it still exists."""
    if path is None:
        return False
    try:
        target = Path(path)
    except Exception:
        return False
    if not target.exists() or not target.is_dir() or not target.name.startswith(YOUTUBE_DOWNLOAD_DIR_PREFIX):
        return False
    try:
        shutil.rmtree(target, ignore_errors=False)
        print(f"[YouTube Cleanup] Removed {target}")
        return True
    except Exception as exc:
        print(f"[YouTube Cleanup] Failed to remove {target}: {exc}")
        return False


def _cleanup_old_youtube_downloads() -> int:
    """Remove leftover managed YouTube downloads from prior unfinished runs."""
    temp_root = Path(tempfile.gettempdir())
    removed = 0
    for path in temp_root.glob(f"{YOUTUBE_DOWNLOAD_DIR_PREFIX}*"):
        if _cleanup_managed_youtube_dir(path):
            removed += 1
    if removed:
        print(f"[YouTube Cleanup] Removed {removed} stale download folder(s) on startup")
    return removed


def _build_youtube_progress_hook(progress, title_hint: str = ""):
    """Create a yt-dlp progress hook that forwards progress into Gradio."""
    title_hint = _clean_text(title_hint)

    def _hook(update: dict):
        if progress is None or not isinstance(update, dict):
            return
        status = str(update.get("status", "")).strip().lower()
        label = title_hint or "YouTube match"

        if status == "downloading":
            downloaded = float(update.get("downloaded_bytes") or 0.0)
            total = float(
                update.get("total_bytes")
                or update.get("total_bytes_estimate")
                or 0.0
            )
            if total > 0:
                fraction = max(0.0, min(downloaded / total, 1.0))
                progress(0.1 + (0.8 * fraction), desc=f"Downloading {label}... {fraction * 100:.1f}%")
            else:
                progress(0.5, desc=f"Downloading {label}...")
        elif status == "finished":
            progress(0.93, desc=f"Finalizing {label}...")
        elif status == "error":
            progress(0.1, desc=f"Download failed for {label}")

    return _hook


def _download_youtube_video(youtube_url: str, progress=None) -> tuple:
    """Download a YouTube match video and return its local path plus parsed metadata."""
    url = str(youtube_url or "").strip()
    if not url:
        raise gr.Error("Please enter a YouTube URL.")
    if not YOUTUBE_URL_PATTERN.search(url):
        raise gr.Error("Please enter a valid YouTube URL.")

    try:
        from yt_dlp import YoutubeDL
        from yt_dlp.utils import DownloadError
    except ImportError as exc:
        raise gr.Error("yt-dlp is not installed. Please reinstall dependencies.") from exc

    ffmpeg_exe = _ensure_ffmpeg_executable()
    download_dir = Path(tempfile.mkdtemp(prefix=YOUTUBE_DOWNLOAD_DIR_PREFIX))

    base_ydl_opts = {
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    if ffmpeg_exe:
        base_ydl_opts["ffmpeg_location"] = ffmpeg_exe

    metadata_ydl_opts = dict(base_ydl_opts)
    download_ydl_opts = dict(base_ydl_opts)
    download_ydl_opts.update({
        "outtmpl": str(download_dir / "%(title).180B [%(id)s].%(ext)s"),
        "format": "bestvideo*[ext=mp4]+bestaudio[ext=m4a]/bestvideo*+bestaudio/best[ext=mp4]/best",
        "merge_output_format": "mp4",
    })

    if progress is not None:
        progress(0.05, desc="Fetching YouTube match metadata...")

    try:
        with YoutubeDL(metadata_ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            metadata = _parse_youtube_match_metadata(
                info.get("title", ""),
                info.get("description", ""),
            )
    except DownloadError:
        _cleanup_managed_youtube_dir(download_dir)
        raise
    except Exception:
        _cleanup_managed_youtube_dir(download_dir)
        raise

    if progress is not None:
        progress(0.12, desc=f"Starting download for {metadata.get('match_title') or 'YouTube match'}...")
    download_ydl_opts["progress_hooks"] = [_build_youtube_progress_hook(progress, metadata.get("match_label") or metadata.get("match_title") or "YouTube match")]

    try:
        with YoutubeDL(download_ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_path = _resolve_downloaded_video_path(info, download_dir, ydl=ydl)
    except DownloadError:
        _cleanup_managed_youtube_dir(download_dir)
        raise
    except Exception:
        _cleanup_managed_youtube_dir(download_dir)
        raise

    if not video_path:
        _cleanup_managed_youtube_dir(download_dir)
        raise gr.Error("The YouTube download completed, but the video file could not be found.")

    metadata.update({
        "match_label": _clean_text(
            _parse_match_title_parts(info.get("title", ""))[0] or metadata.get("match_label", "")
        ),
        "match_title": _clean_text(info.get("title", metadata.get("match_title", ""))),
        "regional_name": _clean_text(
            _parse_match_title_parts(info.get("title", ""))[2] or metadata.get("regional_name", "")
        ),
    })
    if progress is not None:
        progress(0.98, desc=f"Download complete: {metadata.get('match_title') or 'YouTube match'}")
    return video_path, metadata


def _get_page_title_for_match(metadata: dict) -> str:
    """Return the short browser-tab title for a downloaded match."""
    if isinstance(metadata, dict):
        return _clean_text(metadata.get("match_label") or metadata.get("match_title") or DEFAULT_PAGE_TITLE)
    return DEFAULT_PAGE_TITLE


def _create_scaled_video_preview(video_path: str, target_width: int = MANUAL_PREVIEW_TARGET_WIDTH,
                                 target_height: int = MANUAL_PREVIEW_TARGET_HEIGHT,
                                 progress=None) -> str:
    """Create a lighter-weight 720p preview video for manual playback."""
    if not video_path:
        return ""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error("Could not open preview video source.")

    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    if source_width <= 0 or source_height <= 0:
        raise gr.Error("Could not determine preview video dimensions.")
    if source_width <= target_width and source_height <= target_height:
        return video_path

    scaled_width = max(2, int(target_width))
    scaled_height = max(2, int(target_height))

    output_path = tempfile.NamedTemporaryFile(suffix="_manual_preview.mp4", delete=False).name
    ffmpeg_exe = _ensure_ffmpeg_executable()

    if ffmpeg_exe:
        if progress:
            progress(0.02, desc="Preparing 720p-class manual preview...")
        cmd = [
            ffmpeg_exe, "-y",
            "-i", video_path,
            "-vf", f"scale={scaled_width}:{scaled_height}:flags=lanczos",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "24",
            "-an",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise gr.Error(f"Preview video creation failed: {result.stderr[-500:]}")
        return output_path

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error("Could not reopen preview video source.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (scaled_width, scaled_height))
    if not out.isOpened():
        cap.release()
        raise gr.Error("Could not create manual preview video.")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
        out.write(resized)
        frame_idx += 1
        if progress and frame_idx % 100 == 0 and total_frames > 0:
            progress(min(0.98, frame_idx / total_frames), desc=f"Preparing manual preview... {frame_idx}/{total_frames}")

    cap.release()
    out.release()
    return output_path


def _serialize_calibration_points(points: list, image_size: tuple) -> list:
    """Persist calibration points as normalized coordinates."""
    if not points or not image_size:
        return []
    img_w, img_h = image_size
    if not img_w or not img_h:
        return []

    serialized = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        x = max(0.0, min(float(point[0]), float(img_w))) / float(img_w)
        y = max(0.0, min(float(point[1]), float(img_h))) / float(img_h)
        serialized.append([round(x, 6), round(y, 6)])
    return serialized


def _restore_calibration_points(points: list, image_size: tuple) -> list:
    """Restore normalized calibration points into the current image size."""
    if not points or not image_size:
        return []
    img_w, img_h = image_size
    if not img_w or not img_h:
        return []

    restored = []
    max_x = max(int(img_w) - 1, 0)
    max_y = max(int(img_h) - 1, 0)
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            x = min(max(int(round(float(point[0]) * float(img_w))), 0), max_x)
            y = min(max(int(round(float(point[1]) * float(img_h))), 0), max_y)
        except Exception:
            continue
        restored.append((x, y))
    return restored


def _load_field_calibration_cache() -> dict:
    """Load cached field calibration data from disk."""
    if not FIELD_CALIBRATION_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(FIELD_CALIBRATION_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[Calibration Cache] Failed to load cache: {exc}")
        return {}


def _save_field_calibration_cache(cache: dict) -> None:
    """Atomically save cached field calibration data to disk."""
    temp_path = FIELD_CALIBRATION_CACHE_PATH.with_suffix(".tmp")
    temp_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(FIELD_CALIBRATION_CACHE_PATH)


def _get_saved_regional_calibration(regional_name: str) -> dict:
    """Return a cached calibration entry for the given regional / event name."""
    cache_key = _normalize_regional_name(regional_name)
    if not cache_key:
        return {}
    cache = _load_field_calibration_cache()
    entry = cache.get(cache_key)
    return entry if isinstance(entry, dict) else {}


def _persist_regional_calibration(
    regional_name: str,
    calibration_points: list = None,
    calibration_image_size: tuple = None,
    blue_side_box_points: list = None,
    blue_side_box_image_size: tuple = None,
    red_side_box_points: list = None,
    red_side_box_image_size: tuple = None,
) -> bool:
    """Save completed calibration details for a regional so future matches prefill."""
    cache_key = _normalize_regional_name(regional_name)
    display_name = _clean_text(regional_name)
    if not cache_key or not display_name:
        return False

    cache = _load_field_calibration_cache()
    entry = cache.get(cache_key, {})
    if not isinstance(entry, dict):
        entry = {}
    entry["regional_name"] = display_name

    updated = False
    if calibration_points and calibration_image_size and len(calibration_points) >= CALIBRATION_REQUIRED_POINTS:
        entry["center_points"] = _serialize_calibration_points(calibration_points, calibration_image_size)
        updated = True
    if blue_side_box_points and blue_side_box_image_size and len(blue_side_box_points) >= SIDE_CAMERA_BOX_POINT_COUNT:
        entry["blue_side_points"] = _serialize_calibration_points(blue_side_box_points, blue_side_box_image_size)
        updated = True
    if red_side_box_points and red_side_box_image_size and len(red_side_box_points) >= SIDE_CAMERA_BOX_POINT_COUNT:
        entry["red_side_points"] = _serialize_calibration_points(red_side_box_points, red_side_box_image_size)
        updated = True

    if not updated:
        return False

    cache[cache_key] = entry
    _save_field_calibration_cache(cache)
    print(f"[Calibration Cache] Saved calibration for {display_name}")
    return True


def _get_saved_calibration_points(regional_name: str, center_frame=None, blue_frame=None, red_frame=None) -> tuple:
    """Restore any saved calibration points for the current extracted frames."""
    entry = _get_saved_regional_calibration(regional_name)
    if not entry:
        return [], [], [], False

    center_points = _restore_calibration_points(
        entry.get("center_points", []),
        center_frame.size if center_frame is not None else None,
    )
    blue_points = _restore_calibration_points(
        entry.get("blue_side_points", []),
        blue_frame.size if blue_frame is not None else None,
    )
    red_points = _restore_calibration_points(
        entry.get("red_side_points", []),
        red_frame.size if red_frame is not None else None,
    )
    return center_points, blue_points, red_points, bool(center_points or blue_points or red_points)


def _prepare_composite_video_calibration_state(video_path: str, start_seconds: float = 0, regional_name: str = "") -> tuple:
    """Extract preview frames and apply any saved calibration for the regional."""
    if video_path is None:
        return (
            None, None, [], None, "*Upload a video to begin calibration*",
            None, None, [], None, "*Upload a video to calibrate blue side boxes*",
            None, None, [], None, "*Upload a video to calibrate red side boxes*",
            False,
        )

    center_frame, blue_frame, red_frame = _extract_composite_calibration_frames(video_path, start_seconds or 0)
    if center_frame is None:
        return (
            None, None, [], None, "Failed to extract frame from video",
            None, None, [], None, "Failed to extract blue side frame",
            None, None, [], None, "Failed to extract red side frame",
            False,
        )

    center_points, blue_points, red_points, loaded_saved = _get_saved_calibration_points(
        regional_name,
        center_frame=center_frame,
        blue_frame=blue_frame,
        red_frame=red_frame,
    )

    center_image = _redraw_calibration_image(center_frame, center_points) if center_points else center_frame
    blue_image = (
        _redraw_side_calibration_image(blue_frame, blue_points, "blue")
        if blue_frame is not None and blue_points else blue_frame
    )
    red_image = (
        _redraw_side_calibration_image(red_frame, red_points, "red")
        if red_frame is not None and red_points else red_frame
    )
    blue_status = _get_side_box_calibration_status_text("blue", len(blue_points)) if blue_frame is not None else "Failed to extract blue side frame"
    red_status = _get_side_box_calibration_status_text("red", len(red_points)) if red_frame is not None else "Failed to extract red side frame"

    return (
        center_image, center_frame, center_points, center_frame.size, _get_calibration_status_text(len(center_points)),
        blue_image, blue_frame, blue_points, (blue_frame.size if blue_frame is not None else None), blue_status,
        red_image, red_frame, red_points, (red_frame.size if red_frame is not None else None), red_status,
        loaded_saved,
    )


def _prepare_manual_video_calibration_state(video_path: str, start_seconds: float = 0, regional_name: str = "") -> tuple:
    """Extract manual-mode calibration data and apply any saved regional calibration."""
    if video_path is None:
        return None, None, None, None, [], None, "*Upload a video to begin calibration*", "{}", False

    center_frame, _, _ = _extract_composite_calibration_frames(video_path, start_seconds or 0)
    if center_frame is None:
        return None, None, None, None, [], None, "Failed to extract center calibration frame", "{}", False

    center_video_path = extract_center_video_from_composite(video_path)
    preview_video_path = _create_scaled_video_preview(center_video_path, target_width=MANUAL_PREVIEW_TARGET_WIDTH)
    center_points, _, _, loaded_saved = _get_saved_calibration_points(
        regional_name,
        center_frame=center_frame,
        blue_frame=None,
        red_frame=None,
    )
    center_image = _redraw_calibration_image(center_frame, center_points) if center_points else center_frame
    return (
        preview_video_path,
        center_video_path,
        center_image,
        center_frame,
        center_points,
        center_frame.size,
        _get_calibration_status_text(len(center_points)),
        "{}",
        loaded_saved,
    )


def _merge_prefilled_robot_numbers(prefilled: list, current_values: list) -> list:
    """Overwrite robot numbers only when parsed metadata provides a value."""
    merged = list(current_values or [])
    while len(merged) < 3:
        merged.append("")
    for idx, value in enumerate((prefilled or [])[:3]):
        cleaned = _clean_text(value)
        if cleaned:
            merged[idx] = cleaned
    return merged[:3]


def _format_video_source_status(source_label: str, regional_name: str = "", match_title: str = "",
                                blue_robots: list = None, red_robots: list = None,
                                calibration_loaded: bool = False) -> str:
    """Build a short markdown summary for the current video source / metadata."""
    lines = [f"**{source_label}**"]
    regional_name = _clean_text(regional_name)
    match_title = _clean_text(match_title)
    blue_labels = _normalize_robot_numbers(blue_robots or [])
    red_labels = _normalize_robot_numbers(red_robots or [])

    if match_title:
        lines.append(f"Match: {match_title}")
    if regional_name:
        lines.append(f"Regional: {regional_name}")
    if blue_labels:
        lines.append(f"Blue teams: {', '.join(blue_labels)}")
    if red_labels:
        lines.append(f"Red teams: {', '.join(red_labels)}")
    if calibration_loaded and regional_name:
        lines.append(f"Loaded saved calibration for {regional_name}.")
    elif regional_name:
        lines.append(f"No saved calibration found yet for {regional_name}.")
    else:
        lines.append("Enter a regional name to reuse saved calibration for uploaded files.")

    return "  \n".join(lines)


class RobotLabelTracker:
    """
    Track robot identities across frames using spatial proximity.
    Maintains history of team number assignments for each tracked robot.
    """
    
    def __init__(self, max_distance: float = 100.0, confident_threshold: int = 2):
        """
        Args:
            max_distance: Maximum distance (in pixels) to match robots between frames
            confident_threshold: Minimum confidence level to skip LLM query
        """
        self.max_distance = max_distance
        self.confident_threshold = confident_threshold
        # Tracked robots: {id: {'bbox': (x1,y1,x2,y2), 'label': str, 'confidence': int}}
        self.tracked_robots = {}
        self.next_id = 0
    
    def _calculate_center(self, bbox: tuple) -> tuple:
        """Calculate center point of bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _distance(self, p1: tuple, p2: tuple) -> float:
        """Euclidean distance between two points."""
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5

    @staticmethod
    def _normalize_allowed_labels(allowed_labels: list) -> set:
        """Normalize an optional list of allowed labels into a set."""
        if allowed_labels is None:
            return set()
        return {
            str(label).strip()
            for label in allowed_labels
            if str(label).strip()
        }

    @classmethod
    def _label_is_allowed(cls, label: str, allowed_labels: list = None) -> bool:
        """Check whether a tracked/OCR label is valid for the current detection."""
        label_text = str(label).strip() if label is not None else ""
        if label_text in ("", "robot", "unknown"):
            return True
        if allowed_labels is None:
            return True
        return label_text in cls._normalize_allowed_labels(allowed_labels)
    
    def check_needs_llm(self, detections: list, allowed_labels_by_detection: list = None) -> tuple:
        """
        Check which detections need LLM queries vs can use tracked labels.
        
        Args:
            detections: List of (x1, y1, x2, y2) bounding boxes in pixels
            allowed_labels_by_detection: Optional valid team-number candidates for
                each detection. Empty candidates force the detection to stay generic.
            
        Returns:
            Tuple of (tracked_labels, needs_llm):
            - tracked_labels: List of labels from tracking (or None if no match)
            - needs_llm: List of booleans indicating if LLM query is needed
        """
        tracked_labels = []
        needs_llm = []
        matched_ids = set()
        
        for idx, bbox in enumerate(detections):
            center = self._calculate_center(bbox)
            allowed_labels = (
                allowed_labels_by_detection[idx]
                if allowed_labels_by_detection is not None and idx < len(allowed_labels_by_detection)
                else None
            )
            force_generic = (
                allowed_labels_by_detection is not None and
                len(self._normalize_allowed_labels(allowed_labels)) == 0
            )
            
            # Find closest existing tracked robot
            best_match_id = None
            best_distance = float('inf')
            
            for robot_id, robot_data in self.tracked_robots.items():
                if robot_id in matched_ids:
                    continue
                old_center = self._calculate_center(robot_data['bbox'])
                dist = self._distance(center, old_center)
                if dist < best_distance and dist < self.max_distance:
                    best_distance = dist
                    best_match_id = robot_id
            
            if best_match_id is not None:
                matched_ids.add(best_match_id)
                robot_data = self.tracked_robots[best_match_id]
                label = robot_data['label']
                confidence = robot_data['confidence']
                
                if force_generic:
                    tracked_labels.append("robot")
                    needs_llm.append(False)
                elif not self._label_is_allowed(label, allowed_labels):
                    tracked_labels.append(None)
                    needs_llm.append(True)
                # Use tracked label if confident (identified before)
                elif label not in ("robot", "unknown") and confidence >= self.confident_threshold:
                    tracked_labels.append(label)
                    needs_llm.append(False)  # Skip LLM - confident track
                else:
                    # Low confidence or still unknown - need LLM
                    tracked_labels.append(label)
                    needs_llm.append(True)
            else:
                # New robot - no match found, need LLM
                if force_generic:
                    tracked_labels.append("robot")
                    needs_llm.append(False)
                else:
                    tracked_labels.append(None)
                    needs_llm.append(True)
        
        return tracked_labels, needs_llm
    
    def update(self, detections: list, new_labels: list, allowed_labels_by_detection: list = None) -> list:
        """
        Update tracking with new detections and labels.
        
        Args:
            detections: List of (x1, y1, x2, y2) bounding boxes in pixels
            new_labels: List of labels from LLM (or "unknown")
            allowed_labels_by_detection: Optional valid team-number candidates for
                each detection. Incompatible labels are forced back to "robot".
            
        Returns:
            List of final labels (using history when LLM returns "unknown")
        """
        if len(detections) != len(new_labels):
            return new_labels
        
        final_labels = []
        new_tracked = {}
        matched_ids = set()
        
        for i, (bbox, label) in enumerate(zip(detections, new_labels)):
            center = self._calculate_center(bbox)
            allowed_labels = (
                allowed_labels_by_detection[i]
                if allowed_labels_by_detection is not None and i < len(allowed_labels_by_detection)
                else None
            )
            force_generic = (
                allowed_labels_by_detection is not None and
                len(self._normalize_allowed_labels(allowed_labels)) == 0
            )
            label = str(label).strip() if label is not None else "unknown"
            if force_generic or not self._label_is_allowed(label, allowed_labels):
                label = "robot"
            
            # Find closest existing tracked robot
            best_match_id = None
            best_distance = float('inf')
            
            for robot_id, robot_data in self.tracked_robots.items():
                if robot_id in matched_ids:
                    continue
                old_center = self._calculate_center(robot_data['bbox'])
                dist = self._distance(center, old_center)
                if dist < best_distance and dist < self.max_distance:
                    best_distance = dist
                    best_match_id = robot_id
            
            if best_match_id is not None:
                # Found match - update or keep previous label
                matched_ids.add(best_match_id)
                old_data = self.tracked_robots[best_match_id]
                old_label_allowed = self._label_is_allowed(old_data['label'], allowed_labels)
                
                if label == "unknown" or label == "robot":
                    # Keep previous label only if it still matches this detection's
                    # allowed alliance candidates.
                    if old_label_allowed and old_data['label'] not in ("robot", "unknown"):
                        final_label = old_data['label']
                        confidence = old_data['confidence']
                    else:
                        final_label = "robot"
                        confidence = 0
                else:
                    # Update with new label
                    final_label = label
                    confidence = old_data['confidence'] + 1
                
                new_tracked[best_match_id] = {
                    'bbox': bbox,
                    'label': final_label,
                    'confidence': confidence
                }
                final_labels.append(final_label)
            else:
                # New robot
                robot_id = self.next_id
                self.next_id += 1
                
                final_label = label if label not in ("unknown", "robot") else "robot"
                new_tracked[robot_id] = {
                    'bbox': bbox,
                    'label': final_label,
                    'confidence': 1 if final_label != "robot" else 0
                }
                final_labels.append(final_label)
        
        self.tracked_robots = new_tracked
        return final_labels
    
    def reset(self):
        """Reset tracker state."""
        self.tracked_robots = {}
        self.next_id = 0


def query_local_llm_for_team_number(
    cropped_image: Image.Image, 
    available_numbers: list, 
    previous_label: str = None,
    timeout: float = 60.0
) -> str:
    """
    Query local LMStudio vision LLM to identify robot team number.
    
    Args:
        cropped_image: PIL Image of cropped robot
        available_numbers: List of valid team numbers in this match
        previous_label: Previous detected label for this robot (if any)
        timeout: Request timeout in seconds
        
    Returns:
        Team number string or "unknown"
    """
    if not LMSTUDIO_ENABLED:
        return "unknown"
    if not available_numbers:
        return "unknown"
    
    # Convert image to base64
    buffered = BytesIO()
    cropped_image.save(buffered, format="JPEG", quality=85)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # Build prompt
    numbers_str = ", ".join(str(n) for n in available_numbers)
    
    prompt = f'What number is the one closest to the center of this image? Choose ONLY from: {numbers_str}. Reply with JUST the number. If unsure, say "none" — accuracy matters more than guessing.'

    try:
        response = requests.post(
            LMSTUDIO_URL,
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                        ]
                    }
                ],
                "temperature": 0.1
            },
            timeout=timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            print(f"[LMStudio Center OCR] Raw response: {answer}")
            
            # Validate answer is a valid team number
            answer_clean = answer.replace(" ", "").strip()
            if answer_clean in [str(n) for n in available_numbers]:
                return answer_clean
            elif "unknown" in answer.lower() or "none" in answer.lower():
                return "unknown"
            else:
                # Try to extract a number from the response
                for num in available_numbers:
                    if str(num) in answer:
                        return str(num)
                return "unknown"
        else:
            print(f"LMStudio error: {response.status_code}")
            return "unknown"
            
    except requests.exceptions.Timeout:
        print("LMStudio timeout")
        return "unknown"
    except requests.exceptions.ConnectionError:
        print("LMStudio not available")
        return "unknown"
    except Exception as e:
        print(f"LMStudio error: {e}")
        return "unknown"


def query_local_llm_batch(
    queries: list,
    max_workers: int = 4,
    timeout: float = 60.0
) -> list:
    """
    Query local LMStudio for multiple robots in parallel.
    
    Args:
        queries: List of dicts with keys:
            - 'cropped_image': PIL Image of cropped robot
            - 'available_numbers': List of valid team numbers
            - 'previous_label': Previous label for context (optional)
        max_workers: Maximum parallel requests (default 4 to avoid overwhelming LMStudio)
        timeout: Timeout in seconds for each individual query
        
    Returns:
        List of team number strings in the same order as input queries
    """
    if not queries:
        return []
    
    if not LMSTUDIO_ENABLED:
        return ["unknown"] * len(queries)
    
    results = [None] * len(queries)
    
    def query_single(idx: int, query: dict) -> tuple:
        """Execute a single query and return (index, result)."""
        label = query_local_llm_for_team_number(
            query['cropped_image'],
            query['available_numbers'],
            query.get('previous_label'),
            timeout=timeout
        )
        return idx, label
    
    # Use ThreadPoolExecutor for parallel I/O-bound queries
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all queries
        futures = {
            executor.submit(query_single, idx, query): idx 
            for idx, query in enumerate(queries)
        }
        
        # Gather results as they complete
        for future in as_completed(futures):
            try:
                idx, label = future.result()
                results[idx] = label
            except Exception as e:
                idx = futures[future]
                print(f"Parallel LLM query {idx} failed: {e}")
                results[idx] = "unknown"
    
    # Fill any None values with "unknown"
    return [r if r is not None else "unknown" for r in results]


def _query_single_side_camera_robot(
    img_base64: str,
    robot_team: str,
    camera_side: str = "blue",
    timeout: float = 60.0
) -> dict:
    """Ask the side-camera LLM about one specific robot and return its position if visible."""
    team = str(robot_team).strip()
    if not team:
        return None

    side_name = str(camera_side).strip().lower()
    if side_name == "red":
        valid_positions = {
            "middle": 1,
            "right": 2,
            "far right": 3,
            "farright": 3,
        }
        bucket_instructions = (
            "Assign this robot to one position: 'middle', 'right', or 'far right'. "
            "Default to 'middle' if between middle and right. "
            "Use 'right' only if clearly in the right lane but not at the edge. "
            "Use 'far right' only if on the right side of the attached frame and close to the ladder. "
            "Base decisions on visual evidence (guide-box, lane, frame side, center) and compare nearby boxes when helpful."
        )
        example_json = (
            f"[{{\"team\":\"{team}\","
            "\"description\":\"This robot is shifted into the right-side lane and aligns better with the right guide box than the middle one, but it is not extreme enough to be far right.\","
            "\"position\":\"right\"}}]"
        )
    else:
        valid_positions = {
            "middle": 1,
            "left": 2,
            "far left": 3,
            "farleft": 3,
        }
        bucket_instructions = (
            "Assign this robot to one position: 'middle', 'left', or 'far left'. "
            "Default to 'middle' if between middle and left. "
            "Use 'left' only if clearly in the left lane but not at the edge. "
            "Use 'far left' only if on the left side of the attached frame and close to the ladder. "
            "Base decisions on visual evidence (guide-box, lane, frame side, center) and compare nearby boxes when helpful."
        )
        example_json = (
            f"[{{\"team\":\"{team}\","
            "\"description\":\"This robot is shifted into the left-side lane and aligns better with the left guide box than the middle one, but it is not extreme enough to be far left.\","
            "\"position\":\"left\"}}]"
        )

    prompt = (
        f"Where is robot {team} in this image? "
        f"Only answer for robot {team}. "
        f"{bucket_instructions} "
        f"Reply with ONLY a JSON array. "
        f"Return either [] if robot {team} is not visible, or a one-item JSON array with keys in this order: 'team', 'description', 'position'. "
        f"Descriptions should be 1-2 sentences describing where the robot is in the frame. "
        f"Example if visible: {example_json}. "
        f"If robot {team} is not visible, return []."
    )

    try:
        response = requests.post(
            LMSTUDIO_URL,
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                        ]
                    }
                ],
                "temperature": 0.1
            },
            timeout=timeout
        )

        if response.status_code != 200:
            print(f"[Side Camera LLM] Error for {team}: {response.status_code}")
            return None

        result = response.json()
        answer = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        print(f"[LMStudio Side Presence {side_name}] Raw response for {team}: {answer}")

        if not answer or answer.strip() == "[]" or "none" in answer.lower():
            return None

        try:
            parsed = json.loads(parse_json(answer))
        except Exception:
            print(f"[Side Camera LLM] Could not parse JSON for {team}: {answer}")
            return None

        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            return None

        for item in parsed:
            if not isinstance(item, dict):
                continue
            returned_team = str(item.get("team", "")).strip() or team
            if returned_team != team:
                continue
            description = " ".join(str(item.get("description", "")).strip().split())
            position = str(item.get("position", "")).strip().lower()
            position = " ".join(position.split())
            x_bucket = valid_positions.get(position)
            if x_bucket is None:
                continue
            return {
                "team": team,
                "description": description,
                "position": position,
                "x_bucket": x_bucket
            }

        return None
    except requests.exceptions.Timeout:
        print(f"[Side Camera LLM] Timeout for {team}")
        return None
    except requests.exceptions.ConnectionError:
        print("[Side Camera LLM] Not available")
        return None
    except Exception as e:
        print(f"[Side Camera LLM] Error for {team}: {e}")
        return None


def query_side_camera_presence(
    frame_image: Image.Image,
    alliance_robots: list,
    camera_side: str = "blue",
    timeout: float = 60.0
) -> list:
    """
    Query local LMStudio one robot at a time to determine which alliance robots
    are visible in a side camera frame and which of the 3 side-camera position
    buckets each visible robot occupies.

    Args:
        frame_image: PIL Image of the full side camera frame
        alliance_robots: List of team numbers for this alliance (up to 3)
        camera_side: "blue" or "red" side camera, used for response ordering
        timeout: Request timeout in seconds

    Returns:
        List of dicts like
        [{'team': '1234', 'description': 'low robot near left wall', 'position': 'left', 'x_bucket': 2}]
    """
    if not LMSTUDIO_ENABLED:
        return []

    valid_robots = [str(r).strip() for r in alliance_robots if r and str(r).strip()]
    if not valid_robots:
        return []

    buffered = BytesIO()
    frame_image.save(buffered, format="JPEG", quality=85)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    results = [None] * len(valid_robots)

    def query_single(idx: int, team: str) -> tuple:
        return idx, _query_single_side_camera_robot(
            img_base64,
            team,
            camera_side=camera_side,
            timeout=timeout
        )

    max_workers = min(3, len(valid_robots))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(query_single, idx, team): idx
            for idx, team in enumerate(valid_robots)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result_idx, found = future.result()
                results[result_idx] = found
            except Exception as e:
                print(f"[Side Camera LLM] Parallel query {valid_robots[idx]} failed: {e}")
                results[idx] = None

    return [item for item in results if item]


# Hidden robot bounding box positions on center camera (reference coords 1918x709)
# When a robot is seen by a side camera but not the center camera, a bounding box is
# placed at this position on the center camera frame for shot attribution.
_HIDDEN_ROBOT_BBOX_BLUE = (502, 360, 574, 417)  # x1, y1, x2, y2
_HIDDEN_ROBOT_BBOX_RED = (1364, 371, 1442, 422)  # x1, y1, x2, y2

# Hidden robot slot centers on the center camera (reference coords 1918x709).
# These align with the 3 visible position buckets from each side camera.
_HIDDEN_ROBOT_SLOT_CENTERS = {
    "blue": {
        1: (445, 482),   # middle blue (1)
        2: (594, 379),   # left blue (2)
        3: (338, 402),   # farthest left blue (3)
    },
    "red": {
        1: (1493, 483),  # middle red (1)
        2: (1367, 400),  # right red (2)
        3: (1618, 406),  # farthest right red (3)
    }
}

_RECENT_ROBOT_BBOX_SKIP_LABELS = {"robot", "unknown", "red", "blue"}


def _hidden_bbox_for_slot(side_name: str, x_bucket: int) -> tuple:
    """Build a hidden robot bbox for a side-camera position bucket."""
    base_bbox = _HIDDEN_ROBOT_BBOX_BLUE if side_name == "blue" else _HIDDEN_ROBOT_BBOX_RED
    base_x1, base_y1, base_x2, base_y2 = base_bbox
    width = base_x2 - base_x1
    height = base_y2 - base_y1

    cx, cy = _HIDDEN_ROBOT_SLOT_CENTERS.get(side_name, {}).get(
        x_bucket,
        ((base_x1 + base_x2) // 2, (base_y1 + base_y2) // 2)
    )

    x1 = int(round(cx - width / 2))
    y1 = int(round(cy - height / 2))
    x2 = x1 + width
    y2 = y1 + height
    return x1, y1, x2, y2


def _bbox_json_to_pixels(bbox: dict, frame_width: int, frame_height: int) -> tuple:
    """Convert a normalized detection bbox into pixel coordinates."""
    box = bbox.get("box_2d", []) if isinstance(bbox, dict) else []
    if len(box) < 4 or frame_width <= 0 or frame_height <= 0:
        return None
    y1 = float(box[0]) / 1000.0 * frame_height
    x1 = float(box[1]) / 1000.0 * frame_width
    y2 = float(box[2]) / 1000.0 * frame_height
    x2 = float(box[3]) / 1000.0 * frame_width
    return (x1, y1, x2, y2)


def _bbox_overlap_ratio(box_a: tuple, box_b: tuple) -> float:
    """
    Return overlap relative to the smaller box.

    This is more forgiving than IoU for center-vs-hidden robot conflicts because
    the hidden proxy box is intentionally small.
    """
    if not box_a or not box_b:
        return 0.0

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return inter_area / min(area_a, area_b)


def inject_hidden_robot_bboxes(base_bboxes_json: str, persistent_hidden_robots: dict,
                               side_camera_visible_robots: dict, frame_count: int,
                               width: int, height: int,
                               edge_persist_frames: int = 60) -> tuple:
    """
    Augment center-camera detections with hidden robots inferred from the latest
    side-camera visibility snapshot for each alliance.

    Returns:
        Tuple of (augmented_bboxes_json, updated_persistent_hidden_robots)
    """
    try:
        center_bboxes = json.loads(parse_json(base_bboxes_json))
    except Exception:
        center_bboxes = []

    center_detected_labels = set()
    for bbox in center_bboxes:
        lbl = str(bbox.get('label', '')).strip()
        if lbl and lbl not in ('robot', 'unknown', 'red', 'blue'):
            center_detected_labels.add(lbl)

    updated_hidden = {}

    for side_name in ('blue', 'red'):
        side_data = side_camera_visible_robots.get(side_name, {}) if side_camera_visible_robots else {}
        if not side_data:
            continue

        latest_side_frame = None
        for sf in sorted(side_data.keys()):
            if sf <= frame_count:
                latest_side_frame = sf
            else:
                break

        if latest_side_frame is None:
            continue

        side_visible = side_data[latest_side_frame]
        for item in side_visible:
            if not isinstance(item, dict):
                continue
            robot_label = str(item.get('team', '')).strip()
            try:
                x_bucket = int(item.get('x_bucket'))
            except Exception:
                continue

            if not robot_label or x_bucket not in (1, 2, 3):
                continue
            if robot_label in center_detected_labels:
                continue

            updated_hidden[robot_label] = {
                'side': side_name,
                'x_bucket': x_bucket,
                'last_side_seen_frame': latest_side_frame
            }

    corner_hidden_overrides = []
    for robot_label, hidden_meta in updated_hidden.items():
        side_name = hidden_meta.get('side')
        x_bucket = hidden_meta.get('x_bucket', 1)
        if x_bucket != 3:
            continue
        hx1, hy1, hx2, hy2 = _hidden_bbox_for_slot(side_name, x_bucket)
        hx1_s, hy1_s = _calibration_transform_point(hx1, hy1, width, height, inverse=False)
        hx2_s, hy2_s = _calibration_transform_point(hx2, hy2, width, height, inverse=False)
        corner_hidden_overrides.append((robot_label, side_name, (hx1_s, hy1_s, hx2_s, hy2_s)))

    filtered_center_bboxes = []
    for bbox in center_bboxes:
        center_label = str(bbox.get('label', '')).strip()
        center_box_px = _bbox_json_to_pixels(bbox, width, height)
        should_drop = False
        for hidden_label, side_name, hidden_box_px in corner_hidden_overrides:
            if center_label == hidden_label:
                continue
            overlap_ratio = _bbox_overlap_ratio(center_box_px, hidden_box_px)
            if overlap_ratio >= 0.25:
                print(
                    f"[Hidden Robot] Corner override: keeping side-camera label {hidden_label} "
                    f"from {side_name} far corner and dropping overlapping center label "
                    f"{center_label or 'robot'}"
                )
                should_drop = True
                break
        if not should_drop:
            filtered_center_bboxes.append(bbox)

    center_bboxes = filtered_center_bboxes
    injected_bboxes = list(center_bboxes)
    slot_counts = {}
    ordered_hidden = sorted(
        updated_hidden.items(),
        key=lambda item: (
            0 if item[1].get('side') == "blue" else 1,
            item[1].get('x_bucket', 99),
            item[0]
        )
    )

    for robot_label, hidden_meta in ordered_hidden:
        side_name = hidden_meta.get('side')
        x_bucket = hidden_meta.get('x_bucket', 1)
        hx1, hy1, hx2, hy2 = _hidden_bbox_for_slot(side_name, x_bucket)

        slot_key = (side_name, x_bucket)
        duplicate_index = slot_counts.get(slot_key, 0)
        slot_counts[slot_key] = duplicate_index + 1
        if duplicate_index > 0:
            bbox_height = hy2 - hy1
            hy1 += duplicate_index * (bbox_height + 5)
            hy2 += duplicate_index * (bbox_height + 5)

        hx1_s, hy1_s = _calibration_transform_point(hx1, hy1, width, height, inverse=False)
        hx2_s, hy2_s = _calibration_transform_point(hx2, hy2, width, height, inverse=False)

        y1_norm = int((hy1_s / height) * 1000)
        x1_norm = int((hx1_s / width) * 1000)
        y2_norm = int((hy2_s / height) * 1000)
        x2_norm = int((hx2_s / width) * 1000)

        injected_bboxes.append({
            "box_2d": [y1_norm, x1_norm, y2_norm, x2_norm],
            "label": robot_label
        })
        print(
            f"[Hidden Robot] Injecting {robot_label} at "
            f"({hx1_s:.0f},{hy1_s:.0f})-({hx2_s:.0f},{hy2_s:.0f}) "
            f"from {side_name} bucket {x_bucket}"
        )

    return json.dumps(injected_bboxes), updated_hidden


def persist_recent_robot_bboxes(current_bboxes_json: str, recent_robot_bboxes: dict,
                                frame_count: int, max_age_frames: int) -> tuple:
    """
    Keep the last seen labeled robot bbox alive briefly when detections disappear.

    This runs after center detections and side-camera hidden injections are merged,
    so a robot only persists when neither source currently sees it.
    """
    try:
        current_bboxes = json.loads(parse_json(current_bboxes_json))
    except Exception:
        current_bboxes = []

    output_bboxes = list(current_bboxes)
    updated_recent = {}
    current_labels = set()

    for bbox in current_bboxes:
        label = str(bbox.get('label', '')).strip()
        box = bbox.get('box_2d', [])
        if not label or label.lower() in _RECENT_ROBOT_BBOX_SKIP_LABELS or len(box) < 4:
            continue

        current_labels.add(label)
        updated_recent[label] = {
            'bbox': {
                'box_2d': list(box[:4]),
                'label': label
            },
            'last_seen_frame': frame_count
        }

    for label, meta in (recent_robot_bboxes or {}).items():
        if label in current_labels:
            continue

        bbox = meta.get('bbox')
        last_seen_frame = meta.get('last_seen_frame')
        if bbox is None or last_seen_frame is None:
            continue

        if (frame_count - last_seen_frame) > max_age_frames:
            continue

        output_bboxes.append({
            'box_2d': list(bbox.get('box_2d', [])[:4]),
            'label': str(bbox.get('label', label)).strip() or label
        })
        updated_recent[label] = {
            'bbox': {
                'box_2d': list(bbox.get('box_2d', [])[:4]),
                'label': str(bbox.get('label', label)).strip() or label
            },
            'last_seen_frame': last_seen_frame
        }

    return json.dumps(output_bboxes), updated_recent


_SIDE_CAMERA_REF_SIZE = (940, 339)
_SIDE_CAMERA_RED_ZONE_RECTS = [
    ("MIDDLE", (257, 10, 509, 337)),
    ("RIGHT", (509, 10, 705, 337)),
    ("FAR RIGHT", (705, 10, 937, 337)),
]


def _get_side_camera_zone_rects(camera_side: str, frame_width: int, frame_height: int) -> list:
    """Return side-camera guidance rectangles scaled to the current cropped frame."""
    ref_w, ref_h = _SIDE_CAMERA_REF_SIZE
    sx = frame_width / ref_w if ref_w else 1.0
    sy = frame_height / ref_h if ref_h else 1.0
    is_blue = str(camera_side).strip().lower() == "blue"
    shift_px = 50 if is_blue else -50

    rects = []
    for label, (x1, y1, x2, y2) in _SIDE_CAMERA_RED_ZONE_RECTS:
        if is_blue:
            flipped_label = label.replace("RIGHT", "LEFT")
            x1_f = ref_w - x2
            x2_f = ref_w - x1
            x1_use, x2_use = x1_f, x2_f
            label_use = flipped_label
        else:
            x1_use, x2_use = x1, x2
            label_use = label

        x1_use = max(0, min(ref_w, x1_use + shift_px))
        x2_use = max(0, min(ref_w, x2_use + shift_px))

        rects.append((
            label_use,
            (
                int(round(x1_use * sx)),
                int(round(y1 * sy)),
                int(round(x2_use * sx)),
                int(round(y2 * sy)),
            )
        ))

    return rects


def annotate_side_camera_guides(frame: Image.Image, camera_side: str,
                                calibrated_boxes: list = None) -> Image.Image:
    """
    Draw lightweight side-camera lane guides to help the local LLM reason about
    middle / left / right / far-left / far-right positions.
    """
    if frame is None:
        return frame

    boxes = calibrated_boxes or _get_side_camera_zone_rects(camera_side, frame.width, frame.height)
    return _draw_side_camera_box_overlay(frame, boxes, camera_side)




# Color palette for bounding boxes
additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]
COLORS = [
    'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 'brown',
    'gray', 'turquoise', 'cyan', 'magenta', 'lime', 'navy', 'maroon', 'teal',
    'olive', 'coral', 'lavender', 'violet', 'gold', 'silver'
] + additional_colors

# Map configuration
MAP_IMAGE_PATH = r"C:\Users\derek\OneDrive\Documents\GitHub\Rebuilt-Scouting\map.png"

# Alliance-based colors (RGB tuples)
# Blue alliance: light blue -> medium blue -> dark blue
BLUE_ALLIANCE_COLORS = [
    (0, 150, 255),    # Blue 1 - Light blue
    (0, 100, 200),    # Blue 2 - Medium blue
    (0, 50, 150),     # Blue 3 - Dark blue
]

# Red alliance: light red -> medium red -> dark red
RED_ALLIANCE_COLORS = [
    (255, 50, 50),    # Red 1 - Light red
    (200, 0, 0),      # Red 2 - Medium red
    (140, 0, 0),      # Red 3 - Dark red
]

# Default color for unknown robots
DEFAULT_COLOR = (0, 0, 0)  # Black
BALL_HIGHLIGHT_ALL_OPTION = "All Robots"
CENTER_SCORE_COUNTER_REF_SIZE = (1918, 709)
CENTER_SCORE_COUNTER_RECTS = {
    "blue": (81, 67, 228, 114),
    "red": (1738, 70, 1885, 112),
}
CENTER_SCORE_OCR_SAMPLE_FPS = 5.0
CENTER_SCORE_OCR_MIN_CONFIRMATIONS = 2
CENTER_SCORE_OCR_EVENT_WINDOW_SECONDS = 4.0
CENTER_SCORE_OCR_POST_ROLL_SECONDS = 4.0
CENTER_SCORE_OCR_ATTRIBUTION_MIN_LAG_SECONDS = 1.0
CENTER_SCORE_OCR_ATTRIBUTION_MAX_LAG_SECONDS = 4.0
CENTER_SCORE_OCR_ATTRIBUTION_PREFERRED_LAG_SECONDS = 2.5
CENTER_SCORE_OCR_ATTRIBUTION_RELAXED_MAX_LAG_SECONDS = 6.0
MANUAL_TRACK_SHOOTING_PERSIST_SECONDS = 4.0
MULTI_CAMERA_SHOT_DEDUP_WINDOW_SECONDS = 3.0
CENTER_MATCH_CLOCK_RECT = (900, 61, 1020, 128)
CENTER_MATCH_CLOCK_OCR_SAMPLE_FPS = 2.0
CENTER_MATCH_CLOCK_OCR_MIN_CONFIRMATIONS = 2
MATCH_CLOCK_LOOKUP_MAX_GAP_SECONDS = 6.0
_CENTER_SCORE_OCR_DISABLED = pytesseract is None
_CENTER_SCORE_OCR_DISABLED_REASON_PRINTED = pytesseract is None
_CENTER_MATCH_CLOCK_OCR_DISABLED = pytesseract is None
_CENTER_MATCH_CLOCK_OCR_DISABLED_REASON_PRINTED = pytesseract is None


def get_robot_color(robot_label: str, blue_robots: list = None, red_robots: list = None) -> tuple:
    """
    Get the color for a robot based on its team number and alliance.
    
    Args:
        robot_label: The team number as a string
        blue_robots: List of blue alliance team numbers [robot1, robot2, robot3]
        red_robots: List of red alliance team numbers [robot1, robot2, robot3]
        
    Returns:
        RGB tuple for the robot's color
    """
    blue_robots = blue_robots or []
    red_robots = red_robots or []
    
    # Clean the label
    label = str(robot_label).strip()
    
    # Check blue alliance
    for i, blue_num in enumerate(blue_robots):
        if blue_num and str(blue_num).strip() == label:
            return BLUE_ALLIANCE_COLORS[i] if i < len(BLUE_ALLIANCE_COLORS) else BLUE_ALLIANCE_COLORS[-1]
    
    # Check red alliance
    for i, red_num in enumerate(red_robots):
        if red_num and str(red_num).strip() == label:
            return RED_ALLIANCE_COLORS[i] if i < len(RED_ALLIANCE_COLORS) else RED_ALLIANCE_COLORS[-1]
    
    # Unknown robot - return black
    return DEFAULT_COLOR


def _normalize_robot_numbers(robots: list) -> list:
    """Return cleaned team-number strings, preserving input order."""
    return [str(robot).strip() for robot in (robots or []) if str(robot).strip()]


def _scale_ref_rect(rect: tuple, frame_width: int, frame_height: int, ref_size: tuple = CENTER_SCORE_COUNTER_REF_SIZE) -> tuple:
    """Scale a rectangle from the reference center-camera resolution to the current frame."""
    ref_w, ref_h = ref_size
    if frame_width <= 0 or frame_height <= 0 or ref_w <= 0 or ref_h <= 0:
        return rect
    x1, y1, x2, y2 = rect
    sx = frame_width / ref_w
    sy = frame_height / ref_h
    return (
        int(round(x1 * sx)),
        int(round(y1 * sy)),
        int(round(x2 * sx)),
        int(round(y2 * sy)),
    )


def _parse_center_score_counter_text(text: str):
    """Parse a scoreboard snippet like '12 / 40' and return the left-hand number."""
    cleaned = re.sub(r"[^0-9/]", "", str(text or ""))
    if not cleaned:
        return None
    if "/" in cleaned:
        cleaned = cleaned.split("/", 1)[0]
    match = re.search(r"\d+", cleaned)
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _ocr_center_roi_text(
    frame_bgr: np.ndarray,
    rect: tuple,
    whitelist: str,
    scale: float = 4.0,
    variant_mode: str = "all",
):
    """Run OCR on a scaled center-camera ROI and return raw OCR reads."""
    if pytesseract is None or frame_bgr is None or rect is None:
        return []

    frame_height, frame_width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = _scale_ref_rect(rect, frame_width, frame_height)
    x1 = max(0, min(frame_width - 1, x1))
    x2 = max(x1 + 1, min(frame_width, x2))
    y1 = max(0, min(frame_height - 1, y1))
    y2 = max(y1 + 1, min(frame_height, y2))
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return []

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    if variant_mode == "single":
        _, single_variant = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants = [single_variant]
    else:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, thresh_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            5,
        )
        variants = [gray, thresh, thresh_inv, adaptive]
    outputs = []

    for variant in variants:
        text = pytesseract.image_to_string(
            Image.fromarray(variant),
            config=f"--psm 7 -c tessedit_char_whitelist={whitelist}"
        )
        if text is not None:
            outputs.append(str(text))

    return outputs


def _ocr_center_score_counter(frame_bgr: np.ndarray, alliance: str, return_debug: bool = False):
    """Run lightweight OCR on the center-camera alliance score counter."""
    global _CENTER_SCORE_OCR_DISABLED, _CENTER_SCORE_OCR_DISABLED_REASON_PRINTED

    debug_info = {
        "alliance": str(alliance).strip().lower(),
        "value": None,
        "raw_texts": [],
        "parsed_values": [],
        "error": None,
        "disabled": bool(_CENTER_SCORE_OCR_DISABLED or pytesseract is None),
    }

    if _CENTER_SCORE_OCR_DISABLED or pytesseract is None or frame_bgr is None:
        return debug_info if return_debug else None

    rect = CENTER_SCORE_COUNTER_RECTS.get(str(alliance).strip().lower())
    if rect is None:
        debug_info["error"] = "missing_rect"
        return debug_info if return_debug else None

    parsed_counts = []

    try:
        raw_texts = _ocr_center_roi_text(frame_bgr, rect, "0123456789/")
        debug_info["raw_texts"] = list(raw_texts or [])
    except TesseractNotFoundError as e:
        _CENTER_SCORE_OCR_DISABLED = True
        debug_info["disabled"] = True
        debug_info["error"] = str(e)
        if not _CENTER_SCORE_OCR_DISABLED_REASON_PRINTED:
            print(f"Center score OCR disabled: {e}")
            _CENTER_SCORE_OCR_DISABLED_REASON_PRINTED = True
        return debug_info if return_debug else None
    except Exception as e:
        debug_info["error"] = str(e)
        return debug_info if return_debug else None

    for text in debug_info["raw_texts"]:
        parsed = _parse_center_score_counter_text(text)
        if parsed is not None:
            parsed_counts.append(parsed)
    debug_info["parsed_values"] = list(parsed_counts)

    if not parsed_counts:
        return debug_info if return_debug else None

    debug_info["value"] = Counter(parsed_counts).most_common(1)[0][0]
    return debug_info if return_debug else debug_info["value"]


class CenterScoreOCRTracker:
    """Track the scoreboard counters shown on the cropped center camera feed."""

    def __init__(self, min_confirmations: int = CENTER_SCORE_OCR_MIN_CONFIRMATIONS):
        self.min_confirmations = max(1, int(min_confirmations))
        self.confirmed_counts = {"blue": None, "red": None}
        self.start_counts = {"blue": None, "red": None}
        self.pending_counts = {"blue": None, "red": None}
        self.pending_hits = {"blue": 0, "red": 0}
        self.events = {"blue": [], "red": []}
        self.observation_counts = {"blue": 0, "red": 0}
        self.latest_debug = {
            "blue": {"value": None, "raw_texts": [], "parsed_values": [], "disabled": bool(pytesseract is None)},
            "red": {"value": None, "raw_texts": [], "parsed_values": [], "disabled": bool(pytesseract is None)},
        }

    def _accept_count(self, alliance: str, count: int, elapsed_seconds: float):
        previous = self.confirmed_counts[alliance]
        if previous is None:
            self.start_counts[alliance] = count
        elif count > previous:
            self.events[alliance].append((float(elapsed_seconds), int(count - previous), int(count)))
        self.confirmed_counts[alliance] = int(count)
        self.pending_counts[alliance] = None
        self.pending_hits[alliance] = 0

    def update(self, frame_bgr: np.ndarray, elapsed_seconds: float):
        for alliance in ("blue", "red"):
            read_result = _ocr_center_score_counter(frame_bgr, alliance, return_debug=True)
            self.latest_debug[alliance] = {
                **(read_result or {}),
                "elapsed_seconds": float(elapsed_seconds),
                "confirmed": self.confirmed_counts[alliance],
                "pending": self.pending_counts[alliance],
                "pending_hits": int(self.pending_hits[alliance]),
            }
            count = None if not isinstance(read_result, dict) else read_result.get("value")
            if count is None:
                continue

            self.observation_counts[alliance] += 1
            confirmed = self.confirmed_counts[alliance]

            if confirmed is not None and count < confirmed:
                continue

            if confirmed is not None and count == confirmed:
                self.pending_counts[alliance] = None
                self.pending_hits[alliance] = 0
                continue

            if self.pending_counts[alliance] == count:
                self.pending_hits[alliance] += 1
            else:
                self.pending_counts[alliance] = count
                self.pending_hits[alliance] = 1

            if self.pending_hits[alliance] >= self.min_confirmations:
                self._accept_count(alliance, count, elapsed_seconds)
                self.latest_debug[alliance]["confirmed"] = self.confirmed_counts[alliance]
                self.latest_debug[alliance]["pending"] = self.pending_counts[alliance]
                self.latest_debug[alliance]["pending_hits"] = int(self.pending_hits[alliance])

    def summary(self) -> dict:
        summary = {}
        for alliance in ("blue", "red"):
            start = self.start_counts[alliance]
            end = self.confirmed_counts[alliance]
            scored = None if start is None or end is None else max(0, int(end) - int(start))
            summary[alliance] = {
                "start": start,
                "end": end,
                "scored": scored,
                "events": list(self.events[alliance]),
                "observations": int(self.observation_counts[alliance]),
                "latest_debug": dict(self.latest_debug.get(alliance, {})),
            }
        return summary


def _parse_center_match_clock_text(text: str):
    """Parse a center-camera match clock OCR snippet like '2:20' or '0:08'."""
    normalized = str(text or "").upper()
    replacements = {
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "|": "1",
        ";": ":",
        ".": ":",
        ",": ":",
        " ": "",
    }
    for src, dst in replacements.items():
        normalized = normalized.replace(src, dst)
    normalized = re.sub(r"[^0-9:]", "", normalized)
    if not normalized:
        return None

    candidates = []
    colon_match = re.search(r"([0-2]):([0-5]\d)", normalized)
    if colon_match:
        candidates.append((int(colon_match.group(1)), int(colon_match.group(2))))

    compact_match = re.search(r"\b([0-2])([0-5]\d)\b", normalized)
    if compact_match:
        candidates.append((int(compact_match.group(1)), int(compact_match.group(2))))

    if not candidates and ":" in normalized:
        loose_match = re.search(r"([0-2]):(\d{1,2})", normalized)
        if loose_match:
            minute = int(loose_match.group(1))
            seconds = int(loose_match.group(2))
            if 0 <= seconds <= 59:
                candidates.append((minute, seconds))

    for minute, seconds in candidates:
        total_seconds = (minute * 60) + seconds
        if 0 <= total_seconds <= 140:
            return total_seconds

    return None


def _format_clock_seconds(clock_seconds: int) -> str:
    """Format a match clock value like 134 as 2:14."""
    try:
        total_seconds = max(0, int(clock_seconds))
    except (TypeError, ValueError):
        return ""
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:02d}"


def _ocr_center_match_clock(frame_bgr: np.ndarray, return_debug: bool = False):
    """Run OCR on the center-camera match clock."""
    global _CENTER_MATCH_CLOCK_OCR_DISABLED, _CENTER_MATCH_CLOCK_OCR_DISABLED_REASON_PRINTED

    debug_info = {
        "value": None,
        "raw_texts": [],
        "parsed_values": [],
        "error": None,
        "disabled": bool(_CENTER_MATCH_CLOCK_OCR_DISABLED or pytesseract is None),
    }

    if _CENTER_MATCH_CLOCK_OCR_DISABLED or pytesseract is None or frame_bgr is None:
        return debug_info if return_debug else None

    parsed_values = []

    try:
        raw_texts = _ocr_center_roi_text(
            frame_bgr,
            CENTER_MATCH_CLOCK_RECT,
            "0123456789:",
            scale=5.0,
            variant_mode="single",
        )
        debug_info["raw_texts"] = list(raw_texts or [])
    except TesseractNotFoundError as e:
        _CENTER_MATCH_CLOCK_OCR_DISABLED = True
        debug_info["disabled"] = True
        debug_info["error"] = str(e)
        if not _CENTER_MATCH_CLOCK_OCR_DISABLED_REASON_PRINTED:
            print(f"Center match clock OCR disabled: {e}")
            _CENTER_MATCH_CLOCK_OCR_DISABLED_REASON_PRINTED = True
        return debug_info if return_debug else None
    except Exception as e:
        debug_info["error"] = str(e)
        return debug_info if return_debug else None

    for text in debug_info["raw_texts"]:
        parsed = _parse_center_match_clock_text(text)
        if parsed is not None:
            parsed_values.append(parsed)
    debug_info["parsed_values"] = list(parsed_values)

    if not parsed_values:
        return debug_info if return_debug else None

    debug_info["value"] = Counter(parsed_values).most_common(1)[0][0]
    return debug_info if return_debug else debug_info["value"]


class CenterMatchClockOCRTracker:
    """Track the on-screen match clock from the cropped center camera feed."""

    def __init__(self, min_confirmations: int = CENTER_MATCH_CLOCK_OCR_MIN_CONFIRMATIONS):
        self.min_confirmations = max(1, int(min_confirmations))
        self.confirmed_clock_seconds = None
        self.pending_clock_seconds = None
        self.pending_hits = 0
        self.observations = []
        self.observation_count = 0
        self.latest_debug = {"value": None, "raw_texts": [], "parsed_values": [], "disabled": bool(pytesseract is None)}

    def _record_observation(self, elapsed_seconds: float, clock_seconds: int):
        clock_seconds = int(clock_seconds)
        elapsed_seconds = float(elapsed_seconds)
        if (
            self.observations
            and int(self.observations[-1][1]) == clock_seconds
            and abs(float(self.observations[-1][0]) - elapsed_seconds) < 0.75
        ):
            return
        self.observations.append((elapsed_seconds, clock_seconds))

    def update(self, frame_bgr: np.ndarray, elapsed_seconds: float):
        read_result = _ocr_center_match_clock(frame_bgr, return_debug=True)
        self.latest_debug = {
            **(read_result or {}),
            "elapsed_seconds": float(elapsed_seconds),
            "confirmed": self.confirmed_clock_seconds,
            "pending": self.pending_clock_seconds,
            "pending_hits": int(self.pending_hits),
        }
        clock_seconds = None if not isinstance(read_result, dict) else read_result.get("value")
        if clock_seconds is None:
            return

        self.observation_count += 1
        if self.confirmed_clock_seconds is not None and int(clock_seconds) == int(self.confirmed_clock_seconds):
            self.pending_clock_seconds = None
            self.pending_hits = 0
            self._record_observation(elapsed_seconds, clock_seconds)
            return

        if self.pending_clock_seconds == clock_seconds:
            self.pending_hits += 1
        else:
            self.pending_clock_seconds = int(clock_seconds)
            self.pending_hits = 1

        if self.pending_hits >= self.min_confirmations:
            self.confirmed_clock_seconds = int(clock_seconds)
            self.pending_clock_seconds = None
            self.pending_hits = 0
            self._record_observation(elapsed_seconds, clock_seconds)
            self.latest_debug["confirmed"] = self.confirmed_clock_seconds
            self.latest_debug["pending"] = self.pending_clock_seconds
            self.latest_debug["pending_hits"] = int(self.pending_hits)

    def summary(self) -> dict:
        return {
            "observations": list(self.observations),
            "observations_count": int(self.observation_count),
            "last_confirmed": self.confirmed_clock_seconds,
            "latest_debug": dict(self.latest_debug),
        }


def _allocate_proportional_integers(raw_values: dict, target_total: int) -> dict:
    """Allocate an integer total proportionally using largest remainders."""
    target_total = max(0, int(round(target_total or 0)))
    keys = list(raw_values.keys())
    if not keys:
        return {}

    sanitized = {key: max(0.0, float(raw_values.get(key, 0) or 0)) for key in keys}
    current_total = sum(sanitized.values())
    if current_total <= 0:
        return {key: 0 for key in keys}

    scaled = {key: (sanitized[key] * target_total) / current_total for key in keys}
    allocation = {key: int(np.floor(value)) for key, value in scaled.items()}
    remaining = target_total - sum(allocation.values())
    if remaining > 0:
        order = sorted(
            keys,
            key=lambda key: (scaled[key] - allocation[key], sanitized[key], str(key)),
            reverse=True,
        )
        for key in order[:remaining]:
            allocation[key] += 1
    return allocation


def _get_marked_shooters_for_ocr_event(shooting_snapshots: list, alliance: str, event_time: float,
                                       max_gap_seconds: float = CENTER_SCORE_OCR_EVENT_WINDOW_SECONDS) -> list:
    """Return shooters marked during the plausible pre-OCR launch window."""
    if not shooting_snapshots:
        return []

    try:
        target_time = float(event_time)
    except (TypeError, ValueError):
        return []

    snapshot_times = [float(snapshot[0]) for snapshot in shooting_snapshots]
    if not snapshot_times:
        return []

    min_lag = max(0.0, float(CENTER_SCORE_OCR_ATTRIBUTION_MIN_LAG_SECONDS))
    preferred_lag = max(min_lag, float(CENTER_SCORE_OCR_ATTRIBUTION_PREFERRED_LAG_SECONDS))
    window_end = target_time - min_lag
    window_start = target_time - max(0.0, float(max_gap_seconds or 0.0))
    preferred_time = target_time - preferred_lag
    if window_end < window_start:
        window_start = window_end

    candidate_indices = [
        idx for idx, snapshot_time in enumerate(snapshot_times)
        if window_start <= snapshot_time <= window_end
    ]
    if not candidate_indices:
        insert_idx = bisect_right(snapshot_times, preferred_time)
        if insert_idx > 0:
            candidate_indices.append(insert_idx - 1)
        if insert_idx < len(shooting_snapshots):
            candidate_indices.append(insert_idx)
    if not candidate_indices:
        return []

    best_snapshot = None
    best_gap = None
    for idx in candidate_indices:
        snapshot = shooting_snapshots[idx]
        snapshot_time = float(snapshot[0])
        if (
            not (window_start <= snapshot_time <= window_end) and
            abs(snapshot_time - preferred_time) > max(0.0, float(max_gap_seconds or 0.0))
        ):
            continue
        gap = abs(snapshot_time - preferred_time)
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_snapshot = snapshot

    if best_snapshot is None:
        return []

    labels = best_snapshot[1] if str(alliance).strip().lower() == "blue" else best_snapshot[2]
    return [str(label).strip() for label in (labels or []) if str(label).strip()]


def _rank_ocr_attempt_indices(attempts: list, candidate_indices: list, event_time: float,
                              preferred_lag_seconds: float = CENTER_SCORE_OCR_ATTRIBUTION_PREFERRED_LAG_SECONDS) -> list:
    """Prefer made-hinted attempts whose timestamps fit the expected OCR lag best."""
    preferred_lag = max(0.0, float(preferred_lag_seconds or 0.0))
    return sorted(
        candidate_indices,
        key=lambda idx: (
            not bool(attempts[idx].get("made_hint")),
            abs((float(event_time) - float(attempts[idx].get("time", 0.0))) - preferred_lag),
            abs(float(event_time) - float(attempts[idx].get("time", 0.0))),
            -float(attempts[idx].get("time", 0.0)),
            str(attempts[idx].get("label", "")),
        )
    )


def _select_ocr_attempt_indices(attempts: list, event_time: float, target_count: int, allowed_labels: set = None,
                                min_lag_seconds: float = CENTER_SCORE_OCR_ATTRIBUTION_MIN_LAG_SECONDS,
                                max_lag_seconds: float = CENTER_SCORE_OCR_ATTRIBUTION_MAX_LAG_SECONDS,
                                relaxed_max_lag_seconds: float = CENTER_SCORE_OCR_ATTRIBUTION_RELAXED_MAX_LAG_SECONDS) -> list:
    """Pick unmatched prior attempts that could realistically explain an OCR score jump."""
    target_count = max(0, int(target_count or 0))
    if target_count <= 0 or not attempts:
        return []

    min_lag = max(0.0, float(min_lag_seconds or 0.0))
    max_lag = max(min_lag, float(max_lag_seconds or 0.0))
    relaxed_max = max(max_lag, float(relaxed_max_lag_seconds or 0.0))

    strict_candidates = []
    relaxed_candidates = []
    for idx, attempt in enumerate(attempts):
        if attempt.get("assigned"):
            continue
        label = str(attempt.get("label", "")).strip()
        if allowed_labels and label not in allowed_labels:
            continue

        lag = float(event_time) - float(attempt.get("time", 0.0))
        if lag < 0:
            continue
        if min_lag <= lag <= max_lag:
            strict_candidates.append(idx)
        elif lag <= relaxed_max:
            relaxed_candidates.append(idx)

    selected = []
    for pool in (strict_candidates, relaxed_candidates):
        for idx in _rank_ocr_attempt_indices(attempts, pool, event_time):
            if idx in selected:
                continue
            selected.append(idx)
            if len(selected) >= target_count:
                return selected
    return selected


def _apply_ocr_score_correction(stats: dict, center_score_ocr: dict, blue_robots: list, red_robots: list,
                                all_shot_events: list = None, shooting_snapshots: list = None,
                                manual_mode: bool = False, match_clock_ocr: dict = None) -> dict:
    """
    Ground per-robot makes to the center-score OCR timeline.

    Each positive OCR delta is treated as the hard budget for made shots at that
    moment, and those makes are only assigned to plausible prior attempts from
    the same alliance. This keeps robot totals within what the scoreboard says
    was actually possible, while still preserving the original attempt counts.
    """
    if not isinstance(center_score_ocr, dict):
        return stats

    configured_labels = []
    for label in _normalize_robot_numbers((blue_robots or []) + (red_robots or [])):
        if label not in configured_labels:
            configured_labels.append(label)

    corrected = {}
    baseline_labels = []
    for robot_label in list((stats or {}).keys()) + configured_labels:
        label = str(robot_label).strip()
        if label and label not in baseline_labels:
            baseline_labels.append(label)

    for robot_label in baseline_labels:
        robot_data = (stats or {}).get(robot_label, {}) if isinstance((stats or {}).get(robot_label, {}), dict) else {}
        by_period = {}
        original_periods = robot_data.get("by_period") or {}
        for period_name, _, _ in MATCH_PERIODS:
            period_data = original_periods.get(period_name, {}) if isinstance(original_periods, dict) else {}
            by_period[period_name] = {
                "attempts": int(period_data.get("attempts", 0) or 0),
                "made": int(period_data.get("made", 0) or 0),
            }
        corrected[robot_label] = {
            "attempts": int(robot_data.get("attempts", 0) or 0),
            "made": int(robot_data.get("made", 0) or 0),
            "by_period": by_period,
        }

    deduped_shot_events = (
        list(all_shot_events or [])
        if manual_mode else
        _dedupe_shot_events(all_shot_events or [], dedup_window_seconds=MULTI_CAMERA_SHOT_DEDUP_WINDOW_SECONDS)
    )

    for alliance, robot_labels in (
        ("blue", _normalize_robot_numbers(blue_robots)),
        ("red", _normalize_robot_numbers(red_robots)),
    ):
        ocr_data = center_score_ocr.get(alliance, {}) if isinstance(center_score_ocr.get(alliance, {}), dict) else {}
        ocr_total = ocr_data.get("scored")

        participating_labels = [label for label in robot_labels if label in corrected]
        if not participating_labels:
            continue

        sam_total = sum(corrected[label]["made"] for label in participating_labels)
        if ocr_total is not None:
            ocr_total = max(0, int(ocr_total))
        ocr_events = list(ocr_data.get("events") or [])
        positive_ocr_events = [
            (float(event_time), max(0, int(delta or 0)), raw_total)
            for event_time, delta, raw_total in ocr_events
            if max(0, int(delta or 0)) > 0
        ]
        if ocr_total is None and not positive_ocr_events:
            continue

        alliance_attempts = sorted(
            [
                {
                    "time": float(elapsed),
                    "label": str(robot_label).strip(),
                    "made_hint": bool(made),
                    "assigned": False,
                }
                for elapsed, robot_label, made in deduped_shot_events
                if str(robot_label).strip() in participating_labels
            ],
            key=lambda attempt: (float(attempt["time"]), str(attempt["label"])),
        )
        scaled_makes = {label: 0 for label in participating_labels}
        scaled_period_makes = {
            label: {period_name: 0 for period_name, _, _ in MATCH_PERIODS}
            for label in participating_labels
        }
        unmatched_ocr_total = 0
        ocr_only_total = 0
        allocated_total = 0

        for event_time, delta, _ in positive_ocr_events:
            remaining_budget = delta
            if ocr_total is not None:
                remaining_budget = min(remaining_budget, max(0, ocr_total - allocated_total))
            if remaining_budget <= 0:
                continue

            marked_shooters = list(dict.fromkeys(
                label for label in _get_marked_shooters_for_ocr_event(
                    shooting_snapshots,
                    alliance,
                    event_time,
                )
                if label in participating_labels
            ))
            allowed_labels = set(marked_shooters) if marked_shooters else None
            candidate_indices = _select_ocr_attempt_indices(
                alliance_attempts,
                event_time,
                remaining_budget,
                allowed_labels=allowed_labels,
            )

            for attempt_idx in candidate_indices:
                attempt = alliance_attempts[attempt_idx]
                attempt["assigned"] = True
                label = attempt["label"]
                scaled_makes[label] = scaled_makes.get(label, 0) + 1
                allocated_total += 1
                period_name = get_match_period_for_elapsed(float(attempt["time"]), match_clock_ocr=match_clock_ocr)
                if period_name in scaled_period_makes.get(label, {}):
                    scaled_period_makes[label][period_name] += 1

            remaining_budget -= len(candidate_indices)
            if remaining_budget > 0 and manual_mode and allowed_labels and len(allowed_labels) == 1:
                fallback_label = next(iter(allowed_labels))
                fallback_period = get_match_period_for_elapsed(float(event_time), match_clock_ocr=match_clock_ocr)
                scaled_makes[fallback_label] = scaled_makes.get(fallback_label, 0) + remaining_budget
                allocated_total += remaining_budget
                ocr_only_total += remaining_budget
                if fallback_period in scaled_period_makes.get(fallback_label, {}):
                    scaled_period_makes[fallback_label][fallback_period] += remaining_budget
                remaining_budget = 0

            if remaining_budget > 0:
                unmatched_ocr_total += remaining_budget

        print(
            f"[Center Score OCR] {alliance.title()} alliance grounding"
            f"{' (manual)' if manual_mode else ''}: "
            f"SAM made={sam_total}, OCR made={ocr_total}, attributed={allocated_total}, "
            f"unmatched_ocr={unmatched_ocr_total}, ocr_only={ocr_only_total}, grounded={scaled_makes}"
        )

        for label in participating_labels:
            robot_data = corrected[label]
            original_attempts = int(robot_data["attempts"])
            original_period_attempts = {
                period_name: int(robot_data["by_period"].get(period_name, {}).get("attempts", 0))
                for period_name, _, _ in MATCH_PERIODS
            }

            corrected_made = int(scaled_makes.get(label, 0))
            corrected_period_made = {
                period_name: int(scaled_period_makes.get(label, {}).get(period_name, 0))
                for period_name, _, _ in MATCH_PERIODS
            }
            remaining_period_made = max(0, corrected_made - sum(corrected_period_made.values()))
            if remaining_period_made > 0:
                period_weights = {
                    period_name: int(corrected_period_made.get(period_name, 0))
                    for period_name, _, _ in MATCH_PERIODS
                }
                if sum(period_weights.values()) <= 0:
                    period_weights = dict(original_period_attempts)
                if sum(period_weights.values()) <= 0:
                    period_weights = {period_name: 0 for period_name, _, _ in MATCH_PERIODS}
                    fallback_period = next(
                        (period_name for period_name, amount in corrected_period_made.items() if amount > 0),
                        MATCH_PERIODS[0][0],
                    )
                    period_weights[fallback_period] = remaining_period_made
                extra_period_made = _allocate_proportional_integers(period_weights, remaining_period_made)
                for period_name, amount in extra_period_made.items():
                    corrected_period_made[period_name] = corrected_period_made.get(period_name, 0) + int(amount)
            robot_data["made"] = corrected_made
            robot_data["attempts"] = max(original_attempts, corrected_made)

            for period_name, _, _ in MATCH_PERIODS:
                period_made = int(corrected_period_made.get(period_name, 0))
                robot_data["by_period"][period_name] = {
                    "attempts": max(original_period_attempts[period_name], period_made),
                    "made": period_made,
                }

    return corrected


def _normalize_highlight_ball_robot(highlight_ball_robot: str = None) -> str:
    """Return the selected robot label for ball highlighting, or empty for no filter."""
    label = str(highlight_ball_robot or "").strip()
    if not label or label == BALL_HIGHLIGHT_ALL_OPTION:
        return ""
    return label


def _summarize_ocr_raw_texts(raw_texts: list, max_items: int = 3, max_length: int = 28) -> str:
    """Create a compact one-line summary of the latest OCR raw strings."""
    cleaned = []
    seen = set()
    for item in list(raw_texts or []):
        text = _clean_text(str(item or ""))
        if not text or text in seen:
            continue
        cleaned.append(text)
        seen.add(text)
        if len(cleaned) >= max_items:
            break
    if not cleaned:
        return "no raw read"
    summary = " | ".join(cleaned)
    if len(summary) > max_length:
        return summary[:max_length - 3] + "..."
    return summary


def _build_center_ocr_debug_region_specs(frame_width: int, frame_height: int,
                                         center_score_ocr_tracker: CenterScoreOCRTracker = None,
                                         center_match_clock_ocr_tracker: CenterMatchClockOCRTracker = None) -> list:
    """Describe the center-camera OCR debug badges to render on top of the video."""
    specs = []
    if frame_width <= 0 or frame_height <= 0:
        return specs

    if center_score_ocr_tracker is not None:
        score_colors = {"blue": (70, 130, 255), "red": (255, 95, 95)}
        for alliance in ("blue", "red"):
            rect = CENTER_SCORE_COUNTER_RECTS.get(alliance)
            if rect is None:
                continue
            scaled_rect = _scale_ref_rect(rect, frame_width, frame_height)
            debug = center_score_ocr_tracker.latest_debug.get(alliance, {}) if hasattr(center_score_ocr_tracker, "latest_debug") else {}
            latest_value = debug.get("value")
            confirmed_value = debug.get("confirmed")
            if debug.get("disabled"):
                read_text = "OCR disabled"
            elif debug.get("error"):
                read_text = "OCR error"
            elif latest_value is None:
                read_text = "read --"
            else:
                read_text = f"read {int(latest_value)}"
            if confirmed_value is not None:
                read_text += f" | conf {int(confirmed_value)}"
            raw_summary = _summarize_ocr_raw_texts(debug.get("raw_texts"))
            specs.append({
                "rect": scaled_rect,
                "title": f"{alliance.title()} Score OCR",
                "value_line": read_text,
                "raw_line": raw_summary,
                "color": score_colors.get(alliance, (255, 255, 255)),
                "align": "left" if alliance == "blue" else "right",
            })

    if center_match_clock_ocr_tracker is not None:
        scaled_rect = _scale_ref_rect(CENTER_MATCH_CLOCK_RECT, frame_width, frame_height)
        debug = center_match_clock_ocr_tracker.latest_debug if hasattr(center_match_clock_ocr_tracker, "latest_debug") else {}
        latest_value = debug.get("value")
        confirmed_value = debug.get("confirmed")
        if debug.get("disabled"):
            read_text = "OCR disabled"
        elif debug.get("error"):
            read_text = "OCR error"
        elif latest_value is None:
            read_text = "read --:--"
        else:
            read_text = f"read {_format_clock_seconds(latest_value)}"
        if confirmed_value is not None:
            read_text += f" | conf {_format_clock_seconds(confirmed_value)}"
        raw_summary = _summarize_ocr_raw_texts(debug.get("raw_texts"))
        specs.append({
            "rect": scaled_rect,
            "title": "Match Clock OCR",
            "value_line": read_text,
            "raw_line": raw_summary,
            "color": (255, 220, 110),
            "align": "center",
        })

    return specs


def draw_center_ocr_debug_overlay(frame: Image.Image, center_score_ocr_tracker: CenterScoreOCRTracker = None,
                                  center_match_clock_ocr_tracker: CenterMatchClockOCRTracker = None) -> Image.Image:
    """Draw persistent OCR debug badges near the center-camera OCR regions."""
    if frame is None:
        return frame

    width, height = frame.size
    specs = _build_center_ocr_debug_region_specs(
        width,
        height,
        center_score_ocr_tracker=center_score_ocr_tracker,
        center_match_clock_ocr_tracker=center_match_clock_ocr_tracker,
    )
    if not specs:
        return frame

    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    title_font = get_font(15)
    body_font = get_font(13)
    padding_x = 8
    padding_y = 6
    line_gap = 3
    region_pad = 6

    for spec in specs:
        x1, y1, x2, y2 = spec["rect"]
        color = spec.get("color", (255, 255, 255))
        draw.rectangle([x1, y1, x2, y2], outline=color + (220,), width=2)

        title = str(spec.get("title", "OCR"))
        value_line = str(spec.get("value_line", ""))
        raw_line = f"raw {str(spec.get('raw_line', ''))}"
        lines = [title, value_line, raw_line]

        line_heights = []
        max_text_width = 0
        for idx, line in enumerate(lines):
            font = title_font if idx == 0 else body_font
            bbox = draw.textbbox((0, 0), line, font=font)
            max_text_width = max(max_text_width, bbox[2] - bbox[0])
            line_heights.append(bbox[3] - bbox[1])
        box_width = max_text_width + (padding_x * 2)
        box_height = sum(line_heights) + (padding_y * 2) + (line_gap * (len(lines) - 1))

        align = spec.get("align", "left")
        if align == "right":
            box_x1 = x2 - box_width
        elif align == "center":
            box_x1 = int(round(((x1 + x2) / 2.0) - (box_width / 2.0)))
        else:
            box_x1 = x1
        box_x1 = max(8, min(width - box_width - 8, box_x1))

        preferred_y1 = y2 + region_pad
        if preferred_y1 + box_height > height - 8:
            preferred_y1 = y1 - box_height - region_pad
        box_y1 = max(8, min(height - box_height - 8, preferred_y1))
        box_x2 = box_x1 + box_width
        box_y2 = box_y1 + box_height

        draw.rounded_rectangle(
            [box_x1, box_y1, box_x2, box_y2],
            radius=10,
            fill=(16, 18, 24, 180),
            outline=color + (235,),
            width=2,
        )

        cursor_y = box_y1 + padding_y
        for idx, line in enumerate(lines):
            font = title_font if idx == 0 else body_font
            fill = color + (255,) if idx == 0 else (255, 255, 255, 235)
            draw.text((box_x1 + padding_x, cursor_y), line, fill=fill, font=font)
            cursor_y += line_heights[idx] + line_gap

    return Image.alpha_composite(frame.convert("RGBA"), overlay).convert("RGB")


def _build_ball_highlight_choices(blue_robots: list = None, red_robots: list = None) -> list:
    """Build dropdown choices for the ball-highlight export filter."""
    choices = [BALL_HIGHLIGHT_ALL_OPTION]
    seen = {BALL_HIGHLIGHT_ALL_OPTION}
    for label in _normalize_robot_numbers((blue_robots or []) + (red_robots or [])):
        if label not in seen:
            choices.append(label)
            seen.add(label)
    return choices


def _update_ball_highlight_dropdown(blue_robot_1: str = "", blue_robot_2: str = "", blue_robot_3: str = "",
                                    red_robot_1: str = "", red_robot_2: str = "", red_robot_3: str = "",
                                    current_value: str = None):
    """Keep the ball-highlight dropdown synced with the configured robot numbers."""
    blue_robots = [blue_robot_1, blue_robot_2, blue_robot_3]
    red_robots = [red_robot_1, red_robot_2, red_robot_3]
    choices = _build_ball_highlight_choices(blue_robots, red_robots)
    selected = _normalize_highlight_ball_robot(current_value)
    value = selected if selected in choices else BALL_HIGHLIGHT_ALL_OPTION
    return gr.update(choices=choices, value=value)


def _get_center_robot_scan_alliance(bbox: tuple, frame_width: int) -> str:
    """
    Split the center camera into scan zones.

    Left third scans blue only, right third scans red only, and the middle third
    can still be either alliance.
    """
    if bbox is None or frame_width <= 0:
        return None
    x1, _, x2, _ = bbox
    center_x = (x1 + x2) / 2.0
    third_width = frame_width / 3.0
    if center_x < third_width:
        return "blue"
    if center_x >= (2.0 * third_width):
        return "red"
    return None


def _get_allowed_robot_numbers_for_detection(camera_side: str, bbox: tuple, mask_color: str,
                                             frame_width: int, blue_robots: list = None,
                                             red_robots: list = None) -> list:
    """
    Return the only team numbers this detection is allowed to become.

    Side cameras stay alliance-locked. For the center camera, the outer thirds
    only scan the alliance on that side of the field. The mask color is also a
    hard constraint: blue-mask detections cannot become red teams and vice versa.
    """
    blue_labels = _normalize_robot_numbers(blue_robots)
    red_labels = _normalize_robot_numbers(red_robots)
    all_labels = blue_labels + red_labels
    if not all_labels:
        return []

    side_name = str(camera_side).strip().lower()
    field_allowed = set(all_labels)
    if side_name == "blue":
        field_allowed = set(blue_labels)
    elif side_name == "red":
        field_allowed = set(red_labels)
    elif side_name == "center":
        center_alliance = _get_center_robot_scan_alliance(bbox, frame_width)
        if center_alliance == "blue":
            field_allowed = set(blue_labels)
        elif center_alliance == "red":
            field_allowed = set(red_labels)

    mask_color_name = str(mask_color).strip().lower()
    if mask_color_name == "blue":
        color_allowed = set(blue_labels)
    elif mask_color_name == "red":
        color_allowed = set(red_labels)
    else:
        color_allowed = set(all_labels)

    allowed = field_allowed & color_allowed
    return [robot for robot in all_labels if robot in allowed]


def rgb_to_hex(rgb: tuple) -> str:
    """Convert RGB tuple to hex color string."""
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])


# Legacy PATH_COLORS kept for backwards compatibility
PATH_COLORS = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (255, 128, 0),    # Orange
    (128, 0, 255),    # Purple
]


def parse_json(json_output: str) -> str:
    """Parse JSON output, removing markdown fencing if present."""
    import re
    
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])
            json_output = json_output.split("```")[0]
            break
    
    # Fix malformed JSON with duplicate "label": "label": pattern
    # Gemini sometimes outputs: "label": "label": "value"
    # This fixes it to: "label": "value"
    json_output = re.sub(r'"label":\s*"label":\s*', '"label": ', json_output)
    
    return json_output


# Match time periods (name, start_seconds, end_seconds)
MATCH_PERIODS = [
    ("Auto", 0, 15),
    ("Transition", 15, 25),
    ("Shift 1", 25, 50),
    ("Shift 2", 50, 75),
    ("Shift 3", 75, 100),
    ("Shift 4", 100, 125),
    ("Endgame", 125, float('inf'))
]


def get_match_period(elapsed_seconds: float) -> str:
    """Get the match period name for a given elapsed time."""
    for name, start, end in MATCH_PERIODS:
        if start <= elapsed_seconds < end:
            return name
    return "Endgame"  # Default to endgame for any time beyond defined periods


def _get_match_period_from_clock_seconds(clock_seconds: int, teleop_started: bool = False) -> str:
    """Map an OCR-read match clock value to the program's period buckets."""
    try:
        seconds = int(clock_seconds)
    except (TypeError, ValueError):
        return None

    if seconds < 0:
        return None

    if not teleop_started:
        return "Auto"

    if seconds > 130:
        return "Transition"
    if seconds > 105:
        return "Shift 1"
    if seconds > 80:
        return "Shift 2"
    if seconds > 55:
        return "Shift 3"
    if seconds >= 30:
        return "Shift 4"
    return "Endgame"


def _resolve_match_clock_for_elapsed(elapsed_seconds: float, match_clock_ocr: dict = None):
    """
    Return the OCR-derived clock state nearest this elapsed time.

    Prefers the most recent observation at or before the requested time. If no
    prior observation exists, the earliest future observation is used only when
    it is close enough to plausibly describe the same on-screen clock state.
    """
    if not isinstance(match_clock_ocr, dict):
        return None, False

    observations = list(match_clock_ocr.get("observations") or [])
    if not observations:
        return None, False

    try:
        target = float(elapsed_seconds)
    except (TypeError, ValueError):
        return None, False

    observation_times = [float(item[0]) for item in observations]
    idx = bisect_right(observation_times, target) - 1

    if idx >= 0:
        observed_elapsed, observed_clock = observations[idx]
        if abs(float(observed_elapsed) - target) <= MATCH_CLOCK_LOOKUP_MAX_GAP_SECONDS:
            teleop_started = any(int(clock) > 20 for _, clock in observations[:idx + 1])
            return int(observed_clock), teleop_started

    next_idx = idx + 1
    if 0 <= next_idx < len(observations):
        observed_elapsed, observed_clock = observations[next_idx]
        if abs(float(observed_elapsed) - target) <= MATCH_CLOCK_LOOKUP_MAX_GAP_SECONDS:
            teleop_started = any(int(clock) > 20 for _, clock in observations[:next_idx + 1])
            return int(observed_clock), teleop_started

    return None, False


def get_match_period_for_elapsed(elapsed_seconds: float, match_clock_ocr: dict = None) -> str:
    """
    Determine the match period for an elapsed timestamp.

    Uses OCR-derived center-clock observations when available, falling back to
    the original elapsed-time bucket mapping otherwise.
    """
    clock_seconds, teleop_started = _resolve_match_clock_for_elapsed(elapsed_seconds, match_clock_ocr)
    clock_period = _get_match_period_from_clock_seconds(clock_seconds, teleop_started=teleop_started)
    if clock_period:
        return clock_period
    return get_match_period(elapsed_seconds)


class FerryTracker:
    """
    Track when robots ferry fuel by detecting line crossings on the 2D map.
    
    A ferry is counted when a robot:
    1. Leaves its alliance's home side (crosses the ferry line going out)
    2. Returns to its home side (crosses the ferry line coming back)
    3. Shoots
    
    Ferry lines are defined in unrotated map coordinates (574x961 PNG).
    Since transform_to_map() returns rotated coords (961x574), and the
    rotation is (x_orig, y_orig) -> (y_orig, 574 - x_orig), the unrotated
    y-coordinate equals the rotated map_x. So ferry thresholds are checked
    against map_x from transform_to_map().
    
    Ferry lines (unrotated map y-coordinates):
    - Red Ferry Line: y = 270  (rotated map_x = 270, red home is map_x < 270)
    - Blue Ferry Line: y = 694 (rotated map_x = 694, blue home is map_x > 694)
    """
    
    # Ferry line thresholds in rotated map x-coordinates
    # These correspond to horizontal lines on the unrotated map PNG
    RED_FERRY_LINE_MAP_X = 270   # Unrotated map y=270
    BLUE_FERRY_LINE_MAP_X = 694  # Unrotated map y=694
    
    # Hysteresis buffer in map pixels to prevent jitter near the line
    HYSTERESIS_PX = 20
    
    def __init__(self, blue_robots: list = None, red_robots: list = None):
        """
        Initialize ferry tracker.
        
        Args:
            blue_robots: List of blue alliance team numbers
            red_robots: List of red alliance team numbers
        """
        self.blue_robots = [str(r).strip() for r in (blue_robots or []) if r]
        self.red_robots = [str(r).strip() for r in (red_robots or []) if r]
        
        # State machine for each robot: 'idle', 'crossed_out', 'ready_to_ferry'
        # {robot_label: {'state': str, 'in_home': bool|None, 'ferry_count': int}}
        self.robot_states = {}
    
    def _get_robot_state(self, label: str) -> dict:
        """Get or create state for a robot."""
        if label not in self.robot_states:
            self.robot_states[label] = {
                'state': 'idle',
                'in_home': None,  # None = unknown, True/False = last committed side
                'ferry_count': 0
            }
        return self.robot_states[label]
    
    def _get_alliance(self, robot_label: str) -> str:
        """
        Get the alliance for a robot.
        
        Returns:
            'blue', 'red', or None if unknown
        """
        label = str(robot_label).strip()
        if label in self.blue_robots:
            return 'blue'
        elif label in self.red_robots:
            return 'red'
        return None
    
    def _is_in_home(self, map_x: float, alliance: str) -> bool:
        """
        Check if a robot is on its home side of the ferry line with hysteresis.
        
        Returns True/False only if the robot is clearly past the threshold
        (beyond the hysteresis buffer). Returns None if in the buffer zone.
        """
        if alliance == 'blue':
            # Blue home is map_x > BLUE line
            if map_x > self.BLUE_FERRY_LINE_MAP_X + self.HYSTERESIS_PX:
                return True
            elif map_x < self.BLUE_FERRY_LINE_MAP_X - self.HYSTERESIS_PX:
                return False
            return None  # In buffer zone, no change
        elif alliance == 'red':
            # Red home is map_x < RED line
            if map_x < self.RED_FERRY_LINE_MAP_X - self.HYSTERESIS_PX:
                return True
            elif map_x > self.RED_FERRY_LINE_MAP_X + self.HYSTERESIS_PX:
                return False
            return None  # In buffer zone, no change
        return None
    
    def update_position(self, robot_label: str, map_x: float, map_y: float):
        """
        Update robot position and detect line crossings for ferry tracking.
        Uses rotated map coordinates from transform_to_map().
        
        Args:
            robot_label: The robot's team number
            map_x: The robot's x-coordinate on the rotated map (961x574)
            map_y: The robot's y-coordinate on the rotated map (961x574)
        """
        alliance = self._get_alliance(robot_label)
        if alliance is None:
            return
        
        state = self._get_robot_state(robot_label)
        in_home = self._is_in_home(map_x, alliance)
        
        # If in hysteresis buffer zone, keep previous state (no transition)
        if in_home is None:
            return
        
        was_in_home = state['in_home']
        
        if was_in_home is not None:
            if was_in_home and not in_home:
                # Robot left its home side (going out to collect fuel)
                state['state'] = 'crossed_out'
            elif not was_in_home and in_home:
                # Robot re-entered its home side (returning with fuel)
                if state['state'] == 'crossed_out':
                    state['state'] = 'ready_to_ferry'
                # If idle and entering home, stay idle (incomplete cycle)
        
        state['in_home'] = in_home
    
    def on_shot(self, robot_label: str) -> bool:
        """
        Called when a robot shoots. Returns True if this was a ferry.
        
        Args:
            robot_label: The robot's team number
            
        Returns:
            True if this shot completes a ferry cycle, False otherwise
        """
        state = self._get_robot_state(robot_label)
        
        if state['state'] == 'ready_to_ferry':
            state['ferry_count'] += 1
            state['state'] = 'idle'
            return True
        else:
            # Shot without completing ferry cycle - reset state
            state['state'] = 'idle'
            return False
    
    def get_ferry_count(self, robot_label: str) -> int:
        """Get the ferry count for a robot."""
        state = self._get_robot_state(robot_label)
        return state['ferry_count']
    
    def get_all_ferry_counts(self) -> dict:
        """Get ferry counts for all tracked robots."""
        return {label: data['ferry_count'] for label, data in self.robot_states.items()}


class DisabledTracker:
    """
    Track whether robots are disabled by detecting lack of movement.
    
    Disabled status:
    - "Full": Robot didn't move for 80%+ of the match
    - "Partially": Robot didn't move for 20+ consecutive seconds at some point
    - "None": Robot was moving normally
    
    Uses map coordinates to detect movement with tolerance for micro-movements
    from imperfect bounding box detection.
    """
    
    # Movement threshold in map pixels - movements smaller than this are ignored
    MOVEMENT_THRESHOLD = 8  # pixels on map (reduced from 15 to be less sensitive)
    
    # Thresholds for disabled detection
    FULL_DISABLED_PERCENT = 0.90  # 90% of video stationary = fully disabled
    PARTIAL_DISABLED_SECONDS = 20  # 20 consecutive seconds = partially disabled
    
    def __init__(self, fps: float = 3.0):
        """
        Initialize disabled tracker.
        
        Args:
            fps: Frame rate at which robot positions are sampled
        """
        self.fps = fps
        
        # Track each robot's position history and movement status
        # {robot_label: {'positions': [(map_x, map_y, frame_num), ...], 
        #                'stationary_frames': int, 'total_frames': int,
        #                'current_stationary_streak': int, 'max_stationary_streak': int}}
        self.robot_data = {}
    
    def _get_robot_data(self, label: str) -> dict:
        """Get or create tracking data for a robot."""
        if label not in self.robot_data:
            self.robot_data[label] = {
                'last_pos': None,
                'stationary_frames': 0,
                'total_frames': 0,
                'current_stationary_streak': 0,
                'max_stationary_streak': 0
            }
        return self.robot_data[label]
    
    def update_position(self, robot_label: str, map_x: float, map_y: float):
        """
        Update robot position and track movement.
        
        Args:
            robot_label: The robot's team number
            map_x: Robot's x-coordinate on the map
            map_y: Robot's y-coordinate on the map
        """
        data = self._get_robot_data(robot_label)
        data['total_frames'] += 1
        
        if data['last_pos'] is not None:
            last_x, last_y = data['last_pos']
            
            # Calculate movement distance
            distance = ((map_x - last_x) ** 2 + (map_y - last_y) ** 2) ** 0.5
            
            if distance < self.MOVEMENT_THRESHOLD:
                # Robot is stationary
                data['stationary_frames'] += 1
                data['current_stationary_streak'] += 1
                
                # Update max streak
                if data['current_stationary_streak'] > data['max_stationary_streak']:
                    data['max_stationary_streak'] = data['current_stationary_streak']
            else:
                # Robot moved - reset current streak
                data['current_stationary_streak'] = 0
        
        data['last_pos'] = (map_x, map_y)
    
    def get_disabled_status(self, robot_label: str) -> tuple:
        """
        Get the disabled status for a robot.
        
        Returns:
            (status, max_stationary_seconds) where:
            - status: "Full", "Partially", or "None"
            - max_stationary_seconds: Longest consecutive stationary period in seconds
        """
        data = self._get_robot_data(robot_label)
        
        if data['total_frames'] == 0:
            return ("None", 0)
        
        # Calculate stationary percentage
        stationary_percent = data['stationary_frames'] / data['total_frames']
        
        # Calculate max stationary seconds
        max_stationary_seconds = data['max_stationary_streak'] / self.fps
        
        # Determine status (Full trumps Partially)
        if stationary_percent >= self.FULL_DISABLED_PERCENT:
            return ("Full", max_stationary_seconds)
        elif max_stationary_seconds >= self.PARTIAL_DISABLED_SECONDS:
            return ("Partially", max_stationary_seconds)
        else:
            return ("None", max_stationary_seconds)
    
    def get_all_disabled_statuses(self) -> dict:
        """
        Get disabled statuses for all tracked robots.
        
        Returns:
            Dict of {robot_label: (status, max_stationary_seconds)}
        """
        return {label: self.get_disabled_status(label) for label in self.robot_data}


class BallTracker:
    """
    Track individual balls across frames, detect shots, and attribute to robots.
    
    A ball is considered "shot" if it moves UP (negative y change) by at least 10 pixels
    in approximately 0.034 seconds (1 frame at 30fps).
    """
    
    def __init__(self, fps: float = 30.0, shot_label_duration: float = 2.0, 
                 min_upward_pixels: int = 10, max_matching_distance: int = 50,
                 max_frames_lost: int = 30, camera_side: str = "blue",
                 blue_robots: list = None, red_robots: list = None,
                 start_seconds: float = 0.0, ferry_tracker: FerryTracker = None,
                 frame_width: int = 0, frame_height: int = 0):
        """
        Initialize ball tracker.
        
        Args:
            fps: Frame rate for ball detection
            shot_label_duration: How long to keep robot label on ball after shot (seconds)
            min_upward_pixels: Minimum upward movement to count as shot (pixels)
            max_matching_distance: Maximum distance to match balls between frames
            max_frames_lost: How many frames to keep a ball in memory after losing it
            camera_side: "blue", "red", or "center" - determines which alliance's shots are tracked
            blue_robots: List of blue alliance team numbers
            red_robots: List of red alliance team numbers
            start_seconds: Start time of video processing (for period calculation)
            ferry_tracker: FerryTracker instance for counting ferried fuel
            frame_width: Actual video frame width (for polygon scaling)
            frame_height: Actual video frame height (for polygon scaling)
        """
        self.fps = max(1.0, float(fps))
        # Lower tracking FPS means each observation is farther apart in time, so widen
        # per-frame spatial tolerances and gravity integration proportionally.
        self.frame_step_scale = max(1.0, BALL_TRACKER_BASELINE_FPS / self.fps)
        self.shot_label_duration = shot_label_duration
        self.min_upward_pixels = max(4.0, float(min_upward_pixels) * self.frame_step_scale)
        self.max_matching_distance = min(
            180.0,
            max(30.0, float(max_matching_distance) * self.frame_step_scale),
        )
        self.max_frames_lost = max_frames_lost
        self.camera_side = camera_side
        self.blue_robots = [str(r).strip() for r in (blue_robots or []) if r]
        self.red_robots = [str(r).strip() for r in (red_robots or []) if r]
        self.start_seconds = start_seconds  # Video start time for period calculation
        self.ferry_tracker = ferry_tracker  # Reference to ferry tracker
        self.frame_width = frame_width  # Stored for edge-robot detection
        self.frame_height = frame_height  # Stored for edge-robot detection
        self.possession_memory_frames = max(2, int(round(self.fps * 0.30)))
        self.launch_anchor_y_ratio = 0.35
        self.launch_zone_y_ratio = 0.65
        self.launch_zone_x_ratio = 0.15
        self.launch_anchor_max_distance = min(325.0, 175.0 * self.frame_step_scale)
        self.min_launch_rise_pixels = max(4, int(round(self.min_upward_pixels * 0.5)))
        self.min_launch_window_gain = max(8, int(round(self.min_upward_pixels * 1.5)))
        self.motion_history_size = 4
        self.trajectory_gravity = 0.5 * (self.frame_step_scale ** 2)
        self.prediction_horizon_frames = max(18, int(round(self.fps * 2.0)))
        self.prediction_substeps = max(3, int(round(3 * self.frame_step_scale)))
        self.prediction_bounds_padding = 40.0
        self.min_shot_progress_pixels = max(12.0, float(self.min_upward_pixels) * 2.0)
        self.min_predicted_make_stable_frames = max(2, int(round(self.fps * 0.10)))
        self.min_predicted_make_progress_pixels = max(24.0, self.min_shot_progress_pixels * 1.5)
        
        # Track balls: {ball_id: {'pos': (x, y, r), 'prev_pos': (x, y, r), 'shot_by': robot_label, ...}}
        self.tracked_balls = {}
        
        # Lost balls (temporarily unmatched): {ball_id: {'data': ball_data, 'frames_lost': int, 'predicted_pos': (x, y)}}
        self.lost_balls = {}
        
        self.next_ball_id = 0
        self.current_frame = 0
        
        # Store robot bounding boxes from nearest Gemini/manual detection.
        # Format: [(y1, x1, y2, x2, label, is_shooting, has_explicit_shooting), ...]
        self.robot_bboxes = []
        
        # Shot statistics: {robot_label: {'attempts': 0, 'made': 0, 'by_period': {...}}}
        self.robot_stats = {}
        
        # Shot event log for cross-camera deduplication: [(elapsed_seconds, robot_label, made_bool), ...]
        self.shot_events = []

        # Per-frame shooting state sampled at ball-tracking FPS.
        # Format: [(elapsed_seconds, active_blue_labels, active_red_labels), ...]
        self.shooting_snapshots = []
        
        # Set up goal polygons based on camera type, scaled to actual resolution
        if camera_side == "center":
            # Center camera reference (cropped from composite): 1918x709
            ref_w, ref_h = 1918, 709
            # Blue side goal: rect (470,197) -> (637,287)
            blue_goal = [(470, 197), (637, 197), (637, 287), (470, 287)]
            # Red side goal: rect (1301,197) -> (1469,299)
            red_goal = [(1301, 197), (1469, 197), (1469, 299), (1301, 299)]
            self.goal_polygons = [
                self._scale_polygon(blue_goal, ref_w, ref_h, frame_width, frame_height),
                self._scale_polygon(red_goal, ref_w, ref_h, frame_width, frame_height)
            ]
        elif camera_side == "blue":
            # Blue side camera reference (cropped from composite): 940x339
            ref_w, ref_h = 940, 339
            # Goal: rect (435,152) -> (588,224)
            goal = [(435, 152), (588, 152), (588, 224), (435, 224)]
            self.goal_polygons = [
                self._scale_polygon(goal, ref_w, ref_h, frame_width, frame_height)
            ]
        else:
            # Red side camera reference (cropped from composite): 940x339
            ref_w, ref_h = 940, 339
            # Goal: rect (221,149) -> (387,232)
            goal = [(221, 149), (387, 149), (387, 232), (221, 232)]
            self.goal_polygons = [
                self._scale_polygon(goal, ref_w, ref_h, frame_width, frame_height)
            ]

    def _get_robot_stats(self, label: str) -> dict:
        if label not in self.robot_stats:
            # Initialize with period breakdown
            by_period = {name: {'attempts': 0, 'made': 0} for name, _, _ in MATCH_PERIODS}
            self.robot_stats[label] = {'attempts': 0, 'made': 0, 'by_period': by_period}
        return self.robot_stats[label]
    
    def _get_elapsed_seconds(self) -> float:
        """Get elapsed match time based on current frame."""
        return self.start_seconds + (self.current_frame / self.fps)
    
    def _record_shot(self, robot_label: str, made: bool, event_elapsed: float = None):
        """
        Record a shot attempt for a robot, tracking both total and by period.
        
        Args:
            robot_label: The robot's team number
            made: True if shot was made, False if missed
            event_elapsed: Timestamp of the actual launch event when known
        """
        stats = self._get_robot_stats(robot_label)
        try:
            elapsed = float(event_elapsed) if event_elapsed is not None else self._get_elapsed_seconds()
        except (TypeError, ValueError):
            elapsed = self._get_elapsed_seconds()
        period = get_match_period(elapsed)
        
        # Log shot event for cross-camera deduplication
        self.shot_events.append((elapsed, robot_label, made))
        
        # Update totals
        stats['attempts'] += 1
        if made:
            stats['made'] += 1
        
        # Update period stats
        if period in stats['by_period']:
            stats['by_period'][period]['attempts'] += 1
            if made:
                stats['by_period'][period]['made'] += 1
        
        # Notify ferry tracker that this robot shot
        if self.ferry_tracker:
            self.ferry_tracker.on_shot(robot_label)

    @staticmethod
    def _scale_polygon(polygon, ref_w, ref_h, actual_w, actual_h):
        """Scale polygon coordinates from reference resolution to actual resolution."""
        if actual_w <= 0 or actual_h <= 0 or ref_w <= 0 or ref_h <= 0:
            return polygon  # No scaling if dimensions unknown
        sx = actual_w / ref_w
        sy = actual_h / ref_h
        return [(x * sx, y * sy) for x, y in polygon]

    @staticmethod
    def _polygon_center(polygon) -> tuple:
        """Return the centroid-like average of polygon vertices."""
        if not polygon:
            return None
        xs = [pt[0] for pt in polygon]
        ys = [pt[1] for pt in polygon]
        return sum(xs) / len(xs), sum(ys) / len(ys)
    
    def _get_unshifted_point(self, x, y):
        """Un-shift a point from video coords back to base reference coords if tracking center camera."""
        if self.camera_side == "center" and self.frame_width > 0 and self.frame_height > 0:
            return _calibration_transform_point(x, y, self.frame_width, self.frame_height, inverse=True)
        return x, y


    
    def _is_in_goal(self, x, y):
        """Check if point is in any goal polygon."""
        ux, uy = self._get_unshifted_point(x, y)
        for polygon in self.goal_polygons:
            if self._is_point_in_polygon(ux, uy, polygon):
                return True
        return False
    
    def _is_point_in_polygon(self, x, y, polygon):
        """Check if point (x,y) is inside polygon using ray casting."""
        n = len(polygon)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            
            # Check if the ray from (x,y) going right crosses this edge
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            
            j = i
        
        return inside
    
    def update_robot_bboxes(self, bboxes_json: str, frame_width: int, frame_height: int):
        """
        Update robot bounding boxes from Gemini detection.
        
        Args:
            bboxes_json: JSON string with robot detections
            frame_width: Frame width for coordinate conversion
            frame_height: Frame height for coordinate conversion
        """
        self.robot_bboxes = []
        try:
            bboxes = json.loads(parse_json(bboxes_json))
            for bbox in bboxes:
                label = bbox.get('label', 'Unknown')
                box = bbox.get('box_2d', [])
                if len(box) >= 4:
                    # Convert from 0-1000 normalized to pixel coordinates
                    y1 = float(box[0]) / 1000 * frame_height
                    x1 = float(box[1]) / 1000 * frame_width
                    y2 = float(box[2]) / 1000 * frame_height
                    x2 = float(box[3]) / 1000 * frame_width
                    has_explicit_shooting = 'shooting' in bbox
                    is_shooting = bool(bbox.get('shooting', True))
                    self.robot_bboxes.append((y1, x1, y2, x2, label, is_shooting, has_explicit_shooting))
        except Exception as e:
            print(f"Error parsing robot bboxes: {e}")

    def _is_robot_marked_shooting(self, robot_label: str) -> bool:
        """
        Manual center tracking can explicitly mark which robots are shooting.
        If that metadata is absent, fall back to the legacy permissive behavior.
        """
        label = str(robot_label).strip()
        matched = [bbox for bbox in self.robot_bboxes if len(bbox) >= 5 and str(bbox[4]).strip() == label]
        if not matched:
            return True
        explicit_matches = [bbox for bbox in matched if len(bbox) >= 7 and bool(bbox[6])]
        if not explicit_matches:
            return True
        return any(bool(bbox[5]) for bbox in explicit_matches)

    def any_robot_marked_shooting(self) -> bool:
        """Return True if the current frame has at least one robot allowed to shoot."""
        if not self.robot_bboxes:
            return True
        explicit_matches = [bbox for bbox in self.robot_bboxes if len(bbox) >= 7 and bool(bbox[6])]
        if not explicit_matches:
            return True
        return any(bool(bbox[5]) for bbox in explicit_matches)

    def _get_active_shooting_labels(self) -> dict:
        """Return the currently marked shooters grouped by alliance."""
        active = {"blue": [], "red": []}

        for robot_bbox in self.robot_bboxes:
            if len(robot_bbox) < 5:
                continue
            label = str(robot_bbox[4]).strip()
            if not label:
                continue
            has_explicit_shooting = len(robot_bbox) >= 7 and bool(robot_bbox[6])
            if not has_explicit_shooting or not bool(robot_bbox[5]):
                continue
            if label in self.blue_robots:
                active["blue"].append(label)
            elif label in self.red_robots:
                active["red"].append(label)

        return {
            alliance: tuple(sorted(set(labels)))
            for alliance, labels in active.items()
        }

    def _record_shooting_snapshot(self):
        """Sample the current set of marked shooters for later OCR reconciliation."""
        active = self._get_active_shooting_labels()
        self.shooting_snapshots.append((
            float(self._get_elapsed_seconds()),
            active["blue"],
            active["red"],
        ))
    
    def _is_robot_in_camera_alliance(self, robot_label: str) -> bool:
        """
        Check if a robot belongs to the same alliance as the camera.
        
        Blue camera should only track shots from blue robots.
        Red camera should only track shots from red robots.
        
        Args:
            robot_label: The robot's team number/label
            
        Returns:
            True if robot is in the camera's alliance, False otherwise
        """
        label = str(robot_label).strip()
        
        if self.camera_side == "blue":
            return label in self.blue_robots
        elif self.camera_side == "red":
            return label in self.red_robots
        else:
            # Unknown camera side - allow all robots
            return True

    def _get_center_ball_alliance(self, ball_x: int) -> str:
        """Determine which alliance owns the current half of the center camera."""
        if self.camera_side != "center" or self.frame_width <= 0:
            return None
        midpoint = self.frame_width / 2.0
        return "blue" if ball_x < midpoint else "red"

    def _is_robot_eligible_for_ball(self, robot_label: str, ball_x: int) -> bool:
        """
        Filter candidate robots for attribution.

        Side cameras are alliance-specific. For the center camera, the ball's x
        position determines which alliance can own the shot: left half is blue,
        right half is red.
        """
        label = str(robot_label).strip()
        if not self._is_robot_marked_shooting(label):
            return False
        if self.camera_side == "center":
            ball_alliance = self._get_center_ball_alliance(ball_x)
            if ball_alliance == "blue":
                return label in self.blue_robots
            if ball_alliance == "red":
                return label in self.red_robots
            return False
        return self._is_robot_in_camera_alliance(label)
    
    def _ball_overlaps_robot(self, ball_x: int, ball_y: int, ball_radius: int) -> str:
        """
        Check if ball center is inside any robot bounding box.
        For robots near frame edges (partially visible), the bbox is extended
        horizontally to compensate for the clipped portion.
        All robot bboxes are expanded generously for shot attribution.
        Only returns robots that belong to the camera's alliance.
        
        Returns:
            Robot label if ball center is inside an alliance robot's box, None otherwise
        """
        edge_margin = self.frame_width * 0.05 if self.frame_width > 0 else 0
        
        for robot_bbox in self.robot_bboxes:
            y1, x1, y2, x2, label = robot_bbox[:5]
            # Only consider robots eligible for the current ball position.
            if not self._is_robot_eligible_for_ball(label, ball_x):
                continue
            
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            
            # Check if robot bbox is near frame edges (partially visible)
            near_left_edge = x1 < edge_margin
            near_right_edge = x2 > (self.frame_width - edge_margin) if self.frame_width > 0 else False
            is_edge_robot = near_left_edge or near_right_edge
            
            if is_edge_robot:
                # Extend bbox outward toward the edge the robot is clipped at
                extend_x = bbox_w * 0.30
                extend_y = bbox_h * 0.40  # Large top extension for edge robots
                x1_ext = x1 - (extend_x if near_left_edge else 0)
                x2_ext = x2 + (extend_x if near_right_edge else 0)
            else:
                extend_y = bbox_h * 0.40  # Large top extension to catch balls above robot
                x1_ext = x1
                x2_ext = x2
            
            y1_ext = y1 - extend_y
            
            # Require ball CENTER to be inside the (possibly extended) box
            if x1_ext <= ball_x <= x2_ext and y1_ext <= ball_y <= y2:
                return label
        
        return None

    def _get_robot_launch_geometry(self, y1: float, x1: float, y2: float, x2: float):
        """
        Build a launch anchor slightly above the robot plus a forgiving launch zone.

        The anchor is where we measure "nearest robot" for shot credit, and the zone
        lets us prefer robots that the ball is visibly lifting out of.
        """
        bbox_w = max(1.0, x2 - x1)
        bbox_h = max(1.0, y2 - y1)
        anchor_x = (x1 + x2) / 2.0
        anchor_y = y1 - (bbox_h * self.launch_anchor_y_ratio)
        zone_x1 = x1 - (bbox_w * self.launch_zone_x_ratio)
        zone_x2 = x2 + (bbox_w * self.launch_zone_x_ratio)
        zone_y1 = y1 - (bbox_h * self.launch_zone_y_ratio)
        zone_y2 = y2
        return anchor_x, anchor_y, zone_x1, zone_y1, zone_x2, zone_y2

    def _is_probable_side_source(self, ball_x: int) -> bool:
        """Heuristic guard for balls originating from the human-player side edges."""
        if self.camera_side != "center" or self.frame_width <= 0:
            return False
        edge_margin = self.frame_width * 0.08
        return ball_x <= edge_margin or ball_x >= (self.frame_width - edge_margin)

    def _find_nearest_alliance_robot(self, ball_x: int, ball_y: int, max_dist: float = None) -> str:
        """
        Find the nearest alliance robot using a launch anchor above each robot bbox.
        More forgiving than bbox overlap — works even when the ball has exited the bbox.
        
        Args:
            ball_x: Ball center x
            ball_y: Ball center y
            max_dist: Maximum distance to consider (pixels)
            
        Returns:
            Robot label of nearest alliance robot, or None if none within max_dist
        """
        if max_dist is None:
            max_dist = self.launch_anchor_max_distance

        best_label = None
        best_dist = max_dist
        
        for robot_bbox in self.robot_bboxes:
            y1, x1, y2, x2, label = robot_bbox[:5]
            if not self._is_robot_eligible_for_ball(label, ball_x):
                continue
            anchor_x, anchor_y, zone_x1, zone_y1, zone_x2, zone_y2 = self._get_robot_launch_geometry(y1, x1, y2, x2)
            dist = ((ball_x - anchor_x) ** 2 + (ball_y - anchor_y) ** 2) ** 0.5
            in_launch_zone = zone_x1 <= ball_x <= zone_x2 and zone_y1 <= ball_y <= zone_y2
            score = dist * 0.75 if in_launch_zone else dist + 20.0
            if score < best_dist:
                best_dist = score
                best_label = label
        
        return best_label

    def _get_shot_origin_robot(self, ball_x: int, overlapping_robot: str, last_overlap_robot: str,
                               last_overlap_frame: int, last_near_robot: str) -> str:
        """
        Choose which robot should own a newly detected shot.

        We prefer a recent true overlap/possession signal over nearest-robot fallback
        so airborne balls don't get reassigned to a robot behind the shooter.
        """
        if overlapping_robot and self._is_robot_eligible_for_ball(overlapping_robot, ball_x):
            return overlapping_robot

        if last_overlap_robot and last_overlap_frame is not None:
            frames_since_overlap = self.current_frame - last_overlap_frame
            if (frames_since_overlap <= self.possession_memory_frames and
                    self._is_robot_eligible_for_ball(last_overlap_robot, ball_x)):
                return last_overlap_robot

        if (last_near_robot
                and self._is_robot_eligible_for_ball(last_near_robot, ball_x)
                and not self._is_probable_side_source(ball_x)):
            return last_near_robot

        return None

    def _get_shot_launch_metrics(self, prev_pos, recent_positions: list, current_y: int) -> tuple:
        """
        Measure whether the ball is lifting into a shot over a short window.
        This is intentionally forgiving so a missed SAM 3 frame does not erase the event.
        """
        instant_rise = 0.0
        if prev_pos is not None:
            instant_rise = prev_pos[1] - current_y

        recent_positions = list(recent_positions or [])
        y_history = [pos[1] for pos in recent_positions[-(self.motion_history_size - 1):]]
        y_history.append(current_y)

        window_gain = 0.0
        rising_steps = 0
        if len(y_history) >= 2:
            window_gain = y_history[0] - y_history[-1]
            rising_steps = sum(
                1 for prev_y, next_y in zip(y_history, y_history[1:])
                if (prev_y - next_y) >= self.min_launch_rise_pixels
            )

        return instant_rise, window_gain, rising_steps

    def _dedupe_frame_detections(self, detections: list) -> list:
        """
        Collapse near-identical detections in the same frame into a single ball.

        This is especially important for head-on views where one physical ball can
        generate multiple nested circles/boxes with almost the same center.
        """
        if len(detections or []) <= 1:
            return list(detections or [])

        merged = []
        original_count = len(detections)

        for x, y, radius in sorted(detections, key=lambda item: item[2], reverse=True):
            x = float(x)
            y = float(y)
            radius = float(radius)
            matched_index = None

            for idx, (mx, my, mr, votes) in enumerate(merged):
                dist = ((x - mx) ** 2 + (y - my) ** 2) ** 0.5
                same_center_limit = max(4.0, min(radius, mr) * 0.75)
                nested_detection = dist + min(radius, mr) <= max(radius, mr) * 1.15
                if dist <= same_center_limit or nested_detection:
                    weight_old = max(1.0, mr * mr) * votes
                    weight_new = max(1.0, radius * radius)
                    total_weight = weight_old + weight_new
                    merged[idx] = (
                        ((mx * weight_old) + (x * weight_new)) / total_weight,
                        ((my * weight_old) + (y * weight_new)) / total_weight,
                        max(mr, radius),
                        votes + 1,
                    )
                    matched_index = idx
                    break

            if matched_index is None:
                merged.append((x, y, radius, 1))

        deduped = [
            (int(round(mx)), int(round(my)), int(round(mr)))
            for mx, my, mr, _ in merged
        ]

        if len(deduped) < original_count:
            print(f"[BALL DEDUPE] Collapsed {original_count} raw detections into {len(deduped)} unique balls")

        return deduped

    def _estimate_motion_vector(self, prev_pos, recent_positions: list, current_pos, velocity_hint: tuple = None) -> tuple:
        """
        Estimate current per-frame velocity from recent observations.

        If a velocity hint is provided (used for lost balls), prefer it so the
        prediction continues smoothly through occlusions.
        """
        if velocity_hint is not None:
            vx, vy = velocity_hint
            if abs(vx) > 0.01 or abs(vy) > 0.01:
                return float(vx), float(vy)

        positions = list(recent_positions or [])
        if current_pos is not None:
            positions.append(current_pos)
        elif prev_pos is not None:
            positions.append(prev_pos)

        if len(positions) < 2 and prev_pos is not None and current_pos is not None:
            positions = [prev_pos, current_pos]

        deltas = []
        for start, end in zip(positions, positions[1:]):
            deltas.append((float(end[0] - start[0]), float(end[1] - start[1])))

        if not deltas:
            return 0.0, 0.0

        weighted_vx = 0.0
        weighted_vy = 0.0
        total_weight = 0.0
        for idx, (dx, dy) in enumerate(deltas, start=1):
            weighted_vx += dx * idx
            weighted_vy += dy * idx
            total_weight += idx

        if total_weight <= 0:
            return 0.0, 0.0

        return weighted_vx / total_weight, weighted_vy / total_weight

    def _get_goal_center_for_origin(self, shot_origin_pos) -> tuple:
        """Pick the goal center this shot is most plausibly traveling toward."""
        if shot_origin_pos is None or not self.goal_polygons:
            return None

        goal_centers = []
        for polygon in self.goal_polygons:
            center = self._polygon_center(polygon)
            if center is None:
                continue
            cx, cy = center
            if self.camera_side == "center" and self.frame_width > 0 and self.frame_height > 0:
                cx, cy = _calibration_transform_point(cx, cy, self.frame_width, self.frame_height, inverse=False)
            goal_centers.append((cx, cy))

        if not goal_centers:
            return None

        ox, oy = float(shot_origin_pos[0]), float(shot_origin_pos[1])
        return min(goal_centers, key=lambda center: ((center[0] - ox) ** 2 + (center[1] - oy) ** 2))

    def _shot_progress_from_origin(self, current_pos, shot_origin_pos) -> float:
        """Measure forward travel specifically along the path toward the goal zone."""
        if current_pos is None or shot_origin_pos is None:
            return 0.0
        goal_center = self._get_goal_center_for_origin(shot_origin_pos)
        dx = float(current_pos[0]) - float(shot_origin_pos[0])
        dy = float(current_pos[1]) - float(shot_origin_pos[1])
        if goal_center is None:
            return (dx * dx + dy * dy) ** 0.5

        goal_dx = float(goal_center[0]) - float(shot_origin_pos[0])
        goal_dy = float(goal_center[1]) - float(shot_origin_pos[1])
        goal_len = (goal_dx * goal_dx + goal_dy * goal_dy) ** 0.5
        if goal_len <= 1e-6:
            return (dx * dx + dy * dy) ** 0.5

        projected_progress = ((dx * goal_dx) + (dy * goal_dy)) / goal_len
        return max(0.0, projected_progress)

    def _shot_has_launch_progress(self, current_pos, shot_origin_pos) -> bool:
        """Only arm trajectory prediction after the ball has visibly displaced."""
        return self._shot_progress_from_origin(current_pos, shot_origin_pos) >= self.min_shot_progress_pixels

    @staticmethod
    def _goal_spawn_lock_active(spawned_in_goal_zone: bool, exited_spawn_goal_zone: bool) -> bool:
        """Prevent brand-new goal-zone detections from immediately becoming shots."""
        return bool(spawned_in_goal_zone and not exited_spawn_goal_zone)

    def _is_prediction_in_bounds(self, x: float, y: float) -> bool:
        """Stop trajectory simulation once the ball is well outside the frame."""
        pad = self.prediction_bounds_padding
        if self.frame_width > 0 and (x < -pad or x > self.frame_width + pad):
            return False
        if self.frame_height > 0 and (y < -pad or y > self.frame_height + pad):
            return False
        return True

    def _simulate_trajectory(self, start_x: float, start_y: float, vx: float, vy: float,
                             max_steps: int = None) -> tuple:
        """
        Simulate a short future path using constant horizontal velocity and a
        simple gravity term for vertical acceleration.

        Returns:
            Tuple of (path_points, enters_goal, goal_frame_index)
        """
        if max_steps is None:
            max_steps = self.prediction_horizon_frames

        x = float(start_x)
        y = float(start_y)
        draw_path = [(x, y)]
        enters_goal = self._is_in_goal(x, y)
        goal_frame_index = 0 if enters_goal else None

        if enters_goal:
            return draw_path, enters_goal, goal_frame_index

        for step in range(1, max_steps + 1):
            next_x = x + vx
            next_y = y + vy

            for substep in range(1, self.prediction_substeps + 1):
                t = substep / float(self.prediction_substeps)
                sample_x = x + ((next_x - x) * t)
                sample_y = y + ((next_y - y) * t)
                if self._is_in_goal(sample_x, sample_y):
                    enters_goal = True
                    goal_frame_index = step
                    next_x, next_y = sample_x, sample_y
                    break

            draw_path.append((next_x, next_y))
            x, y = next_x, next_y
            if enters_goal:
                break

            vy += self.trajectory_gravity
            if not self._is_prediction_in_bounds(x, y):
                break

        return draw_path, enters_goal, goal_frame_index

    def _build_shot_prediction(self, current_pos, prev_pos=None, recent_positions: list = None,
                               velocity_hint: tuple = None, previous_prediction: dict = None) -> dict:
        """
        Build a predicted future path for a shot ball and determine whether it
        intersects a goal polygon.
        """
        if current_pos is None:
            return None

        vx, vy = self._estimate_motion_vector(prev_pos, recent_positions, current_pos, velocity_hint=velocity_hint)
        if abs(vx) < 0.01 and abs(vy) < 0.01:
            return None

        path, will_score, goal_frame_index = self._simulate_trajectory(
            current_pos[0],
            current_pos[1],
            vx,
            vy,
        )

        stable_frames = 1
        if previous_prediction and previous_prediction.get('will_score') == will_score:
            stable_frames = int(previous_prediction.get('stable_frames', 1)) + 1

        return {
            'path': [(int(round(px)), int(round(py))) for px, py in path],
            'will_score': will_score,
            'goal_frame_index': goal_frame_index,
            'velocity': (vx, vy),
            'stable_frames': stable_frames,
            'updated_at_frame': self.current_frame,
        }

    def _prediction_will_score(self, prediction: dict, launch_progress: float = 0.0) -> bool:
        """Return True only when the predicted make is strong enough to trust."""
        if not prediction or not prediction.get('will_score'):
            return False
        if int(prediction.get('stable_frames', 0)) < self.min_predicted_make_stable_frames:
            return False
        return launch_progress >= self.min_predicted_make_progress_pixels

    def _resolve_shot_result(self, robot_label: str, x: float, y: float,
                             was_ever_in_goal: bool = False, prediction: dict = None,
                             shot_origin_pos: tuple = None, trace_visible: bool = False,
                             context: str = "", shot_frame: int = None) -> bool:
        """
        Finalize a shot using observed goal entry if available, otherwise the
        latest trajectory prediction.
        """
        raw_observed_make = bool(was_ever_in_goal or self._is_in_goal(x, y))
        observed_make = raw_observed_make and trace_visible
        launch_progress = self._shot_progress_from_origin((x, y), shot_origin_pos)
        raw_predicted_make = bool(prediction and prediction.get('will_score'))
        predicted_make = (
            self._prediction_will_score(prediction, launch_progress=launch_progress)
            if trace_visible else False
        )
        if not observed_make and not predicted_make and launch_progress < self.min_shot_progress_pixels:
            suffix = f" ({context})" if context else ""
            print(
                f"[SHOT DROPPED] Robot {robot_label}{suffix}: "
                f"only moved {launch_progress:.1f}px from launch point"
            )
            return False
        made = observed_make or predicted_make
        try:
            event_elapsed = (
                self.start_seconds + (float(shot_frame) / max(1.0, float(self.fps)))
                if shot_frame is not None else
                self._get_elapsed_seconds()
            )
        except (TypeError, ValueError, ZeroDivisionError):
            event_elapsed = self._get_elapsed_seconds()
        self._record_shot(robot_label, made=made, event_elapsed=event_elapsed)

        period = get_match_period(event_elapsed)
        if raw_observed_make and not trace_visible:
            reason = "goal entry ignored because no shot trace was ever armed"
        elif raw_predicted_make and not trace_visible:
            reason = "predicted make ignored because no shot trace was ever armed"
        elif observed_make:
            reason = "observed goal entry"
        elif predicted_make:
            reason = "predicted path intersects goal"
        else:
            reason = (
                "predicted make was too weak to trust" if raw_predicted_make
                else "predicted path misses goal"
            )
        outcome = "SHOT MADE" if made else "SHOT MISSED"
        suffix = f" ({context})" if context else ""
        print(f"[{outcome}] Robot {robot_label} @ {period}{suffix}: {reason}")
        return made
    
    def get_predicted_positions(self) -> list:
        """
        Return predicted positions of all currently tracked and lost balls.
        Used to create a relaxed detection zone around expected ball locations.
        
        Returns:
            List of (x, y, radius) tuples for predicted ball positions.
        """
        positions = []
        
        # Active tracked balls — predict via velocity
        for ball_data in self.tracked_balls.values():
            cx, cy, cr = ball_data['pos']
            prev = ball_data['prev_pos']
            if prev:
                vx = cx - prev[0]
                vy = cy - prev[1]
                positions.append((cx + vx, cy + vy, cr))
            else:
                positions.append((cx, cy, cr))
        
        # Lost balls — use their extrapolated prediction
        for lost_data in self.lost_balls.values():
            px, py = lost_data['predicted_pos']
            _, _, cr = lost_data['data']['pos']
            positions.append((px, py, cr))
        
        return positions
    
    def _match_balls(self, new_detections: list) -> tuple:
        """
        Match new ball detections to existing tracked balls AND lost balls
        using linear assignment (Hungarian algorithm) and velocity prediction.
        
        Args:
            new_detections: List of (x, y, radius) tuples
            
        Returns:
            Tuple of (matches dict, recovered_lost dict)
            - matches: Dict mapping detection index to ball_id (or None for new balls)
            - recovered_lost: Dict mapping detection index to lost_ball_id (for re-identified balls)
        """
        from scipy.optimize import linear_sum_assignment
        
        matches = {}
        recovered_lost = {}
        
        # Combine active and lost balls for matching
        # Active balls have priority (lower cost penalty)
        all_ball_ids = []
        all_predicted_positions = []
        all_current_positions = []  # Actual last-known positions (fallback for deceleration)
        is_lost_ball = []
        lost_frames_count = []  # How many frames each ball has been lost
        
        # Add active tracked balls
        for ball_id, data in self.tracked_balls.items():
            curr_pos = data['pos']
            prev_pos = data['prev_pos']
            
            # Simple velocity prediction: pos + velocity
            if prev_pos:
                vx = curr_pos[0] - prev_pos[0]
                vy = curr_pos[1] - prev_pos[1]
                pred_x = curr_pos[0] + vx
                pred_y = curr_pos[1] + vy
                all_predicted_positions.append((pred_x, pred_y))
            else:
                all_predicted_positions.append((curr_pos[0], curr_pos[1]))
            
            all_current_positions.append((curr_pos[0], curr_pos[1]))
            all_ball_ids.append(ball_id)
            is_lost_ball.append(False)
            lost_frames_count.append(0)
        
        # Add lost balls (extrapolate their predicted position)
        for ball_id, lost_data in self.lost_balls.items():
            pred_pos = lost_data['predicted_pos']
            all_predicted_positions.append(pred_pos)
            # Also store the last-known position before the ball was lost
            last_pos = lost_data['data']['pos']
            all_current_positions.append((last_pos[0], last_pos[1]))
            all_ball_ids.append(ball_id)
            is_lost_ball.append(True)
            lost_frames_count.append(lost_data['frames_lost'])
        
        if not all_ball_ids or not new_detections:
            # Trivial case: all new or no existing
            return ({idx: None for idx in range(len(new_detections))}, {})
            
        # Create cost matrix (distances)
        cost_matrix = np.zeros((len(new_detections), len(all_ball_ids)))
        
        for i, (nx, ny, nr) in enumerate(new_detections):
            for j, (ex, ey) in enumerate(all_predicted_positions):
                # Distance to velocity-predicted position
                dist_pred = np.sqrt((nx - ex) ** 2 + (ny - ey) ** 2)
                
                # Distance to actual last-known position (handles deceleration/direction changes)
                cx, cy = all_current_positions[j]
                dist_curr = np.sqrt((nx - cx) ** 2 + (ny - cy) ** 2)
                
                # Use the MINIMUM — if ball is near either predicted or current pos, it's a match
                dist = min(dist_pred, dist_curr)
                
                # Add penalty for lost balls (prefer matching to active balls)
                if is_lost_ball[j]:
                    dist += 10  # Small penalty to prefer active balls
                
                cost_matrix[i, j] = dist
        
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Process assignments
        assigned_detections = set()
        
        for r, c in zip(row_ind, col_ind):
            dist = cost_matrix[r, c]
            ball_id = all_ball_ids[c]
            is_lost = is_lost_ball[c]
            
            # Calculate threshold
            if is_lost:
                # For lost balls, scale threshold with how long they've been gone
                # Base: 2x distance, growing by 0.3x per lost frame (accounts for prediction drift)
                frames_lost = lost_frames_count[c]
                threshold = self.max_matching_distance * (2.0 + frames_lost * 0.3)
            else:
                # For active balls, use dynamic threshold based on speed
                data = self.tracked_balls[ball_id]
                prev_pos = data['prev_pos']
                curr_pos = data['pos']
                
                speed = 0
                if prev_pos:
                    speed = np.sqrt((curr_pos[0]-prev_pos[0])**2 + (curr_pos[1]-prev_pos[1])**2)
                
                threshold = max(self.max_matching_distance, speed * 2.0)
            
            # Remove penalty from dist for comparison
            actual_dist = dist - (10 if is_lost else 0)
            
            if actual_dist < threshold:
                if is_lost:
                    recovered_lost[r] = ball_id
                else:
                    matches[r] = ball_id
                assigned_detections.add(r)
            else:
                matches[r] = None  # Too far, treat as new
                
        # Handle unassigned detections
        for i in range(len(new_detections)):
            if i not in assigned_detections:
                matches[i] = None
        
        return matches, recovered_lost
    
    def update(self, fuel_detections: list) -> list:
        """
        Update ball tracking with new detections and detect shots.
        Handles ball occlusion by keeping lost balls in memory for re-identification.
        
        Args:
            fuel_detections: List of (x, y, radius) tuples
            
        Returns:
            List of visualization dicts for tracked and predicted balls.
        """
        self.current_frame += 1
        self._record_shooting_snapshot()
        fuel_detections = self._dedupe_frame_detections(fuel_detections)
        
        # Match new detections to existing balls (and lost balls)
        matches, recovered_lost = self._match_balls(fuel_detections)
        
        # Determine which active balls were matched
        matched_active_ids = set(v for v in matches.values() if v is not None)
        matched_lost_ids = set(recovered_lost.values())
        
        # Move unmatched active balls to lost_balls (with prediction)
        new_lost_balls = {}
        for ball_id, ball_data in self.tracked_balls.items():
            if ball_id not in matched_active_ids:
                # Ball lost this frame - move to lost pool
                curr_pos = ball_data['pos']
                prev_pos = ball_data['prev_pos']
                
                # Predict where ball will be
                if prev_pos:
                    vx = curr_pos[0] - prev_pos[0]
                    vy = curr_pos[1] - prev_pos[1]
                    pred_x = curr_pos[0] + vx
                    pred_y = curr_pos[1] + vy
                else:
                    vx, vy = 0, 0
                    pred_x, pred_y = curr_pos[0], curr_pos[1]
                
                new_lost_balls[ball_id] = {
                    'data': ball_data,
                    'frames_lost': 1,
                    'predicted_pos': (pred_x, pred_y),
                    'velocity': (vx, vy)  # Store velocity for gravity simulation
                }
        
        # Update existing lost balls (increment frames, update prediction)
        updated_lost_balls = {}
        for ball_id, lost_data in self.lost_balls.items():
            if ball_id in matched_lost_ids:
                # This lost ball was recovered - don't keep it in lost pool
                continue
            
            lost_data['frames_lost'] += 1
            
            # Update predicted position (continue extrapolating)
            pred_x, pred_y = lost_data['predicted_pos']
            ball_data = lost_data['data']
            curr_pos = ball_data['pos']
            prev_pos = ball_data['prev_pos']
            
            # Use stored velocity with gravity for parabolic prediction
            vx, vy = lost_data.get('velocity', (0, 0))
            if vx != 0 or vy != 0:
                vy += self.trajectory_gravity
                pred_x += vx
                pred_y += vy
                lost_data['velocity'] = (vx, vy)
                lost_data['predicted_pos'] = (pred_x, pred_y)

            shot_by = ball_data.get('shot_by')
            shot_evaluated = ball_data.get('shot_evaluated', False)
            candidate_shot = ball_data.get('candidate_shot')
            shot_origin_pos = ball_data.get('shot_origin_pos')
            trace_visible_once = bool(ball_data.get('trace_visible_once', False))
            if shot_by and not shot_evaluated and trace_visible_once:
                predicted_ball = (pred_x, pred_y, ball_data['pos'][2])
                if self._shot_has_launch_progress(predicted_ball, shot_origin_pos):
                    ball_data['candidate_shot'] = self._build_shot_prediction(
                        predicted_ball,
                        prev_pos=curr_pos,
                        recent_positions=ball_data.get('recent_positions'),
                        velocity_hint=lost_data.get('velocity'),
                        previous_prediction=candidate_shot,
                    )
                else:
                    ball_data['candidate_shot'] = None
            
            # Check if ball has been lost too long
            if lost_data['frames_lost'] <= self.max_frames_lost:
                updated_lost_balls[ball_id] = lost_data
            else:
                # Ball lost for too long - finalize shot stats if applicable
                shot_by = lost_data['data'].get('shot_by')
                shot_evaluated = lost_data['data'].get('shot_evaluated', False)
                
                if shot_by and not shot_evaluated:
                    x, y, _ = lost_data['data']['pos']
                    self._resolve_shot_result(
                        shot_by,
                        x,
                        y,
                        was_ever_in_goal=lost_data['data'].get('last_seen_in_goal', False),
                        prediction=lost_data['data'].get('candidate_shot'),
                        shot_origin_pos=lost_data['data'].get('shot_origin_pos'),
                        trace_visible=bool(lost_data['data'].get('trace_visible_once', False)),
                        context="lost timeout",
                        shot_frame=lost_data['data'].get('shot_time'),
                    )
        
        # Merge new and updated lost balls
        # Save reference to original for recovery lookup
        original_lost_balls = self.lost_balls
        self.lost_balls = {**updated_lost_balls, **new_lost_balls}
        
        # Update tracked balls
        new_tracked = {}
        results = []
        
        for det_idx, (x, y, r) in enumerate(fuel_detections):
            ball_id = matches.get(det_idx)
            recovered_id = recovered_lost.get(det_idx)
            
            if recovered_id is not None:
                # Recovered a lost ball!
                ball_id = recovered_id
                old_data = original_lost_balls[recovered_id]['data']
                prev_pos = old_data['pos']
                
                # Restore all shot state from the lost ball
                shot_by = old_data.get('shot_by')
                shot_time = old_data.get('shot_time')
                shot_evaluated = old_data.get('shot_evaluated', False)
                candidate_shot = old_data.get('candidate_shot')
                shot_origin_pos = old_data.get('shot_origin_pos')
                overlapping_robot = old_data.get('overlapping_robot')
                last_near_robot = old_data.get('last_near_robot')
                last_overlap_robot = old_data.get('last_overlap_robot')
                last_overlap_frame = old_data.get('last_overlap_frame')
                was_ever_in_goal = old_data.get('last_seen_in_goal', False)
                spawned_in_goal_zone = bool(old_data.get('spawned_in_goal_zone', False))
                exited_spawn_goal_zone = bool(old_data.get('exited_spawn_goal_zone', not spawned_in_goal_zone))
                trace_visible_once = bool(old_data.get('trace_visible_once', False))
                recent_positions = list(old_data.get('recent_positions') or ([prev_pos] if prev_pos else []))
                
            elif ball_id is None:
                # New ball
                cur_in_goal = self._is_in_goal(x, y)
                if cur_in_goal:
                    print(f"[BALL IGNORED] New detection inside goal zone at ({x:.0f},{y:.0f})")
                    continue

                ball_id = self.next_ball_id
                self.next_ball_id += 1
                cur_overlap = self._ball_overlaps_robot(x, y, r)
                cur_nearest = self._find_nearest_alliance_robot(x, y)
                new_tracked[ball_id] = {
                    'pos': (x, y, r),
                    'prev_pos': None,
                    'shot_by': None,
                    'shot_time': None,
                    'shot_evaluated': False,
                    'overlapping_robot': cur_overlap,
                    'last_near_robot': cur_overlap or cur_nearest,
                    'last_overlap_robot': cur_overlap,
                    'last_overlap_frame': self.current_frame if cur_overlap else None,
                    'candidate_shot': None,
                    'shot_origin_pos': None,
                    'last_seen_in_goal': cur_in_goal,
                    'spawned_in_goal_zone': cur_in_goal,
                    'exited_spawn_goal_zone': not cur_in_goal,
                    'trace_visible_once': False,
                    'recent_positions': [(x, y, r)]
                }
                results.append({
                    'x': x,
                    'y': y,
                    'radius': r,
                    'robot_label': None,
                    'predicted_path': [],
                    'predicted_make': None,
                    'predicted_only': False,
                })
                continue
            else:
                # Existing active ball
                old_data = self.tracked_balls[ball_id]
                prev_pos = old_data['pos']
                shot_by = old_data.get('shot_by')
                shot_time = old_data.get('shot_time')
                shot_evaluated = old_data.get('shot_evaluated', False)
                candidate_shot = old_data.get('candidate_shot')
                shot_origin_pos = old_data.get('shot_origin_pos')
                overlapping_robot = old_data.get('overlapping_robot')
                last_near_robot = old_data.get('last_near_robot')
                last_overlap_robot = old_data.get('last_overlap_robot')
                last_overlap_frame = old_data.get('last_overlap_frame')
                was_ever_in_goal = old_data.get('last_seen_in_goal', False)
                spawned_in_goal_zone = bool(old_data.get('spawned_in_goal_zone', False))
                exited_spawn_goal_zone = bool(old_data.get('exited_spawn_goal_zone', not spawned_in_goal_zone))
                trace_visible_once = bool(old_data.get('trace_visible_once', False))
                recent_positions = list(old_data.get('recent_positions') or ([prev_pos] if prev_pos else []))
            
            recent_positions = recent_positions[-(self.motion_history_size - 1):]
            cur_overlap = self._ball_overlaps_robot(x, y, r)
            cur_nearest = self._find_nearest_alliance_robot(x, y)

            # Check for shot initiation using a short upward-motion window.
            instant_rise, window_gain, rising_steps = self._get_shot_launch_metrics(prev_pos, recent_positions, y)

            # As soon as the ball is clearly lifting, credit the nearest launch anchor.
            shot_detection_locked = self._goal_spawn_lock_active(
                spawned_in_goal_zone,
                exited_spawn_goal_zone,
            )
            if not shot_by and not shot_detection_locked:
                nearby_robot = self._get_shot_origin_robot(
                    x,
                    cur_overlap or overlapping_robot,
                    last_overlap_robot,
                    last_overlap_frame,
                    cur_nearest,
                )
                launch_detected = (
                    instant_rise >= self.min_launch_rise_pixels or
                    window_gain >= self.min_launch_window_gain or
                    rising_steps >= 2
                )
                if nearby_robot and launch_detected:
                    shot_by = nearby_robot
                    shot_time = self.current_frame
                    shot_evaluated = False
                    candidate_shot = None
                    shot_origin_pos = (x, y, r)
                    print(
                        f"[SHOT DETECTED] Ball {ball_id} shot by {shot_by} at "
                        f"pos=({x:.0f},{y:.0f}), rise={instant_rise:.0f}px, "
                        f"window_gain={window_gain:.0f}px, steps={rising_steps}"
                    )

            updated_recent_positions = recent_positions + [(x, y, r)]
            launch_progress = self._shot_progress_from_origin((x, y, r), shot_origin_pos)
            shot_trace_ready = bool(shot_by) and launch_progress >= self.min_shot_progress_pixels
            trace_visible_once = trace_visible_once or shot_trace_ready
            if shot_by and shot_trace_ready:
                candidate_shot = self._build_shot_prediction(
                    (x, y, r),
                    prev_pos=prev_pos,
                    recent_positions=recent_positions,
                    previous_prediction=candidate_shot,
                )
            else:
                candidate_shot = None
            
            # Check if 2 seconds have passed since shot - time to evaluate!
            # Only count MADE shots (ball in goal)
            if shot_time is not None and not shot_evaluated:
                frames_since_shot = self.current_frame - shot_time
                seconds_since_shot = frames_since_shot / self.fps
                
                if seconds_since_shot >= 2.0:
                    self._resolve_shot_result(
                        shot_by,
                        x,
                        y,
                        was_ever_in_goal=was_ever_in_goal,
                        prediction=candidate_shot,
                        shot_origin_pos=shot_origin_pos,
                        trace_visible=trace_visible_once,
                        context="2sec eval",
                        shot_frame=shot_time,
                    )
                    shot_evaluated = True
            
            # Check if shot label should expire (keep the visual label for duration)
            if shot_time is not None:
                frames_since_shot = self.current_frame - shot_time
                seconds_since_shot = frames_since_shot / self.fps
                if seconds_since_shot > self.shot_label_duration:
                    shot_by = None
                    shot_time = None
            
            is_in_goal = self._is_in_goal(x, y)
            exited_spawn_goal_zone = exited_spawn_goal_zone or not is_in_goal
            
            # Sticky flags: once True, stays True
            ever_in_goal = was_ever_in_goal or is_in_goal
            
            # Debug: track when shot balls enter goal
            if shot_by and is_in_goal:
                print(f"[IN GOAL] Ball {ball_id} (shot by {shot_by}) at pos=({x:.0f},{y:.0f})")
            
            updated_last_overlap_robot = cur_overlap or last_overlap_robot
            updated_last_overlap_frame = self.current_frame if cur_overlap else last_overlap_frame
            # Update last_near_robot: prefer overlap, then nearest, then keep previous
            updated_near = cur_overlap or cur_nearest or last_near_robot
            
            new_tracked[ball_id] = {
                'pos': (x, y, r),
                'prev_pos': prev_pos,
                'shot_by': shot_by,
                'shot_time': shot_time,
                'shot_evaluated': shot_evaluated,
                'overlapping_robot': cur_overlap,
                'last_near_robot': updated_near,
                'last_overlap_robot': updated_last_overlap_robot,
                'last_overlap_frame': updated_last_overlap_frame,
                'candidate_shot': candidate_shot,
                'shot_origin_pos': shot_origin_pos,
                'last_seen_in_goal': ever_in_goal,
                'spawned_in_goal_zone': spawned_in_goal_zone,
                'exited_spawn_goal_zone': exited_spawn_goal_zone,
                'trace_visible_once': trace_visible_once,
                'recent_positions': updated_recent_positions
            }
            
            # Add to results
            robot_label = new_tracked[ball_id].get('shot_by') if candidate_shot is not None else None
            predicted_make_state = None
            if candidate_shot is not None:
                if self._prediction_will_score(candidate_shot, launch_progress=launch_progress):
                    predicted_make_state = True
                elif not candidate_shot.get('will_score'):
                    predicted_make_state = False
            results.append({
                'x': x,
                'y': y,
                'radius': r,
                'robot_label': robot_label,
                'predicted_path': list((candidate_shot or {}).get('path') or []),
                'predicted_make': predicted_make_state,
                'predicted_only': False,
            })

        for lost_data in self.lost_balls.values():
            ball_data = lost_data.get('data', {})
            prediction = ball_data.get('candidate_shot')
            if not prediction:
                continue

            robot_label = ball_data.get('shot_by')
            pred_x, pred_y = lost_data.get('predicted_pos', (0, 0))
            pos = ball_data.get('pos', (pred_x, pred_y, 0))
            radius = pos[2] if len(pos) >= 3 else 0
            launch_progress = self._shot_progress_from_origin((pred_x, pred_y, radius), ball_data.get('shot_origin_pos'))
            predicted_make_state = None
            if self._prediction_will_score(prediction, launch_progress=launch_progress):
                predicted_make_state = True
            elif not prediction.get('will_score'):
                predicted_make_state = False
            results.append({
                'x': int(round(pred_x)),
                'y': int(round(pred_y)),
                'radius': radius,
                'robot_label': robot_label,
                'predicted_path': list((prediction or {}).get('path') or []),
                'predicted_make': predicted_make_state,
                'predicted_only': True,
            })
        
        self.tracked_balls = new_tracked
        return results
    
    def reset(self):
        """Reset tracker state."""
        self.tracked_balls = {}
        self.lost_balls = {}
        self.robot_stats = {}
        self.shot_events = []
        self.shooting_snapshots = []
        self.next_ball_id = 0
        self.current_frame = 0
        self.robot_bboxes = []
    
    def finalize_all(self):
        """
        Finalize all remaining tracked and lost balls.
        Call this at the end of video processing to ensure all shots are counted.
        """
        # Finalize all balls currently being tracked — only count MADE shots
        for ball_id, ball_data in self.tracked_balls.items():
            shot_by = ball_data.get('shot_by')
            shot_evaluated = ball_data.get('shot_evaluated', False)
            
            if shot_by and not shot_evaluated:
                x, y, _ = ball_data['pos']
                self._resolve_shot_result(
                    shot_by,
                    x,
                    y,
                    was_ever_in_goal=ball_data.get('last_seen_in_goal', False),
                    prediction=ball_data.get('candidate_shot'),
                    shot_origin_pos=ball_data.get('shot_origin_pos'),
                    trace_visible=bool(ball_data.get('trace_visible_once', False)),
                    context="finalize tracked",
                    shot_frame=ball_data.get('shot_time'),
                )
        
        # Finalize all balls in the lost pool — only count MADE shots
        for ball_id, lost_data in self.lost_balls.items():
            shot_by = lost_data['data'].get('shot_by')
            shot_evaluated = lost_data['data'].get('shot_evaluated', False)
            
            if shot_by and not shot_evaluated:
                x, y, _ = lost_data['data']['pos']
                self._resolve_shot_result(
                    shot_by,
                    x,
                    y,
                    was_ever_in_goal=lost_data['data'].get('last_seen_in_goal', False),
                    prediction=lost_data['data'].get('candidate_shot'),
                    shot_origin_pos=lost_data['data'].get('shot_origin_pos'),
                    trace_visible=bool(lost_data['data'].get('trace_visible_once', False)),
                    context="finalize lost",
                    shot_frame=lost_data['data'].get('shot_time'),
                )
        
        print(f"[FINAL STATS] {self.robot_stats}")


def camera_to_map_coords(bbox_center_x: float, bbox_center_y: float, 
                         frame_width: int, frame_height: int,
                         map_width: int, map_height: int,
                         camera_side: str = "blue") -> tuple:
    """
    Transform camera coordinates to bird's eye map coordinates using homography.
    
    Uses calibrated correspondence points between the video frame and the map
    to compute an accurate perspective transformation.
    
    Note: The map is rotated 90° counterclockwise from the original orientation.
    
    Args:
        bbox_center_x: X center of bounding box (0-frame_width)
        bbox_center_y: Y center of bounding box (0-frame_height)
        frame_width: Width of the video frame (reference: 1068)
        frame_height: Height of the video frame (reference: 836)
        map_width: Width of the map image (reference: 961 after rotation)
        map_height: Height of the map image (reference: 574 after rotation)
        camera_side: "blue" for blue side camera, "red" for red side camera
        
    Returns:
        (map_x, map_y) coordinates on the map
    """
    # Reference dimensions used for calibration (after 90° CCW rotation)
    REF_VIDEO_WIDTH = 1068
    REF_VIDEO_HEIGHT = 836
    REF_MAP_WIDTH = 961   # Was 574 (height becomes width after rotation)
    REF_MAP_HEIGHT = 574  # Was 961 (width becomes height after rotation)
    # ORIGINAL_MAP_WIDTH = 574  (original width for coordinate transformation)
    
    # Map coordinates after 90° CCW rotation
    # Original (x, y) -> Rotated (y, original_width - x)
    # Original points:
    #   [45, 695],    # Trench 1 Blue
    #   [528, 694],   # Trench 2 Blue
    #   [308, 903],   # Climb Blue
    #   [267, 62],    # Climb Red
    #   [46, 269],    # Trench 1 Red
    #   [528, 270],   # Trench 2 Red
    #   [287, 483],   # Center of Field
    MAP_POINTS = np.array([
        [695, 574 - 45],    # Trench 1 Blue: (45, 695) -> (695, 529)
        [694, 574 - 528],   # Trench 2 Blue: (528, 694) -> (694, 46)
        [903, 574 - 308],   # Climb Blue: (308, 903) -> (903, 266)
        [62, 574 - 267],    # Climb Red: (267, 62) -> (62, 307)
        [269, 574 - 46],    # Trench 1 Red: (46, 269) -> (269, 528)
        [270, 574 - 528],   # Trench 2 Red: (528, 270) -> (270, 46)
        [483, 574 - 287],   # Center of Field: (287, 483) -> (483, 287)
    ], dtype=np.float32)
    
    if camera_side == "blue":
        # Blue camera calibration points
        VIDEO_POINTS = np.array([
            [143, 532],   # Trench 1 Blue
            [623, 364],   # Trench 2 Blue
            [801, 496],   # Climb Blue
            [172, 323],   # Climb Red
            [23, 370],    # Trench 1 Red
            [377, 318],   # Trench 2 Red
            [328, 361],   # Center of Field
        ], dtype=np.float32)
    else:  # red camera
        # Red camera calibration points
        VIDEO_POINTS = np.array([
            [377, 318],   # Trench 1 Blue
            [23, 370],    # Trench 2 Blue
            [172, 323],   # Climb Blue
            [801, 496],   # Climb Red
            [623, 364],   # Trench 1 Red
            [143, 532],   # Trench 2 Red
            [328, 361],   # Center of Field
        ], dtype=np.float32)
    
    # Compute homography matrix
    homography_matrix, _ = cv2.findHomography(VIDEO_POINTS, MAP_POINTS)
    
    # Scale input coordinates to reference frame dimensions
    scaled_x = bbox_center_x * REF_VIDEO_WIDTH / frame_width
    scaled_y = bbox_center_y * REF_VIDEO_HEIGHT / frame_height
    
    # Apply perspective transform
    point = np.array([[[scaled_x, scaled_y]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, homography_matrix)
    
    # Extract and scale to actual map dimensions
    map_x_ref = transformed[0][0][0]
    map_y_ref = transformed[0][0][1]
    
    map_x = int(map_x_ref * map_width / REF_MAP_WIDTH)
    map_y = int(map_y_ref * map_height / REF_MAP_HEIGHT)
    
    # Clamp to map bounds
    map_x = max(0, min(map_width - 1, map_x))
    map_y = max(0, min(map_height - 1, map_y))
    
    return (map_x, map_y)


def center_camera_to_map_coords(bbox_center_x: float, bbox_center_y: float, 
                                frame_width: int, frame_height: int,
                                map_width: int, map_height: int) -> tuple:
    """
    Transform center camera coordinates to bird's eye map coordinates using homography.
    
    The center camera (1918x709) captures both red and blue sides of the field.
    Left side shows blue, right side shows red.
    
    Uses 8-point calibration (4 blue-side + 4 red-side field landmarks) between
    the video frame and the map to compute an accurate perspective transformation.
    
    Note: The map is rotated 90° counterclockwise from the original orientation.
    
    Args:
        bbox_center_x: X center of bounding box (0-frame_width)
        bbox_center_y: Y center of bounding box (0-frame_height)
        frame_width: Width of the video frame (reference: 1918)
        frame_height: Height of the video frame (reference: 709)
        map_width: Width of the map image (reference: 961 after rotation)
        map_height: Height of the map image (reference: 574 after rotation)
        
    Returns:
        (map_x, map_y) coordinates on the map
    """
    # Reference dimensions for center camera
    REF_VIDEO_WIDTH = 1918
    REF_VIDEO_HEIGHT = 709
    REF_MAP_WIDTH = 961   # After 90° CCW rotation
    REF_MAP_HEIGHT = 574  # After 90° CCW rotation
    # ORIGINAL_MAP_WIDTH = 574  (original width before rotation)
    # ORIGINAL_MAP_HEIGHT = 961  (original height before rotation)
    
    # Center camera calibration points (video coordinates, reference 1918x709)
    # 8-point calibration using field landmarks on both sides
    VIDEO_POINTS = np.array([
        [164, 496],    # BlueSide1
        [310, 366],    # BlueSide2
        [611, 496],    # BlueSide3
        [693, 387],    # BlueSide4
        [1736, 489],   # RedSide1
        [1636, 376],   # RedSide2
        [1308, 502],   # RedSide3
        [1234, 405],   # RedSide4
    ], dtype=np.float32)
    
    # Corresponding map points (unrotated map is 574 x 961)
    # After 90° CCW rotation: (x, y) -> (y, original_width - x)
    # BlueSide1: (338, 900) -> (900, 236)
    # BlueSide2: (1, 958)   -> (958, 573)
    # BlueSide3: (327, 661) -> (661, 247)
    # BlueSide4: (92, 660)  -> (660, 482)
    # RedSide1: (296, 61)   -> (61, 278)
    # RedSide2: (3, 3)      -> (3, 571)
    # RedSide3: (324, 302)  -> (302, 250)
    # RedSide4: (92, 304)   -> (304, 482)
    MAP_POINTS = np.array([
        [900, 574 - 338],   # BlueSide1: (338, 900) -> (900, 236)
        [958, 574 - 1],     # BlueSide2: (1, 958)   -> (958, 573)
        [661, 574 - 327],   # BlueSide3: (327, 661) -> (661, 247)
        [660, 574 - 92],    # BlueSide4: (92, 660)  -> (660, 482)
        [61, 574 - 296],    # RedSide1: (296, 61)   -> (61, 278)
        [3, 574 - 3],       # RedSide2: (3, 3)      -> (3, 571)
        [302, 574 - 324],   # RedSide3: (324, 302)  -> (302, 250)
        [304, 574 - 92],    # RedSide4: (92, 304)   -> (304, 482)
    ], dtype=np.float32)
    
    homography_matrix, _ = cv2.findHomography(VIDEO_POINTS, MAP_POINTS, cv2.RANSAC)
    
    # Un-shift coordinates using calibration homography if available
    scaled_x = bbox_center_x * REF_VIDEO_WIDTH / frame_width
    scaled_y = bbox_center_y * REF_VIDEO_HEIGHT / frame_height
    
    H_inv = getattr(center_camera_to_map_coords, 'calibration_homography_inv', None)
    if H_inv is not None:
        pt = np.array([[[scaled_x, scaled_y]]], dtype=np.float32)
        unshifted = cv2.perspectiveTransform(pt, H_inv)
        scaled_x = unshifted[0][0][0]
        scaled_y = unshifted[0][0][1]
        
    # Apply standard perspective transform to the un-shifted base coords
    point = np.array([[[scaled_x, scaled_y]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, homography_matrix)
    
    # Extract and scale to actual map dimensions
    map_x_ref = transformed[0][0][0]
    map_y_ref = transformed[0][0][1]
    
    map_x = int(map_x_ref * map_width / REF_MAP_WIDTH)
    map_y = int(map_y_ref * map_height / REF_MAP_HEIGHT)
    
    # Clamp to map bounds
    map_x = max(0, min(map_width - 1, map_x))
    map_y = max(0, min(map_height - 1, map_y))
    
    return (map_x, map_y)


def _calibration_transform_point(x, y, frame_w, frame_h, inverse=True):
    """
    Transform a point using the calibration homography.
    
    Args:
        x, y: Point coordinates in actual frame resolution
        frame_w, frame_h: Actual frame dimensions
        inverse: If True, map current→reference (un-shift). If False, map reference→current (shift).
        
    Returns:
        (new_x, new_y) in actual frame resolution, or original (x, y) if no calibration available.
    """
    fn = globals().get('center_camera_to_map_coords')
    if fn is None:
        return x, y
    H = getattr(fn, 'calibration_homography_inv' if inverse else 'calibration_homography', None)
    if H is None:
        return x, y
    
    # Scale to reference resolution
    REF_W, REF_H = 1918, 709
    ref_x = x * REF_W / frame_w if frame_w > 0 else x
    ref_y = y * REF_H / frame_h if frame_h > 0 else y
    
    # Apply homography
    pt = np.array([[[ref_x, ref_y]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pt, H)
    out_x = transformed[0][0][0]
    out_y = transformed[0][0][1]
    
    # Scale back to actual resolution
    out_x = out_x * frame_w / REF_W
    out_y = out_y * frame_h / REF_H
    return float(out_x), float(out_y)


def _calibration_transform_point_ref(ref_x, ref_y, inverse=True):
    """
    Transform a point in reference resolution (1918x709) using the calibration homography.
    Returns result in reference resolution. Used for ROI/zone operations already in ref coords.
    """
    fn = globals().get('center_camera_to_map_coords')
    if fn is None:
        return ref_x, ref_y
    H = getattr(fn, 'calibration_homography_inv' if inverse else 'calibration_homography', None)
    if H is None:
        return ref_x, ref_y
    
    pt = np.array([[[float(ref_x), float(ref_y)]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pt, H)
    return float(transformed[0][0][0]), float(transformed[0][0][1])


class CenterCameraCalibrator:
    """
    Auto-calibrates the center camera using Gemini API landmark detection.
    Sends a reference image with known landmark positions and the current frame
    to Gemini, asking it to locate the same landmarks. Computes a full homography
    from matched points to handle translation, rotation, scale, and perspective changes.
    """
    
    # Reference landmark points on the center camera reference frame (1918x709)
    # These are the 8 calibration points from center_camera_to_map_coords
    REFERENCE_POINTS = {
        'B1': (164, 496),    # BlueSide1
        'B2': (310, 366),    # BlueSide2
        'B3': (611, 496),    # BlueSide3
        'B4': (693, 387),    # BlueSide4
        'R1': (1736, 489),   # RedSide1
        'R2': (1636, 376),   # RedSide2
        'R3': (1308, 502),   # RedSide3
        'R4': (1234, 405),   # RedSide4
    }
    
    # Reference image path
    REFERENCE_IMAGE_PATH = Path(__file__).parent / "reference_image.png"
    
    def __init__(self, fps: float, gather_duration_sec: float = 5.0, display_duration_sec: float = 5.0):
        self.fps = fps
        self.max_gather_frames = int(fps * gather_duration_sec)
        self.max_display_frames = int(fps * display_duration_sec)
        self.frame_count = 0
        self.is_calibrating = True
        self.is_displaying = False
        
        self.calibration_homography = None      # 3x3 matrix: reference → current
        self.calibration_homography_inv = None   # 3x3 matrix: current → reference
        self.found_points = {}                   # {label: (x, y)} in reference resolution
        self.last_visualization_data = None
    
    @property
    def is_active(self):
        """True while the calibrator still needs to process frames (gather + display)."""
        return self.frame_count < (self.max_gather_frames + self.max_display_frames)
        
    def process_frame(self, frame_bgr: np.ndarray, frame_width: int, frame_height: int) -> dict:
        """
        Track frame count and return visualization data during display phase.
        Calibration is pre-computed via Gemini before processing starts.
        """
        self.frame_count += 1
        
        if self.is_calibrating:
            # Calibration is done via user clicks in the UI,
            # but we still count frames. At the gather boundary, finalize.
            if self.frame_count >= self.max_gather_frames:
                self.is_calibrating = False
                self.is_displaying = True
            return None
            
        elif self.is_displaying:
            # Phase 2: Displaying locked overlays
            if self.frame_count >= (self.max_gather_frames + self.max_display_frames):
                self.is_displaying = False
                return None
                
            self.last_visualization_data = {
                'reference_points': self.REFERENCE_POINTS,
                'found_points': self.found_points,
                'frame_count': self.frame_count - self.max_gather_frames,
                'max_frames': self.max_display_frames,
                'homography': self.calibration_homography,
            }
            return self.last_visualization_data
            
        else:
            # Check if we were fast-forwarded (calibration pre-calculated) and just waiting for 5s mark
            if self.frame_count == self.max_gather_frames and self.max_display_frames > 0:
                self.is_displaying = True
            return None
    
    @classmethod
    def extract_calibration_frame(cls, video_path: str, start_seconds: float = 0) -> Image.Image:
        """Extract the center camera portion of a frame from the composite video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("[Calibration] Failed to open video for frame extraction")
            return None
        
        target_ms = (start_seconds + 4) * 1000  # 4 seconds after start, in milliseconds
        cap.set(cv2.CAP_PROP_POS_MSEC, target_ms)
        
        ret, frame = cap.read()
        actual_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        cap.release()
        
        print(f"[Calibration] Requested frame at {target_ms:.0f}ms, got frame at {actual_ms:.0f}ms")
        
        if not ret:
            print("[Calibration] Failed to read frame")
            return None
        
        h, w = frame.shape[:2]
        center_x1, center_y1, center_x2, center_y2 = _build_composite_crop_layout(w, h)["center"]["rect"]
        center_frame = frame[center_y1:center_y2, center_x1:center_x2]
        
        frame_rgb = cv2.cvtColor(center_frame, cv2.COLOR_BGR2RGB)
        print(f"[Calibration] Extracted center camera frame: {center_frame.shape[1]}x{center_frame.shape[0]} from composite {w}x{h}")
        return Image.fromarray(frame_rgb)
    
    @classmethod
    def compute_homography_from_points(cls, clicked_points: list, image_width: int, image_height: int) -> tuple:
        """
        Compute homography from user-clicked points.
        
        Args:
            clicked_points: List of 8 (x, y) tuples in the displayed image resolution
            image_width, image_height: Dimensions of the displayed image
            
        Returns:
            (H, H_inv, found_points) or (None, None, {}) on failure
        """
        if len(clicked_points) < 4:
            print(f"[Calibration] Not enough points ({len(clicked_points)}<4). Using default calibration.")
            return None, None, {}
        
        REF_W, REF_H = 1918, 709
        labels = list(cls.REFERENCE_POINTS.keys())
        
        ref_pts = []
        cur_pts = []
        found_points = {}
        
        for i, (click_x, click_y) in enumerate(clicked_points):
            if i >= len(labels):
                break
            label = labels[i]
            ref_x, ref_y = cls.REFERENCE_POINTS[label]
            
            # Scale clicked coords to reference resolution
            cur_x = click_x * REF_W / image_width if image_width > 0 else click_x
            cur_y = click_y * REF_H / image_height if image_height > 0 else click_y
            
            print(f"[Calibration]   {label}: click=({click_x:.1f}, {click_y:.1f}) → scaled=({cur_x:.1f}, {cur_y:.1f}), ref=({ref_x}, {ref_y})")
            
            found_points[label] = (cur_x, cur_y)
            ref_pts.append([ref_x, ref_y])
            cur_pts.append([cur_x, cur_y])
        
        num_matched = len(ref_pts)
        print(f"[Calibration] Computing full affine from {num_matched} clicked points (image_size={image_width}x{image_height}, ref_size={REF_W}x{REF_H})")
        
        # ref_pts and cur_pts are used directly via zip below
        
        # Solve 6-DOF affine via numpy least squares (no OpenCV version issues)
        # Affine: x' = a*x + b*y + tx,  y' = c*x + d*y + ty
        # Build system: A * [a,b,tx,c,d,ty]^T = b
        A_rows = []
        b_vec = []
        for (rx, ry), (cx, cy) in zip(ref_pts, cur_pts):
            A_rows.append([rx, ry, 1, 0, 0, 0])
            A_rows.append([0, 0, 0, rx, ry, 1])
            b_vec.append(cx)
            b_vec.append(cy)
        
        A_mat = np.array(A_rows, dtype=np.float64)
        b_arr = np.array(b_vec, dtype=np.float64)
        params, residuals, rank, sv = np.linalg.lstsq(A_mat, b_arr, rcond=None)
        
        affine_2x3 = np.array([
            [params[0], params[1], params[2]],
            [params[3], params[4], params[5]]
        ], dtype=np.float64)
        
        # Convert 2x3 to 3x3 for perspectiveTransform/warpPerspective compatibility
        H = np.eye(3, dtype=np.float64)
        H[:2, :] = affine_2x3
        
        H_inv = np.linalg.inv(H)
        
        print(f"[Calibration] Success! Affine computed from {num_matched} points.")
        print(f"[Calibration]   Affine matrix:\n{affine_2x3}")
        
        # Log per-point residuals
        max_err = 0
        for i, label in enumerate(labels[:num_matched]):
            rx, ry = ref_pts[i]
            cx, cy = cur_pts[i]
            tx = affine_2x3[0, 0] * rx + affine_2x3[0, 1] * ry + affine_2x3[0, 2]
            ty = affine_2x3[1, 0] * rx + affine_2x3[1, 1] * ry + affine_2x3[1, 2]
            err = np.sqrt((tx - cx)**2 + (ty - cy)**2)
            max_err = max(max_err, err)
            print(f"[Calibration]   {label}: ref=({rx:.0f},{ry:.0f}) → xform=({tx:.1f},{ty:.1f}), clicked=({cx:.0f},{cy:.0f}), err={err:.1f}px")
        
        print(f"[Calibration]   Max residual: {max_err:.1f}px")
        
        return H, H_inv, found_points


# --- Calibration UI Helper Functions ---

CALIBRATION_POINT_LABELS = list(CenterCameraCalibrator.REFERENCE_POINTS.keys())
CALIBRATION_REQUIRED_POINTS = len(CALIBRATION_POINT_LABELS)
NO_SCAN_POINTS_PER_BOX = 4
SIDE_CAMERA_BOX_LABELS = {
    "blue": ["MIDDLE", "LEFT", "FAR LEFT"],
    "red": ["MIDDLE", "RIGHT", "FAR RIGHT"],
}
SIDE_CAMERA_BOX_POINT_COUNT = 6


def _extract_composite_calibration_frames(video_path: str, start_seconds: float = 0) -> tuple:
    """Extract center / blue / red preview frames from the composite video for calibration."""
    if video_path is None:
        return None, None, None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[Calibration] Failed to open video for composite preview extraction")
        return None, None, None

    target_ms = (start_seconds + 4) * 1000
    cap.set(cv2.CAP_PROP_POS_MSEC, target_ms)
    ret, frame = cap.read()
    actual_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    cap.release()

    print(f"[Calibration] Requested preview frame at {target_ms:.0f}ms, got {actual_ms:.0f}ms")
    if not ret:
        print("[Calibration] Failed to read composite preview frame")
        return None, None, None

    h, w = frame.shape[:2]
    crops = _build_composite_crop_layout(w, h)

    images = {}
    for name, crop_info in crops.items():
        x1, y1, x2, y2 = crop_info["rect"]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            images[name] = None
            continue
        images[name] = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    return images.get("center"), images.get("blue"), images.get("red")


def _get_side_box_point_labels(camera_side: str) -> list:
    """Return ordered top-left / bottom-right click labels for side-box calibration."""
    side_name = str(camera_side).strip().lower()
    labels = SIDE_CAMERA_BOX_LABELS.get(side_name, SIDE_CAMERA_BOX_LABELS["blue"])
    point_labels = []
    for label in labels:
        point_labels.extend([f"{label} TL", f"{label} BR"])
    return point_labels


def _get_side_box_calibration_status_text(camera_side: str, num_points: int) -> str:
    """Instruction text for side-camera box calibration."""
    point_labels = _get_side_box_point_labels(camera_side)
    side_title = "Blue" if str(camera_side).strip().lower() == "blue" else "Red"
    if num_points <= 0:
        return f"**{side_title} side:** Click **{point_labels[0]}** (1 of {len(point_labels)})"
    if num_points < len(point_labels):
        return f"**{side_title} side:** Click **{point_labels[num_points]}** ({num_points + 1} of {len(point_labels)})"
    return f"**{side_title} side boxes set!** Click Process Video to use them."


def _side_box_pairs_to_rects(clicked_points: list, camera_side: str) -> list:
    """Convert side-camera click pairs into normalized rectangles."""
    labels = SIDE_CAMERA_BOX_LABELS.get(str(camera_side).strip().lower(), SIDE_CAMERA_BOX_LABELS["blue"])
    rects = []
    points = list(clicked_points or [])
    for idx, label in enumerate(labels):
        pair = points[idx * 2:(idx * 2) + 2]
        if len(pair) < 2:
            continue
        (x1, y1), (x2, y2) = pair
        rects.append((label, (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))))
    return rects


def _extract_side_camera_calibration_boxes(clicked_points: list, image_size: tuple,
                                           frame_width: int, frame_height: int,
                                           camera_side: str) -> list:
    """Scale side-camera calibration rectangles from UI image coords to frame coords."""
    if not clicked_points or not image_size or frame_width <= 0 or frame_height <= 0:
        return []

    src_w, src_h = image_size
    if src_w <= 0 or src_h <= 0:
        return []

    boxes = []
    for label, (x1, y1, x2, y2) in _side_box_pairs_to_rects(clicked_points, camera_side):
        sx1 = int(round((x1 / src_w) * frame_width))
        sy1 = int(round((y1 / src_h) * frame_height))
        sx2 = int(round((x2 / src_w) * frame_width))
        sy2 = int(round((y2 / src_h) * frame_height))
        boxes.append((label, (min(sx1, sx2), min(sy1, sy2), max(sx1, sx2), max(sy1, sy2))))
    return boxes


def _draw_side_camera_box_overlay(base_image: Image.Image, boxes: list, camera_side: str) -> Image.Image:
    """Draw side-camera guidance boxes on top of an image."""
    if base_image is None:
        return None

    img = base_image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = get_font(18)
    is_blue = str(camera_side).strip().lower() == "blue"
    line_color = (80, 180, 255, 180) if is_blue else (255, 110, 110, 180)
    fill_color = (80, 180, 255, 30) if is_blue else (255, 110, 110, 30)
    tag_fill = (18, 18, 18, 175)
    text_fill = (255, 255, 255, 235)

    for label, (x1, y1, x2, y2) in boxes or []:
        draw.rectangle([x1, y1, x2, y2], outline=line_color, width=3, fill=fill_color)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        tag_pad_x = 10
        tag_pad_y = 6
        tag_x1 = max(x1 + 8, min(x2 - text_w - tag_pad_x * 2 - 8, x1 + 14))
        tag_y1 = max(6, y1 + 6)
        tag_x2 = tag_x1 + text_w + tag_pad_x * 2
        tag_y2 = tag_y1 + text_h + tag_pad_y * 2
        draw.rounded_rectangle([tag_x1, tag_y1, tag_x2, tag_y2], radius=8, fill=tag_fill)
        draw.text((tag_x1 + tag_pad_x, tag_y1 + tag_pad_y - 1), label, fill=text_fill, font=font)

    return Image.alpha_composite(img, overlay).convert("RGB")


def _redraw_side_calibration_image(base_image: Image.Image, clicked_points: list, camera_side: str) -> Image.Image:
    """Redraw a side-camera calibration image with completed boxes and click markers."""
    if base_image is None:
        return None

    img = _draw_side_camera_box_overlay(base_image, _side_box_pairs_to_rects(clicked_points, camera_side), camera_side)
    draw = ImageDraw.Draw(img)
    font = get_font(15)
    point_labels = _get_side_box_point_labels(camera_side)
    is_blue = str(camera_side).strip().lower() == "blue"
    point_color = (80, 180, 255) if is_blue else (255, 110, 110)

    for i, (px, py) in enumerate(clicked_points or []):
        label = point_labels[i] if i < len(point_labels) else f"P{i + 1}"
        radius = 6
        draw.ellipse([px - radius, py - radius, px + radius, py + radius], fill=point_color, outline=(255, 255, 255), width=2)
        draw.text((px + 10, py - 10), label, fill=point_color, font=font)

    return img


def _get_calibration_status_text(num_points: int) -> str:
    """Return the next-step instruction for calibration / no-scan clicks."""
    if num_points <= 0:
        return "**Click point B1** (1 of 8)"

    if num_points < CALIBRATION_REQUIRED_POINTS:
        next_label = CALIBRATION_POINT_LABELS[num_points]
        return f"**Click point {next_label}** ({num_points + 1} of 8)"

    if num_points == CALIBRATION_REQUIRED_POINTS:
        return (
            "**Calibration points set!** Optional: click **Z1.1** to start the first "
            "center-camera no-scan box, or click 'Process Video' now."
        )

    extra_points = num_points - CALIBRATION_REQUIRED_POINTS
    next_label = _get_center_calibration_click_label(num_points)
    current_box = (extra_points // NO_SCAN_POINTS_PER_BOX) + 1
    next_point_in_box = (extra_points % NO_SCAN_POINTS_PER_BOX) + 1
    if extra_points % NO_SCAN_POINTS_PER_BOX == 0:
        completed_boxes = extra_points // NO_SCAN_POINTS_PER_BOX
        return (
            f"**{completed_boxes} no-scan box(es) set!** Optional: click **{next_label}** "
            "to start another box, or click 'Process Video' now."
        )

    return (
        f"**Click point {next_label}** "
        f"(no-scan box {current_box}, point {next_point_in_box} of 4)"
    )


def _get_center_calibration_click_label(index: int) -> str:
    """Return the UI label for a center-camera calibration / exclusion click."""
    if index < CALIBRATION_REQUIRED_POINTS:
        return CALIBRATION_POINT_LABELS[index]
    extra_index = index - CALIBRATION_REQUIRED_POINTS
    box_index = (extra_index // NO_SCAN_POINTS_PER_BOX) + 1
    point_index = (extra_index % NO_SCAN_POINTS_PER_BOX) + 1
    return f"Z{box_index}.{point_index}"


def _split_calibration_and_exclusion_points(clicked_points: list) -> tuple:
    """Split UI clicks into homography points and optional robot no-scan polygons."""
    points = list(clicked_points or [])
    calibration_points = points[:CALIBRATION_REQUIRED_POINTS]
    extra_points = points[CALIBRATION_REQUIRED_POINTS:]

    polygons = []
    for idx in range(0, len(extra_points), NO_SCAN_POINTS_PER_BOX):
        polygon = extra_points[idx:idx + NO_SCAN_POINTS_PER_BOX]
        if len(polygon) == NO_SCAN_POINTS_PER_BOX:
            polygons.append(polygon)

    return calibration_points, polygons


def _scale_polygon_points(points: list, src_size: tuple, dst_size: tuple) -> list:
    """Scale polygon points from UI image coordinates to frame coordinates."""
    if not points or not src_size or not dst_size:
        return []

    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
        return []

    scaled = []
    for px, py in points:
        sx = int(round((px / src_w) * dst_w))
        sy = int(round((py / src_h) * dst_h))
        scaled.append((sx, sy))
    return scaled


def _extract_robot_exclusion_polygons(clicked_points: list, image_size: tuple,
                                      frame_width: int, frame_height: int) -> list:
    """Return completed no-scan polygons scaled to the actual video frame."""
    _, polygons = _split_calibration_and_exclusion_points(clicked_points)
    return [
        _scale_polygon_points(poly, image_size, (frame_width, frame_height))
        for poly in polygons
        if len(poly) == 4
    ]


def _redraw_calibration_image(base_image: Image.Image, clicked_points: list) -> Image.Image:
    """Redraw the calibration image with all clicked points marked."""
    if base_image is None:
        return None
    img = base_image.copy()
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size=18)
    except:
        font = ImageFont.load_default()
    
    _, exclusion_polygons = _split_calibration_and_exclusion_points(clicked_points)

    for polygon in exclusion_polygons:
        draw.line(polygon + [polygon[0]], fill=(255, 220, 0), width=3)

    for i, (px, py) in enumerate(clicked_points):
        label = _get_center_calibration_click_label(i)
        color = (0, 200, 255) if label.startswith('B') else (255, 100, 100)
        radius = 8
        draw.ellipse([px - radius, py - radius, px + radius, py + radius], fill=color, outline=(255, 255, 255), width=2)
        draw.text((px + 12, py - 10), label, fill=color, font=font)
    
    return img


def _on_video_upload(video_path, start_seconds):
    """When video is uploaded, extract a frame for calibration."""
    if video_path is None:
        return None, [], "Upload a video to begin calibration"
    
    frame = CenterCameraCalibrator.extract_calibration_frame(video_path, start_seconds or 0)
    if frame is None:
        return None, [], "Failed to extract frame from video"
    
    return frame, [], "**Click point B1** (1 of 8) — Blue side, bottom-left field landmark"


def _on_image_click(base_image, clicked_points, evt: gr.SelectData):
    """Handle a click on the calibration image."""
    if base_image is None:
        return None, clicked_points, "Upload a video first"
    
    x, y = evt.index
    clicked_points = list(clicked_points) + [(x, y)]
    
    n = len(clicked_points)
    annotated = _redraw_calibration_image(base_image, clicked_points)
    
    if n >= 8:
        status = "**All 8 points set!** ✅ Click 'Process Video' to start."
    else:
        next_label = CALIBRATION_POINT_LABELS[n]
        status = f"**Click point {next_label}** ({n + 1} of 8)"
    
    return annotated, clicked_points, status


def _on_undo_click(base_image, clicked_points):
    """Remove the last clicked point."""
    if not clicked_points:
        return base_image, clicked_points, "No points to undo"
    
    clicked_points = list(clicked_points)[:-1]
    
    n = len(clicked_points)
    if n == 0:
        annotated = base_image
    else:
        annotated = _redraw_calibration_image(base_image, clicked_points)
    
    next_label = CALIBRATION_POINT_LABELS[n]
    status = f"**Click point {next_label}** ({n + 1} of 8) — Undid last point"
    
    return annotated, clicked_points, status


def _on_skip_click():
    """Skip calibration entirely."""
    return [], "**Calibration skipped** — processing will use default alignment"

def transform_to_map(bbox_center_x: float, bbox_center_y: float,
                     frame_width: int, frame_height: int,
                     map_width: int, map_height: int,
                     camera_side: str = "blue") -> tuple:
    """
    Transform camera coordinates to bird's eye map coordinates.
    
    Automatically selects the correct transformation based on camera_side.
    
    Args:
        bbox_center_x: X center of bounding box
        bbox_center_y: Y center of bounding box
        frame_width: Width of the video frame
        frame_height: Height of the video frame
        map_width: Width of the map image
        map_height: Height of the map image
        camera_side: "blue", "red", or "center"
        
    Returns:
        (map_x, map_y) coordinates on the map
    """
    if camera_side == "center":
        return center_camera_to_map_coords(bbox_center_x, bbox_center_y, 
                                           frame_width, frame_height,
                                           map_width, map_height)
    else:
        return camera_to_map_coords(bbox_center_x, bbox_center_y,
                                    frame_width, frame_height,
                                    map_width, map_height, camera_side)


# get_robot_color is defined above (line ~732) with alliance-specific color shades


def draw_robot_paths(map_image_path: str, robot_tracks: dict, frame_width: int, frame_height: int, camera_side: str = "blue", blue_robots: list = None, red_robots: list = None, max_seconds: float = None, fps: float = 30.0) -> Image.Image:
    """
    Draw robot movement paths on the field map.
    
    Args:
        map_image_path: Path to the map image
        robot_tracks: Dict mapping robot label to list of (bbox_center_x, bbox_center_y, camera_side) over time
        frame_width: Original video frame width
        frame_height: Original video frame height
        camera_side: Default camera side for backwards compatibility
        blue_robots: List of blue alliance team numbers [robot1, robot2, robot3]
        red_robots: List of red alliance team numbers [robot1, robot2, robot3]
        max_seconds: Optional limit on how many seconds of data to show (e.g., 15 for autonomous only)
        fps: Frame rate used for calculating max_frames from max_seconds
        
    Returns:
        PIL Image with paths drawn
    """
    # Load map
    try:
        map_img = Image.open(map_image_path).convert('RGB')
        # Rotate 90° counterclockwise (left)
        map_img = map_img.rotate(90, expand=True)
    except:
        # Create a blank field if map not found (landscape after rotation)
        map_img = Image.new('RGB', (1200, 600), color=(200, 200, 200))
    
    map_width, map_height = map_img.size
    draw = ImageDraw.Draw(map_img)
    
    # Default to empty lists if not provided
    blue_robots = blue_robots or []
    red_robots = red_robots or []
    
    # Calculate max frames if time limit specified
    max_frames = None
    if max_seconds is not None:
        max_frames = int(max_seconds * fps)
    
    # Get font for labels
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size=12)
    except:
        font = ImageFont.load_default()
    
    # Draw each robot's path
    for i, (robot_label, positions) in enumerate(robot_tracks.items()):
        if len(positions) < 1:
            continue
        
        # Limit positions to max_frames if specified
        if max_frames is not None:
            positions = positions[:max_frames]
        
        # Get the robot's alliance color
        robot_color = get_robot_color(robot_label, blue_robots, red_robots)
        
        # Convert all positions to map coordinates
        map_positions = []
        for pos in positions:
            # Handle 2-tuple (cx, cy), 3-tuple (cx, cy, side), and longer tuples
            # such as (cx, cy, side, bbox_area[, shooting]).
            if len(pos) >= 4:
                cx, cy, side = pos[:3]
            elif len(pos) == 3:
                cx, cy, side = pos
            else:
                cx, cy = pos
                side = camera_side
            map_x, map_y = transform_to_map(cx, cy, frame_width, frame_height, map_width, map_height, side)
            map_positions.append((map_x, map_y))
        
        num_positions = len(map_positions)
        
        # Draw lines connecting positions using the robot's alliance color
        if num_positions >= 2:
            for j in range(num_positions - 1):
                # Fade the color slightly based on position in path (older = more faded)
                progress = j / (num_positions - 1)
                # Interpolate from darker (start) to full color (end)
                fade_factor = 0.5 + 0.5 * progress
                faded_color = tuple(int(c * fade_factor) for c in robot_color)
                draw.line([map_positions[j], map_positions[j + 1]], fill=faded_color, width=3)
        
        # Draw points at each position with the robot's color
        for j, (mx, my) in enumerate(map_positions):
            progress = j / max(1, num_positions - 1)
            fade_factor = 0.5 + 0.5 * progress
            point_color = tuple(int(c * fade_factor) for c in robot_color)
            # Point size decreases for older positions
            radius = max(3, 8 - j // 5)
            draw.ellipse([(mx - radius, my - radius), (mx + radius, my + radius)], fill=point_color)
        
        # Draw start marker (darker version of robot color with circle)
        if map_positions:
            mx, my = map_positions[0]
            start_color = tuple(max(0, c - 50) for c in robot_color)
            draw.ellipse([(mx - 10, my - 10), (mx + 10, my + 10)], outline=start_color, width=2)
            draw.text((mx + 12, my - 6), f"Start: {robot_label}", fill=robot_color, font=font)
        
        # Draw end marker (brighter version of robot color with square)
        if num_positions > 1:
            mx, my = map_positions[-1]
            end_color = tuple(min(255, c + 50) for c in robot_color)
            draw.rectangle([(mx - 8, my - 8), (mx + 8, my + 8)], outline=end_color, width=2)
    
    # Add legend with alliance colors
    legend_y = 10
    for robot_label, _ in robot_tracks.items():
        color = get_robot_color(robot_label, blue_robots, red_robots)
        draw.rectangle([(10, legend_y), (25, legend_y + 15)], fill=color)
        draw.text((30, legend_y), robot_label[:30], fill=color, font=font)
        legend_y += 20
    
    return map_img


def interpolate_robot_tracks(robot_tracks_by_frame: list, max_gap: int = 15) -> list:
    """
    Interpolate robot positions to fill gaps when robots aren't detected.
    Creates smooth movement paths instead of jumps.
    
    Args:
        robot_tracks_by_frame: List of dicts, each dict maps robot label to (cx, cy, camera_side) or (cx, cy, camera_side, area)
        max_gap: Maximum number of frames to interpolate across (larger gaps are left as-is)
        
    Returns:
        New list with interpolated positions filled in
    """
    if not robot_tracks_by_frame:
        return robot_tracks_by_frame
    
    # Get all unique robot labels
    all_labels = set()
    for frame_data in robot_tracks_by_frame:
        all_labels.update(frame_data.keys())
    
    # Create a copy of the tracks to modify
    interpolated = [dict(frame_data) for frame_data in robot_tracks_by_frame]
    
    # For each robot, find gaps and interpolate
    for label in all_labels:
        # Find all frames where this robot appears
        appearances = []
        for frame_idx, frame_data in enumerate(robot_tracks_by_frame):
            if label in frame_data:
                appearances.append((frame_idx, frame_data[label]))
        
        if len(appearances) < 2:
            continue  # Need at least 2 points to interpolate
        
        # Fill gaps between consecutive appearances
        for i in range(len(appearances) - 1):
            start_frame, start_pos = appearances[i]
            end_frame, end_pos = appearances[i + 1]
            
            gap_size = end_frame - start_frame - 1
            
            if gap_size <= 0 or gap_size > max_gap:
                continue  # No gap or too large to interpolate
            
            # Extract positions (handle 3-tuple and 4-tuple formats)
            if len(start_pos) >= 4:
                start_x, start_y, start_side, start_area = start_pos[:4]
            elif len(start_pos) == 3:
                start_x, start_y, start_side = start_pos
                start_area = None
            else:
                start_x, start_y = start_pos[:2]
                start_side = "blue"
                start_area = None
            
            if len(end_pos) >= 4:
                end_x, end_y, end_side, end_area = end_pos[:4]
            elif len(end_pos) == 3:
                end_x, end_y, end_side = end_pos
                end_area = None
            else:
                end_x, end_y = end_pos[:2]
                end_area = None
            
            # Linear interpolation for each frame in the gap
            for gap_idx in range(1, gap_size + 1):
                interp_frame = start_frame + gap_idx
                t = gap_idx / (gap_size + 1)  # Interpolation factor [0, 1]
                
                interp_x = start_x + (end_x - start_x) * t
                interp_y = start_y + (end_y - start_y) * t
                
                # Use the camera side from the start position
                if start_area is not None and end_area is not None:
                    interp_area = start_area + (end_area - start_area) * t
                    interpolated[interp_frame][label] = (interp_x, interp_y, start_side, interp_area)
                else:
                    interpolated[interp_frame][label] = (interp_x, interp_y, start_side)
    
    return interpolated





def generate_map_video(map_image_path: str, robot_tracks_by_frame: list, frame_width: int, frame_height: int, target_fps: int = 3, trail_length: int = 10, blue_robots: list = None, red_robots: list = None) -> str:
    """
    Generate a video of the map showing robot positions over time with trailing effect.
    
    Args:
        map_image_path: Path to the map image
        robot_tracks_by_frame: List of dicts, each dict maps robot label to (cx, cy, camera_side) for that frame
        frame_width: Original video frame width
        frame_height: Original video frame height
        target_fps: FPS for output video
        trail_length: Number of previous frames to show as trail
        blue_robots: List of blue alliance team numbers [robot1, robot2, robot3]
        red_robots: List of red alliance team numbers [robot1, robot2, robot3]
        
    Returns:
        Path to the generated map video
    """
    # Load base map
    try:
        base_map = Image.open(map_image_path).convert('RGB')
        # Rotate 90° counterclockwise (left)
        base_map = base_map.rotate(90, expand=True)
    except:
        # Create a blank field if map not found (landscape after rotation)
        base_map = Image.new('RGB', (1200, 600), color=(200, 200, 200))
    
    map_width, map_height = base_map.size
    
    # Default to empty lists if not provided
    blue_robots = blue_robots or []
    red_robots = red_robots or []
    
    # Get font for labels
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size=14)
    except:
        font = ImageFont.load_default()
    
    # Create output video
    output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    fourcc_options = ['avc1', 'H264', 'mp4v', 'XVID']
    out = None
    for codec in fourcc_options:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, float(target_fps), (map_width, map_height))
            if out.isOpened():
                break
        except:
            continue
    
    if out is None or not out.isOpened():
        output_path = tempfile.NamedTemporaryFile(suffix=".avi", delete=False).name
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path, fourcc, float(target_fps), (map_width, map_height))
    
    if not out.isOpened():
        return None
    
    # Track all robots we've seen for the legend
    all_robot_labels = set()
    
    # Generate a frame for each timestep
    for frame_idx, frame_data in enumerate(robot_tracks_by_frame):
        # Start with fresh map
        map_frame = base_map.copy()
        draw = ImageDraw.Draw(map_frame)
        
        # Calculate trail start index
        trail_start = max(0, frame_idx - trail_length)
        
        # Track robots in this frame
        for robot_label in frame_data.keys():
            all_robot_labels.add(robot_label)
        
        # Draw trail points from previous frames
        for trail_idx in range(trail_start, frame_idx + 1):
            if trail_idx >= len(robot_tracks_by_frame):
                continue
                
            trail_data = robot_tracks_by_frame[trail_idx]
            age = frame_idx - trail_idx  # 0 = current, higher = older
            
            for robot_label, pos in trail_data.items():
                all_robot_labels.add(robot_label)
                
                # Get alliance-based color for this robot
                base_color = get_robot_color(robot_label, blue_robots, red_robots)
                
                # Handle 2-tuple, 3-tuple, and longer tuple formats.
                if len(pos) >= 4:
                    cx, cy, side = pos[:3]
                elif len(pos) == 3:
                    cx, cy, side = pos
                else:
                    cx, cy = pos
                    side = "blue"
                
                map_x, map_y = transform_to_map(cx, cy, frame_width, frame_height, map_width, map_height, side)
                
                # Calculate opacity/size based on age (newer = bigger/brighter)
                fade = 1.0 - (age / (trail_length + 1))
                radius = int(4 + 8 * fade)
                
                # Fade color based on age
                faded_color = tuple(int(c * fade + 100 * (1 - fade)) for c in base_color)
                
                # Draw trail point
                draw.ellipse([(map_x - radius, map_y - radius), (map_x + radius, map_y + radius)], fill=faded_color)
                
                # Draw label for current position only
                if age == 0:
                    draw.text((map_x + radius + 2, map_y - 7), robot_label, fill=base_color, font=font)
        
        # Draw connecting lines between trail points for each robot
        for robot_label in all_robot_labels:
            # Get alliance-based color for this robot
            base_color = get_robot_color(robot_label, blue_robots, red_robots)
            
            trail_positions = []
            for trail_idx in range(trail_start, frame_idx + 1):
                if trail_idx >= len(robot_tracks_by_frame):
                    continue
                if robot_label in robot_tracks_by_frame[trail_idx]:
                    pos = robot_tracks_by_frame[trail_idx][robot_label]
                    # Handle 2-tuple, 3-tuple, and longer tuple formats.
                    if len(pos) >= 4:
                        cx, cy, side = pos[:3]
                    elif len(pos) == 3:
                        cx, cy, side = pos
                    else:
                        cx, cy = pos
                        side = "blue"
                    map_x, map_y = transform_to_map(cx, cy, frame_width, frame_height, map_width, map_height, side)
                    trail_positions.append((map_x, map_y))
            
            # Draw lines connecting trail
            if len(trail_positions) >= 2:
                for i in range(len(trail_positions) - 1):
                    age = len(trail_positions) - i - 2
                    fade = 1.0 - (age / (trail_length + 1))
                    faded_color = tuple(int(c * fade + 100 * (1 - fade)) for c in base_color)
                    draw.line([trail_positions[i], trail_positions[i + 1]], fill=faded_color, width=2)
        
        # Add legend with alliance colors
        legend_y = 10
        for robot_label in sorted(all_robot_labels):
            color = get_robot_color(robot_label, blue_robots, red_robots)
            draw.rectangle([(10, legend_y), (25, legend_y + 15)], fill=color)
            draw.text((30, legend_y), robot_label[:20], fill=color, font=font)
            legend_y += 20
        
        # Convert to OpenCV format and write
        frame_bgr = cv2.cvtColor(np.array(map_frame), cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    return output_path


# Pre-allocated kernel for morphology operations (performance optimization)
_MORPH_KERNEL_3x3 = np.ones((3, 3), np.uint8)


def _validate_ball_colors(frame_bgr: np.ndarray, contour: np.ndarray,
                          min_stddev: float = 15.0,
                          max_blue_green_ratio: float = 0.4,
                          rg_ratio_range: tuple = (0.6, 1.4)) -> bool:
    """
    Validate that a contour's pixels look like a real yellow ball.
    
    Uses brightness-invariant colour-ratio checks so both bright and dark
    balls pass, while brown objects and non-yellow surfaces are rejected.
    
    Three checks:
      1. Blue channel must be much lower than Green (B/G < 0.4).
         Yellow balls:  B/G ≈ 0.1–0.3.  Brown: B/G ≈ 0.5+.
      2. Red and Green must be close (R/G between 0.6 and 1.4).
         Yellow balls:  R/G ≈ 0.9–1.05.  Brown: R/G ≈ 1.5+.
      3. Pixel stddev must exceed min_stddev in at least one channel
         (rejects flat-coloured walls; real balls have light/dark shading).
    
    Args:
        frame_bgr: Full BGR frame.
        contour: The contour to validate.
        min_stddev: Minimum stddev required in at least one channel.
        max_blue_green_ratio: Maximum allowed mean_B / mean_G.
        rg_ratio_range: (min, max) allowed mean_R / mean_G.
    
    Returns:
        True if the contour looks like a real ball.
    """
    # Build a mask covering only this contour's bounding box
    bx, by, bw, bh = cv2.boundingRect(contour)
    roi = frame_bgr[by:by+bh, bx:bx+bw]
    mask = np.zeros((bh, bw), dtype=np.uint8)
    shifted = contour - [bx, by]
    cv2.drawContours(mask, [shifted], 0, 255, -1)

    pixels = roi[mask == 255].astype(np.float32)
    if len(pixels) < 5:
        return False

    mean_b, mean_g, mean_r = pixels.mean(axis=0)

    # Guard against near-black regions (camera noise)
    if mean_g < 5:
        return False

    # Check 1: Blue must be much lower than Green (characteristic of yellow)
    if mean_b / mean_g > max_blue_green_ratio:
        return False

    # Check 2: Red ≈ Green  (yellow), not R >> G (brown/orange)
    rg_ratio = mean_r / mean_g
    if rg_ratio < rg_ratio_range[0] or rg_ratio > rg_ratio_range[1]:
        return False

    # Check 3: enough colour variance (real balls have light/dark shading)
    stddev = pixels.std(axis=0)
    if np.all(stddev < min_stddev):
        return False

    return True



def detect_fuel(frame_bgr: np.ndarray, min_radius: int = 3, max_radius: int = 30,
                tracked_positions: list = None) -> list:
    """
    Detect yellow fuel balls using HSV color-based detection with separation of overlapping balls.
    
    Supports tracking hysteresis: if tracked_positions are provided (predicted locations of
    already-tracked balls), contours near those positions use relaxed colour validation
    so balls don't flicker in and out of detection.
    
    Args:
        frame_bgr: OpenCV BGR image
        min_radius: Minimum radius for fuel detection (pixels) - default 3 for distant balls
        max_radius: Maximum radius for fuel detection (pixels) - default 30 for close balls
        tracked_positions: Optional list of (x, y, radius) from BallTracker.get_predicted_positions()
        
    Returns:
        List of (x, y, radius) tuples for detected fuel
    """
    # Ball colour validation thresholds
    MIN_COLOR_STDDEV = 8.0
    # Relaxed thresholds for contours near already-tracked balls
    RELAXED_BG_RATIO = 0.55          # vs strict 0.4
    RELAXED_RG_RANGE = (0.5, 1.6)    # vs strict (0.6, 1.4)
    RELAXED_STDDEV = 4.0             # vs strict 8.0
    TRACKED_MATCH_DIST = 100         # pixels — how close to a predicted pos to use relaxed mode

    # Define yellow-green color range in HSV
    lower_yellow = np.array([15, 60, 40])
    upper_yellow = np.array([85, 255, 255])
    
    try:
        # GPU Acceleration Path (using OpenCV T-API / OpenCL)
        # Uploading to UMat automatically uses GPU if available
        umat_frame = cv2.UMat(frame_bgr)
        hsv_umat = cv2.cvtColor(umat_frame, cv2.COLOR_BGR2HSV)
        
        # Thresholding on GPU
        mask_umat = cv2.inRange(hsv_umat, lower_yellow, upper_yellow)
        
        # Morphology on GPU - use pre-allocated kernel
        mask_umat = cv2.morphologyEx(mask_umat, cv2.MORPH_OPEN, _MORPH_KERNEL_3x3, iterations=1)
        mask_umat = cv2.morphologyEx(mask_umat, cv2.MORPH_CLOSE, _MORPH_KERNEL_3x3, iterations=1)
        
        # Download mask back to CPU for contour finding
        mask = mask_umat.get()
        
    except Exception:
        # CPU Fallback Path
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        
        # Create mask for yellow objects
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Apply morphological operations to reduce noise - use pre-allocated kernel
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _MORPH_KERNEL_3x3, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL_3x3, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    fuel_detections = []
    
    for contour in contours:
        # Calculate contour area
        area = cv2.contourArea(contour)
        
        # Skip if too small
        min_area = np.pi * (min_radius ** 2)
        max_area = np.pi * (max_radius ** 2)
        
        if area < min_area:
            continue
        
        # Calculate circularity to check if it's a single ball or multiple overlapping
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            continue
        
        # Determine if this contour is near a tracked ball (use relaxed thresholds)
        near_tracked = False
        if tracked_positions:
            (cx_c, cy_c), _ = cv2.minEnclosingCircle(contour)
            for tx, ty, tr in tracked_positions:
                if (cx_c - tx) ** 2 + (cy_c - ty) ** 2 < TRACKED_MATCH_DIST ** 2:
                    near_tracked = True
                    break
        
        # Validate colour ratios + variance (relaxed if near a tracked ball)
        if near_tracked:
            if not _validate_ball_colors(frame_bgr, contour,
                                        min_stddev=RELAXED_STDDEV,
                                        max_blue_green_ratio=RELAXED_BG_RATIO,
                                        rg_ratio_range=RELAXED_RG_RANGE):
                continue
        else:
            if not _validate_ball_colors(frame_bgr, contour, min_stddev=MIN_COLOR_STDDEV):
                continue

        # If area is within single ball range AND highly circular, accept as single ball
        if min_area <= area <= max_area and circularity > 0.75:
            # High circularity = likely a single ball
            (x, y), radius = cv2.minEnclosingCircle(contour)
            fuel_detections.append((int(x), int(y), int(radius)))
        
        # If area is too large OR circularity is low, it's likely overlapping balls
        elif area > max_area or (min_area <= area <= max_area and circularity <= 0.75):
            # Try to separate overlapping balls using distance transform with local maxima
            # OPTIMIZATION: Use bounding box instead of full-frame mask (10-50x smaller)
            bx, by, bw, bh = cv2.boundingRect(contour)
            
            # Create small mask just for this contour's bounding box region
            contour_mask = np.zeros((bh, bw), dtype=np.uint8)
            # Shift contour to bounding box origin
            shifted_contour = contour - [bx, by]
            cv2.drawContours(contour_mask, [shifted_contour], 0, 255, -1)
            
            # Distance transform on small mask
            dist_transform = cv2.distanceTransform(contour_mask, cv2.DIST_L2, 5)
            
            # Light smoothing to remove simple noise but preserve valleys between balls
            dist_transform = cv2.GaussianBlur(dist_transform, (3, 3), 0)
            
            # Estimate typical ball radius based on expected size (use conservative estimate)
            # Using median of range to be safe
            expected_ball_radius = (min_radius + max_radius) / 2
            
            # Find local maxima using dilation - kernel size roughly ball radius (not diameter)
            # Smaller kernel helps find close peaks
            kernel_size = max(3, int(expected_ball_radius * 0.7)) 
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            dilated = cv2.dilate(dist_transform, np.ones((kernel_size, kernel_size)))
            
            # Local maxima are where original equals dilated and distance is significant
            # Lower threshold (0.3) to allow smaller balls next to big ones
            max_val = dist_transform.max()
            local_max = (dist_transform == dilated) & (dist_transform > max(min_radius, max_val * 0.3))
            
            # Get coordinates of local maxima (in bounding box space)
            max_coords = np.where(local_max)
            
            # Filter close peaks
            final_peaks = []
            if len(max_coords[0]) > 0:
                points = list(zip(max_coords[1], max_coords[0])) # x, y (in bbox space)
                
                # Sort by distance value (radius) descending
                points.sort(key=lambda p: dist_transform[p[1], p[0]], reverse=True)
                
                for p in points:
                    x_local, y_local = p
                    radius = dist_transform[y_local, x_local]
                    
                    # Convert to frame coordinates by adding bounding box offset
                    x_frame = x_local + bx
                    y_frame = y_local + by
                    
                    # Check if too close to existing peak
                    # Uses dynamic threshold: if centers are closer than the sum of their radii * 0.6
                    # This implies significant overlap (>40%) is needed to merge
                    too_close = False
                    for existing_p, existing_r in final_peaks:
                        dist = np.sqrt((x_frame - existing_p[0])**2 + (y_frame - existing_p[1])**2)
                        
                        # Collision distance would be existing_r + radius
                        # We merge if they are essentially the same object (dist small)
                        min_dist_to_separate = (existing_r + radius) * 0.6
                        
                        if dist < min_dist_to_separate: 
                            too_close = True
                            break
                    
                    if not too_close and min_radius <= radius <= max_radius:
                        final_peaks.append(((x_frame, y_frame), int(radius)))
                        fuel_detections.append((int(x_frame), int(y_frame), int(radius)))
            else:
                # Fallback: if no local maxima found, use the centroid
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    _, radius = cv2.minEnclosingCircle(contour)
                    if min_radius <= radius <= max_radius:
                        fuel_detections.append((cx, cy, int(radius)))
    
    return fuel_detections


def _run_sam3_on_region(frame_bgr: np.ndarray, predictor,
                        min_radius: int, max_radius: int,
                        x_offset: int = 0, y_offset: int = 0) -> list:
    """
    Run SAM 3 on a single image region and return detections with coordinate offsets applied.
    
    Args:
        frame_bgr: OpenCV BGR image (the region to scan)
        predictor: SAM3SemanticPredictor instance
        min_radius: Minimum ball radius in pixels
        max_radius: Maximum ball radius in pixels
        x_offset: X offset to add back to detections (for ROI remapping)
        y_offset: Y offset to add back to detections (for ROI remapping)
        
    Returns:
        List of (x, y, radius) tuples in full-frame coordinates
    """
    temp_path = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name
    cv2.imwrite(temp_path, frame_bgr)
    
    detections = []
    try:
        predictor.set_image(temp_path)
        results = predictor(text=["yellow ball"])
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx = int((x1 + x2) / 2) + x_offset
                    cy = int((y1 + y2) / 2) + y_offset
                    radius = int(max(x2 - x1, y2 - y1) / 2)
                    if min_radius <= radius <= max_radius:
                        detections.append((cx, cy, radius))
    except Exception as e:
        print(f"SAM 3 ball detection error: {e}")
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
    
    return detections


def _apply_sam3_exclusion_polygons(frame_bgr: np.ndarray, exclusion_polygons: list = None,
                                   x_offset: int = 0, y_offset: int = 0) -> np.ndarray:
    """Black out excluded polygons before passing an image region to SAM3."""
    if frame_bgr is None or frame_bgr.size == 0 or not exclusion_polygons:
        return frame_bgr

    masked = frame_bgr.copy()
    region_h, region_w = masked.shape[:2]
    local_polygons = []

    for polygon in exclusion_polygons:
        if not polygon or len(polygon) < 3:
            continue

        points = []
        for px, py in polygon:
            lx = int(round(px - x_offset))
            ly = int(round(py - y_offset))
            points.append((lx, ly))

        polygon_np = np.array(points, dtype=np.int32)
        if polygon_np.size == 0:
            continue

        min_x = int(np.min(polygon_np[:, 0]))
        max_x = int(np.max(polygon_np[:, 0]))
        min_y = int(np.min(polygon_np[:, 1]))
        max_y = int(np.max(polygon_np[:, 1]))

        if max_x < 0 or max_y < 0 or min_x >= region_w or min_y >= region_h:
            continue

        local_polygons.append(polygon_np)

    if local_polygons:
        cv2.fillPoly(masked, local_polygons, (0, 0, 0))

    return masked


# Center camera ROI regions (1918x709 frame)
# Only the bottom-left and bottom-right corners contain scoring areas
_CENTER_CAM_ROIS = [
    (0,    129, 730,  709),   # Left side  — 730x580
    (1188, 129, 1918, 709),   # Right side — 730x580
]


def detect_fuel_sam3(frame_bgr: np.ndarray, predictor,
                     min_radius: int = 3, max_radius: int = 30,
                     camera_side: str = "blue",
                     exclusion_polygons: list = None) -> list:
    """
    Detect yellow fuel balls using SAM 3 semantic segmentation.
    
    Uses text-prompted segmentation with the query "yellow ball" for
    more robust detection across varying lighting conditions.
    
    For center camera, only scans two ROI regions (bottom-left and
    bottom-right corners) where the scoring areas are located.
    
    Args:
        frame_bgr: OpenCV BGR image
        predictor: Initialized SAM3SemanticPredictor instance
        min_radius: Minimum radius for fuel detection (pixels)
        max_radius: Maximum radius for fuel detection (pixels)
        camera_side: "blue", "red", or "center" camera perspective
        
    Returns:
        List of (x, y, radius) tuples for detected fuel
    """
    if camera_side == "center":
        h, w = frame_bgr.shape[:2]
        sx = w / 1918 if w > 0 else 1.0
        sy = h / 709 if h > 0 else 1.0

        fuel_detections = []
        for (rx1, ry1, rx2, ry2) in _CENTER_CAM_ROIS:
            x1 = int(max(0, min(w, rx1 * sx)))
            y1 = int(max(0, min(h, ry1 * sy)))
            x2 = int(max(0, min(w, rx2 * sx)))
            y2 = int(max(0, min(h, ry2 * sy)))

            if x2 > x1 and y2 > y1:
                roi = frame_bgr[y1:y2, x1:x2]
                roi = _apply_sam3_exclusion_polygons(
                    roi,
                    exclusion_polygons=exclusion_polygons,
                    x_offset=x1,
                    y_offset=y1,
                )
                detections = _run_sam3_on_region(
                    roi,
                    predictor,
                    min_radius,
                    max_radius,
                    x_offset=x1,
                    y_offset=y1,
                )
                fuel_detections.extend(detections)
        return fuel_detections
    else:
        # Side cameras: scan the full frame
        return _run_sam3_on_region(frame_bgr, predictor, min_radius, max_radius)


def _draw_fuel_detections_legacy(frame: Image.Image, fuel_detections: list, blue_robots: list = None, red_robots: list = None) -> Image.Image:
    """
    Draw bounding boxes around detected fuel, including robot labels for shot balls.
    
    Args:
        frame: PIL Image
        fuel_detections: List of (x, y, radius) or (x, y, radius, robot_label) tuples
        blue_robots: List of blue alliance team numbers for color coding
        red_robots: List of red alliance team numbers for color coding
        
    Returns:
        PIL Image with fuel detections drawn
    """
    frame = frame.copy()
    draw = ImageDraw.Draw(frame)
    
    # Default color
    fuel_color = "#FFD700"  # Gold
    
    font = get_font(12)
    label_font = get_font(14)
    
    blue_robots = blue_robots or []
    red_robots = red_robots or []
    
    for detection in fuel_detections:
        # Handle both 3-tuple and 4-tuple formats
        if len(detection) == 4:
            x, y, radius, robot_label = detection
        else:
            x, y, radius = detection
            robot_label = None
        
        # Choose color based on whether ball was shot
        if robot_label:
            # Ball was shot - use robot's alliance color
            color_rgb = get_robot_color(robot_label, blue_robots, red_robots)
            circle_color = rgb_to_hex(color_rgb)
            outline_width = 3
        else:
            circle_color = fuel_color
            outline_width = 2
        
        # Draw circle around fuel
        draw.ellipse(
            [(x - radius, y - radius), (x + radius, y + radius)],
            outline=circle_color,
            width=outline_width
        )
        
        # Draw label
        if robot_label:
            # Draw robot number above ball
            label_text = f"🎯 {robot_label}"
            draw.text((x - 20, y - radius - 18), label_text, fill=circle_color, font=label_font)
        else:
            # Draw simple fuel indicator
            label = "⚽"
            draw.text((x - 8, y - radius - 15), label, fill=circle_color, font=font)
    
    return frame


def draw_fuel_detections(frame: Image.Image, fuel_detections: list, blue_robots: list = None,
                         red_robots: list = None, highlight_robot_label: str = None) -> Image.Image:
    """
    Draw fuel detections plus optional predicted trajectories.

    Args:
        frame: PIL Image
        fuel_detections: List of tuples or dict payloads from BallTracker.update()
        blue_robots: List of blue alliance team numbers for color coding
        red_robots: List of red alliance team numbers for color coding
        highlight_robot_label: Only this robot's attributed balls stay fully highlighted

    Returns:
        PIL Image with fuel detections drawn
    """
    frame = frame.copy().convert("RGBA")
    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font = get_font(12)
    label_font = get_font(14)

    blue_robots = blue_robots or []
    red_robots = red_robots or []
    highlight_robot_label = _normalize_highlight_ball_robot(highlight_robot_label)

    def _normalize_detection(detection):
        if isinstance(detection, dict):
            return {
                'x': detection.get('x', 0),
                'y': detection.get('y', 0),
                'radius': detection.get('radius', 0),
                'robot_label': detection.get('robot_label'),
                'predicted_path': list(detection.get('predicted_path') or []),
                'predicted_make': detection.get('predicted_make'),
                'predicted_only': bool(detection.get('predicted_only', False)),
            }

        if len(detection) == 4:
            x, y, radius, robot_label = detection
        else:
            x, y, radius = detection
            robot_label = None

        return {
            'x': x,
            'y': y,
            'radius': radius,
            'robot_label': robot_label,
            'predicted_path': [],
            'predicted_make': None,
            'predicted_only': False,
        }

    for detection in fuel_detections:
        payload = _normalize_detection(detection)
        x = int(round(payload['x']))
        y = int(round(payload['y']))
        radius = max(1, int(round(payload['radius'])))
        robot_label = payload['robot_label']
        predicted_path = [
            (int(round(px)), int(round(py)))
            for px, py in payload['predicted_path']
        ]
        predicted_make = payload['predicted_make']
        predicted_only = payload['predicted_only']
        is_selected_robot = (
            not highlight_robot_label
            or (robot_label is not None and str(robot_label).strip() == highlight_robot_label)
        )

        if robot_label and is_selected_robot:
            color_rgb = get_robot_color(robot_label, blue_robots, red_robots)
            outline_width = 3
        elif highlight_robot_label:
            color_rgb = (205, 205, 205) if robot_label else (185, 185, 165)
            outline_width = 1 if predicted_only else 2
        else:
            color_rgb = (255, 215, 0)
            outline_width = 2

        if highlight_robot_label and not is_selected_robot:
            circle_rgba = (*color_rgb, 110 if not predicted_only else 80)
            path_rgba = (*color_rgb, 90 if not predicted_only else 60)
        else:
            circle_rgba = (*color_rgb, 220 if not predicted_only else 150)
            if predicted_make is True:
                path_rgba = (90, 255, 120, 210 if not predicted_only else 150)
            else:
                path_rgba = (*color_rgb, 170 if not predicted_only else 110)

        if len(predicted_path) >= 2:
            draw.line(predicted_path, fill=path_rgba, width=3)
            for idx, (px, py) in enumerate(predicted_path[1:], start=1):
                if predicted_only and idx % 2 == 1:
                    continue
                dot_radius = 2 if idx < len(predicted_path) - 1 else 3
                draw.ellipse(
                    [(px - dot_radius, py - dot_radius), (px + dot_radius, py + dot_radius)],
                    fill=path_rgba
                )

        draw.ellipse(
            [(x - radius, y - radius), (x + radius, y + radius)],
            outline=circle_rgba,
            width=outline_width
        )

        if robot_label and is_selected_robot:
            if predicted_make is True:
                suffix = " IN"
            else:
                suffix = ""
            prefix = "PRED " if predicted_only else ""
            label_text = f"{prefix}{robot_label}{suffix}"
            draw.text((x - 24, y - radius - 18), label_text, fill=circle_rgba, font=label_font)
        elif not highlight_robot_label:
            draw.text((x - 10, y - radius - 15), "fuel", fill=circle_rgba, font=font)

    return Image.alpha_composite(frame, overlay).convert("RGB")


def extract_bbox_centers(bounding_boxes_json: str, frame_width: int, frame_height: int, filter_unknown: bool = True) -> dict:
    """
    Extract center points of detected bounding boxes along with their areas.
    
    Args:
        bounding_boxes_json: JSON string with detections
        frame_width: Frame width
        frame_height: Frame height
        filter_unknown: If True, skip robots labeled 'robot', 'unknown', 'Unknown' (for map display)
        
    Returns:
        Dict mapping label to (center_x, center_y, bbox_area)
    """
    centers = {}
    # Labels to exclude from map (still shown in video)
    unknown_labels = {'robot', 'unknown', 'Unknown'}
    
    try:
        bboxes = json.loads(parse_json(bounding_boxes_json))
        for bbox in bboxes:
            label = bbox.get('label', 'Unknown')
            
            # Skip unknown/unidentified robots for map display
            if filter_unknown and label in unknown_labels:
                continue
                
            box = bbox.get('box_2d', [])
            if len(box) >= 4:
                # box_2d format: [y1, x1, y2, x2] normalized to 1000
                y1 = float(box[0]) / 1000 * frame_height
                x1 = float(box[1]) / 1000 * frame_width
                y2 = float(box[2]) / 1000 * frame_height
                x2 = float(box[3]) / 1000 * frame_width
                
                center_x = (x1 + x2) / 2
                # Use 1/3 from bottom instead of center for better ground-plane estimation
                center_y = y2 - (y2 - y1) / 3
                # Calculate bounding box area for weighted averaging
                bbox_area = (x2 - x1) * (y2 - y1)
                centers[label] = (center_x, center_y, bbox_area)
    except Exception as e:
        print(f"Error extracting bbox centers: {e}")
    
    return centers


def get_font(size: int = 14):
    """Get a font for drawing text, with fallbacks."""
    try:
        # Try common Windows fonts
        font_paths = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "arial.ttf",
        ]
        for font_path in font_paths:
            try:
                return ImageFont.truetype(font_path, size=size)
            except:
                continue
        # Fallback to default
        return ImageFont.load_default()
    except:
        return ImageFont.load_default()


def plot_bounding_boxes(img: Image.Image, bounding_boxes_json: str, blue_robots: list = None, red_robots: list = None, stats: dict = None, show_unlabeled: bool = True) -> Image.Image:
    """
    Plots bounding boxes on an image with markers for each label and optional stats.
    
    Args:
        img: PIL Image to draw on
        bounding_boxes_json: JSON string containing bounding boxes
        blue_robots: List of blue alliance team numbers [robot1, robot2, robot3]
        red_robots: List of red alliance team numbers [robot1, robot2, robot3]
        stats: Dict mapping robot label to {'made': int, 'attempts': int}
        
    Returns:
        PIL Image with bounding boxes drawn
    """
    img = img.copy()
    width, height = img.size
    draw = ImageDraw.Draw(img)
    
    # Default to empty lists if not provided
    blue_robots = blue_robots or []
    red_robots = red_robots or []
    stats = stats or {}
    
    # Parse the JSON
    bounding_boxes_str = parse_json(bounding_boxes_json)
    
    try:
        bounding_boxes = json.loads(bounding_boxes_str)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Raw response: {bounding_boxes_json}")
        return img
    
    font = get_font(16)
    
    for i, bounding_box in enumerate(bounding_boxes):
        # Get team number and determine color based on alliance
        team_number = bounding_box.get("label", f"Object {i+1}")
        
        # Skip unlabeled robots if the user chose to hide them
        if not show_unlabeled and team_number in ("robot", "unknown"):
            continue
        
        color_rgb = get_robot_color(team_number, blue_robots, red_robots)
        color_hex = rgb_to_hex(color_rgb)
        
        # Format label with stats if available
        label_text = str(team_number)
        if team_number in stats:
            made = stats[team_number]['made']
            if made > 0:
                label_text += f" - {made} made"
        
        # Convert normalized coordinates to absolute coordinates
        # Format: [y1, x1, y2, x2] normalized to 1000
        try:
            # Handle both string and numeric values from API
            if "box_2d" not in bounding_box:
                continue  # Skip malformed detections (e.g. from overloaded API)
            box = bounding_box["box_2d"]
            abs_y1 = int(float(box[0]) / 1000 * height)
            abs_x1 = int(float(box[1]) / 1000 * width)
            abs_y2 = int(float(box[2]) / 1000 * height)
            abs_x2 = int(float(box[3]) / 1000 * width)
            
            # Ensure correct order
            if abs_x1 > abs_x2:
                abs_x1, abs_x2 = abs_x2, abs_x1
            if abs_y1 > abs_y2:
                abs_y1, abs_y2 = abs_y2, abs_y1
            
            # Draw bounding box
            draw.rectangle([abs_x1, abs_y1, abs_x2, abs_y2], outline=color_hex, width=3)
            
            # Draw label background
            text_bbox = draw.textbbox((abs_x1, abs_y1), label_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            draw.rectangle([abs_x1, abs_y1 - text_height - 4, abs_x1 + text_width + 8, abs_y1], fill=color_hex)
            
            # Draw label text
            draw.text((abs_x1 + 4, abs_y1 - text_height - 2), label_text, fill="white", font=font)
            
        except (ValueError, IndexError, TypeError) as e:
            print(f"Error drawing bbox {i}: {e}")
            continue
            
    return img


# Pre-allocated morphology kernels for bumper detection
_BUMPER_MORPH_KERNEL = np.ones((5, 5), np.uint8)
_BUMPER_BRIDGE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
_BUMPER_WHITE_PROXIMITY_KERNEL = np.ones((30, 30), np.uint8)
_BUMPER_STRUCTURE_MARGIN_KERNEL = np.ones((9, 9), np.uint8)

# White/near-white HSV range for team number text on bumpers
# Covers pure white rgb(255,255,255) and pinkish-white rgb(203,181,199)
_BUMPER_WHITE_LOWER = np.array([0, 0, 160])
_BUMPER_WHITE_UPPER = np.array([180, 60, 255])

# Center-camera blue bumper tuning.
# The extra dark-navy range keeps very dark blue bumpers visible while the
# channel-dominance gate prevents near-black field structures from matching.
_BUMPER_BLUE_LOWER = np.array([92, 100, 30])
_BUMPER_BLUE_UPPER = np.array([122, 255, 200])
_BUMPER_DARK_BLUE_LOWER = np.array([96, 65, 18])
_BUMPER_DARK_BLUE_UPPER = np.array([132, 255, 90])
_BUMPER_BLUE_MIN_DOMINANCE = 10
_BUMPER_BLUE_NEAR_BLACK_VALUE_MAX = 55
_BUMPER_BLUE_NEAR_BLACK_SPREAD_MAX = 10

# Center-camera red bumper tuning.
# Includes darker maroon bumpers such as rgb(68, 19, 30) while using the
# field-structure exclusion mask to keep persistent deep-red structures out.
_BUMPER_RED1_LOWER = np.array([0, 80, 80])
_BUMPER_RED1_UPPER = np.array([10, 255, 220])
_BUMPER_RED2_LOWER = np.array([160, 80, 80])
_BUMPER_RED2_UPPER = np.array([180, 255, 220])
_BUMPER_DARK_RED1_LOWER = np.array([0, 110, 35])
_BUMPER_DARK_RED1_UPPER = np.array([12, 255, 120])
_BUMPER_DARK_RED2_LOWER = np.array([165, 110, 35])
_BUMPER_DARK_RED2_UPPER = np.array([180, 255, 120])
_BUMPER_RED_MIN_DOMINANCE = 10
_BUMPER_RED_NEAR_BLACK_VALUE_MAX = 40
_BUMPER_RED_NEAR_BLACK_SPREAD_MAX = 10

# Minimum contour area / color pixels for bumper detection (reject small noise)
_BUMPER_MIN_AREA = 90
_BUMPER_MIN_COLOR_PIXELS = 65
_BUMPER_MERGE_GAP_X = 90
_BUMPER_MERGE_GAP_Y = 45
_BUMPER_MAX_BOX_WIDTH = 170
_BUMPER_MAX_BOX_HEIGHT = 110

# Center camera playing field ROI (x1, y1, x2, y2) — excludes audience areas
_BUMPER_FIELD_ROI = (0, 315, 1918, 705)


def _build_center_red_mask(field_region_bgr: np.ndarray, hsv_region: np.ndarray) -> np.ndarray:
    """
    Detect center-camera red bumpers, including darker maroon shades such as
    rgb(68, 19, 30), while rejecting nearly neutral black structures.
    """
    base_red_mask = cv2.bitwise_or(
        cv2.inRange(hsv_region, _BUMPER_RED1_LOWER, _BUMPER_RED1_UPPER),
        cv2.inRange(hsv_region, _BUMPER_RED2_LOWER, _BUMPER_RED2_UPPER)
    )
    dark_red_mask = cv2.bitwise_or(
        cv2.inRange(hsv_region, _BUMPER_DARK_RED1_LOWER, _BUMPER_DARK_RED1_UPPER),
        cv2.inRange(hsv_region, _BUMPER_DARK_RED2_LOWER, _BUMPER_DARK_RED2_UPPER)
    )

    if field_region_bgr is None or field_region_bgr.size == 0:
        return cv2.bitwise_or(base_red_mask, dark_red_mask)

    blue = field_region_bgr[:, :, 0].astype(np.int16)
    green = field_region_bgr[:, :, 1].astype(np.int16)
    red = field_region_bgr[:, :, 2].astype(np.int16)
    value = hsv_region[:, :, 2].astype(np.int16)

    channel_max = np.maximum(np.maximum(blue, green), red)
    channel_min = np.minimum(np.minimum(blue, green), red)
    channel_spread = channel_max - channel_min

    red_dominant = (
        (red >= (green + _BUMPER_RED_MIN_DOMINANCE)) &
        (red >= (blue + _BUMPER_RED_MIN_DOMINANCE))
    )
    near_black_structure = (
        (value <= _BUMPER_RED_NEAR_BLACK_VALUE_MAX) &
        (channel_spread <= _BUMPER_RED_NEAR_BLACK_SPREAD_MAX)
    )

    red_gate = np.where(red_dominant & ~near_black_structure, 255, 0).astype(np.uint8)
    return cv2.bitwise_and(cv2.bitwise_or(base_red_mask, dark_red_mask), red_gate)


def _build_center_blue_mask(field_region_bgr: np.ndarray, hsv_region: np.ndarray) -> np.ndarray:
    """
    Detect center-camera blue bumpers, including dark navy shades such as
    rgb(18, 21, 38), while rejecting nearly neutral black structures.
    """
    base_blue_mask = cv2.inRange(hsv_region, _BUMPER_BLUE_LOWER, _BUMPER_BLUE_UPPER)
    dark_blue_mask = cv2.inRange(hsv_region, _BUMPER_DARK_BLUE_LOWER, _BUMPER_DARK_BLUE_UPPER)

    if field_region_bgr is None or field_region_bgr.size == 0:
        return cv2.bitwise_or(base_blue_mask, dark_blue_mask)

    blue = field_region_bgr[:, :, 0].astype(np.int16)
    green = field_region_bgr[:, :, 1].astype(np.int16)
    red = field_region_bgr[:, :, 2].astype(np.int16)
    value = hsv_region[:, :, 2].astype(np.int16)

    channel_max = np.maximum(np.maximum(blue, green), red)
    channel_min = np.minimum(np.minimum(blue, green), red)
    channel_spread = channel_max - channel_min

    blue_dominant = (
        (blue >= (green + _BUMPER_BLUE_MIN_DOMINANCE)) &
        (blue >= (red + _BUMPER_BLUE_MIN_DOMINANCE))
    )
    near_black_structure = (
        (value <= _BUMPER_BLUE_NEAR_BLACK_VALUE_MAX) &
        (channel_spread <= _BUMPER_BLUE_NEAR_BLACK_SPREAD_MAX)
    )

    blue_gate = np.where(blue_dominant & ~near_black_structure, 255, 0).astype(np.uint8)
    return cv2.bitwise_and(cv2.bitwise_or(base_blue_mask, dark_blue_mask), blue_gate)


def _bumper_far_corner_sensitivity(x1: int, y1: int, x2: int, y2: int,
                                   roi_x1: int, roi_y1: int,
                                   roi_x2: int, roi_y2: int) -> float:
    """
    Return a 0-1 sensitivity boost for distant robots near the top-left/top-right
    of the calibrated field ROI, where robots appear smaller.
    """
    roi_w = max(1, roi_x2 - roi_x1)
    roi_h = max(1, roi_y2 - roi_y1)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    nx = (cx - roi_x1) / roi_w
    ny = (cy - roi_y1) / roi_h

    top_factor = float(np.clip((0.55 - ny) / 0.55, 0.0, 1.0))
    side_distance = abs(nx - 0.5)
    side_factor = float(np.clip((side_distance - 0.18) / 0.32, 0.0, 1.0))
    return top_factor * side_factor


def _get_calibrated_field_roi() -> tuple:
    """Return _BUMPER_FIELD_ROI adjusted by the calibration homography.

    Transforms the four corners of the reference ROI through the forward
    homography, then takes the axis-aligned bounding box.  Falls back to
    the static ROI when no calibration is active.
    """
    x1, y1, x2, y2 = _BUMPER_FIELD_ROI

    fn = globals().get('center_camera_to_map_coords')
    H_fwd = getattr(fn, 'calibration_homography', None) if fn else None
    if H_fwd is None:
        return _BUMPER_FIELD_ROI

    # ROI corners in reference resolution (1918x709)
    corners = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(corners, H_fwd.astype(np.float32))
    t = transformed[0]

    # Axis-aligned bounding box, clamped to reference frame
    new_x1 = max(0, int(np.floor(t[:, 0].min())))
    new_y1 = max(0, int(np.floor(t[:, 1].min())))
    new_x2 = min(1918, int(np.ceil(t[:, 0].max())))
    new_y2 = min(709,  int(np.ceil(t[:, 1].max())))

    return (new_x1, new_y1, new_x2, new_y2)


def _merge_bumper_boxes(boxes: list,
                        gap_x: int = _BUMPER_MERGE_GAP_X,
                        gap_y: int = _BUMPER_MERGE_GAP_Y) -> list:
    """
    Merge nearby bumper fragments into robot-sized boxes without widening color thresholds.
    """
    merged = [list(box) for box in boxes]
    changed = True

    while changed:
        changed = False
        next_boxes = []
        used = [False] * len(merged)

        for i, box_a in enumerate(merged):
            if used[i]:
                continue

            current = list(box_a)
            used[i] = True

            expanded = True
            while expanded:
                expanded = False
                cx1, cy1, cx2, cy2 = current
                ccy = (cy1 + cy2) / 2.0

                for j, box_b in enumerate(merged):
                    if used[j]:
                        continue

                    bx1, by1, bx2, by2 = box_b
                    bcy = (by1 + by2) / 2.0

                    overlaps_x = not (bx1 > cx2 or bx2 < cx1)
                    overlaps_y = not (by1 > cy2 or by2 < cy1)
                    horiz_gap = max(0, max(bx1 - cx2, cx1 - bx2))
                    vert_gap = max(0, max(by1 - cy2, cy1 - by2))
                    center_y_gap = abs(bcy - ccy)

                    if (overlaps_x and overlaps_y) or (
                        horiz_gap <= gap_x and
                        vert_gap <= gap_y and
                        center_y_gap <= gap_y
                    ):
                        current = [
                            min(cx1, bx1),
                            min(cy1, by1),
                            max(cx2, bx2),
                            max(cy2, by2),
                        ]
                        used[j] = True
                        expanded = True
                        changed = True
                        break

            next_boxes.append(tuple(current))

        merged = next_boxes

    return merged


def _has_large_internal_horizontal_gap(mask_slice: np.ndarray,
                                       min_gap_px: int = 24,
                                       min_gap_fraction: float = 0.30) -> bool:
    """
    Detect merged boxes that actually contain two disconnected side blobs with a
    wide empty middle section.

    Real robot bumpers can have white team-number gaps, but the bridged contour
    mask should keep those connected. Large empty interior spans usually mean we
    merged unrelated fragments across open space.
    """
    if mask_slice is None or mask_slice.size == 0:
        return False

    col_has_signal = np.any(mask_slice > 0, axis=0)
    active_cols = np.flatnonzero(col_has_signal)
    if active_cols.size < 2:
        return False

    left = int(active_cols[0])
    right = int(active_cols[-1])
    span_width = right - left + 1
    if span_width < (min_gap_px * 2):
        return False

    interior = col_has_signal[left:right + 1]
    longest_gap = 0
    current_gap = 0
    for has_signal in interior:
        if has_signal:
            current_gap = 0
            continue
        current_gap += 1
        if current_gap > longest_gap:
            longest_gap = current_gap

    return longest_gap >= min_gap_px and (longest_gap / span_width) >= min_gap_fraction


def _split_oversized_bumper_box(box: tuple, contour_mask: np.ndarray, color_mask: np.ndarray,
                                max_width: int = _BUMPER_MAX_BOX_WIDTH,
                                max_height: int = _BUMPER_MAX_BOX_HEIGHT) -> list:
    """
    Break oversized merged boxes into smaller robot-sized boxes.

    Splits along the lowest-signal seam in the original color mask so merged
    robots are separated before final validation.
    """
    pending = [tuple(box)]
    final_boxes = []

    while pending:
        x1, y1, x2, y2 = pending.pop(0)
        box_w = max(0, x2 - x1)
        box_h = max(0, y2 - y1)
        if box_w <= max_width and box_h <= max_height:
            final_boxes.append((x1, y1, x2, y2))
            continue

        split_vertical = box_w > max_width and (
            box_h <= max_height or
            (box_w / max(1, max_width)) >= (box_h / max(1, max_height))
        )

        signal_slice = color_mask[y1:y2, x1:x2]
        if signal_slice.size == 0 or cv2.countNonZero(signal_slice) == 0:
            signal_slice = contour_mask[y1:y2, x1:x2]

        if signal_slice.size == 0:
            continue

        if split_vertical:
            projection = np.sum(signal_slice > 0, axis=0)
            if projection.size < 2:
                final_boxes.append((x1, y1, x2, y2))
                continue
            candidate_offsets = np.arange(1, projection.size)
            seam_scores = projection[:-1] + projection[1:]
            midpoint = projection.size / 2.0
            seam_scores = seam_scores + (np.abs(candidate_offsets - midpoint) * 0.25)
            split_offset = int(candidate_offsets[np.argmin(seam_scores)])
            split_x = x1 + split_offset
            child_boxes = [
                (x1, y1, split_x, y2),
                (split_x, y1, x2, y2),
            ]
        else:
            projection = np.sum(signal_slice > 0, axis=1)
            if projection.size < 2:
                final_boxes.append((x1, y1, x2, y2))
                continue
            candidate_offsets = np.arange(1, projection.size)
            seam_scores = projection[:-1] + projection[1:]
            midpoint = projection.size / 2.0
            seam_scores = seam_scores + (np.abs(candidate_offsets - midpoint) * 0.25)
            split_offset = int(candidate_offsets[np.argmin(seam_scores)])
            split_y = y1 + split_offset
            child_boxes = [
                (x1, y1, x2, split_y),
                (x1, split_y, x2, y2),
            ]

        valid_children = []
        for child_x1, child_y1, child_x2, child_y2 in child_boxes:
            if child_x2 <= child_x1 or child_y2 <= child_y1:
                continue
            child_signal = color_mask[child_y1:child_y2, child_x1:child_x2]
            if child_signal.size == 0 or cv2.countNonZero(child_signal) == 0:
                child_signal = contour_mask[child_y1:child_y2, child_x1:child_x2]
                if child_signal.size == 0 or cv2.countNonZero(child_signal) == 0:
                    continue
            valid_children.append((child_x1, child_y1, child_x2, child_y2))

        if not valid_children:
            final_boxes.append((x1, y1, x2, y2))
            continue

        pending = valid_children + pending

    return final_boxes



def compute_field_pixel_mask(video_path: str, start_seconds: float = 0,
                             sample_fps: float = 3.0,
                             threshold: float = 0.40) -> np.ndarray:
    """
    Pre-scan the center camera video to build a per-pixel field exclusion mask.

    Any pixel in the field ROI that is red or blue in >= `threshold` fraction of
    sampled frames is considered a static field element and will be excluded from
    robot bumper detection.  The first 3 seconds of the video are always skipped
    (robots may still be settling into position).

    Args:
        video_path: Path to the center camera video file.
        start_seconds: Global start offset already applied to the video (so we
                       skip an additional 3 s on top of this).
        sample_fps: How many frames per second to sample (default 3).
        threshold: Fraction of frames a pixel must be red/blue to be excluded.

    Returns:
        Binary mask the same size as the field ROI (h, w) where
        0 = excluded (field element) and 255 = allowed (potential robot).
    """
    roi_x1, roi_y1, roi_x2, roi_y2 = _get_calibrated_field_roi()
    roi_h = roi_y2 - roi_y1
    roi_w = roi_x2 - roi_x1

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[FieldMask] Could not open video – returning empty mask")
        return np.ones((roi_h, roi_w), dtype=np.uint8) * 255

    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Skip the first 3 seconds (on top of any user-specified start offset)
    skip_seconds = 3.0
    first_valid_frame = int((start_seconds + skip_seconds) * original_fps)
    sample_interval = max(1, int(original_fps / sample_fps))

    # ROI already computed above via _get_calibrated_field_roi()

    # Accumulators (float32 to avoid overflow for long videos)
    red_blue_count = np.zeros((roi_h, roi_w), dtype=np.float32)
    grey_count = np.zeros((roi_h, roi_w), dtype=np.float32)
    yellow_count = np.zeros((roi_h, roi_w), dtype=np.float32)
    frame_count = 0

    # Wide HSV ranges to catch field element shades but NOT yellow balls.
    # Yellow balls → HSV H≈25-35, high S/V.
    #   Red1 upper hue capped at 20 to avoid catching yellow.
    lower_red1 = np.array([0, 25, 15])
    upper_red1 = np.array([20, 255, 255])
    lower_red2 = np.array([140, 25, 15])
    upper_red2 = np.array([180, 255, 255])
    # Grey carpet range (low saturation, mid-range value).
    # Covers greys from dark ~(90,90,85) to light ~(170,170,165).
    # Any hue is OK since saturation is nearly zero for true greys.
    lower_grey = np.array([0, 0, 70])
    upper_grey = np.array([180, 35, 190])
    # Yellow fuel can temporarily cover carpet, so persistent yellow should also
    # protect those pixels from being classified as static field structure.
    lower_yellow = np.array([15, 60, 40])
    upper_yellow = np.array([85, 255, 255])
    # Fraction of frames a pixel must be grey to be protected from exclusion
    grey_protect_threshold = 0.10

    cap.set(cv2.CAP_PROP_POS_FRAMES, first_valid_frame)
    current_frame = first_valid_frame

    print(f"[FieldMask] Scanning video for field pixel mask "
          f"(frames {first_valid_frame}–{total_frames}, sample every {sample_interval} frames) ...")

    while current_frame < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if (current_frame - first_valid_frame) % sample_interval == 0:
            # Crop to field ROI (clamp to actual frame dimensions)
            h_full, w_full = frame.shape[:2]
            rx1 = max(0, min(roi_x1, w_full))
            ry1 = max(0, min(roi_y1, h_full))
            rx2 = max(0, min(roi_x2, w_full))
            ry2 = max(0, min(roi_y2, h_full))
            field_region = frame[ry1:ry2, rx1:rx2]

            fh, fw = field_region.shape[:2]
            if fh == 0 or fw == 0:
                current_frame += 1
                continue

            hsv = cv2.cvtColor(field_region, cv2.COLOR_BGR2HSV)

            # Red, including deeper maroon shades that should also count as
            # persistent field structure if they remain static in the center view.
            red_pix = cv2.bitwise_or(
                cv2.bitwise_or(
                    cv2.inRange(hsv, lower_red1, upper_red1),
                    cv2.inRange(hsv, lower_red2, upper_red2)
                ),
                _build_center_red_mask(field_region, hsv)
            )

            # Blue, including dark navy bumpers while filtering nearly-black structures.
            blue_pix = _build_center_blue_mask(field_region, hsv)

            # Combined: pixel is red OR blue in this frame
            combined = cv2.bitwise_or(red_pix, blue_pix)

            # Grey carpet pixels
            grey_pix = cv2.inRange(hsv, lower_grey, upper_grey)
            yellow_pix = cv2.inRange(hsv, lower_yellow, upper_yellow)

            # Accumulate (only the overlapping region in case of size mismatch)
            ah = min(fh, roi_h)
            aw = min(fw, roi_w)
            red_blue_count[:ah, :aw] += (combined[:ah, :aw] > 0).astype(np.float32)
            grey_count[:ah, :aw] += (grey_pix[:ah, :aw] > 0).astype(np.float32)
            yellow_count[:ah, :aw] += (yellow_pix[:ah, :aw] > 0).astype(np.float32)
            frame_count += 1

        current_frame += 1

    cap.release()

    if frame_count == 0:
        print("[FieldMask] No frames sampled – returning empty mask")
        return np.ones((roi_h, roi_w), dtype=np.uint8) * 255

    # Compute per-pixel frequency
    frequency = red_blue_count / frame_count

    # Build exclusion mask: 0 = field element (excluded), 255 = allowed
    mask = np.where(frequency >= threshold, 0, 255).astype(np.uint8)

    # Carpet protection: force-allow pixels that are frequently grey or covered by
    # yellow fuel, preventing pooled balls from causing carpet to be treated as
    # static field structure.
    grey_frequency = grey_count / frame_count
    yellow_frequency = yellow_count / frame_count
    carpet_protected = (
        (grey_frequency >= grey_protect_threshold) |
        (yellow_frequency >= grey_protect_threshold)
    )
    mask[carpet_protected] = 255

    excluded_pixels = int(np.sum(mask == 0))
    protected_pixels = int(np.sum(carpet_protected & (frequency >= threshold)))
    total_pixels = mask.size
    excluded_pct = excluded_pixels / total_pixels * 100
    print(f"[FieldMask] Computed field pixel mask from {frame_count} frames: "
          f"{excluded_pct:.1f}% of ROI excluded ({excluded_pixels}/{total_pixels} pixels), "
          f"{protected_pixels} pixels protected by grey carpet filter")

    return mask


def detect_people_yolo(frame_bgr: np.ndarray, confidence: float = 0.35) -> tuple:
    """
    Detect and segment people in the playing field region using YOLO-seg (class 0 = person).
    
    Args:
        frame_bgr: OpenCV BGR image (full frame)
        confidence: Minimum confidence for person detections
        
    Returns:
        Tuple of (person_mask, person_count):
        - person_mask: Binary mask (full frame size) where detected people are 255
        - person_count: Number of people detected
    """
    if YOLO_PERSON_MODEL is None:
        h, w = frame_bgr.shape[:2]
        return np.zeros((h, w), dtype=np.uint8), 0
    
    h_full, w_full = frame_bgr.shape[:2]
    roi_x1, roi_y1, roi_x2, roi_y2 = _get_calibrated_field_roi()
    
    # Clamp ROI to frame
    roi_x1 = max(0, min(roi_x1, w_full))
    roi_y1 = max(0, min(roi_y1, h_full))
    roi_x2 = max(0, min(roi_x2, w_full))
    roi_y2 = max(0, min(roi_y2, h_full))
    roi_h = roi_y2 - roi_y1
    roi_w = roi_x2 - roi_x1
    
    # Crop to field ROI
    roi_frame = frame_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
    
    # Run YOLO segmentation on the ROI
    results = YOLO_PERSON_MODEL(roi_frame, verbose=False, conf=confidence, classes=[0])
    
    person_mask = np.zeros((h_full, w_full), dtype=np.uint8)
    person_count = 0
    
    for result in results:
        if result.masks is not None:
            for i, mask_data in enumerate(result.masks.data):
                if int(result.boxes[i].cls[0]) == 0:  # class 0 = person
                    # mask_data is a tensor at model resolution, resize to ROI size
                    seg_mask = mask_data.cpu().numpy().astype(np.uint8)
                    seg_mask = cv2.resize(seg_mask, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
                    # Place into full-frame mask
                    person_mask[roi_y1:roi_y2, roi_x1:roi_x2] = np.maximum(
                        person_mask[roi_y1:roi_y2, roi_x1:roi_x2],
                        seg_mask * 255
                    )
                    person_count += 1
    
    return person_mask, person_count


def _build_robot_exclusion_mask(polygons: list, frame_width: int, frame_height: int) -> np.ndarray:
    """Build a binary allow-mask where user no-scan polygons are zeroed out."""
    mask = np.ones((frame_height, frame_width), dtype=np.uint8) * 255
    if not polygons:
        return mask

    for polygon in polygons:
        if len(polygon) < 3:
            continue
        pts = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 0)

    return mask


def detect_robots_by_bumper_color(frame_bgr: np.ndarray, person_mask: np.ndarray = None,
                                  field_pixel_mask: np.ndarray = None,
                                  robot_exclusion_polygons: list = None) -> tuple:
    """
    Detect robots by finding red and blue bumper regions using HSV color matching.
    
    Uses two HSV ranges for red (wraps around 0° in HSV) and one for blue.
    Returns bounding boxes in the same JSON format as other detection backends,
    plus raw masks for visual highlighting.
    
    Args:
        frame_bgr: OpenCV BGR image (center camera frame)
        person_mask: Optional binary mask of detected people (255 = person, 0 = not)
        field_pixel_mask: Optional per-pixel exclusion mask from compute_field_pixel_mask().
        robot_exclusion_polygons: Optional list of user-drawn no-scan polygons in frame coordinates.
        
    Returns:
        Tuple of (bounding_boxes_json, red_mask, blue_mask):
        - bounding_boxes_json: JSON string with detections in standard format
        - red_mask: Binary mask of red bumper pixels
        - blue_mask: Binary mask of blue bumper pixels
    """
    # Crop to playing field ROI (calibration-adjusted) to exclude audience areas
    roi_x1, roi_y1, roi_x2, roi_y2 = _get_calibrated_field_roi()
    h_full, w_full = frame_bgr.shape[:2]
    # Clamp ROI to actual frame dimensions
    roi_x1 = max(0, min(roi_x1, w_full))
    roi_y1 = max(0, min(roi_y1, h_full))
    roi_x2 = max(0, min(roi_x2, w_full))
    roi_y2 = max(0, min(roi_y2, h_full))
    field_region = frame_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
    robot_exclusion_mask_full = _build_robot_exclusion_mask(robot_exclusion_polygons, w_full, h_full)
    
    # Use dynamic field pixel mask if available
    fh, fw = field_region.shape[:2]
    if field_pixel_mask is not None:
        active_exc_mask = field_pixel_mask[:fh, :fw]
        # Expand excluded structure regions slightly so edge-adjacent color noise
        # does not turn into robot detections hugging field elements.
        active_exc_mask = cv2.erode(active_exc_mask, _BUMPER_STRUCTURE_MARGIN_KERNEL, iterations=1)
    else:
        active_exc_mask = np.ones((fh, fw), dtype=np.uint8) * 255

    exclusion_roi = robot_exclusion_mask_full[roi_y1:roi_y2, roi_x1:roi_x2][:fh, :fw]
    active_exc_mask = cv2.bitwise_and(active_exc_mask, exclusion_roi)

    try:
        # GPU Acceleration Path (using OpenCV T-API / OpenCL)
        umat_roi = cv2.UMat(field_region)
        hsv = cv2.cvtColor(umat_roi, cv2.COLOR_BGR2HSV)
        
        # Pull HSV back to CPU for the shared color builders.
        hsv_cpu = hsv.get()
        red_mask = _build_center_red_mask(field_region, hsv_cpu)
        
        # Blue bumper
        blue_mask = _build_center_blue_mask(field_region, hsv_cpu)

        # Apply the field-structure exclusion mask before morphology so excluded
        # deep-red structures cannot seed dilated/bridged robot contours.
        red_mask = cv2.bitwise_and(red_mask, active_exc_mask)
        blue_mask = cv2.bitwise_and(blue_mask, active_exc_mask)
        red_mask = cv2.UMat(red_mask)
        blue_mask = cv2.UMat(blue_mask)
        
        # Morphology on GPU
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, _BUMPER_MORPH_KERNEL, iterations=1)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, _BUMPER_MORPH_KERNEL, iterations=1)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, _BUMPER_BRIDGE_KERNEL, iterations=2)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, _BUMPER_BRIDGE_KERNEL, iterations=2)
        red_mask = cv2.dilate(red_mask, _BUMPER_MORPH_KERNEL, iterations=1)
        blue_mask = cv2.dilate(blue_mask, _BUMPER_MORPH_KERNEL, iterations=1)
        
        # Download back to CPU for contour finding
        red_mask = red_mask.get()
        blue_mask = blue_mask.get()
        
        # Zero out field element regions
        red_mask = cv2.bitwise_and(red_mask, active_exc_mask)
        blue_mask = cv2.bitwise_and(blue_mask, active_exc_mask)
        
        # Zero out person regions
        if person_mask is not None:
            person_roi = person_mask[roi_y1:roi_y2, roi_x1:roi_x2]
            person_roi = person_roi[:fh, :fw]
            inv_person = cv2.bitwise_not(person_roi)
            red_mask = cv2.bitwise_and(red_mask, inv_person)
            blue_mask = cv2.bitwise_and(blue_mask, inv_person)
        
    except Exception:
        # CPU Fallback Path
        hsv = cv2.cvtColor(field_region, cv2.COLOR_BGR2HSV)
        
        red_mask = _build_center_red_mask(field_region, hsv)
        blue_mask = _build_center_blue_mask(field_region, hsv)

        red_mask = cv2.bitwise_and(red_mask, active_exc_mask)
        blue_mask = cv2.bitwise_and(blue_mask, active_exc_mask)
        
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, _BUMPER_MORPH_KERNEL, iterations=1)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, _BUMPER_MORPH_KERNEL, iterations=1)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, _BUMPER_BRIDGE_KERNEL, iterations=2)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, _BUMPER_BRIDGE_KERNEL, iterations=2)
        red_mask = cv2.dilate(red_mask, _BUMPER_MORPH_KERNEL, iterations=1)
        blue_mask = cv2.dilate(blue_mask, _BUMPER_MORPH_KERNEL, iterations=1)
        
        # Zero out field element regions
        red_mask = cv2.bitwise_and(red_mask, active_exc_mask)
        blue_mask = cv2.bitwise_and(blue_mask, active_exc_mask)
        
        # Zero out person regions
        if person_mask is not None:
            person_roi = person_mask[roi_y1:roi_y2, roi_x1:roi_x2]
            person_roi = person_roi[:fh, :fw]
            inv_person = cv2.bitwise_not(person_roi)
            red_mask = cv2.bitwise_and(red_mask, inv_person)
            blue_mask = cv2.bitwise_and(blue_mask, inv_person)
    
    # Build white-bridged contour masks (for bounding box computation only).
    # White team number text on bumpers creates gaps between same-color bumper
    # sections (red-white-red). The bridge masks connect them into one bounding box
    # but do NOT modify the visual overlay masks (red_mask / blue_mask stay pure).
    hsv_cpu = cv2.cvtColor(field_region, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv_cpu, _BUMPER_WHITE_LOWER, _BUMPER_WHITE_UPPER)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, _BUMPER_MORPH_KERNEL, iterations=1)
    # Apply same exclusions to white mask
    exc_slice = active_exc_mask[:white_mask.shape[0], :white_mask.shape[1]] if field_pixel_mask is not None else active_exc_mask
    white_mask = cv2.bitwise_and(white_mask, exc_slice)
    if person_mask is not None:
        person_roi_w = person_mask[roi_y1:roi_y2, roi_x1:roi_x2][:fh, :fw]
        white_mask = cv2.bitwise_and(white_mask, cv2.bitwise_not(person_roi_w))
    # White near red/blue → bridge masks (color + adjacent white), then close gaps
    red_dilated = cv2.dilate(red_mask, _BUMPER_WHITE_PROXIMITY_KERNEL, iterations=1)
    blue_dilated = cv2.dilate(blue_mask, _BUMPER_WHITE_PROXIMITY_KERNEL, iterations=1)
    red_contour_mask = cv2.bitwise_or(red_mask, cv2.bitwise_and(white_mask, red_dilated))
    blue_contour_mask = cv2.bitwise_or(blue_mask, cv2.bitwise_and(white_mask, blue_dilated))
    red_contour_mask = cv2.morphologyEx(red_contour_mask, cv2.MORPH_CLOSE, _BUMPER_BRIDGE_KERNEL, iterations=1)
    blue_contour_mask = cv2.morphologyEx(blue_contour_mask, cv2.MORPH_CLOSE, _BUMPER_BRIDGE_KERNEL, iterations=1)
    
    # Expand ROI-sized masks back to full frame size (zeros outside the field)
    red_mask_full = np.zeros((h_full, w_full), dtype=np.uint8)
    red_mask_full[roi_y1:roi_y2, roi_x1:roi_x2] = red_mask
    red_mask = red_mask_full
    
    blue_mask_full = np.zeros((h_full, w_full), dtype=np.uint8)
    blue_mask_full[roi_y1:roi_y2, roi_x1:roi_x2] = blue_mask
    blue_mask = blue_mask_full
    
    # Expand contour masks to full frame size
    red_contour_full = np.zeros((h_full, w_full), dtype=np.uint8)
    red_contour_full[roi_y1:roi_y2, roi_x1:roi_x2] = red_contour_mask
    blue_contour_full = np.zeros((h_full, w_full), dtype=np.uint8)
    blue_contour_full[roi_y1:roi_y2, roi_x1:roi_x2] = blue_contour_mask
    active_exc_mask_full = np.zeros((h_full, w_full), dtype=np.uint8)
    active_exc_mask_full[roi_y1:roi_y2, roi_x1:roi_x2] = active_exc_mask
    
    # Find contours on bridged masks (wider bboxes) but validate each has real color pixels
    height, width = frame_bgr.shape[:2]
    detections = []
    raw_bboxes = []  # Raw pixel coordinates (x1, y1, x2, y2) for LLM cropping
    
    for contour_mask, color_mask, label in [
        (red_contour_full, red_mask, "red"),
        (blue_contour_full, blue_mask, "blue"),
    ]:
        contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidate_boxes = []

        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            sensitivity = _bumper_far_corner_sensitivity(
                x, y, x + w, y + h, roi_x1, roi_y1, roi_x2, roi_y2
            )
            min_area = _BUMPER_MIN_AREA * (1.0 - 0.45 * sensitivity)
            min_color_pixels = _BUMPER_MIN_COLOR_PIXELS * (1.0 - 0.45 * sensitivity)

            if area < min_area:
                continue
            
            # Verify this contour actually contains original color pixels (not just white)
            roi_slice = color_mask[y:y+h, x:x+w]
            color_pixels = cv2.countNonZero(roi_slice)
            if color_pixels < min_color_pixels:
                continue

            candidate_boxes.append((x, y, x + w, y + h))

        for merged_box in _merge_bumper_boxes(candidate_boxes):
            split_boxes = _split_oversized_bumper_box(merged_box, contour_mask, color_mask)
            for x1, y1, x2, y2 in split_boxes:
                sensitivity = _bumper_far_corner_sensitivity(
                    x1, y1, x2, y2, roi_x1, roi_y1, roi_x2, roi_y2
                )
                min_color_pixels = _BUMPER_MIN_COLOR_PIXELS * (1.0 - 0.45 * sensitivity)
                fill_ratio_min = max(0.008, 0.015 * (1.0 - 0.35 * sensitivity))
                allowed_ratio_min = max(0.45, 0.55 - 0.10 * sensitivity)
                soft_allowed_ratio_min = max(0.65, 0.75 - 0.10 * sensitivity)
                contour_slice = contour_mask[y1:y2, x1:x2]
                roi_slice = color_mask[y1:y2, x1:x2]
                color_pixels = cv2.countNonZero(roi_slice)
                box_area = max(1, (x2 - x1) * (y2 - y1))
                fill_ratio = color_pixels / box_area
                allowed_slice = active_exc_mask_full[y1:y2, x1:x2]
                allowed_pixels = cv2.countNonZero(allowed_slice)
                allowed_ratio = allowed_pixels / box_area

                # Keep robot-sized, color-supported regions while still rejecting sparse noise.
                if (x2 - x1) > _BUMPER_MAX_BOX_WIDTH or (y2 - y1) > _BUMPER_MAX_BOX_HEIGHT:
                    continue
                if color_pixels < min_color_pixels:
                    continue
                if fill_ratio < fill_ratio_min and color_pixels < (min_color_pixels * 2):
                    continue
                if allowed_ratio < allowed_ratio_min:
                    continue
                if allowed_ratio < soft_allowed_ratio_min and color_pixels < (min_color_pixels * 3):
                    continue
                if _has_large_internal_horizontal_gap(contour_slice):
                    continue

                raw_bboxes.append((x1, y1, x2, y2))

                # Convert to normalized 0-1000 format [y1, x1, y2, x2]
                y1_norm = int(y1 / height * 1000)
                x1_norm = int(x1 / width * 1000)
                y2_norm = int(y2 / height * 1000)
                x2_norm = int(x2 / width * 1000)

                detections.append({
                    "box_2d": [y1_norm, x1_norm, y2_norm, x2_norm],
                    "label": label
                })
    
    bounding_boxes_json = json.dumps(detections)
    return bounding_boxes_json, red_mask, blue_mask, raw_bboxes


def draw_bumper_highlights(frame: Image.Image, red_mask: np.ndarray, blue_mask: np.ndarray,
                           field_pixel_mask: np.ndarray = None) -> Image.Image:
    """
    Draw semi-transparent color overlays on detected bumper regions and field elements.
    Uses cv2.addWeighted for SIMD-optimized blending.
    
    Args:
        frame: PIL Image to draw on
        red_mask: Binary mask of red bumper pixels
        blue_mask: Binary mask of blue bumper pixels
        field_pixel_mask: Optional per-pixel field exclusion mask (0 = field element, 255 = allowed).
                          Field element pixels are tinted brown.
        
    Returns:
        PIL Image with bumper and field element highlights drawn
    """
    if red_mask is None and blue_mask is None and field_pixel_mask is None:
        return frame
    
    frame_np = np.array(frame)  # RGB
    overlay = frame_np.copy()
    
    # Paint brown on field element pixels (excluded regions)
    if field_pixel_mask is not None:
        roi_x1, roi_y1, roi_x2, roi_y2 = _get_calibrated_field_roi()
        h_frame, w_frame = frame_np.shape[:2]
        # Clamp ROI to frame
        ry1 = max(0, min(roi_y1, h_frame))
        ry2 = max(0, min(roi_y2, h_frame))
        rx1 = max(0, min(roi_x1, w_frame))
        rx2 = max(0, min(roi_x2, w_frame))
        fh = ry2 - ry1
        fw = rx2 - rx1
        # Slice mask to match ROI region (handle size mismatches)
        mask_h = min(fh, field_pixel_mask.shape[0])
        mask_w = min(fw, field_pixel_mask.shape[1])
        field_region = field_pixel_mask[:mask_h, :mask_w]
        # Pixels where mask == 0 are field elements → paint brown
        overlay_roi = overlay[ry1:ry1+mask_h, rx1:rx1+mask_w]
        overlay_roi[field_region == 0] = (0, 0, 0)  # Black (RGB)
    
    # Paint solid color on overlay where bumpers are detected
    if red_mask is not None:
        overlay[red_mask > 0] = (255, 60, 60)   # Bright red (RGB)
    if blue_mask is not None:
        overlay[blue_mask > 0] = (60, 100, 255)  # Bright blue (RGB)
    
    # Blend: result = frame * 0.6 + overlay * 0.4 (SIMD-optimized)
    blended = cv2.addWeighted(frame_np, 0.6, overlay, 0.4, 0)
    
    return Image.fromarray(blended)



import threading
import queue


class ThreadedVideoReader:
    """
    Read video frames in a background thread so decoding overlaps with processing.
    OpenCV releases the GIL during cap.read(), enabling true parallelism.
    """
    
    def __init__(self, cap: cv2.VideoCapture, start_frame: int, end_frame: int, queue_size: int = 128):
        self.cap = cap
        self.end_frame = end_frame
        self.queue = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.frame_count = start_frame
        self.thread = threading.Thread(target=self._read_frames, daemon=True)
        self.thread.start()
    
    def _read_frames(self):
        while not self.stopped:
            if self.frame_count >= self.end_frame:
                self.queue.put((False, None, self.frame_count))
                return
            ret, frame = self.cap.read()
            if not ret:
                self.queue.put((False, None, self.frame_count))
                return
            self.queue.put((True, frame, self.frame_count))
            self.frame_count += 1
    
    def read(self):
        """Get next frame. Returns (success, frame, frame_number)."""
        return self.queue.get()
    
    def stop(self):
        self.stopped = True
        # Drain queue to unblock writer thread
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break


class ThreadedVideoWriter:
    """
    Write video frames in a background thread so encoding overlaps with processing.
    OpenCV releases the GIL during out.write(), enabling true parallelism.
    """
    
    def __init__(self, out: cv2.VideoWriter, queue_size: int = 128):
        self.out = out
        self.queue = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.thread = threading.Thread(target=self._write_frames, daemon=True)
        self.thread.start()
    
    def _write_frames(self):
        while True:
            frame = self.queue.get()
            if frame is None:  # Sentinel to stop
                return
            self.out.write(frame)
    
    def write(self, frame):
        """Queue a frame for writing."""
        self.queue.put(frame)
    
    def stop(self):
        """Signal the writer to finish and wait for all frames to be written."""
        self.queue.put(None)  # Sentinel
        self.thread.join()


MANUAL_TRACK_SLOT_IDS = [
    ("blue_1", 0, "blue"),
    ("blue_2", 1, "blue"),
    ("blue_3", 2, "blue"),
    ("red_1", 0, "red"),
    ("red_2", 1, "red"),
    ("red_3", 2, "red"),
]


def _iter_manual_track_slots(blue_robots: list, red_robots: list):
    blue_robots = list(blue_robots or [])
    red_robots = list(red_robots or [])
    while len(blue_robots) < 3:
        blue_robots.append("")
    while len(red_robots) < 3:
        red_robots.append("")

    for slot_id, index, alliance in MANUAL_TRACK_SLOT_IDS:
        label = blue_robots[index] if alliance == "blue" else red_robots[index]
        yield slot_id, str(label).strip(), alliance


def _dedupe_manual_track_samples(samples: list) -> list:
    samples = sorted(samples, key=lambda item: item[0])
    deduped = []
    for t, x, y, shooting in samples:
        if deduped and abs(t - deduped[-1][0]) <= 0.03 and bool(shooting) == bool(deduped[-1][3]):
            deduped[-1] = (t, x, y, shooting)
            continue
        if (deduped
                and abs(t - deduped[-1][0]) <= 0.20
                and abs(x - deduped[-1][1]) <= 0.5
                and abs(y - deduped[-1][2]) <= 0.5
                and bool(shooting) == bool(deduped[-1][3])):
            deduped[-1] = (t, x, y, shooting)
            continue
        deduped.append((t, x, y, shooting))
    return deduped


def _parse_manual_robot_tracks_json(manual_tracks_json: str, blue_robots: list, red_robots: list) -> dict:
    """
    Parse the browser-recorded manual robot tracks.

    Returns:
        Dict mapping robot label -> {'samples': [(t, x, y, shooting), ...], 'times': [t, ...], ...}
    """
    if not manual_tracks_json or not str(manual_tracks_json).strip():
        raise gr.Error("Manual tracking mode is enabled, but no robot tracks were recorded.")

    try:
        payload = json.loads(parse_json(str(manual_tracks_json)))
    except Exception as e:
        raise gr.Error(f"Could not parse manual robot tracks: {e}")

    slot_payload = payload.get("slots")
    if not isinstance(slot_payload, dict):
        raise gr.Error("Manual robot tracks are missing slot data.")

    video_payload = payload.get("video") if isinstance(payload.get("video"), dict) else {}
    try:
        source_width = float(video_payload.get("width") or 0)
        source_height = float(video_payload.get("height") or 0)
    except (TypeError, ValueError):
        source_width = 0.0
        source_height = 0.0

    parsed_tracks = {}
    missing_labels = []

    for slot_id, label, alliance in _iter_manual_track_slots(blue_robots, red_robots):
        if not label:
            continue

        slot_data = slot_payload.get(slot_id, {}) if isinstance(slot_payload.get(slot_id, {}), dict) else {}
        if slot_data.get("skipped"):
            continue

        raw_samples = slot_data.get("samples") or []
        cleaned_samples = []
        for sample in raw_samples:
            if not isinstance(sample, dict):
                continue
            try:
                t = float(sample.get("t"))
                x = float(sample.get("x"))
                y = float(sample.get("y"))
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(t) and np.isfinite(x) and np.isfinite(y)):
                continue
            if t < 0:
                continue
            cleaned_samples.append((t, x, y, bool(sample.get("shooting", True))))

        deduped = _dedupe_manual_track_samples(cleaned_samples)
        if not deduped:
            missing_labels.append(label)
            continue

        parsed_tracks[label] = {
            "slot_id": slot_id,
            "alliance": alliance,
            "samples": deduped,
            "times": [sample[0] for sample in deduped],
            "source_width": source_width,
            "source_height": source_height,
        }

    if missing_labels:
        joined = ", ".join(missing_labels)
        raise gr.Error(
            f"Manual tracks are missing for: {joined}. "
            "Drag those robots for the match or mark them skipped in the manual tracker."
        )

    return parsed_tracks


def _manual_track_shooting_active_at_time(
    robot_track: dict,
    target_time: float,
    persist_seconds: float = MANUAL_TRACK_SHOOTING_PERSIST_SECONDS,
) -> bool:
    """
    Keep a manual "shooting" mark alive briefly after the last explicit sample.

    In manual mode the operator marks the launch moment, but the center-score OCR
    can confirm the make a few seconds later. This helper keeps the robot
    score-eligible during that gap.
    """
    times = robot_track.get("times") or []
    samples = robot_track.get("samples") or []
    if not times or not samples:
        return False

    try:
        target = float(target_time)
    except (TypeError, ValueError):
        return False

    persist = max(0.0, float(persist_seconds or 0.0))
    idx = min(len(samples) - 1, bisect_right(times, target) - 1)
    if idx < 0:
        return bool(samples[0][3])

    window_start = target - persist
    while idx >= 0:
        sample_time, _, _, shooting = samples[idx]
        if float(sample_time) < window_start:
            break
        if bool(shooting):
            return True
        idx -= 1

    return False


def _interpolate_manual_robot_position(robot_track: dict, target_time: float):
    times = robot_track.get("times") or []
    samples = robot_track.get("samples") or []
    if not times or not samples:
        return None

    if target_time <= times[0]:
        _, x, y, shooting = samples[0]
        return x, y, bool(shooting)
    if target_time >= times[-1]:
        _, x, y, _ = samples[-1]
        return x, y, _manual_track_shooting_active_at_time(robot_track, target_time)

    idx = bisect_left(times, target_time)
    if idx <= 0:
        _, x, y, shooting = samples[0]
        return x, y, bool(shooting)
    if idx >= len(samples):
        _, x, y, _ = samples[-1]
        return x, y, _manual_track_shooting_active_at_time(robot_track, target_time)

    t0, x0, y0, _ = samples[idx - 1]
    t1, x1, y1, _ = samples[idx]
    if t1 <= t0:
        return x1, y1, _manual_track_shooting_active_at_time(robot_track, target_time)

    alpha = (target_time - t0) / (t1 - t0)
    x = x0 + ((x1 - x0) * alpha)
    y = y0 + ((y1 - y0) * alpha)
    return x, y, _manual_track_shooting_active_at_time(robot_track, target_time)


def _estimate_manual_robot_bbox(center_x: float, center_y: float, frame_width: int, frame_height: int) -> tuple:
    """
    Convert a human-tracked robot center into a synthetic bbox.

    We scale the box by vertical position because closer robots appear larger in the
    center camera. This keeps shot attribution and map projection reasonably stable
    without needing automatic robot detection.
    """
    y_norm = float(np.clip(center_y / max(1.0, frame_height), 0.0, 1.0))
    bbox_h = frame_height * (0.042 + (0.103 * y_norm))
    bbox_h = float(np.clip(bbox_h, frame_height * 0.045, frame_height * 0.16))
    bbox_w = bbox_h * 1.04
    bbox_w = float(np.clip(bbox_w, frame_width * 0.028, frame_width * 0.125))

    x1 = max(0.0, center_x - (bbox_w / 2.0))
    y1 = max(0.0, center_y - (bbox_h / 2.0))
    x2 = min(float(frame_width - 1), center_x + (bbox_w / 2.0))
    y2 = min(float(frame_height - 1), center_y + (bbox_h / 2.0))
    return x1, y1, x2, y2


def build_manual_robot_bboxes_json(manual_robot_tracks: dict, target_time: float, frame_width: int, frame_height: int, camera_side: str = "center") -> tuple:
    """
    Interpolate human-labeled robot tracks for a given video timestamp.

    Returns:
        (bounding_boxes_json, frame_tracks_dict)
    """
    detections = []
    frame_tracks = {}

    for label, robot_track in (manual_robot_tracks or {}).items():
        interp = _interpolate_manual_robot_position(robot_track, target_time)
        if interp is None:
            continue

        center_x, center_y, is_shooting = interp
        source_width = float(robot_track.get("source_width") or 0)
        source_height = float(robot_track.get("source_height") or 0)
        if source_width > 0 and source_height > 0:
            center_x = (float(center_x) / source_width) * float(frame_width)
            center_y = (float(center_y) / source_height) * float(frame_height)
        x1, y1, x2, y2 = _estimate_manual_robot_bbox(center_x, center_y, frame_width, frame_height)
        bbox_area = (x2 - x1) * (y2 - y1)
        track_y = min(float(frame_height - 1), center_y + ((y2 - y1) / 6.0))

        detection = {
            "box_2d": [
                int((y1 / max(1, frame_height)) * 1000),
                int((x1 / max(1, frame_width)) * 1000),
                int((y2 / max(1, frame_height)) * 1000),
                int((x2 / max(1, frame_width)) * 1000),
            ],
            "label": label,
            "shooting": bool(is_shooting),
        }
        detections.append(detection)
        frame_tracks[label] = (float(center_x), float(track_y), camera_side, float(bbox_area), bool(is_shooting))

    return json.dumps(detections), frame_tracks


def process_single_video(video_path: str, camera_side: str = "blue", target_fps: int = 30, start_seconds: float = 0, end_seconds: float = 0, blue_robots: list = None, red_robots: list = None, enable_robot_detection: bool = True, enable_fuel_detection: bool = True, progress=gr.Progress(), camera_name: str = "Camera", enable_person_detection: bool = True, calibration_points: list = None, calibration_image_size: tuple = None, side_box_points: list = None, side_box_image_size: tuple = None, side_camera_visible_robots: dict = None, show_unlabeled_robots: bool = True, manual_robot_tracks: dict = None, highlight_ball_robot: str = "", render_output_video: bool = True) -> tuple:
    """
    Process a single video, tracking objects at specified FPS.
    Uses bumper color detection for robot identification.
    
    Args:
        video_path: Path to input video
        camera_side: "blue", "red", or "center" for camera perspective
        target_fps: Target FPS for robot + ball tracking / output (default 30)
        start_seconds: Start processing at this time (0 = from beginning)
        end_seconds: Stop processing at this time (0 = process to end)
        blue_robots: List of blue alliance team numbers [robot1, robot2, robot3]
        red_robots: List of red alliance team numbers [robot1, robot2, robot3]
        enable_robot_detection: Whether to detect robots (default True)
        enable_fuel_detection: Whether to detect yellow fuel balls (default True)
        progress: Gradio progress tracker
        camera_name: Display name for the camera (e.g., "Blue Camera")
        side_camera_visible_robots: Dict of side camera visibility data (for center camera hidden robot injection)
            Format: {'blue': {frame_num: [robot_labels]}, 'red': {frame_num: [robot_labels]}}
        manual_robot_tracks: Optional dict of human-provided center-camera robot tracks.
        highlight_ball_robot: Optional team number whose balls should remain highlighted in output.
        
    Returns:
        Tuple of (output_video_path, robot_tracks, tracks_by_frame, width, height, robot_stats,
                  ferry_counts, disabled_statuses, shot_events, shooting_snapshots,
                  side_visible_robots, center_score_ocr)
    """
    if not video_path:
        raise gr.Error("Please upload a video file.")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error("Could not open video file.")
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = float(DEFAULT_TRACKING_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_fps = _normalize_tracking_fps(target_fps)
    
    # Frame interval is computed per-detection type below (robot, ball, person)
    
    # Calculate start and end frame numbers (handle None from Gradio)
    start_seconds = start_seconds or 0
    end_seconds = end_seconds or 0
    start_frame = int(start_seconds * original_fps) if start_seconds > 0 else 0
    if end_seconds > 0:
        end_frame = int(end_seconds * original_fps)
    else:
        end_frame = total_frames
    
    # Clamp to valid range
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(start_frame + 1, min(end_frame, total_frames))
    using_manual_robot_tracks = manual_robot_tracks is not None and camera_side == "center"
    
    # Skip to start frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    output_path = None
    ball_fps = min(float(target_fps), float(original_fps))
    output_fps = ball_fps
    out = None
    if render_output_video:
        # Create output video at the ball-tracking rate - use H264 codec for better compatibility
        output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        # Try multiple codecs for Windows compatibility
        fourcc_options = ['avc1', 'H264', 'mp4v', 'XVID']
        for codec in fourcc_options:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
                if out.isOpened():
                    print(f"Using codec: {codec}")
                    break
            except Exception:
                continue

        if out is None or not out.isOpened():
            # Fallback to avi format
            output_path = tempfile.NamedTemporaryFile(suffix=".avi", delete=False).name
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))

        if not out.isOpened():
            cap.release()
            raise gr.Error("Could not create output video file.")
    
    # Calculate sampling schedules.
    robot_sample_fps = float(target_fps)
    robot_frame_stride = _compute_sampling_stride(original_fps, robot_sample_fps)
    ball_frame_stride = _compute_sampling_stride(original_fps, ball_fps)
    next_robot_frame = float(start_frame)
    next_ball_frame = float(start_frame)
    score_ocr_frame_interval = max(1, round(original_fps / CENTER_SCORE_OCR_SAMPLE_FPS))
    match_clock_ocr_frame_interval = max(1, round(original_fps / CENTER_MATCH_CLOCK_OCR_SAMPLE_FPS))
    # Person detection at 6fps (independent of robot FPS)
    person_frame_interval = max(1, int(original_fps / 6))
    
    frame_count = start_frame
    processed_frames = 0
    total_ball_frames = max(1, int(np.ceil((end_frame - start_frame) / ball_frame_stride)))
    

    
    # Robot tracking for map visualization
    robot_tracks = {}  # label -> list of (center_x, center_y, camera_side)
    tracks_by_frame = []  # List of dicts {label: (cx, cy, side)} for each frame
    center_score_ocr_tracker = CenterScoreOCRTracker() if camera_side == "center" else None
    center_match_clock_overlay_tracker = CenterMatchClockOCRTracker() if camera_side == "center" else None
    score_ocr_delay_frames = int(round((original_fps or 30.0) * CENTER_SCORE_OCR_POST_ROLL_SECONDS))
    reader_end_frame = max(
        end_frame,
        min(total_frames, end_frame + score_ocr_delay_frames) if center_score_ocr_tracker else end_frame
    )
    
    # Ferry tracker for counting fuel ferries (cross out, cross back, shoot)
    ferry_tracker = FerryTracker(blue_robots=blue_robots, red_robots=red_robots)
    
    # Latest hidden robot tracking from the side cameras.
    hidden_side_robots = {}

    # Keep labeled robot boxes alive for 2 seconds when neither the center nor
    # side cameras currently sees them.
    recent_robot_bboxes = {}
    robot_bbox_persist_frames = max(1, int(round((original_fps or 30.0) * 2.0)))
    
    # Disabled tracker for detecting when robots stop moving
    disabled_tracker = DisabledTracker(fps=robot_sample_fps)
    
    # Ball tracker for shot detection - filtered by camera alliance
    ball_tracker = BallTracker(
        fps=ball_fps, 
        shot_label_duration=2.0, 
        min_upward_pixels=8,
        camera_side=camera_side,
        blue_robots=blue_robots,
        red_robots=red_robots,
        start_seconds=start_seconds,
        ferry_tracker=ferry_tracker,
        frame_width=width,
        frame_height=height
    )
    
    # Center Camera Auto-Calibrator (uses user-clicked points for homography)
    center_calibrator = None
    robot_exclusion_polygons = []
    side_calibration_boxes = []
    if camera_side == "center":
        # Reset any previous calibration
        center_camera_to_map_coords.calibration_homography = None
        center_camera_to_map_coords.calibration_homography_inv = None
        center_camera_to_map_coords.dynamic_homography = None
        center_calibrator = CenterCameraCalibrator(fps=ball_fps, gather_duration_sec=5.0, display_duration_sec=5.0)
        
        if calibration_points and calibration_image_size:
            robot_exclusion_polygons = _extract_robot_exclusion_polygons(
                calibration_points, calibration_image_size, width, height
            )

        calibration_homography_points = (calibration_points or [])[:CALIBRATION_REQUIRED_POINTS]

        if calibration_homography_points and len(calibration_homography_points) >= 4 and calibration_image_size:
            img_w, img_h = calibration_image_size
            if progress is not None:
                progress(0, desc="Computing calibration homography from clicked points...")
            H, H_inv, found_points = CenterCameraCalibrator.compute_homography_from_points(
                calibration_homography_points, img_w, img_h
            )
            if H is not None:
                print(f"[Pre-Calibration] Click calibration success. Homography computed from {len(found_points)} points.")
                center_camera_to_map_coords.calibration_homography = H
                center_camera_to_map_coords.calibration_homography_inv = H_inv
                center_calibrator.calibration_homography = H
                center_calibrator.calibration_homography_inv = H_inv
                center_calibrator.found_points = found_points
                center_calibrator.is_calibrating = False
            else:
                print("[Pre-Calibration] Homography computation failed. Using default calibration.")
        else:
            print(f"[Pre-Calibration] No calibration points provided ({len(calibration_homography_points) if calibration_homography_points else 0} points). Using default calibration.")
    elif camera_side in ("blue", "red"):
        side_calibration_boxes = _extract_side_camera_calibration_boxes(
            side_box_points,
            side_box_image_size,
            width,
            height,
            camera_side
        )
        expected_box_count = len(SIDE_CAMERA_BOX_LABELS.get(camera_side, []))
        if side_calibration_boxes and len(side_calibration_boxes) != expected_box_count:
            print(
                f"[Side Calibration] Incomplete {camera_side} side calibration "
                f"({len(side_calibration_boxes)}/{expected_box_count} boxes). "
                "Falling back to default guide boxes."
            )
            side_calibration_boxes = []

    # Store latest robot detection for use with ball frames
    current_bboxes_json = "[]"
    
    # Pre-compute field pixel mask for bumper detection (center camera only)
    field_pixel_mask = None
    if camera_side == "center" and not using_manual_robot_tracks:
        if progress is not None:
            progress(0, desc="Scanning video to build field pixel mask...")
        field_pixel_mask = compute_field_pixel_mask(video_path, start_seconds=start_seconds)
    
    # Bumper detection masks (stored for rendering on ball frames)
    current_bumper_red_mask = None
    current_bumper_blue_mask = None
    
    # Person detection state (stored for rendering and bumper exclusion)
    current_person_mask = None
    
    # Side camera LLM visibility data (for side cameras in bumper mode)
    # Maps frame_number -> list of visible robot labels
    side_visible_robots_by_frame = {}
    
    # Robot label tracker for YOLO + LLM mode (maintains identity across frames)
    robot_label_tracker = RobotLabelTracker(max_distance=100.0)
    
    if progress is not None:
        progress(0, desc=f"Processing {camera_name} - Frame 0/{total_ball_frames}")

    # Use threaded reader/writer to overlap I/O with processing
    reader = ThreadedVideoReader(cap, frame_count, reader_end_frame)
    writer = ThreadedVideoWriter(out) if render_output_video else None
    
    while True:
        ret, frame, frame_count = reader.read()
        if not ret:
            break

        if center_score_ocr_tracker and frame_count % score_ocr_frame_interval == 0:
            center_score_ocr_tracker.update(frame, frame_count / max(1.0, original_fps))
        if (
            center_match_clock_overlay_tracker
            and frame_count < end_frame
            and frame_count % match_clock_ocr_frame_interval == 0
        ):
            center_match_clock_overlay_tracker.update(frame, frame_count / max(1.0, original_fps))

        if frame_count >= end_frame:
            continue
        
        # Person detection at 6fps (center camera only)
        if (frame_count % person_frame_interval == 0 and enable_person_detection
                and camera_side == "center"
                and not using_manual_robot_tracks
                and YOLO_PERSON_MODEL is not None):
            current_person_mask, current_person_count = detect_people_yolo(frame)
            if current_person_count > 0:
                print(f"[Person Detection] Found {current_person_count} people at frame {frame_count}")
        
        # Robot detection at the requested tracking FPS.
        should_run_robot = False
        if enable_robot_detection:
            should_run_robot, next_robot_frame = _consume_frame_schedule(
                frame_count,
                next_robot_frame,
                robot_frame_stride,
            )
        if should_run_robot:
            # Convert BGR (OpenCV) to RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            
            # Combine robot numbers
            robot_numbers = (blue_robots or []) + (red_robots or [])
            
            if using_manual_robot_tracks:
                current_bumper_red_mask = None
                current_bumper_blue_mask = None
                video_time = frame_count / max(1.0, original_fps)
                bounding_boxes_json, frame_tracks = build_manual_robot_bboxes_json(
                    manual_robot_tracks,
                    video_time,
                    width,
                    height,
                    camera_side=camera_side,
                )
            # Bumper detection
            elif camera_side in ("blue", "red"):
                # Side camera: LLM presence query (no bounding boxes)
                alliance_robots = blue_robots if camera_side == "blue" else red_robots
                guided_pil_frame = annotate_side_camera_guides(
                    pil_frame,
                    camera_side,
                    calibrated_boxes=side_calibration_boxes
                )
                visible_robots = query_side_camera_presence(
                    guided_pil_frame,
                    alliance_robots,
                    camera_side=camera_side
                )
                side_visible_robots_by_frame[frame_count] = visible_robots
                if visible_robots:
                    print(f"[Side Camera LLM] {camera_name} sees robots: {visible_robots}")
                bounding_boxes_json = "[]"
            else:
                # Center camera: bumper color detection + LLM labeling
                detector_bboxes_json, current_bumper_red_mask, current_bumper_blue_mask, raw_bboxes = detect_robots_by_bumper_color(
                    frame,
                    person_mask=current_person_mask,
                    field_pixel_mask=field_pixel_mask,
                    robot_exclusion_polygons=robot_exclusion_polygons
                )
                
                # Get frame dimensions for crop clamping
                img_height, img_width = frame.shape[:2]
                try:
                    detector_entries = json.loads(parse_json(detector_bboxes_json))
                except Exception:
                    detector_entries = []
                raw_bbox_colors = [
                    str(detector_entries[i].get("label", "robot")).strip().lower()
                    if i < len(detector_entries) and isinstance(detector_entries[i], dict)
                    else "robot"
                    for i in range(len(raw_bboxes))
                ]
                allowed_numbers_by_detection = [
                    _get_allowed_robot_numbers_for_detection(
                        camera_side,
                        bbox,
                        raw_bbox_colors[i] if i < len(raw_bbox_colors) else "robot",
                        img_width,
                        blue_robots,
                        red_robots
                    )
                    for i, bbox in enumerate(raw_bboxes)
                ]
                
                # Use RobotLabelTracker + local LLM OCR to assign team numbers
                if robot_numbers and robot_label_tracker and raw_bboxes:
                    tracked_labels, needs_query = robot_label_tracker.check_needs_llm(
                        raw_bboxes,
                        allowed_numbers_by_detection
                    )
                    if len(tracked_labels) != len(raw_bboxes) or len(needs_query) != len(raw_bboxes):
                        print(
                            f"[Bumper+LLM] Length mismatch: raw_bboxes={len(raw_bboxes)}, "
                            f"tracked_labels={len(tracked_labels)}, needs_query={len(needs_query)}. "
                            "Padding to stay aligned."
                        )
                        tracked_labels = (list(tracked_labels) + [None] * len(raw_bboxes))[:len(raw_bboxes)]
                        needs_query = (list(needs_query) + [True] * len(raw_bboxes))[:len(raw_bboxes)]
                    
                    skipped = sum(1 for n in needs_query if not n)
                    if skipped > 0:
                        print(f"[Bumper+LLM] Skipping {skipped}/{len(needs_query)} robots (confident tracking)")
                    
                    # Collect crops for robots that need OCR
                    llm_queries = []
                    llm_indices = []
                    for i, bbox in enumerate(raw_bboxes):
                        if needs_query[i]:
                            x1, y1, x2, y2 = bbox
                            # Expand crop: 50px on each side
                            cx1 = max(0, x1 - 50)
                            cy1 = max(0, y1 - 50)
                            cx2 = min(img_width, x2 + 50)
                            cy2 = min(img_height, y2 + 50)
                            cropped = pil_frame.crop((cx1, cy1, cx2, cy2))
                            llm_queries.append({
                                'cropped_image': cropped,
                                'available_numbers': allowed_numbers_by_detection[i],
                                'previous_label': tracked_labels[i] if tracked_labels[i] else None
                            })
                            llm_indices.append(i)
                    
                    # Send all OCR queries to local LLM in parallel
                    if llm_queries:
                        print(f"[Bumper+LLM] Querying {len(llm_queries)} robots via local LLM...")
                        parallel_results = query_local_llm_batch(llm_queries, max_workers=min(50, len(llm_queries)))
                    else:
                        parallel_results = []
                    
                    # Build labels combining tracked + OCR results
                    all_labels = []
                    parallel_idx = 0
                    for i in range(len(raw_bboxes)):
                        if needs_query[i]:
                            all_labels.append(parallel_results[parallel_idx])
                            parallel_idx += 1
                        else:
                            all_labels.append(tracked_labels[i])
                    
                    # Update tracker for identity persistence
                    final_labels = robot_label_tracker.update(
                        raw_bboxes,
                        all_labels,
                        allowed_numbers_by_detection
                    )
                else:
                    final_labels = ["robot"] * len(raw_bboxes)
                
                # Rebuild bounding_boxes_json with team number labels
                detections = []

                for i, (x1, y1, x2, y2) in enumerate(raw_bboxes):
                    y1_norm = int((y1 / img_height) * 1000)
                    x1_norm = int((x1 / img_width) * 1000)
                    y2_norm = int((y2 / img_height) * 1000)
                    x2_norm = int((x2 / img_width) * 1000)
                    detections.append({
                        "box_2d": [y1_norm, x1_norm, y2_norm, x2_norm],
                        "label": final_labels[i] if i < len(final_labels) else "robot"
                    })
                bounding_boxes_json = json.dumps(detections)

                if camera_side == "center" and side_camera_visible_robots:
                    bounding_boxes_json, hidden_side_robots = inject_hidden_robot_bboxes(
                        bounding_boxes_json,
                        hidden_side_robots,
                        side_camera_visible_robots,
                        frame_count,
                        width,
                        height,
                        edge_persist_frames=robot_bbox_persist_frames
                    )

                if camera_side == "center":
                    bounding_boxes_json, recent_robot_bboxes = persist_recent_robot_bboxes(
                        bounding_boxes_json,
                        recent_robot_bboxes,
                        frame_count,
                        max_age_frames=robot_bbox_persist_frames
                    )
            
            # Store for tracking
            current_bboxes_json = bounding_boxes_json
            
            # Update ball tracker with robot bounding boxes
            ball_tracker.update_robot_bboxes(bounding_boxes_json, width, height)
            
            # Extract and track robot positions
            if using_manual_robot_tracks:
                frame_tracks = dict(frame_tracks)
            else:
                bbox_centers = extract_bbox_centers(bounding_boxes_json, width, height)
                frame_tracks = {
                    label: (cx, cy, camera_side, bbox_area)
                    for label, (cx, cy, bbox_area) in bbox_centers.items()
                }

            for label, track_data in frame_tracks.items():
                cx, cy = track_data[0], track_data[1]
                bbox_area = track_data[3] if len(track_data) >= 4 else None
                if label not in robot_tracks:
                    robot_tracks[label] = []
                if bbox_area is not None:
                    robot_tracks[label].append((cx, cy, camera_side, bbox_area))
                else:
                    robot_tracks[label].append((cx, cy, camera_side))
                
                # Update disabled tracker and ferry tracker with map coordinates
                # Use rotated map dimensions (961x574)
                map_x, map_y = transform_to_map(cx, cy, width, height, 961, 574, camera_side)
                if map_x is not None:
                    disabled_tracker.update_position(label, map_x, map_y)
                    ferry_tracker.update_position(label, map_x, map_y)
            
            tracks_by_frame.append(frame_tracks)
        
        # Ball detection and output at the requested tracking FPS.
        should_run_ball, next_ball_frame = _consume_frame_schedule(
            frame_count,
            next_ball_frame,
            ball_frame_stride,
        )
        if should_run_ball:
            if progress is not None:
                progress(
                    processed_frames / max(1, total_ball_frames),
                    desc=f"Processing {camera_name} - Frame {processed_frames + 1}/{total_ball_frames}"
                )
            
            # Process Calibration Visualization (Center Camera only)
            calib_viz_data = None
            if center_calibrator and center_calibrator.is_active:
                calib_viz_data = center_calibrator.process_frame(frame, width, height)
            
            render_bboxes_json = current_bboxes_json
            if using_manual_robot_tracks and enable_robot_detection:
                video_time = frame_count / max(1.0, original_fps)
                render_bboxes_json, _ = build_manual_robot_bboxes_json(
                    manual_robot_tracks,
                    video_time,
                    width,
                    height,
                    camera_side=camera_side,
                )
            
            # Update ball tracker with best available robot bboxes (interpolated on non-keyframes)
            # This ensures shot attribution uses accurate robot positions every ball frame,
            # not just stale keyframe data
            if enable_robot_detection:
                ball_tracker.update_robot_bboxes(render_bboxes_json, width, height)
            
            # Detect and draw fuel using color-based detection if enabled
            if enable_fuel_detection:
                should_scan_for_balls = True
                if using_manual_robot_tracks and camera_side == "center":
                    should_scan_for_balls = ball_tracker.any_robot_marked_shooting()

                if should_scan_for_balls:
                    if SAM3_PREDICTOR is not None:
                        fuel_detections = detect_fuel_sam3(frame, SAM3_PREDICTOR,
                                                           min_radius=3, max_radius=30,
                                                           camera_side=camera_side,
                                                           exclusion_polygons=robot_exclusion_polygons if camera_side == "center" else None)
                    else:
                        fuel_detections = detect_fuel(frame, min_radius=3, max_radius=30,
                                                      tracked_positions=ball_tracker.get_predicted_positions())
                else:
                    fuel_detections = []
                
                # Track balls and detect shots
                tracked_balls = ball_tracker.update(fuel_detections)
            
            if render_output_video:
                # Convert BGR (OpenCV) to RGB (PIL) only when rendering is enabled.
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)

                # Draw bounding boxes with alliance colors (for robots) - only if robot detection enabled
                annotated_frame = pil_frame.copy()
                if camera_side in ("blue", "red"):
                    annotated_frame = annotate_side_camera_guides(
                        annotated_frame,
                        camera_side,
                        calibrated_boxes=side_calibration_boxes
                    )
                if enable_robot_detection:
                    # Draw bumper color highlights on center camera
                    if camera_side == "center" and not using_manual_robot_tracks:
                        # Reuse cached masks from last robot detection frame (fast)
                        annotated_frame = draw_bumper_highlights(annotated_frame, current_bumper_red_mask, current_bumper_blue_mask, field_pixel_mask=field_pixel_mask)

                    # Draw person segmentation in grey
                    if (current_person_mask is not None and enable_person_detection and np.any(current_person_mask)
                            and not using_manual_robot_tracks):
                        frame_np = np.array(annotated_frame)  # RGB
                        overlay = frame_np.copy()
                        overlay[current_person_mask > 0] = (128, 128, 128)  # Grey
                        blended = cv2.addWeighted(frame_np, 0.6, overlay, 0.4, 0)
                        annotated_frame = Image.fromarray(blended)

                    annotated_frame = plot_bounding_boxes(
                        annotated_frame,
                        render_bboxes_json,
                        blue_robots,
                        red_robots,
                        stats=ball_tracker.robot_stats,
                        show_unlabeled=show_unlabeled_robots
                    )

                if enable_fuel_detection:
                    # Draw with shot attribution
                    annotated_frame = draw_fuel_detections(
                        annotated_frame,
                        tracked_balls,
                        blue_robots,
                        red_robots,
                        highlight_robot_label=highlight_ball_robot,
                    )

                # Draw Gemini Calibration Visualization (Center Camera only)
                if calib_viz_data is not None:
                    draw = ImageDraw.Draw(annotated_frame)

                    # Define font (fallback to default if necessary)
                    try:
                        font = ImageFont.truetype("arial.ttf", 20)
                    except IOError:
                        font = ImageFont.load_default()

                    frames_left = calib_viz_data['max_frames'] - calib_viz_data['frame_count']
                    has_homography = calib_viz_data.get('homography') is not None
                    status_text = "CALIBRATION LOCKED" if has_homography else "CALIBRATION (no homography)"
                    draw.text((10, 10), f"{status_text} - DISPLAYING: {frames_left} frames remaining", fill=(255, 255, 0), font=font)

                    # Factors to scale reference coords (1918x709) to actual frame
                    sx = width / 1918 if width > 0 else 1.0
                    sy = height / 709 if height > 0 else 1.0

                    # Helper to transform reference point to current frame position
                    def ref_to_current(rx, ry):
                        """Transform reference coords to current frame coords using forward homography."""
                        tx, ty = _calibration_transform_point_ref(rx, ry, inverse=False)
                        return tx * sx, ty * sy

                    # Draw Reference Points (cyan) - where landmarks SHOULD be if camera hasn't moved
                    for label, (ref_x, ref_y) in calib_viz_data['reference_points'].items():
                        act_x, act_y = ref_to_current(ref_x, ref_y)
                        pt_radius = 5
                        color = (0, 200, 255) if label.startswith('B') else (255, 100, 100)
                        draw.ellipse([act_x - pt_radius, act_y - pt_radius, act_x + pt_radius, act_y + pt_radius], fill=color)
                        draw.text((act_x + 8, act_y - 8), f"Ref {label}", fill=color, font=font)

                    # Draw Found Points (green) - where Gemini detected the landmarks
                    for label, (found_x, found_y) in calib_viz_data['found_points'].items():
                        act_x, act_y = found_x * sx, found_y * sy
                        box_size = 10
                        draw.rectangle([act_x - box_size, act_y - box_size, act_x + box_size, act_y + box_size], outline=(0, 255, 0), width=3)
                        draw.text((act_x - box_size, act_y + box_size + 5), f"Found {label}", fill=(0, 255, 0), font=font)

                    # Draw Ball Tracker Goal Zones (transformed from reference to current)
                    if ball_tracker:
                        for poly in ball_tracker.goal_polygons:
                            # Goal polygons are in actual frame resolution, transform via actual-res helper
                            shifted_poly = [_calibration_transform_point(px, py, width, height, inverse=False) for px, py in poly]
                            draw.polygon(shifted_poly, outline=(255, 0, 255), width=3)
                            draw.text((shifted_poly[0][0], shifted_poly[0][1] - 25), "Goal Zone", fill=(255, 0, 255), font=font)

                    # Draw SAM 3 scanner regions at fixed positions
                    roi_sx = width / 1918 if width > 0 else 1.0
                    roi_sy = height / 709 if height > 0 else 1.0
                    for (rx1, ry1, rx2, ry2) in _CENTER_CAM_ROIS:
                        roi_poly = [
                            (rx1 * roi_sx, ry1 * roi_sy),
                            (rx2 * roi_sx, ry1 * roi_sy),
                            (rx2 * roi_sx, ry2 * roi_sy),
                            (rx1 * roi_sx, ry2 * roi_sy),
                        ]
                        draw.polygon(roi_poly, outline=(255, 255, 255), width=2)
                        draw.text((roi_poly[0][0] + 5, roi_poly[0][1] + 5), "SAM 3 ROI", fill=(255, 255, 255), font=font)

                if camera_side == "center":
                    annotated_frame = draw_center_ocr_debug_overlay(
                        annotated_frame,
                        center_score_ocr_tracker=center_score_ocr_tracker,
                        center_match_clock_ocr_tracker=center_match_clock_overlay_tracker,
                    )

                # Convert back to BGR for OpenCV
                annotated_bgr = cv2.cvtColor(np.array(annotated_frame), cv2.COLOR_RGB2BGR)

                # Write frame to output (non-blocking, queued to writer thread)
                writer.write(annotated_bgr)

            processed_frames += 1
        
    
    # Wait for all frames to be written, then release resources
    reader.stop()
    if writer is not None:
        writer.stop()
    cap.release()
    if out is not None:
        out.release()
    
    # Finalize all remaining tracked balls to ensure all shots are counted
    # This is critical for counting misses that exit the frame
    ball_tracker.finalize_all()
    
    # Get ferry counts from the ferry tracker
    ferry_counts = ferry_tracker.get_all_ferry_counts()
    
    # Get disabled statuses from the disabled tracker
    disabled_statuses = disabled_tracker.get_all_disabled_statuses()
    
    center_score_ocr = center_score_ocr_tracker.summary() if center_score_ocr_tracker else None
    center_match_clock_ocr = center_match_clock_overlay_tracker.summary() if center_match_clock_overlay_tracker else None
    return (
        output_path,
        robot_tracks,
        tracks_by_frame,
        width,
        height,
        ball_tracker.robot_stats,
        ferry_counts,
        disabled_statuses,
        ball_tracker.shot_events,
        ball_tracker.shooting_snapshots,
        side_visible_robots_by_frame,
        center_score_ocr,
        center_match_clock_ocr,
    )


def merge_robot_tracks(blue_tracks: dict, red_tracks: dict, frame_width: int = 1068, frame_height: int = 836) -> dict:
    """
    Merge robot tracks from blue and red cameras, using weighted averaging based on bounding box area
    AND field position. Cameras are trusted more for robots on their side of the field.
    
    Args:
        blue_tracks: Dict of {label: [(cx, cy, 'blue', bbox_area), ...]} from blue camera
        red_tracks: Dict of {label: [(cx, cy, 'red', bbox_area), ...]} from red camera
        frame_width: Video frame width for coordinate conversion
        frame_height: Video frame height for coordinate conversion
        
    Returns:
        Merged dict of {label: [(cx, cy, camera_side, bbox_area), ...]}
    """
    # Field center x-coordinate on the map (after 90° rotation)
    FIELD_CENTER_X = 483
    # Penalty for camera on opposite side of field (0.05 = almost ignore)
    OPPOSITE_SIDE_PENALTY = 0.05
    # Map dimensions (after 90° rotation)
    MAP_WIDTH = 961
    MAP_HEIGHT = 574
    
    merged = {}
    all_labels = set(blue_tracks.keys()) | set(red_tracks.keys())
    
    for label in all_labels:
        blue_positions = blue_tracks.get(label, [])
        red_positions = red_tracks.get(label, [])
        
        # If only one camera sees this robot, use that track
        if not blue_positions:
            merged[label] = red_positions
        elif not red_positions:
            merged[label] = blue_positions
        else:
            # Both cameras see this robot - merge frame by frame with weighted average
            merged_positions = []
            max_frames = max(len(blue_positions), len(red_positions))
            
            for i in range(max_frames):
                if i < len(blue_positions) and i < len(red_positions):
                    # Both cameras have data for this frame
                    blue_cx, blue_cy, blue_side, blue_area = blue_positions[i]
                    red_cx, red_cy, red_side, red_area = red_positions[i]
                    
                    # Convert both positions to map coordinates to determine field side
                    blue_map_x, _ = camera_to_map_coords(blue_cx, blue_cy, frame_width, frame_height, MAP_WIDTH, MAP_HEIGHT, "blue")
                    red_map_x, _ = camera_to_map_coords(red_cx, red_cy, frame_width, frame_height, MAP_WIDTH, MAP_HEIGHT, "red")
                    
                    # Use average map_x to determine which side of field robot is on
                    avg_map_x = (blue_map_x + red_map_x) / 2
                    
                    # Calculate camera trust based on field position
                    # Blue camera is trusted more for robots on blue side (HIGHER map_x)
                    # Red camera is trusted more for robots on red side (LOWER map_x)
                    if avg_map_x > FIELD_CENTER_X:
                        # Robot on blue side - trust blue camera, penalize red
                        blue_trust = 1.0
                        red_trust = OPPOSITE_SIDE_PENALTY
                    else:
                        # Robot on red side - trust red camera, penalize blue
                        blue_trust = OPPOSITE_SIDE_PENALTY
                        red_trust = 1.0
                    
                    # Calculate weights: area^2 * camera_trust
                    blue_weight = (blue_area ** 2) * blue_trust
                    red_weight = (red_area ** 2) * red_trust
                    total_weight = blue_weight + red_weight
                    
                    if total_weight > 0:
                        blue_ratio = blue_weight / total_weight
                        red_ratio = red_weight / total_weight
                        
                        # Weighted average position
                        weighted_cx = blue_cx * blue_ratio + red_cx * red_ratio
                        weighted_cy = blue_cy * blue_ratio + red_cy * red_ratio
                        
                        # Use the camera side with larger weight
                        primary_side = blue_side if blue_weight >= red_weight else red_side
                        combined_area = (blue_area + red_area) / 2
                        
                        merged_positions.append((weighted_cx, weighted_cy, primary_side, combined_area))
                    else:
                        # Fallback if both weights are 0 (shouldn't happen)
                        merged_positions.append(blue_positions[i])
                elif i < len(blue_positions):
                    # Only blue has data
                    merged_positions.append(blue_positions[i])
                else:
                    # Only red has data
                    merged_positions.append(red_positions[i])
            
            merged[label] = merged_positions
    
    return merged


def merge_frame_tracks(blue_frames: list, red_frames: list, frame_width: int = 1068, frame_height: int = 836) -> list:
    """
    Merge frame-by-frame tracks from both cameras using weighted averaging based on bounding box area
    AND field position. Cameras are trusted more for robots on their side of the field.
    
    Args:
        blue_frames: List of dicts {label: (cx, cy, 'blue', bbox_area)}
        red_frames: List of dicts {label: (cx, cy, 'red', bbox_area)}
        frame_width: Video frame width for coordinate conversion
        frame_height: Video frame height for coordinate conversion
        
    Returns:
        List of dicts containing merged frame data with weighted positions
    """
    # Field center x-coordinate on the map (after 90° rotation)
    FIELD_CENTER_X = 483
    # Penalty for camera on opposite side of field (0.05 = almost ignore)
    OPPOSITE_SIDE_PENALTY = 0.05
    # Map dimensions (after 90° rotation)
    MAP_WIDTH = 961
    MAP_HEIGHT = 574
    
    merged_frames = []
    max_frames = max(len(blue_frames), len(red_frames)) if blue_frames or red_frames else 0
    
    for i in range(max_frames):
        frame_data = {}
        
        # Get data from both cameras for this frame
        blue_data = blue_frames[i] if i < len(blue_frames) else {}
        red_data = red_frames[i] if i < len(red_frames) else {}
        
        # Get all labels seen by either camera
        all_labels = set(blue_data.keys()) | set(red_data.keys())
        
        for label in all_labels:
            blue_pos = blue_data.get(label)
            red_pos = red_data.get(label)
            
            if blue_pos and red_pos:
                # Both cameras see this robot
                blue_cx, blue_cy, blue_side, blue_area = blue_pos
                red_cx, red_cy, red_side, red_area = red_pos
                
                # Convert both positions to map coordinates to determine field side
                blue_map_x, _ = camera_to_map_coords(blue_cx, blue_cy, frame_width, frame_height, MAP_WIDTH, MAP_HEIGHT, "blue")
                red_map_x, _ = camera_to_map_coords(red_cx, red_cy, frame_width, frame_height, MAP_WIDTH, MAP_HEIGHT, "red")
                
                # Use average map_x to determine which side of field robot is on
                avg_map_x = (blue_map_x + red_map_x) / 2
                
                # Calculate camera trust based on field position
                # Blue side has HIGHER map_x (> 483), red side has LOWER map_x (< 483)
                if avg_map_x > FIELD_CENTER_X:
                    # Robot on blue side - trust blue camera, penalize red
                    blue_trust = 1.0
                    red_trust = OPPOSITE_SIDE_PENALTY
                else:
                    # Robot on red side - trust red camera, penalize blue
                    blue_trust = OPPOSITE_SIDE_PENALTY
                    red_trust = 1.0
                
                # Calculate weights: area^2 * camera_trust
                blue_weight = (blue_area ** 2) * blue_trust
                red_weight = (red_area ** 2) * red_trust
                total_weight = blue_weight + red_weight
                
                if total_weight > 0:
                    blue_ratio = blue_weight / total_weight
                    red_ratio = red_weight / total_weight
                    
                    # Weighted average position
                    weighted_cx = blue_cx * blue_ratio + red_cx * red_ratio
                    weighted_cy = blue_cy * blue_ratio + red_cy * red_ratio
                    
                    # Use the camera side with larger weight
                    primary_side = blue_side if blue_weight >= red_weight else red_side
                    combined_area = (blue_area + red_area) / 2
                    
                    frame_data[label] = (weighted_cx, weighted_cy, primary_side, combined_area)
                else:
                    frame_data[label] = blue_pos
            elif blue_pos:
                frame_data[label] = blue_pos
            else:
                frame_data[label] = red_pos
            
        merged_frames.append(frame_data)
    
    return merged_frames


def split_composite_video(composite_path: str, progress=None) -> tuple:
    """
    Split a 720p or 1080p composite video into 3 separate camera feeds.

    The crop rectangles are scaled from the known 1920x1080 reference layout so
    both resolutions follow the same code path.

    Uses FFmpeg subprocess for speed (parallel, hardware-friendly).
    Falls back to OpenCV frame loop if FFmpeg is unavailable.
    
    Args:
        composite_path: Path to the composite video
        progress: Optional Gradio progress tracker
        
    Returns:
        Tuple of (center_path, blue_path, red_path) temp file paths
    """
    probe = cv2.VideoCapture(composite_path)
    if not probe.isOpened():
        raise gr.Error("Could not open composite video file.")
    source_width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    source_height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = probe.get(cv2.CAP_PROP_FPS)
    total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    probe.release()

    crops = _build_composite_crop_layout(source_width, source_height)
    
    # Create temp output paths
    paths = {}
    for name in crops:
        tmp = tempfile.NamedTemporaryFile(suffix=f'_{name}.mp4', delete=False)
        tmp.close()
        paths[name] = tmp.name
    
    # Try to find an FFmpeg binary
    ffmpeg_exe = shutil.which('ffmpeg')
    
    # Try static_ffmpeg package (pip install static-ffmpeg)
    if ffmpeg_exe is None:
        try:
            import static_ffmpeg
            static_ffmpeg.add_paths()
            ffmpeg_exe = shutil.which('ffmpeg')
        except ImportError:
            pass
    
    if ffmpeg_exe:
        # ── Fast path: parallel FFmpeg subprocesses ──
        if progress:
            progress(0.01, desc="Splitting composite video with FFmpeg...")
        
        def run_ffmpeg_crop(name, crop_filter, out_path):
            cmd = [
                ffmpeg_exe, '-y',
                '-i', composite_path,
                '-vf', crop_filter,
                '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
                '-an',  # drop audio
                out_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg crop failed for {name}: {result.stderr[-500:]}")
            return name
        
        # Run all 3 crops in parallel
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(run_ffmpeg_crop, name, info['filter'], paths[name]): name
                for name, info in crops.items()
            }
            done_count = 0
            for future in as_completed(futures):
                future.result()  # raises on error
                done_count += 1
                if progress:
                    progress(done_count / 3 * 0.1, desc=f"Split {done_count}/3 camera feeds")
        
        print(f"Composite video split (FFmpeg) into 3 feeds: center={paths['center']}, blue={paths['blue']}, red={paths['red']}")
        return paths['center'], paths['blue'], paths['red']
    
    # ── Fallback: OpenCV frame-by-frame loop ──
    print("FFmpeg not found, falling back to OpenCV split (slower)...")
    cap = cv2.VideoCapture(composite_path)
    if not cap.isOpened():
        raise gr.Error("Could not open composite video file.")

    crop_rects = {name: info["rect"] for name, info in crops.items()}
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writers = {}
    for name, (x1, y1, x2, y2) in crop_rects.items():
        w, h = x2 - x1, y2 - y1
        writers[name] = cv2.VideoWriter(paths[name], fourcc, fps, (w, h))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        for name, (x1, y1, x2, y2) in crop_rects.items():
            cropped = frame[y1:y2, x1:x2]
            writers[name].write(cropped)
        
        frame_idx += 1
        if progress and frame_idx % 100 == 0:
            progress(frame_idx / total_frames * 0.1, desc=f"Splitting composite video... {frame_idx}/{total_frames}")
    
    cap.release()
    for w in writers.values():
        w.release()
    
    print(f"Composite video split (OpenCV) into 3 feeds: center={paths['center']}, blue={paths['blue']}, red={paths['red']}")
    return paths['center'], paths['blue'], paths['red']


def extract_center_video_from_composite(composite_path: str, progress=None) -> str:
    """
    Extract just the center-camera crop from a 720p or 1080p composite match video.
    """
    if not composite_path:
        raise gr.Error("Please upload a match video.")

    output_path = tempfile.NamedTemporaryFile(suffix="_center.mp4", delete=False).name
    probe = cv2.VideoCapture(composite_path)
    if not probe.isOpened():
        raise gr.Error("Could not open composite video file.")
    source_width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    source_height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = probe.get(cv2.CAP_PROP_FPS)
    total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    probe.release()

    center_crop = _build_composite_crop_layout(source_width, source_height)["center"]
    ffmpeg_exe = shutil.which('ffmpeg')

    if ffmpeg_exe is None:
        try:
            import static_ffmpeg
            static_ffmpeg.add_paths()
            ffmpeg_exe = shutil.which('ffmpeg')
        except ImportError:
            pass

    if ffmpeg_exe:
        if progress:
            progress(0.01, desc="Extracting center camera feed...")
        cmd = [
            ffmpeg_exe, '-y',
            '-i', composite_path,
            '-vf', center_crop["filter"],
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
            '-an',
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise gr.Error(f"FFmpeg center extraction failed: {result.stderr[-500:]}")
        return output_path

    cap = cv2.VideoCapture(composite_path)
    if not cap.isOpened():
        raise gr.Error("Could not open composite video file.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, center_crop["size"])
    if not out.isOpened():
        cap.release()
        raise gr.Error("Could not create center camera video.")

    center_x1, center_y1, center_x2, center_y2 = center_crop["rect"]
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cropped = frame[center_y1:center_y2, center_x1:center_x2]
        out.write(cropped)
        frame_idx += 1
        if progress and frame_idx % 100 == 0 and total_frames > 0:
            progress(frame_idx / total_frames * 0.1, desc=f"Extracting center camera... {frame_idx}/{total_frames}")

    cap.release()
    out.release()
    return output_path


def extract_center_match_clock_ocr(video_path: str, start_seconds: float = 0, end_seconds: float = 0, progress=None) -> dict:
    """
    Read the center-camera match clock via OCR and return a time-aligned summary.

    This is a lightweight second pass over the already-cropped center feed used
    to align match sections to the on-screen clock instead of assuming the clip
    starts exactly at autonomous frame 0.
    """
    if not video_path:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    original_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if original_fps <= 0:
        cap.release()
        return None

    start_seconds = start_seconds or 0
    end_seconds = end_seconds or 0
    start_frame = int(start_seconds * original_fps) if start_seconds > 0 else 0
    end_frame = int(end_seconds * original_fps) if end_seconds > 0 else total_frames
    end_frame = max(start_frame, min(total_frames, end_frame if end_frame > 0 else total_frames))
    frame_interval = max(1, round(original_fps / CENTER_MATCH_CLOCK_OCR_SAMPLE_FPS))

    tracker = CenterMatchClockOCRTracker()
    frame_idx = start_frame
    sampled_frames = 0
    sampled_total = max(1, ((end_frame - start_frame) // frame_interval) + 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            tracker.update(frame, frame_idx / max(1.0, original_fps))
            sampled_frames += 1
            if progress is not None and sampled_frames % 50 == 0:
                progress(
                    min(0.995, sampled_frames / sampled_total),
                    desc=f"Reading center match clock... {sampled_frames}/{sampled_total}"
                )

        frame_idx += 1

    cap.release()
    summary = tracker.summary()
    if summary.get("observations"):
        print(
            f"[Center Match Clock OCR] Captured {len(summary['observations'])} confirmed clock observations "
            f"from {summary.get('observations_count', 0)} OCR samples"
        )
    else:
        print("[Center Match Clock OCR] No confirmed match clock observations captured")
    return summary


def _dedupe_shot_events(all_shot_events: list, dedup_window_seconds: float = MULTI_CAMERA_SHOT_DEDUP_WINDOW_SECONDS) -> list:
    """Deduplicate shot events within a short time window and keep the best made flag."""
    events_by_robot = {}
    for elapsed, robot_label, made in all_shot_events:
        events_by_robot.setdefault(robot_label, []).append((elapsed, made))

    deduped_output = []
    for robot_label, events in events_by_robot.items():
        events.sort(key=lambda e: e[0])
        deduped_events = []
        for elapsed, made in events:
            is_duplicate = False
            for idx, (prev_elapsed, prev_made) in enumerate(deduped_events):
                if abs(elapsed - prev_elapsed) <= dedup_window_seconds:
                    is_duplicate = True
                    if made and not prev_made:
                        deduped_events[idx] = (prev_elapsed, True)
                    break
            if not is_duplicate:
                deduped_events.append((elapsed, made))
        deduped_output.extend((elapsed, robot_label, made) for elapsed, made in deduped_events)

    deduped_output.sort(key=lambda event: (event[0], str(event[1])))
    return deduped_output


def _build_stats_from_shot_events(
    all_shot_events: list,
    dedup_window_seconds: float = MULTI_CAMERA_SHOT_DEDUP_WINDOW_SECONDS,
    match_clock_ocr: dict = None,
) -> dict:
    """
    Deduplicate shot events within a short time window and rebuild per-robot stats.
    """
    events_by_robot = {}
    if dedup_window_seconds is None:
        normalized_events = list(all_shot_events or [])
    else:
        normalized_events = _dedupe_shot_events(all_shot_events, dedup_window_seconds=dedup_window_seconds)

    for elapsed, robot_label, made in normalized_events:
        events_by_robot.setdefault(robot_label, []).append((elapsed, made))

    merged_stats = {}
    for robot_label, events in events_by_robot.items():
        by_period = {name: {'attempts': 0, 'made': 0} for name, _, _ in MATCH_PERIODS}
        total_attempts = 0
        total_made = 0

        for elapsed, made in events:
            period = get_match_period_for_elapsed(elapsed, match_clock_ocr=match_clock_ocr)
            total_attempts += 1
            if made:
                total_made += 1
            if period in by_period:
                by_period[period]['attempts'] += 1
                if made:
                    by_period[period]['made'] += 1

        merged_stats[robot_label] = {
            'attempts': total_attempts,
            'made': total_made,
            'by_period': by_period
        }

    return merged_stats


def format_robot_stats_md(stats: dict, robot_label: str, ferry_counts: dict, disabled_statuses: dict) -> str:
    ferry_count = ferry_counts.get(robot_label, 0)
    disabled_status, disabled_time = disabled_statuses.get(robot_label, ("None", 0))

    if disabled_status == "Full":
        disabled_line = f"**Disabled: Full** - Robot was disabled for the entire match ({disabled_time:.1f}s longest)"
    elif disabled_status == "Partially":
        disabled_line = f"**Disabled: Partially** - Robot was disabled for part of the match ({disabled_time:.1f}s longest)"
    else:
        disabled_line = "**Disabled: None** - Robot was not disabled"

    if robot_label not in stats or not stats[robot_label].get('by_period'):
        result = disabled_line + "\n\n"
        if ferry_count > 0:
            result += f"**Ferried Fuel: {ferry_count}x**\n\n"
        result += "*No shots recorded*"
        return result

    robot_data = stats[robot_label]
    total = f"**{robot_data['made']} shots made**"
    if ferry_count > 0:
        total += f" | **Ferried: {ferry_count}x**"

    rows = ["| Period | Made |", "|--------|------|"]
    for period_name, _, _ in MATCH_PERIODS:
        period_data = robot_data['by_period'].get(period_name, {'attempts': 0, 'made': 0})
        if period_data['attempts'] > 0:
            rows.append(f"| {period_name} | {period_data['made']} |")

    if len(rows) == 2:
        result = disabled_line + "\n\n"
        if ferry_count > 0:
            result += f"**Ferried Fuel: {ferry_count}x**\n\n"
        result += "*No shots recorded*"
        return result

    return f"{disabled_line}\n\n{total}\n\n" + "\n".join(rows)


def process_dual_videos(blue_video_path: str, red_video_path: str, center_video_path: str = None, composite_video_path: str = None, target_fps: int = 30, start_seconds: float = 0, end_seconds: float = 0, blue_robot_1: str = "", blue_robot_2: str = "", blue_robot_3: str = "", red_robot_1: str = "", red_robot_2: str = "", red_robot_3: str = "", enable_robot_detection: bool = True, enable_fuel_detection: bool = True, side_ref_image: Image.Image = None, center_ref_image: Image.Image = None, enable_blue_camera: bool = True, enable_center_camera: bool = True, enable_red_camera: bool = True, enable_person_detection: bool = True, calibration_points: list = None, calibration_image_size: tuple = None, blue_side_box_points: list = None, blue_side_box_image_size: tuple = None, red_side_box_points: list = None, red_side_box_image_size: tuple = None, show_unlabeled_robots: bool = True, highlight_ball_robot: str = "", regional_name: str = "", progress=gr.Progress()) -> tuple:
    """
    Process blue, red, and center camera videos using bumper detection.
    
    Args:
        blue_video_path: Path to blue side camera video
        red_video_path: Path to red side camera video
        center_video_path: Path to center camera video (2136x836, views both sides)
        target_fps: Requested 10/20/30 FPS for robot tracking and ball output
        start_seconds: Start processing at this time (0 = from beginning)
        end_seconds: Stop processing at this time (0 = process to end)
        blue_robot_1, blue_robot_2, blue_robot_3: Blue alliance team numbers
        red_robot_1, red_robot_2, red_robot_3: Red alliance team numbers
        enable_robot_detection: Whether to detect robots
        enable_fuel_detection: Whether to detect yellow fuel balls
        highlight_ball_robot: Optional team number whose balls should remain highlighted in output
        progress: Gradio progress tracker
        
    Returns:
        Tuple of (blue_output_path, red_output_path, center_output_path, map_video_path, ...)
    """

    target_fps = _normalize_tracking_fps(target_fps)

    managed_youtube_dir = _get_managed_youtube_download_dir(composite_video_path)
    try:
        # If composite video provided, split it into 3 separate camera feeds
        if composite_video_path:
            progress(0, desc="Splitting composite video into camera feeds...")
            center_video_path, blue_video_path, red_video_path = split_composite_video(composite_video_path, progress)
        
        # Handle single video input (backwards compatibility)
        if not blue_video_path and not red_video_path and not center_video_path:
            raise gr.Error("Please upload at least one video file.")
        
        # Create separate lists for blue and red robots
        blue_robots = [blue_robot_1, blue_robot_2, blue_robot_3]
        red_robots = [red_robot_1, red_robot_2, red_robot_3]

        _persist_regional_calibration(
            regional_name,
            calibration_points=calibration_points,
            calibration_image_size=calibration_image_size,
            blue_side_box_points=blue_side_box_points,
            blue_side_box_image_size=blue_side_box_image_size,
            red_side_box_points=red_side_box_points,
            red_side_box_image_size=red_side_box_image_size,
        )
        
        results = {}
        
        # Process videos sequentially to allow real-time progress updates
        if blue_video_path and enable_blue_camera:
            progress(0, desc="Starting Blue Camera processing...")
            try:
                output_path, robot_tracks, tracks_by_frame, width, height, robot_stats, ferry_counts, disabled_statuses, shot_events, shooting_snapshots, side_visible_robots, center_score_ocr, center_match_clock_ocr = process_single_video(
                    blue_video_path,
                    "blue",
                    target_fps,
                    start_seconds,
                    end_seconds,
                    blue_robots,
                    red_robots,
                    enable_robot_detection,
                    False,  # Side cameras only used for positioning, not shot detection
                    progress,
                    "Blue Camera",
                    enable_person_detection=enable_person_detection,
                    side_box_points=blue_side_box_points,
                    side_box_image_size=blue_side_box_image_size,
                    highlight_ball_robot=highlight_ball_robot,
                )
                results['blue'] = {
                    'output_path': output_path,
                    'robot_tracks': robot_tracks,
                    'tracks_by_frame': tracks_by_frame,
                    'width': width,
                    'height': height,
                    'robot_stats': robot_stats,
                    'ferry_counts': ferry_counts,
                    'disabled_statuses': disabled_statuses,
                    'shot_events': shot_events,
                    'shooting_snapshots': shooting_snapshots,
                    'side_visible_robots': side_visible_robots,
                    'center_score_ocr': center_score_ocr,
                    'center_match_clock_ocr': center_match_clock_ocr,
                }
            except Exception as e:
                import traceback
                print(f"Error processing blue camera: {e}")
                print(traceback.format_exc())
                raise gr.Error(f"Error processing blue camera: {e}")
        
        if red_video_path and enable_red_camera:
            progress(0.5, desc="Starting Red Camera processing...")
            try:
                output_path, robot_tracks, tracks_by_frame, width, height, robot_stats, ferry_counts, disabled_statuses, shot_events, shooting_snapshots, side_visible_robots, center_score_ocr, center_match_clock_ocr = process_single_video(
                    red_video_path,
                    "red",
                    target_fps,
                    start_seconds,
                    end_seconds,
                    blue_robots,
                    red_robots,
                    enable_robot_detection,
                    False,  # Side cameras only used for positioning, not shot detection
                    progress,
                    "Red Camera",
                    enable_person_detection=enable_person_detection,
                    side_box_points=red_side_box_points,
                    side_box_image_size=red_side_box_image_size,
                    highlight_ball_robot=highlight_ball_robot,
                )
                results['red'] = {
                    'output_path': output_path,
                    'robot_tracks': robot_tracks,
                    'tracks_by_frame': tracks_by_frame,
                    'width': width,
                    'height': height,
                    'robot_stats': robot_stats,
                    'ferry_counts': ferry_counts,
                    'disabled_statuses': disabled_statuses,
                    'shot_events': shot_events,
                    'shooting_snapshots': shooting_snapshots,
                    'side_visible_robots': side_visible_robots,
                    'center_score_ocr': center_score_ocr,
                    'center_match_clock_ocr': center_match_clock_ocr,
                }
            except Exception as e:
                import traceback
                print(f"Error processing red camera: {e}")
                print(traceback.format_exc())
                raise gr.Error(f"Error processing red camera: {e}")
        
        # Process center camera
        if center_video_path and enable_center_camera:
            progress(0.4, desc="Starting Center Camera processing...")
            try:
                output_path, robot_tracks, tracks_by_frame, width, height, robot_stats, ferry_counts, disabled_statuses, shot_events, shooting_snapshots, _, center_score_ocr, center_match_clock_ocr = process_single_video(
                    center_video_path,
                    "center",
                    target_fps,
                    start_seconds,
                    end_seconds,
                    blue_robots,
                    red_robots,
                    enable_robot_detection,
                    enable_fuel_detection,
                    progress,
                    "Center Camera",
                    enable_person_detection=enable_person_detection,
                    calibration_points=calibration_points,
                    calibration_image_size=calibration_image_size,
                    side_camera_visible_robots={
                        'blue': results.get('blue', {}).get('side_visible_robots', {}),
                        'red': results.get('red', {}).get('side_visible_robots', {}),
                    },
                    show_unlabeled_robots=show_unlabeled_robots,
                    highlight_ball_robot=highlight_ball_robot,
                )
                results['center'] = {
                    'output_path': output_path,
                    'robot_tracks': robot_tracks,
                    'tracks_by_frame': tracks_by_frame,
                    'width': width,
                    'height': height,
                    'robot_stats': robot_stats,
                    'ferry_counts': ferry_counts,
                    'disabled_statuses': disabled_statuses,
                    'shot_events': shot_events,
                    'shooting_snapshots': shooting_snapshots,
                    'center_score_ocr': center_score_ocr,
                    'center_match_clock_ocr': center_match_clock_ocr,
                }
            except Exception as e:
                import traceback
                print(f"Error processing center camera: {e}")
                print(traceback.format_exc())
                raise gr.Error(f"Error processing center camera: {e}")
    
        # Use dimensions from blue camera (or red, or center if others not available)
        frame_width = results.get('blue', results.get('red', results.get('center', {}))).get('width', 1068)
        frame_height = results.get('blue', results.get('red', results.get('center', {}))).get('height', 836)
    
        # Merge robot tracks from all cameras (with field-position-based camera trust)
        blue_tracks = results.get('blue', {}).get('robot_tracks', {})
        red_tracks = results.get('red', {}).get('robot_tracks', {})
        center_tracks = results.get('center', {}).get('robot_tracks', {})
        
        # Merge blue and red camera tracks first
        merged_tracks = merge_robot_tracks(blue_tracks, red_tracks, frame_width, frame_height)
        
        # Add center camera tracks (center camera provides full-field view)
        # For robots seen by center camera, add those positions to merged_tracks
        for label, positions in center_tracks.items():
            if label not in merged_tracks:
                merged_tracks[label] = []
            # Append center camera positions (they have camera_side="center")
            merged_tracks[label].extend(positions)
        
        # Merge frame-by-frame tracks for video (with field-position-based camera trust)
        blue_frames = results.get('blue', {}).get('tracks_by_frame', [])
        red_frames = results.get('red', {}).get('tracks_by_frame', [])
        center_frames = results.get('center', {}).get('tracks_by_frame', [])
        merged_frames = merge_frame_tracks(blue_frames, red_frames, frame_width, frame_height)
        
        # Merge center camera frame data into merged_frames
        for i, center_data in enumerate(center_frames):
            if i < len(merged_frames):
                # Add center camera detections to merged frames
                for label, pos in center_data.items():
                    if label not in merged_frames[i]:
                        merged_frames[i][label] = pos
                    # If robot already in merged_frames, center provides additional confidence
                    # but we keep the blue/red merged position as primary
            else:
                # Center camera has more frames than blue/red - append
                merged_frames.append(center_data)
        
        # Generate individual robot movement maps (15 seconds each)
        progress(0.75, desc="Generating individual robot maps...")
        all_robot_labels = blue_robots + red_robots
        robot_map_paths = []
        
        for robot_label in all_robot_labels:
            if robot_label and robot_label.strip():
                label = robot_label.strip()
                # Filter merged_tracks to only include this robot
                single_robot_tracks = {label: merged_tracks.get(label, [])}
                
                # Only generate map if robot has position data
                if single_robot_tracks[label]:
                    robot_map = draw_robot_paths(
                        MAP_IMAGE_PATH, single_robot_tracks, frame_width, frame_height, 
                        "blue", blue_robots, red_robots, max_seconds=15, fps=target_fps
                    )
                    robot_map_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
                    robot_map.save(robot_map_path)
                    robot_map_paths.append(robot_map_path)
                else:
                    robot_map_paths.append(None)
            else:
                robot_map_paths.append(None)
        
        # Pad to exactly 6 entries (3 blue + 3 red)
        while len(robot_map_paths) < 6:
            robot_map_paths.append(None)
    
            # Interpolate positions for smooth movement on map
            smoothed_frames = interpolate_robot_tracks(merged_frames, max_gap=15)
        
            # Generate map video (with alliance colors and smooth interpolation)
            progress(0.9, desc="Generating map video...")
            map_video_path = generate_map_video(MAP_IMAGE_PATH, smoothed_frames, frame_width, frame_height, target_fps=target_fps, blue_robots=blue_robots, red_robots=red_robots)

            center_match_clock_ocr = results.get('center', {}).get('center_match_clock_ocr')

            progress(1.0, desc="All processing complete!")
        
            # Merge robot stats from all cameras using shot event deduplication
            # Collect shot events from all cameras
            all_shot_events = []  # List of (elapsed_seconds, robot_label, made_bool)
            for camera in ['blue', 'red', 'center']:
                camera_events = results.get(camera, {}).get('shot_events', [])
                all_shot_events.extend(camera_events)
        
            merged_stats = _build_stats_from_shot_events(
                all_shot_events,
                dedup_window_seconds=MULTI_CAMERA_SHOT_DEDUP_WINDOW_SECONDS,
                match_clock_ocr=center_match_clock_ocr,
            )
    
    
        # Legacy dedupe block left inert after switching to the shared stats builder above.
        for robot_label, events in {}.items():
            # Sort by timestamp
            events.sort(key=lambda e: e[0])
        
            # Walk through events, deduplicating within time window
            # Merge events regardless of result — if cameras disagree, prefer "made"
            deduped_events = []
            for elapsed, made in events:
                # Check if this event is a duplicate of a recent one
                is_duplicate = False
                for i, (prev_elapsed, prev_made) in enumerate(deduped_events):
                    if abs(elapsed - prev_elapsed) <= DEDUP_WINDOW_SECONDS:
                        is_duplicate = True
                        # If any camera saw it as made, count as made (optimistic)
                        if made and not prev_made:
                            deduped_events[i] = (prev_elapsed, True)
                        break
                if not is_duplicate:
                    deduped_events.append((elapsed, made))
        
            # Build stats from deduplicated events
            by_period = {name: {'attempts': 0, 'made': 0} for name, _, _ in MATCH_PERIODS}
            total_attempts = 0
            total_made = 0
        
            for elapsed, made in deduped_events:
                period = get_match_period(elapsed)
                total_attempts += 1
                if made:
                    total_made += 1
                if period in by_period:
                    by_period[period]['attempts'] += 1
                    if made:
                        by_period[period]['made'] += 1
        
            merged_stats[robot_label] = {
                'attempts': total_attempts,
                'made': total_made,
                'by_period': by_period
            }
        
            # Debug: show deduplication results
            original_count = len(events)
            deduped_count = len(deduped_events)
            if original_count != deduped_count:
                print(f"[DEDUP] Robot {robot_label}: {original_count} events -> {deduped_count} after dedup ({original_count - deduped_count} duplicates removed)")

        center_score_ocr = results.get('center', {}).get('center_score_ocr')
        center_shooting_snapshots = results.get('center', {}).get('shooting_snapshots')
        merged_stats = _apply_ocr_score_correction(
            merged_stats,
            center_score_ocr,
            blue_robots,
            red_robots,
            all_shot_events=all_shot_events,
            shooting_snapshots=center_shooting_snapshots,
            manual_mode=False,
            match_clock_ocr=center_match_clock_ocr,
        )
    
        # Get ferry counts from all cameras (ferry cycles complete per camera)
        blue_ferry = results.get('blue', {}).get('ferry_counts', {})
        red_ferry = results.get('red', {}).get('ferry_counts', {})
        center_ferry = results.get('center', {}).get('ferry_counts', {})
        merged_ferry_counts = {}
    
        # Merge ferry counts (take max from any camera since same crossing might be seen by multiple)
        for label in set(blue_ferry.keys()) | set(red_ferry.keys()) | set(center_ferry.keys()):
            merged_ferry_counts[label] = max(blue_ferry.get(label, 0), red_ferry.get(label, 0), center_ferry.get(label, 0))
    
        # Get disabled statuses from all cameras
        blue_disabled = results.get('blue', {}).get('disabled_statuses', {})
        red_disabled = results.get('red', {}).get('disabled_statuses', {})
        center_disabled = results.get('center', {}).get('disabled_statuses', {})
        merged_disabled_statuses = {}
    
        # Merge disabled statuses (use worst status, max time)
        status_priority = {"Full": 2, "Partially": 1, "None": 0}
        for label in set(blue_disabled.keys()) | set(red_disabled.keys()) | set(center_disabled.keys()):
            statuses = [
                blue_disabled.get(label, ("None", 0)),
                red_disabled.get(label, ("None", 0)),
                center_disabled.get(label, ("None", 0))
            ]
            # Pick worst status (highest priority) and max time
            best = max(statuses, key=lambda s: status_priority.get(s[0], 0))
            max_time = max(s[1] for s in statuses)
            merged_disabled_statuses[label] = (best[0], max_time)
    
        # Format stats as markdown for Gradio display
        def format_robot_stats_md(stats: dict, robot_label: str, ferry_counts: dict, disabled_statuses: dict) -> str:
            ferry_count = ferry_counts.get(robot_label, 0)
            disabled_status, disabled_time = disabled_statuses.get(robot_label, ("None", 0))
        
            # Format disabled status line
            if disabled_status == "Full":
                disabled_line = f"**🔴 Disabled: Full** - Robot was disabled for the entire match ({disabled_time:.1f}s longest)"
            elif disabled_status == "Partially":
                disabled_line = f"**🟡 Disabled: Partially** - Robot was disabled for part of the match ({disabled_time:.1f}s longest)"
            else:
                disabled_line = "**🟢 Disabled: None** - Robot was not disabled"
        
            if robot_label not in stats or not stats[robot_label].get('by_period'):
                result = disabled_line + "\n\n"
                if ferry_count > 0:
                    result += f"**Ferried Fuel: {ferry_count}x**\n\n"
                result += "*No shots recorded*"
                return result
        
            robot_data = stats[robot_label]
            total = f"**{robot_data['made']} shots made**"
        
            # Add ferry count if any
            if ferry_count > 0:
                total += f" | **Ferried: {ferry_count}x**"
        
            # Build period table
            rows = ["| Period | Made |", "|--------|------|"]
            for period_name, _, _ in MATCH_PERIODS:
                p = robot_data['by_period'].get(period_name, {'attempts': 0, 'made': 0})
                if p['attempts'] > 0:
                    rows.append(f"| {period_name} | {p['made']} |")
        
            if len(rows) == 2:  # Only header rows
                result = disabled_line + "\n\n"
                if ferry_count > 0:
                    result += f"**Ferried Fuel: {ferry_count}x**\n\n"
                result += "*No shots recorded*"
                return result
        
            return f"{disabled_line}\n\n{total}\n\n" + "\n".join(rows)
    
        # Generate markdown for each robot (all 6)
        robot_stats_markdowns = []
        for label in all_robot_labels:
            if label and label.strip():
                robot_stats_markdowns.append(format_robot_stats_md(merged_stats, label.strip(), merged_ferry_counts, merged_disabled_statuses))
            else:
                robot_stats_markdowns.append("*Robot not configured*")
    
        # Pad to exactly 6 entries
        while len(robot_stats_markdowns) < 6:
            robot_stats_markdowns.append("*Robot not configured*")
    
        # Return output paths (None if camera not provided)
        blue_output = results.get('blue', {}).get('output_path', None)
        red_output = results.get('red', {}).get('output_path', None)
        center_output = results.get('center', {}).get('output_path', None)
    
        # Build labels for each robot (use team number as label)
        robot_labels = []
        for label in all_robot_labels:
            if label and label.strip():
                robot_labels.append(f"Team {label.strip()} - Autonomous")
            else:
                robot_labels.append("Not Configured")
        while len(robot_labels) < 6:
            robot_labels.append("Not Configured")
    
        # Return: blue_video, red_video, center_video, map_video, 6x(robot_map with dynamic label, robot_stats)
        return (
            blue_output, red_output, center_output, map_video_path,
            gr.update(value=robot_map_paths[0], label=robot_labels[0]), robot_stats_markdowns[0],  # Blue 1
            gr.update(value=robot_map_paths[1], label=robot_labels[1]), robot_stats_markdowns[1],  # Blue 2
            gr.update(value=robot_map_paths[2], label=robot_labels[2]), robot_stats_markdowns[2],  # Blue 3
            gr.update(value=robot_map_paths[3], label=robot_labels[3]), robot_stats_markdowns[3],  # Red 1
            gr.update(value=robot_map_paths[4], label=robot_labels[4]), robot_stats_markdowns[4],  # Red 2
            gr.update(value=robot_map_paths[5], label=robot_labels[5]), robot_stats_markdowns[5],  # Red 3
        )
    finally:
        _cleanup_managed_youtube_dir(managed_youtube_dir)


def process_manual_center_video(center_video_path: str = None, composite_video_path: str = None, target_fps: int = 30, start_seconds: float = 0, end_seconds: float = 0, blue_robot_1: str = "", blue_robot_2: str = "", blue_robot_3: str = "", red_robot_1: str = "", red_robot_2: str = "", red_robot_3: str = "", enable_fuel_detection: bool = True, calibration_points: list = None, calibration_image_size: tuple = None, manual_tracks_json: str = "", highlight_ball_robot: str = "", regional_name: str = "", progress=gr.Progress(), include_visual_outputs: bool = True, embed_robot_labels_in_stats: bool = False) -> tuple:
    """
    Process only the center camera using human-provided robot tracks and SAM 3 ball detection.
    """
    managed_youtube_dir = _get_managed_youtube_download_dir(composite_video_path)
    try:
        target_fps = _normalize_tracking_fps(target_fps)
        if not center_video_path and composite_video_path:
            progress(0, desc="Extracting center camera feed...")
            center_video_path = extract_center_video_from_composite(composite_video_path, progress=progress)

        if not center_video_path:
            raise gr.Error("Please upload a match video.")

        blue_robots = [blue_robot_1, blue_robot_2, blue_robot_3]
        red_robots = [red_robot_1, red_robot_2, red_robot_3]
        manual_robot_tracks = _parse_manual_robot_tracks_json(manual_tracks_json, blue_robots, red_robots)

        _persist_regional_calibration(
            regional_name,
            calibration_points=calibration_points,
            calibration_image_size=calibration_image_size,
        )

        progress(0.05, desc="Processing center camera with manual robot tracks...")
        center_output, robot_tracks, tracks_by_frame, width, height, _, ferry_counts, disabled_statuses, shot_events, shooting_snapshots, _, center_score_ocr, center_match_clock_ocr = process_single_video(
            center_video_path,
            "center",
            target_fps,
            start_seconds,
            end_seconds,
            blue_robots,
            red_robots,
            True,
            enable_fuel_detection,
            progress,
            "Center Camera",
            enable_person_detection=False,
            calibration_points=calibration_points,
            calibration_image_size=calibration_image_size,
            side_camera_visible_robots=None,
            show_unlabeled_robots=True,
            manual_robot_tracks=manual_robot_tracks,
            highlight_ball_robot=highlight_ball_robot,
            render_output_video=include_visual_outputs,
        )

        all_robot_labels = blue_robots + red_robots
        robot_map_paths = []
        map_video_path = None

        if include_visual_outputs:
            progress(0.75, desc="Generating individual robot maps...")
            for robot_label in all_robot_labels:
                if robot_label and robot_label.strip():
                    label = robot_label.strip()
                    single_robot_tracks = {label: robot_tracks.get(label, [])}
                    if single_robot_tracks[label]:
                        robot_map = draw_robot_paths(
                            MAP_IMAGE_PATH,
                            single_robot_tracks,
                            width,
                            height,
                            "center",
                            blue_robots,
                            red_robots,
                            max_seconds=15,
                            fps=target_fps,
                        )
                        robot_map_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
                        robot_map.save(robot_map_path)
                        robot_map_paths.append(robot_map_path)
                    else:
                        robot_map_paths.append(None)
                else:
                    robot_map_paths.append(None)

            while len(robot_map_paths) < 6:
                robot_map_paths.append(None)

            smoothed_frames = interpolate_robot_tracks(tracks_by_frame, max_gap=15)
            progress(0.9, desc="Generating map video...")
            map_video_path = generate_map_video(
                MAP_IMAGE_PATH,
                smoothed_frames,
                width,
                height,
                target_fps=target_fps,
                blue_robots=blue_robots,
                red_robots=red_robots,
            )
        else:
            while len(robot_map_paths) < 6:
                robot_map_paths.append(None)

        progress(1.0, desc="Manual center-camera processing complete!")

        merged_stats = _build_stats_from_shot_events(
            shot_events,
            dedup_window_seconds=None,
            match_clock_ocr=center_match_clock_ocr,
        )
        merged_stats = _apply_ocr_score_correction(
            merged_stats,
            center_score_ocr,
            blue_robots,
            red_robots,
            all_shot_events=shot_events,
            shooting_snapshots=shooting_snapshots,
            manual_mode=True,
            match_clock_ocr=center_match_clock_ocr,
        )
        robot_stats_markdowns = []
        for label in all_robot_labels:
            if label and label.strip():
                robot_stats_markdowns.append(format_robot_stats_md(merged_stats, label.strip(), ferry_counts, disabled_statuses))
            else:
                robot_stats_markdowns.append("*Robot not configured*")

        while len(robot_stats_markdowns) < 6:
            robot_stats_markdowns.append("*Robot not configured*")

        robot_labels = []
        for label in all_robot_labels:
            if label and label.strip():
                robot_labels.append(f"Team {label.strip()} - Autonomous")
            else:
                robot_labels.append("Not Configured")
        while len(robot_labels) < 6:
            robot_labels.append("Not Configured")

        if embed_robot_labels_in_stats:
            robot_stats_markdowns = [
                f"### {robot_labels[idx]}\n\n{robot_stats_markdowns[idx]}"
                for idx in range(len(robot_stats_markdowns))
            ]

        return (
            center_output,
            map_video_path,
            gr.update(value=robot_map_paths[0], label=robot_labels[0]), robot_stats_markdowns[0],
            gr.update(value=robot_map_paths[1], label=robot_labels[1]), robot_stats_markdowns[1],
            gr.update(value=robot_map_paths[2], label=robot_labels[2]), robot_stats_markdowns[2],
            gr.update(value=robot_map_paths[3], label=robot_labels[3]), robot_stats_markdowns[3],
            gr.update(value=robot_map_paths[4], label=robot_labels[4]), robot_stats_markdowns[4],
            gr.update(value=robot_map_paths[5], label=robot_labels[5]), robot_stats_markdowns[5],
        )
    finally:
        _cleanup_managed_youtube_dir(managed_youtube_dir)


def process_manual_center_video_table_only(center_video_path: str = None, composite_video_path: str = None, target_fps: int = 30, start_seconds: float = 0, end_seconds: float = 0, blue_robot_1: str = "", blue_robot_2: str = "", blue_robot_3: str = "", red_robot_1: str = "", red_robot_2: str = "", red_robot_3: str = "", enable_fuel_detection: bool = True, calibration_points: list = None, calibration_image_size: tuple = None, manual_tracks_json: str = "", highlight_ball_robot: str = "", regional_name: str = "", progress=gr.Progress()) -> tuple:
    """Manual mode variant that only returns the scoring tables."""
    return process_manual_center_video(
        center_video_path=center_video_path,
        composite_video_path=composite_video_path,
        target_fps=target_fps,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        blue_robot_1=blue_robot_1,
        blue_robot_2=blue_robot_2,
        blue_robot_3=blue_robot_3,
        red_robot_1=red_robot_1,
        red_robot_2=red_robot_2,
        red_robot_3=red_robot_3,
        enable_fuel_detection=enable_fuel_detection,
        calibration_points=calibration_points,
        calibration_image_size=calibration_image_size,
        manual_tracks_json=manual_tracks_json,
        highlight_ball_robot=highlight_ball_robot,
        regional_name=regional_name,
        progress=progress,
        include_visual_outputs=False,
        embed_robot_labels_in_stats=True,
    )


MANUAL_TRACKER_HEAD = r"""
<style>
  .gradio-container {
    max-width: min(96vw, 1800px) !important;
  }
  #manual-center-preview-source,
  #manual-tracks-json {
    display: none !important;
  }
  #manual-center-tracker {
    border: 1px solid var(--block-border-color, #d7dde8);
    border-radius: 14px;
    padding: 14px;
    background: var(--block-background-fill, #0f172a);
    color: var(--body-text-color, #e5e7eb);
  }
  #manual-center-tracker p,
  #manual-center-tracker strong,
  #manual-center-tracker span,
  #manual-center-tracker label {
    color: inherit;
  }
  .manual-tracker-toolbar {
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
    margin-bottom: 10px;
  }
  .manual-tracker-toolbar button {
    border: 1px solid var(--button-secondary-border-color, transparent);
    border-radius: 999px;
    padding: 8px 12px;
    background: var(--button-secondary-background-fill, #1f2937);
    color: var(--button-secondary-text-color, #f8fafc);
    cursor: pointer;
    font-weight: 600;
  }
  .manual-tracker-toolbar button:hover {
    background: var(--button-secondary-background-fill-hover, #374151);
  }
  .manual-tracker-toolbar .manual-tracker-readout {
    display: inline-flex;
    align-items: center;
    padding: 7px 12px;
    border-radius: 999px;
    background: var(--input-background-fill, rgba(148, 163, 184, 0.18));
    color: var(--body-text-color, #e5e7eb);
    border: 1px solid var(--block-border-color, rgba(148, 163, 184, 0.25));
    font-weight: 600;
  }
  .manual-tracker-stage {
    position: relative;
    width: 100%;
    overflow: hidden;
    border-radius: 12px;
    background: #0f172a;
    margin-bottom: 12px;
    padding: 10px;
    display: flex;
    justify-content: center;
  }
  .manual-tracker-video-shell {
    position: relative;
    width: min(100%, 1520px);
    aspect-ratio: 16 / 9;
  }
  #manual-tracker-video {
    width: 100%;
    height: 100%;
    display: block;
    max-height: min(82vh, 860px);
    min-height: 0;
    background: #020617;
    object-fit: contain;
  }
  #manual-tracker-overlay {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    cursor: crosshair;
  }
  .manual-tracker-slot-list {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 12px;
    margin-top: 12px;
  }
  .manual-slot-card {
    display: grid;
    grid-template-columns: minmax(96px, auto) minmax(70px, 1fr) auto;
    align-items: center;
    gap: 10px;
    padding: 12px;
    border-radius: 12px;
    background: var(--body-background-fill, #111827);
    border: 1px solid var(--block-border-color, #374151);
  }
  .manual-slot-pick {
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 8px 10px;
    color: white;
    cursor: pointer;
    font-weight: 700;
    min-width: 84px;
  }
  .manual-slot-pick.active {
    outline: 3px solid rgba(96, 165, 250, 0.35);
  }
  .manual-slot-meta {
    display: flex;
    flex-direction: column;
    gap: 4px;
    flex: 1;
    font-size: 12px;
    min-width: 0;
  }
  .manual-slot-meta strong {
    font-size: 12px;
    opacity: 0.7;
  }
  .manual-slot-meta span {
    font-size: 13px;
    font-weight: 600;
  }
  .manual-slot-actions {
    display: flex;
    align-items: center;
    gap: 8px;
    justify-content: flex-end;
    flex-wrap: wrap;
  }
  .manual-slot-actions button {
    border: 1px solid var(--button-secondary-border-color, transparent);
    border-radius: 8px;
    padding: 6px 10px;
    cursor: pointer;
    background: var(--button-secondary-background-fill, #374151);
    color: var(--button-secondary-text-color, #f8fafc);
    font-weight: 600;
  }
  .manual-slot-actions label {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
    opacity: 0.9;
  }
  #manual-tracker-status {
    margin-top: 10px;
    font-size: 13px;
    color: var(--body-text-color-subdued, var(--body-text-color, #cbd5e1));
  }
  @media (max-width: 900px) {
    .manual-tracker-slot-list {
      grid-template-columns: 1fr;
    }
    .manual-slot-card {
      grid-template-columns: minmax(96px, auto) 1fr;
    }
    .manual-slot-actions {
      grid-column: 1 / -1;
      justify-content: flex-start;
    }
  }
  @media (max-width: 1100px) {
    .gradio-container {
      max-width: 98vw !important;
    }
    .manual-tracker-stage {
      padding: 6px;
    }
  }
</style>
<script>
(() => {
  const MANUAL_TRACK_REACTION_SECONDS = 0.025;
  const MANUAL_TRACK_SHOOTING_REACTION_REAL_SECONDS = 1.0;
  const MANUAL_TRACK_AUTO_SAMPLE_SECONDS = 0.04;
  const MANUAL_TRACK_PLAYING_POLL_MS = 40;
  const MANUAL_TRACK_SAMPLE_REPLACE_SECONDS = 0.02;
  const MANUAL_TRACK_DRAG_REPLACE_SECONDS = 0.005;
  const MANUAL_TRACK_FORWARD_REWRITE_SECONDS = 0.2;
  const SLOT_ORDER = [
    { id: "blue_1", selector: "#blue-robot-1-input", color: "#1d4ed8", short: "B1" },
    { id: "blue_2", selector: "#blue-robot-2-input", color: "#2563eb", short: "B2" },
    { id: "blue_3", selector: "#blue-robot-3-input", color: "#3b82f6", short: "B3" },
    { id: "red_1", selector: "#red-robot-1-input", color: "#b91c1c", short: "R1" },
    { id: "red_2", selector: "#red-robot-2-input", color: "#dc2626", short: "R2" },
    { id: "red_3", selector: "#red-robot-3-input", color: "#ef4444", short: "R3" },
  ];

  function getInputValue(selector, fallback) {
    const root = document.querySelector(selector);
    if (!root) return fallback;
    const input = root.querySelector("input, textarea");
    if (!input) return fallback;
    const value = (input.value || "").trim();
    return value || fallback;
  }

  function initManualTracker() {
    const root = document.getElementById("manual-center-tracker");
    if (!root || root.dataset.ready === "1") return;
    root.dataset.ready = "1";

    const video = document.getElementById("manual-tracker-video");
    const canvas = document.getElementById("manual-tracker-overlay");
    const slotList = document.getElementById("manual-tracker-slot-list");
    const status = document.getElementById("manual-tracker-status");
    const timeLabel = document.getElementById("manual-tracker-time");
    const rateLabel = document.getElementById("manual-tracker-rate");
    const playBtn = document.getElementById("manual-tracker-play");
    const restartBtn = document.getElementById("manual-tracker-restart");
    const slowerBtn = document.getElementById("manual-tracker-slower");
    const fasterBtn = document.getElementById("manual-tracker-faster");
    const backBtn = document.getElementById("manual-tracker-back");
    const forwardBtn = document.getElementById("manual-tracker-forward");
    const hiddenFieldRoot = document.querySelector("#manual-tracks-json");
    const hiddenField = hiddenFieldRoot ? hiddenFieldRoot.querySelector("textarea, input") : null;
    const shell = root.querySelector(".manual-tracker-video-shell");

    const state = {
      sourceSrc: null,
      activeSlotId: SLOT_ORDER[0].id,
      dragging: null,
      dragPointerId: null,
      dragInitialButtons: 0,
      dragInitialButton: 0,
      dragAlternateMode: false,
      dragLastPoint: null,
      dragLastClientPoint: null,
      lastSnapshotTime: -1,
      playbackRate: 2.0,
      slots: {},
    };
    SLOT_ORDER.forEach((slot) => {
      state.slots[slot.id] = { x: null, y: null, shooting: false, skipped: false, samples: [] };
    });

    function getSlotLabel(slot) {
      return getInputValue(slot.selector, slot.short);
    }

    function updateRateLabel() {
      if (rateLabel) {
        rateLabel.textContent = `${state.playbackRate.toFixed(2).replace(/\.00$/, "")}x`;
      }
    }

    function setPlaybackRate(nextRate) {
      state.playbackRate = Math.max(0.25, Math.min(4.0, Number(nextRate) || 2.0));
      video.playbackRate = state.playbackRate;
      updateRateLabel();
      updateStatus();
    }

    function getCompensatedTrackTime(rawTime) {
      const numericTime = Number(rawTime) || 0;
      const leadSeconds = MANUAL_TRACK_REACTION_SECONDS * state.playbackRate;
      return Math.max(0, numericTime - leadSeconds);
    }

    function getShootingActivationTime(rawTime) {
      const numericTime = Number(rawTime) || 0;
      const reverseSeconds = MANUAL_TRACK_SHOOTING_REACTION_REAL_SECONDS * state.playbackRate;
      return Math.max(0, numericTime - reverseSeconds);
    }

    function getInterpolatedSlotState(slotId, rawTime) {
      const slotState = state.slots[slotId];
      if (!slotState || slotState.skipped || !slotState.samples.length) return null;
      const targetTime = getCompensatedTrackTime(rawTime);
      const samples = slotState.samples;

      if (targetTime <= samples[0].t) {
        return { x: samples[0].x, y: samples[0].y, shooting: !!samples[0].shooting };
      }
      if (targetTime >= samples[samples.length - 1].t) {
        const last = samples[samples.length - 1];
        return { x: last.x, y: last.y, shooting: !!last.shooting };
      }

      let shootingSample = samples[0];
      for (let i = 1; i < samples.length; i += 1) {
        const prev = samples[i - 1];
        const next = samples[i];
        if (targetTime < next.t) {
          const span = Math.max(next.t - prev.t, 1e-6);
          const alpha = (targetTime - prev.t) / span;
          return {
            x: prev.x + ((next.x - prev.x) * alpha),
            y: prev.y + ((next.y - prev.y) * alpha),
            shooting: !!shootingSample.shooting,
          };
        }
        shootingSample = next;
      }

      return null;
    }

    function syncSlotCursorToTime(slotId, rawTime) {
      const interp = getInterpolatedSlotState(slotId, rawTime);
      if (!interp) return;
      const slotState = state.slots[slotId];
      slotState.x = interp.x;
      slotState.y = interp.y;
      slotState.shooting = !!interp.shooting;
    }

    function getReliableClientPoint(event) {
      const canvasRect = canvas.getBoundingClientRect();
      const maxOverflowX = Math.max(48, canvasRect.width * 0.25);
      const maxOverflowY = Math.max(48, canvasRect.height * 0.25);
      const cornerMarginX = Math.max(42, canvasRect.width * 0.08);
      const cornerMarginY = Math.max(42, canvasRect.height * 0.08);
      const suspiciousCornerJump = Math.max(90, Math.min(canvasRect.width, canvasRect.height) * 0.12);
      const lastClientPoint = state.dragLastClientPoint;
      const candidates = [event];
      if (event && typeof event.getCoalescedEvents === "function") {
        const coalesced = event.getCoalescedEvents();
        for (let i = coalesced.length - 1; i >= 0; i -= 1) {
          candidates.push(coalesced[i]);
        }
      }

      function isUsableSample(sample) {
        if (!sample) return false;
        const clientX = Number(sample.clientX);
        const clientY = Number(sample.clientY);
        const sampleButtons = Number(sample.buttons) || 0;
        const samplePressure = Number(sample.pressure);
        if (!Number.isFinite(clientX) || !Number.isFinite(clientY)) return false;
        const withinExtendedBounds = clientX >= (canvasRect.left - maxOverflowX)
          && clientX <= (canvasRect.right + maxOverflowX)
          && clientY >= (canvasRect.top - maxOverflowY)
          && clientY <= (canvasRect.bottom + maxOverflowY);
        if (!withinExtendedBounds) return false;
        if (
          sample.pointerType === "pen"
          && lastClientPoint
          && sampleButtons === 0
          && Number.isFinite(samplePressure)
          && samplePressure <= 0
        ) {
          return false;
        }
        if (lastClientPoint) {
          const dx = clientX - lastClientPoint.x;
          const dy = clientY - lastClientPoint.y;
          const distanceSq = (dx * dx) + (dy * dy);
          const nearLeft = clientX <= (canvasRect.left + cornerMarginX);
          const nearRight = clientX >= (canvasRect.right - cornerMarginX);
          const nearTop = clientY <= (canvasRect.top + cornerMarginY);
          const nearBottom = clientY >= (canvasRect.bottom - cornerMarginY);
          const nearCorner = (nearLeft || nearRight) && (nearTop || nearBottom);
          if (nearCorner && distanceSq > (suspiciousCornerJump * suspiciousCornerJump)) {
            return false;
          }
        }
        return { x: clientX, y: clientY };
      }

      for (let i = 0; i < candidates.length; i += 1) {
        const usable = isUsableSample(candidates[i]);
        if (usable) return usable;
      }

      return state.dragLastClientPoint
        ? { x: state.dragLastClientPoint.x, y: state.dragLastClientPoint.y }
        : null;
    }

    function hasPenAlternateButtons(event) {
      if (!event || event.pointerType !== "pen") return false;
      const currentButtons = Number(event.buttons) || 0;
      if ((currentButtons & ~1) !== 0) {
        return true;
      }
      const currentButton = Number(event.button);
      return Number.isFinite(currentButton) && currentButton > 0;
    }

    function isAlternatePointerMode(event) {
      const currentButton = Number(event.button) || 0;
      const currentButtons = Number(event.buttons) || 0;
      if (currentButton === 2 || (currentButtons & 2) !== 0) {
        return true;
      }
      if (hasPenAlternateButtons(event)) {
        return true;
      }
      const initialButtons = Number(state.dragInitialButtons) || 0;
      if (initialButtons && currentButtons && currentButtons !== initialButtons) {
        return true;
      }
      return false;
    }

    function toPayload() {
      const slotPayload = {};
      SLOT_ORDER.forEach((slot) => {
        const slotState = state.slots[slot.id];
        slotPayload[slot.id] = {
          label: getSlotLabel(slot),
          skipped: !!slotState.skipped,
          samples: slotState.samples.map((sample) => ({
            t: Number(sample.t.toFixed(3)),
            x: Number(sample.x.toFixed(1)),
            y: Number(sample.y.toFixed(1)),
            shooting: !!sample.shooting,
          })),
        };
      });

      return {
        mode: "manual_center",
        video: {
          width: video.videoWidth || 0,
          height: video.videoHeight || 0,
          duration: Number.isFinite(video.duration) ? Number(video.duration.toFixed(3)) : 0,
        },
        slots: slotPayload,
      };
    }

    function syncHiddenField(force = false) {
      const payload = JSON.stringify(toPayload());
      if (hiddenField && (force || hiddenField.value !== payload)) {
        hiddenField.value = payload;
        hiddenField.dispatchEvent(new Event("input", { bubbles: true }));
        hiddenField.dispatchEvent(new Event("change", { bubbles: true }));
      }
      return payload;
    }

    function updateStatus() {
      const tracked = SLOT_ORDER.filter((slot) => !state.slots[slot.id].skipped && state.slots[slot.id].samples.length > 0).length;
      const skipped = SLOT_ORDER.filter((slot) => state.slots[slot.id].skipped).length;
      const shootingNow = SLOT_ORDER.filter((slot) => {
        if (state.slots[slot.id].skipped) return false;
        const slotNow = getInterpolatedSlotState(slot.id, video.currentTime || 0);
        return !!(slotNow && slotNow.shooting);
      }).length;
      const total = SLOT_ORDER.length;
      const durationText = Number.isFinite(video.duration) ? `${video.duration.toFixed(1)}s` : "0.0s";
      const reactionLead = (MANUAL_TRACK_REACTION_SECONDS * state.playbackRate).toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
      const dragMode = state.dragging ? (state.dragAlternateMode ? "shooting" : "tracking") : "idle";
      status.textContent = `Track all 6 robots unless skipped. Active: ${tracked}/${total} tracked, ${skipped} skipped, ${shootingNow} shooting now. Video duration: ${durationText}. Reaction lead: ${reactionLead}s. Drag mode: ${dragMode}.`;
    }

    function resizeCanvas() {
      const width = Math.max(1, Math.round(video.clientWidth || shell.clientWidth || 1));
      const height = Math.max(1, Math.round(video.clientHeight || (width * 709 / 1918) || 1));
      if (canvas.width !== width) canvas.width = width;
      if (canvas.height !== height) canvas.height = height;
      drawOverlay();
    }

    function getDisplayedVideoRect() {
      const sourceWidth = Math.max(1, video.videoWidth || canvas.width || 1);
      const sourceHeight = Math.max(1, video.videoHeight || canvas.height || 1);
      const scale = Math.min(canvas.width / sourceWidth, canvas.height / sourceHeight);
      const drawWidth = sourceWidth * scale;
      const drawHeight = sourceHeight * scale;
      return {
        scale,
        width: drawWidth,
        height: drawHeight,
        offsetX: (canvas.width - drawWidth) / 2,
        offsetY: (canvas.height - drawHeight) / 2,
      };
    }

    function sourceToCanvas(x, y) {
      const rect = getDisplayedVideoRect();
      return {
        x: rect.offsetX + (x * rect.scale),
        y: rect.offsetY + (y * rect.scale),
      };
    }

    function pointerToSource(event) {
      const canvasRect = canvas.getBoundingClientRect();
      const displayed = getDisplayedVideoRect();
      const clientPoint = getReliableClientPoint(event);
      if (!clientPoint) {
        return state.dragLastPoint
          ? { x: state.dragLastPoint.x, y: state.dragLastPoint.y }
          : { x: 0, y: 0 };
      }
      state.dragLastClientPoint = { x: clientPoint.x, y: clientPoint.y };
      const canvasX = clientPoint.x - canvasRect.left;
      const canvasY = clientPoint.y - canvasRect.top;
      const localX = Math.min(displayed.width, Math.max(0, canvasX - displayed.offsetX));
      const localY = Math.min(displayed.height, Math.max(0, canvasY - displayed.offsetY));
      return {
        x: localX / Math.max(displayed.scale, 1e-6),
        y: localY / Math.max(displayed.scale, 1e-6),
      };
    }

    function drawOverlay() {
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      SLOT_ORDER.forEach((slot) => {
        if (slot.id !== state.activeSlotId) return;
        const slotState = state.slots[slot.id];
        if (slotState.skipped || slotState.x === null || slotState.y === null) return;

        const point = sourceToCanvas(slotState.x, slotState.y);
        const radius = 7;
        ctx.fillStyle = slotState.shooting ? "#f59e0b" : slot.color;
        ctx.beginPath();
        ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
        ctx.fill();

        if (slot.id === state.activeSlotId) {
          ctx.strokeStyle = "rgba(255,255,255,0.9)";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(point.x, point.y, radius + 3, 0, Math.PI * 2);
          ctx.stroke();
        }

        const label = getSlotLabel(slot);
        ctx.font = "bold 10px sans-serif";
        const textWidth = ctx.measureText(label).width;
        ctx.fillStyle = "rgba(15, 23, 42, 0.85)";
        ctx.fillRect(point.x - textWidth / 2 - 5, point.y - 27, textWidth + 10, 15);
        ctx.fillStyle = "#ffffff";
        ctx.fillText(label, point.x - textWidth / 2, point.y - 16);
      });
    }

    function refreshSlotCards() {
      SLOT_ORDER.forEach((slot) => {
        const slotState = state.slots[slot.id];
        const pickBtn = slotList.querySelector(`[data-pick="${slot.id}"]`);
        const labelNode = slotList.querySelector(`[data-label="${slot.id}"]`);
        const countNode = slotList.querySelector(`[data-count="${slot.id}"]`);
        const skipInput = slotList.querySelector(`[data-skip="${slot.id}"]`);
        if (pickBtn) {
          pickBtn.textContent = getSlotLabel(slot);
          pickBtn.classList.toggle("active", slot.id === state.activeSlotId);
        }
        if (labelNode) {
          labelNode.textContent = slot.short;
        }
        if (countNode) {
          const slotNow = getInterpolatedSlotState(slot.id, video.currentTime || 0);
          countNode.textContent = slotState.skipped
            ? "Skipped"
            : `${slotState.samples.length} samples${slotNow && slotNow.shooting ? " | shooting now" : ""}`;
        }
        if (skipInput) {
          skipInput.checked = !!slotState.skipped;
        }
      });
      updateStatus();
    }

    function recordSlotSample(
      slotId,
      time,
      x,
      y,
      shooting = false,
      compensatedTime = null,
      replaceWindow = MANUAL_TRACK_SAMPLE_REPLACE_SECONDS,
      rewriteForwardWindow = MANUAL_TRACK_FORWARD_REWRITE_SECONDS
    ) {
      const slotState = state.slots[slotId];
      if (!slotState || slotState.skipped) return;
      const adjustedTime = compensatedTime === null ? getCompensatedTrackTime(time) : Math.max(0, Number(compensatedTime) || 0);
      const entry = {
        t: Number(adjustedTime.toFixed(3)),
        x: Number(x.toFixed(1)),
        y: Number(y.toFixed(1)),
        shooting: !!shooting,
      };
      if (rewriteForwardWindow > 0) {
        slotState.samples = slotState.samples.filter((sample) => {
          if (sample.t <= entry.t) return true;
          if (sample.t > (entry.t + rewriteForwardWindow)) return true;
          return Math.abs(sample.t - entry.t) <= replaceWindow;
        });
      }
      let inserted = false;
      for (let i = 0; i < slotState.samples.length; i += 1) {
        const sample = slotState.samples[i];
        if (Math.abs(sample.t - entry.t) <= replaceWindow) {
          if (!!sample.shooting !== !!entry.shooting) {
            entry.t = Number((Math.max(entry.t, sample.t) + 0.001).toFixed(3));
            continue;
          }
          slotState.samples[i] = entry;
          inserted = true;
          break;
        }
        if (sample.t > entry.t) {
          slotState.samples.splice(i, 0, entry);
          inserted = true;
          break;
        }
      }
      if (!inserted) {
        slotState.samples.push(entry);
      }
    }

    function captureAllSlots(force = false) {
      if (!video.src) return;
      const time = Number((video.currentTime || 0).toFixed(3));
      if (!force && Math.abs(time - state.lastSnapshotTime) < MANUAL_TRACK_AUTO_SAMPLE_SECONDS) return;
      state.lastSnapshotTime = time;
      const slotState = state.slots[state.activeSlotId];
      if (slotState && !slotState.skipped && slotState.x !== null && slotState.y !== null) {
        recordSlotSample(state.activeSlotId, time, slotState.x, slotState.y, slotState.shooting);
      }
      syncHiddenField();
      refreshSlotCards();
      drawOverlay();
    }

    function resetForNewVideo() {
      SLOT_ORDER.forEach((slot) => {
        const slotState = state.slots[slot.id];
        slotState.x = null;
        slotState.y = null;
        slotState.shooting = false;
        slotState.samples = [];
      });
      state.dragging = null;
      state.dragPointerId = null;
      state.dragInitialButtons = 0;
      state.dragInitialButton = 0;
      state.dragAlternateMode = false;
      state.dragLastPoint = null;
      state.dragLastClientPoint = null;
      state.lastSnapshotTime = -1;
      syncHiddenField(true);
      refreshSlotCards();
      drawOverlay();
    }

    function syncFromPreview() {
      const previewVideo = document.querySelector("#manual-center-preview-source video");
      if (!previewVideo) return;
      const nextSrc = previewVideo.currentSrc || previewVideo.src || (previewVideo.querySelector("source") ? previewVideo.querySelector("source").src : "");
      if (nextSrc && nextSrc !== state.sourceSrc) {
        state.sourceSrc = nextSrc;
        video.src = nextSrc;
        video.load();
        resetForNewVideo();
      }
    }

    window.manualTrackerSync = function () {
      syncFromPreview();
      return syncHiddenField(true);
    };

    SLOT_ORDER.forEach((slot) => {
      const card = document.createElement("div");
      card.className = "manual-slot-card";
      card.innerHTML = `
        <button type="button" class="manual-slot-pick" data-pick="${slot.id}" style="background:${slot.color};"></button>
        <div class="manual-slot-meta">
          <strong data-label="${slot.id}">${slot.short}</strong>
          <span data-count="${slot.id}">0 samples</span>
        </div>
        <div class="manual-slot-actions">
          <label><input type="checkbox" data-skip="${slot.id}"> Skip</label>
          <button type="button" data-clear="${slot.id}">Clear</button>
        </div>
      `;
      slotList.appendChild(card);
    });

    slotList.addEventListener("click", (event) => {
      const pick = event.target.closest("[data-pick]");
      if (pick) {
        state.activeSlotId = pick.getAttribute("data-pick");
        syncSlotCursorToTime(state.activeSlotId, video.currentTime || 0);
        refreshSlotCards();
        drawOverlay();
        return;
      }

      const clear = event.target.closest("[data-clear]");
      if (clear) {
        const slotId = clear.getAttribute("data-clear");
        const slotState = state.slots[slotId];
        slotState.x = null;
        slotState.y = null;
        slotState.shooting = false;
        slotState.samples = [];
        syncHiddenField(true);
        refreshSlotCards();
        drawOverlay();
      }
    });

    slotList.addEventListener("change", (event) => {
      const skip = event.target.closest("[data-skip]");
      if (!skip) return;
      const slotId = skip.getAttribute("data-skip");
      const slotState = state.slots[slotId];
      slotState.skipped = !!skip.checked;
      if (slotState.skipped) {
        slotState.x = null;
        slotState.y = null;
        slotState.shooting = false;
        slotState.samples = [];
      }
      syncHiddenField(true);
      refreshSlotCards();
      drawOverlay();
    });

    playBtn.addEventListener("click", () => {
      if (!video.src) return;
      if (video.paused) {
        video.playbackRate = state.playbackRate;
        video.play();
      } else {
        video.pause();
      }
    });
    restartBtn.addEventListener("click", () => {
      if (!video.src) return;
      video.pause();
      state.lastSnapshotTime = -1;
      video.currentTime = 0;
      syncSlotCursorToTime(state.activeSlotId, 0);
      drawOverlay();
    });
    backBtn.addEventListener("click", () => {
      state.lastSnapshotTime = -1;
      video.currentTime = Math.max(0, (video.currentTime || 0) - 5);
      syncSlotCursorToTime(state.activeSlotId, video.currentTime || 0);
      drawOverlay();
    });
    forwardBtn.addEventListener("click", () => {
      const duration = Number.isFinite(video.duration) ? video.duration : 0;
      state.lastSnapshotTime = -1;
      video.currentTime = Math.min(duration, (video.currentTime || 0) + 5);
      syncSlotCursorToTime(state.activeSlotId, video.currentTime || 0);
      drawOverlay();
    });
    slowerBtn.addEventListener("click", () => {
      setPlaybackRate(state.playbackRate - 0.25);
    });
    fasterBtn.addEventListener("click", () => {
      setPlaybackRate(state.playbackRate + 0.25);
    });

    canvas.addEventListener("contextmenu", (event) => {
      event.preventDefault();
    });

    canvas.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      resizeCanvas();
      const rect = canvas.getBoundingClientRect();
      let hitSlotId = null;
      const activeSlotState = state.slots[state.activeSlotId];
      if (activeSlotState && !activeSlotState.skipped && activeSlotState.x !== null && activeSlotState.y !== null) {
        const point = sourceToCanvas(activeSlotState.x, activeSlotState.y);
        const dx = (event.clientX - rect.left) - point.x;
        const dy = (event.clientY - rect.top) - point.y;
        if ((dx * dx) + (dy * dy) <= (16 * 16)) {
          hitSlotId = state.activeSlotId;
        }
      }

      if (hitSlotId) state.activeSlotId = hitSlotId;
      const point = hitSlotId ? { x: activeSlotState.x, y: activeSlotState.y } : pointerToSource(event);
      const slotState = state.slots[state.activeSlotId];
      slotState.skipped = false;
      slotState.x = point.x;
      slotState.y = point.y;
      state.dragging = state.activeSlotId;
      state.dragPointerId = event.pointerId;
      state.dragInitialButtons = Number(event.buttons) || 0;
      state.dragInitialButton = Number(event.button) || 0;
      state.dragAlternateMode = isAlternatePointerMode(event);
      state.dragLastPoint = point;
      slotState.shooting = state.dragAlternateMode;
      const dragTime = video.currentTime || 0;
      const activationTime = slotState.shooting ? getShootingActivationTime(dragTime) : null;
      recordSlotSample(
        state.activeSlotId,
        dragTime,
        point.x,
        point.y,
        slotState.shooting,
        activationTime,
        MANUAL_TRACK_DRAG_REPLACE_SECONDS
      );
      canvas.setPointerCapture(event.pointerId);
      syncHiddenField(true);

      refreshSlotCards();
      drawOverlay();
    });

    canvas.addEventListener("pointermove", (event) => {
      if (!state.dragging || event.pointerId !== state.dragPointerId) return;
      event.preventDefault();
      const point = pointerToSource(event);
      const slotState = state.slots[state.dragging];
      const wasShooting = !!slotState.shooting;
      state.dragAlternateMode = isAlternatePointerMode(event);
      state.dragLastPoint = point;
      slotState.x = point.x;
      slotState.y = point.y;
      slotState.shooting = state.dragAlternateMode;
      const dragTime = video.currentTime || 0;
      const activationTime = (!wasShooting && slotState.shooting) ? getShootingActivationTime(dragTime) : null;
      recordSlotSample(
        state.dragging,
        dragTime,
        point.x,
        point.y,
        slotState.shooting,
        activationTime,
        MANUAL_TRACK_DRAG_REPLACE_SECONDS
      );
      syncHiddenField();
      refreshSlotCards();
      drawOverlay();
    });

    function stopDrag(event) {
      if (!state.dragging) return;
      if (event && state.dragPointerId !== null && event.pointerId !== state.dragPointerId) return;
      const slotId = state.dragging;
      const slotState = state.slots[slotId];
      const point = event ? pointerToSource(event) : state.dragLastPoint;
      if (point) {
        slotState.x = point.x;
        slotState.y = point.y;
        state.dragLastPoint = point;
      }
      if (event && event.type === "pointerup" && (Number(event.buttons) || 0) > 0) {
        const wasShooting = !!slotState.shooting;
        state.dragAlternateMode = isAlternatePointerMode(event);
        slotState.shooting = state.dragAlternateMode;
        if (point && wasShooting !== slotState.shooting) {
          const dragTime = video.currentTime || 0;
          const activationTime = slotState.shooting ? getShootingActivationTime(dragTime) : null;
          recordSlotSample(
            slotId,
            dragTime,
            point.x,
            point.y,
            slotState.shooting,
            activationTime,
            0,
            0
          );
        }
        syncHiddenField(true);
        refreshSlotCards();
        drawOverlay();
        return;
      }
      if (slotState) {
        if (state.dragAlternateMode && point) {
          recordSlotSample(
            slotId,
            (video.currentTime || 0) + 0.001,
            point.x,
            point.y,
            false,
            null,
            0,
            0
          );
        }
        slotState.shooting = false;
      }
      if (event && canvas.hasPointerCapture && canvas.hasPointerCapture(event.pointerId)) {
        canvas.releasePointerCapture(event.pointerId);
      }
      state.dragging = null;
      state.dragPointerId = null;
      state.dragInitialButtons = 0;
      state.dragInitialButton = 0;
      state.dragAlternateMode = false;
      state.dragLastPoint = null;
      state.dragLastClientPoint = null;
      syncHiddenField(true);
      refreshSlotCards();
      drawOverlay();
    }
    canvas.addEventListener("pointerup", stopDrag);
    canvas.addEventListener("pointercancel", stopDrag);

    video.addEventListener("loadedmetadata", () => {
      setPlaybackRate(state.playbackRate);
      resizeCanvas();
      syncHiddenField(true);
      refreshSlotCards();
    });
    video.addEventListener("timeupdate", () => {
      const duration = Number.isFinite(video.duration) ? video.duration : 0;
      timeLabel.textContent = `${(video.currentTime || 0).toFixed(1)}s / ${duration.toFixed(1)}s`;
      captureAllSlots(false);
    });

    if (window.ResizeObserver) {
      new ResizeObserver(resizeCanvas).observe(shell);
    }
    window.addEventListener("resize", resizeCanvas);
    setInterval(syncFromPreview, 800);
    setInterval(refreshSlotCards, 500);
    setInterval(() => {
      if (!video.paused && !video.ended) {
        captureAllSlots(false);
      }
    }, MANUAL_TRACK_PLAYING_POLL_MS);

    refreshSlotCards();
    updateStatus();
    resizeCanvas();
    updateRateLabel();
    syncHiddenField(true);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => setTimeout(initManualTracker, 0));
  } else {
    setTimeout(initManualTracker, 0);
  }
  setInterval(initManualTracker, 1000);
})();
</script>
"""


MANUAL_TRACKER_HTML = """
<div id="manual-center-tracker">
  <p><strong>Manual Center Tracking</strong> — Keep the marker on the middle of each robot while the center video plays at 2x. Track all six robots unless you intentionally mark one as skipped. Hold the pen side button, switch pointer buttons mid-drag, or use right-click while dragging to mark that robot as shooting.</p>
  <p>Right-click shooting marks stay score-eligible for 4 seconds after release so delayed OCR makes can still be assigned.</p>
  <div class="manual-tracker-toolbar">
    <button type="button" id="manual-tracker-play">Play / Pause</button>
    <button type="button" id="manual-tracker-restart">Restart</button>
    <button type="button" id="manual-tracker-slower">Slower</button>
    <button type="button" id="manual-tracker-faster">Faster</button>
    <button type="button" id="manual-tracker-back">-5s</button>
    <button type="button" id="manual-tracker-forward">+5s</button>
    <span id="manual-tracker-rate" class="manual-tracker-readout">2x</span>
    <span id="manual-tracker-time">0.0s / 0.0s</span>
  </div>
  <div class="manual-tracker-stage">
    <div class="manual-tracker-video-shell">
      <video id="manual-tracker-video" playsinline preload="metadata"></video>
      <canvas id="manual-tracker-overlay"></canvas>
    </div>
  </div>
  <div id="manual-tracker-slot-list" class="manual-tracker-slot-list"></div>
  <div id="manual-tracker-status">Upload a video to begin.</div>
</div>
"""


def create_manual_demo(limited_mode: bool = False):
    """Create the center-camera manual tracking interface."""

    page_title = "Robot Scouter - Manual Limited Tracking" if limited_mode else "Robot Scouter - Manual Center Tracking"
    with gr.Blocks(title=page_title) as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("<div class='panel-title'>Manual Center Tracking Mode</div>", elem_classes="input-panel")
                mode_help_text = (
                    "This mode is enabled by `config.json` with `\"robot_tracking_mode\": \"manual-limited\"`. "
                    "Only the center camera is used, and non-table outputs are skipped so processing finishes faster."
                    if limited_mode else
                    "This mode is enabled by `config.json` with `\"robot_tracking_mode\": \"manual\"`. "
                    "Only the center camera is used. Side cameras, field-mask robot detection, and LLM robot labeling are skipped."
                )
                gr.Markdown(mode_help_text)

                composite_video_input = gr.Video(
                    label="Match Video (720p or 1080p; manual robot tracking on center camera only)",
                    sources=["upload"],
                )
                center_video_input = gr.State(None)
                video_metadata_state = gr.State(_blank_match_metadata())
                youtube_url_input = gr.Textbox(
                    label="YouTube Match URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                    max_lines=1,
                )
                with gr.Row():
                    youtube_download_btn = gr.Button("Download YouTube Video")
                    regional_input = gr.Textbox(
                        label="Regional / Event",
                        placeholder="Auto-filled from YouTube, or type it for uploads",
                        max_lines=1,
                    )
                video_source_status = gr.Markdown(VIDEO_SOURCE_EMPTY_STATUS)
                page_title_state = gr.Textbox(value=page_title, visible=False, elem_id="page-title-state")

                preview_source_video = gr.Video(
                    label="Center Preview Source",
                    interactive=False,
                    elem_id="manual-center-preview-source",
                )

                gr.Markdown("### Center Camera Calibration")
                gr.Markdown(
                    "Click the 8 field landmarks in order (B1→B4, R1→R4). "
                    "Extra clicks after that are grouped into 4-point no-scan polygons."
                )

                calibration_base_image = gr.State(None)
                calibration_points_state = gr.State([])
                calibration_image_size_state = gr.State(None)

                calibration_image = gr.Image(
                    label="Click calibration points here",
                    type="pil",
                    interactive=False,
                    height=300,
                )
                calibration_status = gr.Markdown("*Upload a video to begin calibration*")
                with gr.Row():
                    undo_btn = gr.Button("Undo Last Point", size="sm")
                    skip_calib_btn = gr.Button("Skip Calibration", size="sm")

                gr.Markdown("### Blue Alliance")
                with gr.Row():
                    blue_robot_1 = gr.Textbox(label="Robot 1", value="1796", max_lines=1, elem_id="blue-robot-1-input")
                    blue_robot_2 = gr.Textbox(label="Robot 2", value="250", max_lines=1, elem_id="blue-robot-2-input")
                    blue_robot_3 = gr.Textbox(label="Robot 3", value="11331", max_lines=1, elem_id="blue-robot-3-input")

                gr.Markdown("### Red Alliance")
                with gr.Row():
                    red_robot_1 = gr.Textbox(label="Robot 1", value="7759", max_lines=1, elem_id="red-robot-1-input")
                    red_robot_2 = gr.Textbox(label="Robot 2", value="6621", max_lines=1, elem_id="red-robot-2-input")
                    red_robot_3 = gr.Textbox(label="Robot 3", value="333", max_lines=1, elem_id="red-robot-3-input")

                with gr.Row():
                    fps_slider = gr.Slider(
                        minimum=10,
                        maximum=30,
                        value=30,
                        step=10,
                        label="Tracking FPS",
                        info="Run manual center tracking at 10, 20, or 30 FPS. Lower FPS widens ball-tracking thresholds automatically."
                    )
                    detect_fuel_checkbox = gr.Checkbox(
                        label="Detect Yellow Fuel",
                        value=True,
                        info="Run SAM 3 ball detection and shot calculations"
                    )

                with gr.Row():
                    start_seconds_input = gr.Number(
                        minimum=0,
                        value=0,
                        label="Start Time (seconds)",
                        info="Start processing at this time (0 = from beginning)"
                    )
                    end_seconds_input = gr.Number(
                        minimum=0,
                        value=0,
                        label="End Time (seconds)",
                        info="Stop processing at this time (0 = process to end)"
                    )

                highlight_ball_robot_dropdown = gr.Dropdown(
                    choices=_build_ball_highlight_choices(["1768", "4909", "5962"], ["2342", "6328", "2877"]),
                    value=BALL_HIGHLIGHT_ALL_OPTION,
                    label="Ball Overlay Highlight",
                    info="Only this robot's attributed balls stay fully colored in the annotated export."
                )

                manual_tracks_json = gr.Textbox(
                    label="Manual Track Cache",
                    elem_id="manual-tracks-json",
                    lines=2,
                    value="{}",
                )

        with gr.Row():
            gr.HTML(MANUAL_TRACKER_HTML)
        gr.HTML(PAGE_TITLE_SYNC_HTML)

        process_btn = gr.Button("Process Video")

        with gr.Row(visible=not limited_mode):
            with gr.Column(scale=1):
                gr.Markdown("<div class='panel-title'>Output</div>", elem_classes="output-panel", visible=not limited_mode)
                center_video_output = gr.Video(label="Center Camera - Annotated", visible=not limited_mode)
                map_video_output = gr.Video(label="Map Time-Lapse - Full Match Movement Overview", visible=not limited_mode)

        gr.Markdown(
            "<div class='panel-title'>Blue Alliance - Scoring Tables</div>"
            if limited_mode else
            "<div class='panel-title'>Blue Alliance - Autonomous Movement (15 sec)</div>"
        )
        with gr.Row():
            with gr.Column():
                blue1_map = gr.Image(label="Blue Robot 1 - Movement", visible=not limited_mode)
                blue1_stats = gr.Markdown("*Waiting for processing...*")
            with gr.Column():
                blue2_map = gr.Image(label="Blue Robot 2 - Movement", visible=not limited_mode)
                blue2_stats = gr.Markdown("*Waiting for processing...*")
            with gr.Column():
                blue3_map = gr.Image(label="Blue Robot 3 - Movement", visible=not limited_mode)
                blue3_stats = gr.Markdown("*Waiting for processing...*")

        gr.Markdown(
            "<div class='panel-title'>Red Alliance - Scoring Tables</div>"
            if limited_mode else
            "<div class='panel-title'>Red Alliance - Autonomous Movement (15 sec)</div>"
        )
        with gr.Row():
            with gr.Column():
                red1_map = gr.Image(label="Red Robot 1 - Movement", visible=not limited_mode)
                red1_stats = gr.Markdown("*Waiting for processing...*")
            with gr.Column():
                red2_map = gr.Image(label="Red Robot 2 - Movement", visible=not limited_mode)
                red2_stats = gr.Markdown("*Waiting for processing...*")
            with gr.Column():
                red3_map = gr.Image(label="Red Robot 3 - Movement", visible=not limited_mode)
                red3_stats = gr.Markdown("*Waiting for processing...*")

        def handle_manual_video_upload(video_path, start_seconds,
                                       current_blue_1, current_blue_2, current_blue_3,
                                       current_red_1, current_red_2, current_red_3,
                                       current_highlight, current_regional):
            metadata = _extract_uploaded_video_match_metadata(video_path)
            merged_blue = _merge_prefilled_robot_numbers(
                metadata.get("blue_robots", []),
                [current_blue_1, current_blue_2, current_blue_3],
            )
            merged_red = _merge_prefilled_robot_numbers(
                metadata.get("red_robots", []),
                [current_red_1, current_red_2, current_red_3],
            )
            resolved_regional = _clean_text(metadata.get("regional_name") or current_regional)
            manual_state = _prepare_manual_video_calibration_state(video_path, start_seconds, resolved_regional)
            loaded_saved = manual_state[-1]
            highlight_update = _update_ball_highlight_dropdown(
                merged_blue[0], merged_blue[1], merged_blue[2],
                merged_red[0], merged_red[1], merged_red[2],
                current_highlight,
            )
            status = VIDEO_SOURCE_EMPTY_STATUS if video_path is None else _format_video_source_status(
                "Uploaded video ready.",
                regional_name=resolved_regional,
                match_title=metadata.get("match_title", ""),
                blue_robots=merged_blue,
                red_robots=merged_red,
                calibration_loaded=loaded_saved,
            )
            resolved_page_title = _get_page_title_for_match(metadata) if _clean_text(metadata.get("match_title") or metadata.get("match_label")) else page_title
            return (
                *manual_state[:-1],
                merged_blue[0], merged_blue[1], merged_blue[2],
                merged_red[0], merged_red[1], merged_red[2],
                resolved_regional,
                highlight_update,
                status,
                metadata,
                resolved_page_title,
            )

        def handle_manual_regional_change(video_path, start_seconds, regional_name,
                                          current_blue_1, current_blue_2, current_blue_3,
                                          current_red_1, current_red_2, current_red_3,
                                          current_metadata):
            manual_state = _prepare_manual_video_calibration_state(video_path, start_seconds, regional_name)
            loaded_saved = manual_state[-1]
            metadata = current_metadata if isinstance(current_metadata, dict) else _blank_match_metadata()
            source_label = "YouTube video ready." if _get_managed_youtube_download_dir(video_path) else "Uploaded video ready."
            status = VIDEO_SOURCE_EMPTY_STATUS if video_path is None else _format_video_source_status(
                source_label,
                regional_name=regional_name,
                match_title=metadata.get("match_title", ""),
                blue_robots=[current_blue_1, current_blue_2, current_blue_3],
                red_robots=[current_red_1, current_red_2, current_red_3],
                calibration_loaded=loaded_saved,
            )
            return (*manual_state[:-1], status)

        def handle_manual_youtube_download(youtube_url, start_seconds,
                                           current_blue_1, current_blue_2, current_blue_3,
                                           current_red_1, current_red_2, current_red_3,
                                           current_highlight, current_regional,
                                           progress=gr.Progress()):
            video_path, metadata = _download_youtube_video(youtube_url, progress=progress)
            merged_blue = _merge_prefilled_robot_numbers(
                metadata.get("blue_robots", []),
                [current_blue_1, current_blue_2, current_blue_3],
            )
            merged_red = _merge_prefilled_robot_numbers(
                metadata.get("red_robots", []),
                [current_red_1, current_red_2, current_red_3],
            )
            resolved_regional = _clean_text(metadata.get("regional_name") or current_regional)
            manual_state = _prepare_manual_video_calibration_state(video_path, start_seconds, resolved_regional)
            loaded_saved = manual_state[-1]
            highlight_update = _update_ball_highlight_dropdown(
                merged_blue[0], merged_blue[1], merged_blue[2],
                merged_red[0], merged_red[1], merged_red[2],
                current_highlight,
            )
            status = _format_video_source_status(
                "YouTube video ready.",
                regional_name=resolved_regional,
                match_title=metadata.get("match_title", ""),
                blue_robots=merged_blue,
                red_robots=merged_red,
                calibration_loaded=loaded_saved,
            )
            return (
                video_path,
                *manual_state[:-1],
                merged_blue[0], merged_blue[1], merged_blue[2],
                merged_red[0], merged_red[1], merged_red[2],
                resolved_regional,
                highlight_update,
                status,
                metadata,
                _get_page_title_for_match(metadata),
            )

        composite_video_input.change(
            fn=handle_manual_video_upload,
            inputs=[
                composite_video_input,
                start_seconds_input,
                blue_robot_1,
                blue_robot_2,
                blue_robot_3,
                red_robot_1,
                red_robot_2,
                red_robot_3,
                highlight_ball_robot_dropdown,
                regional_input,
            ],
            outputs=[
                preview_source_video,
                center_video_input,
                calibration_image,
                calibration_base_image,
                calibration_points_state,
                calibration_image_size_state,
                calibration_status,
                manual_tracks_json,
                blue_robot_1,
                blue_robot_2,
                blue_robot_3,
                red_robot_1,
                red_robot_2,
                red_robot_3,
                regional_input,
                highlight_ball_robot_dropdown,
                video_source_status,
                video_metadata_state,
                page_title_state,
            ]
        )

        regional_input.change(
            fn=handle_manual_regional_change,
            inputs=[
                composite_video_input,
                start_seconds_input,
                regional_input,
                blue_robot_1,
                blue_robot_2,
                blue_robot_3,
                red_robot_1,
                red_robot_2,
                red_robot_3,
                video_metadata_state,
            ],
            outputs=[
                preview_source_video,
                center_video_input,
                calibration_image,
                calibration_base_image,
                calibration_points_state,
                calibration_image_size_state,
                calibration_status,
                manual_tracks_json,
                video_source_status,
            ]
        )

        youtube_download_btn.click(
            fn=handle_manual_youtube_download,
            inputs=[
                youtube_url_input,
                start_seconds_input,
                blue_robot_1,
                blue_robot_2,
                blue_robot_3,
                red_robot_1,
                red_robot_2,
                red_robot_3,
                highlight_ball_robot_dropdown,
                regional_input,
            ],
            outputs=[
                composite_video_input,
                preview_source_video,
                center_video_input,
                calibration_image,
                calibration_base_image,
                calibration_points_state,
                calibration_image_size_state,
                calibration_status,
                manual_tracks_json,
                blue_robot_1,
                blue_robot_2,
                blue_robot_3,
                red_robot_1,
                red_robot_2,
                red_robot_3,
                regional_input,
                highlight_ball_robot_dropdown,
                video_source_status,
                video_metadata_state,
                page_title_state,
            ]
        )

        youtube_url_input.submit(
            fn=handle_manual_youtube_download,
            inputs=[
                youtube_url_input,
                start_seconds_input,
                blue_robot_1,
                blue_robot_2,
                blue_robot_3,
                red_robot_1,
                red_robot_2,
                red_robot_3,
                highlight_ball_robot_dropdown,
                regional_input,
            ],
            outputs=[
                composite_video_input,
                preview_source_video,
                center_video_input,
                calibration_image,
                calibration_base_image,
                calibration_points_state,
                calibration_image_size_state,
                calibration_status,
                manual_tracks_json,
                blue_robot_1,
                blue_robot_2,
                blue_robot_3,
                red_robot_1,
                red_robot_2,
                red_robot_3,
                regional_input,
                highlight_ball_robot_dropdown,
                video_source_status,
                video_metadata_state,
                page_title_state,
            ]
        )

        def handle_image_click(base_image, clicked_points, regional_name, evt: gr.SelectData):
            if base_image is None:
                return None, clicked_points, "Upload a video first"
            x, y = evt.index
            clicked_points = list(clicked_points) + [(x, y)]
            _persist_regional_calibration(
                regional_name,
                calibration_points=clicked_points,
                calibration_image_size=base_image.size,
            )
            annotated = _redraw_calibration_image(base_image, clicked_points)
            return annotated, clicked_points, _get_calibration_status_text(len(clicked_points))

        calibration_image.select(
            fn=handle_image_click,
            inputs=[calibration_base_image, calibration_points_state, regional_input],
            outputs=[calibration_image, calibration_points_state, calibration_status]
        )

        def handle_undo(base_image, clicked_points):
            if not clicked_points:
                return base_image, clicked_points, "No points to undo"
            clicked_points = list(clicked_points)[:-1]
            annotated = _redraw_calibration_image(base_image, clicked_points) if clicked_points else base_image
            return annotated, clicked_points, _get_calibration_status_text(len(clicked_points)) + " — Undid last point"

        undo_btn.click(
            fn=handle_undo,
            inputs=[calibration_base_image, calibration_points_state],
            outputs=[calibration_image, calibration_points_state, calibration_status]
        )

        skip_calib_btn.click(
            fn=lambda: ([], "**Calibration skipped** — will use default alignment"),
            inputs=[],
            outputs=[calibration_points_state, calibration_status]
        )

        manual_ball_highlight_inputs = [
            blue_robot_1,
            blue_robot_2,
            blue_robot_3,
            red_robot_1,
            red_robot_2,
            red_robot_3,
            highlight_ball_robot_dropdown,
        ]
        for robot_input in manual_ball_highlight_inputs[:-1]:
            robot_input.change(
                fn=_update_ball_highlight_dropdown,
                inputs=manual_ball_highlight_inputs,
                outputs=[highlight_ball_robot_dropdown]
            )

        process_btn.click(
            fn=process_manual_center_video_table_only if limited_mode else process_manual_center_video,
            inputs=[
                center_video_input,
                composite_video_input,
                fps_slider,
                start_seconds_input,
                end_seconds_input,
                blue_robot_1,
                blue_robot_2,
                blue_robot_3,
                red_robot_1,
                red_robot_2,
                red_robot_3,
                detect_fuel_checkbox,
                calibration_points_state,
                calibration_image_size_state,
                manual_tracks_json,
                highlight_ball_robot_dropdown,
                regional_input,
            ],
            outputs=[
                center_video_output,
                map_video_output,
                blue1_map, blue1_stats,
                blue2_map, blue2_stats,
                blue3_map, blue3_stats,
                red1_map, red1_stats,
                red2_map, red2_stats,
                red3_map, red3_stats,
            ],
            js="""
            (centerVideoPath, compositeVideoPath, fps, startSeconds, endSeconds,
             blue1, blue2, blue3, red1, red2, red3, detectFuel,
             calibrationPoints, calibrationImageSize, manualTracksJson, highlightBallRobot, regionalName) => {
                const synced = window.manualTrackerSync ? window.manualTrackerSync() : manualTracksJson;
                return [
                    centerVideoPath, compositeVideoPath, fps, startSeconds, endSeconds,
                    blue1, blue2, blue3, red1, red2, red3, detectFuel,
                    calibrationPoints, calibrationImageSize, synced, highlightBallRobot, regionalName
                ];
            }
            """
        )

    return demo


def create_demo():
    """Create and return the Gradio interface."""
    
    with gr.Blocks(title="Robot Scouter") as demo:
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("<div class='panel-title'>Input</div>", elem_classes="input-panel")
                
                composite_video_input = gr.Video(
                    label="Match Video (720p or 1080p — auto-splits into 3 cameras)",
                    sources=["upload"],
                )
                
                youtube_url_input = gr.Textbox(
                    label="YouTube Match URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                    max_lines=1,
                )
                with gr.Row():
                    youtube_download_btn = gr.Button("Download YouTube Video")
                    regional_input = gr.Textbox(
                        label="Regional / Event",
                        placeholder="Auto-filled from YouTube, or type it for uploads",
                        max_lines=1,
                    )
                video_source_status = gr.Markdown(VIDEO_SOURCE_EMPTY_STATUS)
                page_title_state = gr.Textbox(value=DEFAULT_PAGE_TITLE, visible=False, elem_id="page-title-state")

                # Hidden placeholders for individual camera paths (not shown in UI)
                blue_video_input = gr.State(None)
                center_video_input = gr.State(None)
                red_video_input = gr.State(None)
                video_metadata_state = gr.State(_blank_match_metadata())
                
                # --- Center Camera Calibration ---
                gr.Markdown("### Center Camera Calibration")
                gr.Markdown(
                    "After the 8 field points, you can optionally click 4 corners per "
                    "center-camera no-scan box. Keep clicking in groups of 4 to add as "
                    "many robot exclusion boxes as you want."
                )
                gr.Markdown(
                    "Click the 8 field landmarks in order (B1→B4, R1→R4) on the frame below. "
                    "Any extra clicks after that are grouped into 4-point no-scan polygons."
                )
                
                calibration_base_image = gr.State(None)  # Original clean frame
                calibration_points_state = gr.State([])    # List of (x,y) tuples
                calibration_image_size_state = gr.State(None)  # (w, h) of displayed image
                
                calibration_image = gr.Image(
                    label="Click calibration points here",
                    type="pil",
                    interactive=False,
                    height=300,
                )
                calibration_status = gr.Markdown("*Upload a video to begin calibration*")
                with gr.Row():
                    undo_btn = gr.Button("Undo Last Point", size="sm")
                    skip_calib_btn = gr.Button("Skip Calibration", size="sm")

                gr.Markdown("### Side Camera Box Calibration")
                gr.Markdown(
                    "Optional: define 6 side-camera boxes using 2 clicks per box "
                    "(top-left, then bottom-right). "
                    "Blue side order: MIDDLE, LEFT, FAR LEFT. "
                    "Red side order: MIDDLE, RIGHT, FAR RIGHT."
                )

                blue_side_name_state = gr.State("blue")
                red_side_name_state = gr.State("red")
                blue_side_base_image = gr.State(None)
                blue_side_box_points_state = gr.State([])
                blue_side_box_image_size_state = gr.State(None)
                red_side_base_image = gr.State(None)
                red_side_box_points_state = gr.State([])
                red_side_box_image_size_state = gr.State(None)

                with gr.Row():
                    with gr.Column():
                        blue_side_calibration_image = gr.Image(
                            label="Blue Side Boxes",
                            type="pil",
                            interactive=False,
                            height=220,
                        )
                        blue_side_status = gr.Markdown("*Upload a video to calibrate blue side boxes*")
                        blue_side_undo_btn = gr.Button("Undo Blue Side Point", size="sm")
                    with gr.Column():
                        red_side_calibration_image = gr.Image(
                            label="Red Side Boxes",
                            type="pil",
                            interactive=False,
                            height=220,
                        )
                        red_side_status = gr.Markdown("*Upload a video to calibrate red side boxes*")
                        red_side_undo_btn = gr.Button("Undo Red Side Point", size="sm")

                # Robot number inputs
                gr.Markdown("### Blue Alliance")
                with gr.Row():
                    blue_robot_1 = gr.Textbox(
                        label="Robot 1",
                        value="1796",
                        placeholder="e.g., 1919",
                        max_lines=1
                    )
                    blue_robot_2 = gr.Textbox(
                        label="Robot 2",
                        value="250",
                        placeholder="e.g., 334",
                        max_lines=1
                    )
                    blue_robot_3 = gr.Textbox(
                        label="Robot 3",
                        value="11331",
                        placeholder="e.g., 254",
                        max_lines=1
                    )
                
                gr.Markdown("### Red Alliance")
                with gr.Row():
                    red_robot_1 = gr.Textbox(
                        label="Robot 1",
                        value="7759",
                        placeholder="e.g., 118",
                        max_lines=1
                    )
                    red_robot_2 = gr.Textbox(
                        label="Robot 2",
                        value="6621",
                        placeholder="e.g., 973",
                        max_lines=1
                    )
                    red_robot_3 = gr.Textbox(
                        label="Robot 3",
                        value="333",
                        placeholder="e.g., 2056",
                        max_lines=1
                    )
                
                with gr.Row():
                    fps_slider = gr.Slider(
                        minimum=10,
                        maximum=30,
                        value=30,
                        step=10,
                        label="Processing FPS",
                        info="Run at 10, 20, or 30 FPS. Higher FPS means more API calls and slower processing."
                    )
                
                with gr.Row():
                    start_seconds_input = gr.Number(
                        minimum=0,
                        value=0,
                        label="Start Time (seconds)",
                        info="Start processing at this time (0 = from beginning)"
                    )
                    end_seconds_input = gr.Number(
                        minimum=0,
                        value=0,
                        label="End Time (seconds)",
                        info="Stop processing at this time (0 = process to end)"
                    )
                
                gr.Markdown("### Cameras to Process")
                with gr.Row():
                    enable_blue_cam = gr.Checkbox(
                        label="Blue Camera",
                        value=True,
                        info="Process blue side camera feed"
                    )
                    enable_center_cam = gr.Checkbox(
                        label="Center Camera",
                        value=True,
                        info="Process center camera feed"
                    )
                    enable_red_cam = gr.Checkbox(
                        label="Red Camera",
                        value=True,
                        info="Process red side camera feed"
                    )
                
                with gr.Row():
                    detect_robots_checkbox = gr.Checkbox(
                        label="Detect Robots",
                        value=True,
                        info="Enable robot detection using AI"
                    )
                    detect_fuel_checkbox = gr.Checkbox(
                        label="Detect Yellow Fuel",
                        value=True,
                        info="Enable color-based detection of yellow fuel balls (no AI required)"
                    )
                    detect_people_checkbox = gr.Checkbox(
                        label="Detect People",
                        value=True,
                        info="Exclude humans from robot detection using YOLO (center camera)"
                    )
                
                with gr.Row():
                    show_unlabeled_checkbox = gr.Checkbox(
                        label="Show Unlabeled Robots",
                        value=True,
                        info="Show bounding boxes for robots that couldn't be identified by team number"
                    )

                highlight_ball_robot_dropdown = gr.Dropdown(
                    choices=_build_ball_highlight_choices(["1768", "4909", "5962"], ["2342", "6328", "2877"]),
                    value=BALL_HIGHLIGHT_ALL_OPTION,
                    label="Ball Overlay Highlight",
                    info="Only this robot's attributed balls stay fully colored in the annotated export."
                )

                # Hidden placeholders to keep inputs list consistent
                side_ref_image_input = gr.State(None)
                center_ref_image_input = gr.State(None)
                
                
                process_btn = gr.Button(
                    "Process Video"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("<div class='panel-title'>Output</div>", elem_classes="output-panel")
                
                with gr.Row():
                    blue_video_output = gr.Video(
                        label="Blue Side - Annotated",
                    )
                    center_video_output = gr.Video(
                        label="Center Camera - Annotated",
                    )
                    red_video_output = gr.Video(
                        label="Red Side - Annotated",
                    )
                
                with gr.Row():
                    map_video_output = gr.Video(
                        label="Map Time-Lapse - Full Match Movement Overview"
                    )
                
                gr.Markdown("<div class='panel-title'>Blue Alliance - Autonomous Movement (15 sec)</div>")
                with gr.Row():
                    with gr.Column():
                        blue1_map = gr.Image(label="Blue Robot 1 - Movement")
                        blue1_stats = gr.Markdown("*Waiting for processing...*")
                    with gr.Column():
                        blue2_map = gr.Image(label="Blue Robot 2 - Movement")
                        blue2_stats = gr.Markdown("*Waiting for processing...*")
                    with gr.Column():
                        blue3_map = gr.Image(label="Blue Robot 3 - Movement")
                        blue3_stats = gr.Markdown("*Waiting for processing...*")
                
                gr.Markdown("<div class='panel-title'>Red Alliance - Autonomous Movement (15 sec)</div>")
                with gr.Row():
                    with gr.Column():
                        red1_map = gr.Image(label="Red Robot 1 - Movement")
                        red1_stats = gr.Markdown("*Waiting for processing...*")
                    with gr.Column():
                        red2_map = gr.Image(label="Red Robot 2 - Movement")
                        red2_stats = gr.Markdown("*Waiting for processing...*")
                    with gr.Column():
                        red3_map = gr.Image(label="Red Robot 3 - Movement")
                        red3_stats = gr.Markdown("*Waiting for processing...*")
        gr.HTML(PAGE_TITLE_SYNC_HTML)
        
        # --- Calibration Event Wiring ---
        
        def handle_video_upload(video_path, start_seconds,
                                current_blue_1, current_blue_2, current_blue_3,
                                current_red_1, current_red_2, current_red_3,
                                current_highlight, current_regional):
            """Extract calibration frames and apply any saved regional calibration."""
            metadata = _extract_uploaded_video_match_metadata(video_path)
            merged_blue = _merge_prefilled_robot_numbers(
                metadata.get("blue_robots", []),
                [current_blue_1, current_blue_2, current_blue_3],
            )
            merged_red = _merge_prefilled_robot_numbers(
                metadata.get("red_robots", []),
                [current_red_1, current_red_2, current_red_3],
            )
            resolved_regional = _clean_text(metadata.get("regional_name") or current_regional)
            calibration_state = _prepare_composite_video_calibration_state(video_path, start_seconds, resolved_regional)
            loaded_saved = calibration_state[-1]
            highlight_update = _update_ball_highlight_dropdown(
                merged_blue[0], merged_blue[1], merged_blue[2],
                merged_red[0], merged_red[1], merged_red[2],
                current_highlight,
            )
            status = VIDEO_SOURCE_EMPTY_STATUS if video_path is None else _format_video_source_status(
                "Uploaded video ready.",
                regional_name=resolved_regional,
                match_title=metadata.get("match_title", ""),
                blue_robots=merged_blue,
                red_robots=merged_red,
                calibration_loaded=loaded_saved,
            )
            resolved_page_title = _get_page_title_for_match(metadata) if _clean_text(metadata.get("match_title") or metadata.get("match_label")) else DEFAULT_PAGE_TITLE
            return (
                *calibration_state[:-1],
                merged_blue[0], merged_blue[1], merged_blue[2],
                merged_red[0], merged_red[1], merged_red[2],
                resolved_regional,
                highlight_update,
                status,
                metadata,
                resolved_page_title,
            )

        def handle_regional_change(video_path, start_seconds, regional_name,
                                   current_blue_1, current_blue_2, current_blue_3,
                                   current_red_1, current_red_2, current_red_3,
                                   current_metadata):
            calibration_state = _prepare_composite_video_calibration_state(video_path, start_seconds, regional_name)
            loaded_saved = calibration_state[-1]
            metadata = current_metadata if isinstance(current_metadata, dict) else _blank_match_metadata()
            source_label = "YouTube video ready." if _get_managed_youtube_download_dir(video_path) else "Uploaded video ready."
            status = VIDEO_SOURCE_EMPTY_STATUS if video_path is None else _format_video_source_status(
                source_label,
                regional_name=regional_name,
                match_title=metadata.get("match_title", ""),
                blue_robots=[current_blue_1, current_blue_2, current_blue_3],
                red_robots=[current_red_1, current_red_2, current_red_3],
                calibration_loaded=loaded_saved,
            )
            return (*calibration_state[:-1], status)

        def handle_youtube_download(youtube_url, start_seconds,
                                    current_blue_1, current_blue_2, current_blue_3,
                                    current_red_1, current_red_2, current_red_3,
                                    current_highlight, current_regional,
                                    progress=gr.Progress()):
            video_path, metadata = _download_youtube_video(youtube_url, progress=progress)
            merged_blue = _merge_prefilled_robot_numbers(
                metadata.get("blue_robots", []),
                [current_blue_1, current_blue_2, current_blue_3],
            )
            merged_red = _merge_prefilled_robot_numbers(
                metadata.get("red_robots", []),
                [current_red_1, current_red_2, current_red_3],
            )
            resolved_regional = _clean_text(metadata.get("regional_name") or current_regional)
            calibration_state = _prepare_composite_video_calibration_state(video_path, start_seconds, resolved_regional)
            loaded_saved = calibration_state[-1]
            highlight_update = _update_ball_highlight_dropdown(
                merged_blue[0], merged_blue[1], merged_blue[2],
                merged_red[0], merged_red[1], merged_red[2],
                current_highlight,
            )
            status = _format_video_source_status(
                "YouTube video ready.",
                regional_name=resolved_regional,
                match_title=metadata.get("match_title", ""),
                blue_robots=merged_blue,
                red_robots=merged_red,
                calibration_loaded=loaded_saved,
            )
            return (
                video_path,
                *calibration_state[:-1],
                merged_blue[0], merged_blue[1], merged_blue[2],
                merged_red[0], merged_red[1], merged_red[2],
                resolved_regional,
                highlight_update,
                status,
                metadata,
                _get_page_title_for_match(metadata),
            )
        
        composite_video_input.change(
            fn=handle_video_upload,
            inputs=[
                composite_video_input,
                start_seconds_input,
                blue_robot_1,
                blue_robot_2,
                blue_robot_3,
                red_robot_1,
                red_robot_2,
                red_robot_3,
                highlight_ball_robot_dropdown,
                regional_input,
            ],
            outputs=[
                calibration_image, calibration_base_image, calibration_points_state, calibration_image_size_state, calibration_status,
                blue_side_calibration_image, blue_side_base_image, blue_side_box_points_state, blue_side_box_image_size_state, blue_side_status,
                red_side_calibration_image, red_side_base_image, red_side_box_points_state, red_side_box_image_size_state, red_side_status,
                blue_robot_1,
                blue_robot_2,
                blue_robot_3,
                red_robot_1,
                red_robot_2,
                red_robot_3,
                regional_input,
                highlight_ball_robot_dropdown,
                video_source_status,
                video_metadata_state,
                page_title_state,
            ]
        )

        regional_input.change(
            fn=handle_regional_change,
            inputs=[
                composite_video_input,
                start_seconds_input,
                regional_input,
                blue_robot_1,
                blue_robot_2,
                blue_robot_3,
                red_robot_1,
                red_robot_2,
                red_robot_3,
                video_metadata_state,
            ],
            outputs=[
                calibration_image, calibration_base_image, calibration_points_state, calibration_image_size_state, calibration_status,
                blue_side_calibration_image, blue_side_base_image, blue_side_box_points_state, blue_side_box_image_size_state, blue_side_status,
                red_side_calibration_image, red_side_base_image, red_side_box_points_state, red_side_box_image_size_state, red_side_status,
                video_source_status,
            ]
        )

        youtube_download_btn.click(
            fn=handle_youtube_download,
            inputs=[
                youtube_url_input,
                start_seconds_input,
                blue_robot_1,
                blue_robot_2,
                blue_robot_3,
                red_robot_1,
                red_robot_2,
                red_robot_3,
                highlight_ball_robot_dropdown,
                regional_input,
            ],
            outputs=[
                composite_video_input,
                calibration_image, calibration_base_image, calibration_points_state, calibration_image_size_state, calibration_status,
                blue_side_calibration_image, blue_side_base_image, blue_side_box_points_state, blue_side_box_image_size_state, blue_side_status,
                red_side_calibration_image, red_side_base_image, red_side_box_points_state, red_side_box_image_size_state, red_side_status,
                blue_robot_1,
                blue_robot_2,
                blue_robot_3,
                red_robot_1,
                red_robot_2,
                red_robot_3,
                regional_input,
                highlight_ball_robot_dropdown,
                video_source_status,
                video_metadata_state,
                page_title_state,
            ]
        )

        youtube_url_input.submit(
            fn=handle_youtube_download,
            inputs=[
                youtube_url_input,
                start_seconds_input,
                blue_robot_1,
                blue_robot_2,
                blue_robot_3,
                red_robot_1,
                red_robot_2,
                red_robot_3,
                highlight_ball_robot_dropdown,
                regional_input,
            ],
            outputs=[
                composite_video_input,
                calibration_image, calibration_base_image, calibration_points_state, calibration_image_size_state, calibration_status,
                blue_side_calibration_image, blue_side_base_image, blue_side_box_points_state, blue_side_box_image_size_state, blue_side_status,
                red_side_calibration_image, red_side_base_image, red_side_box_points_state, red_side_box_image_size_state, red_side_status,
                blue_robot_1,
                blue_robot_2,
                blue_robot_3,
                red_robot_1,
                red_robot_2,
                red_robot_3,
                regional_input,
                highlight_ball_robot_dropdown,
                video_source_status,
                video_metadata_state,
                page_title_state,
            ]
        )
        
        def handle_image_click(base_image, clicked_points, regional_name, evt: gr.SelectData):
            if base_image is None:
                return None, clicked_points, "Upload a video first"
            x, y = evt.index
            n = len(clicked_points)
            label = _get_center_calibration_click_label(n)
            img_w, img_h = base_image.size
            print(f"[Calibration UI] Click #{n+1} ({label}): raw=({x}, {y}), base_image_size=({img_w}x{img_h})")
            clicked_points = list(clicked_points) + [(x, y)]
            _persist_regional_calibration(
                regional_name,
                calibration_points=clicked_points,
                calibration_image_size=base_image.size,
            )
            n = len(clicked_points)
            annotated = _redraw_calibration_image(base_image, clicked_points)
            status = _get_calibration_status_text(n)
            if False and n >= 8:
                status = "**All 8 points set!** ✅ Click 'Process Video' to start."
            if False:
                next_label = CALIBRATION_POINT_LABELS[n]
                status = f"**Click point {next_label}** ({n + 1} of 8)"
            return annotated, clicked_points, status
        
        calibration_image.select(
            fn=handle_image_click,
            inputs=[calibration_base_image, calibration_points_state, regional_input],
            outputs=[calibration_image, calibration_points_state, calibration_status]
        )

        def handle_side_box_click(base_image, clicked_points, camera_side, regional_name, evt: gr.SelectData):
            if base_image is None:
                return None, clicked_points, "Upload a video first"
            x, y = evt.index
            n = len(clicked_points)
            if n >= SIDE_CAMERA_BOX_POINT_COUNT:
                annotated = _redraw_side_calibration_image(base_image, clicked_points, camera_side)
                return annotated, clicked_points, _get_side_box_calibration_status_text(camera_side, n)
            label = _get_side_box_point_labels(camera_side)[n]
            print(f"[Side Box UI] {camera_side} click #{n+1} ({label}): raw=({x}, {y})")
            clicked_points = list(clicked_points) + [(x, y)]
            if camera_side == "blue":
                _persist_regional_calibration(
                    regional_name,
                    blue_side_box_points=clicked_points,
                    blue_side_box_image_size=base_image.size,
                )
            else:
                _persist_regional_calibration(
                    regional_name,
                    red_side_box_points=clicked_points,
                    red_side_box_image_size=base_image.size,
                )
            annotated = _redraw_side_calibration_image(base_image, clicked_points, camera_side)
            return annotated, clicked_points, _get_side_box_calibration_status_text(camera_side, len(clicked_points))

        blue_side_calibration_image.select(
            fn=handle_side_box_click,
            inputs=[blue_side_base_image, blue_side_box_points_state, blue_side_name_state, regional_input],
            outputs=[blue_side_calibration_image, blue_side_box_points_state, blue_side_status]
        )

        red_side_calibration_image.select(
            fn=handle_side_box_click,
            inputs=[red_side_base_image, red_side_box_points_state, red_side_name_state, regional_input],
            outputs=[red_side_calibration_image, red_side_box_points_state, red_side_status]
        )
        
        def handle_undo(base_image, clicked_points):
            if not clicked_points:
                return base_image, clicked_points, "No points to undo"
            clicked_points = list(clicked_points)[:-1]
            n = len(clicked_points)
            if n == 0:
                annotated = base_image
            else:
                annotated = _redraw_calibration_image(base_image, clicked_points)
            return annotated, clicked_points, _get_calibration_status_text(n) + " — Undid last point"
            next_label = CALIBRATION_POINT_LABELS[n]
            return annotated, clicked_points, f"**Click point {next_label}** ({n + 1} of 8) — Undid last point"
        
        undo_btn.click(
            fn=handle_undo,
            inputs=[calibration_base_image, calibration_points_state],
            outputs=[calibration_image, calibration_points_state, calibration_status]
        )

        def handle_side_undo(base_image, clicked_points, camera_side):
            if not clicked_points:
                return base_image, clicked_points, "No points to undo"
            clicked_points = list(clicked_points)[:-1]
            if clicked_points:
                annotated = _redraw_side_calibration_image(base_image, clicked_points, camera_side)
            else:
                annotated = base_image
            return annotated, clicked_points, _get_side_box_calibration_status_text(camera_side, len(clicked_points)) + " — Undid last point"

        blue_side_undo_btn.click(
            fn=handle_side_undo,
            inputs=[blue_side_base_image, blue_side_box_points_state, blue_side_name_state],
            outputs=[blue_side_calibration_image, blue_side_box_points_state, blue_side_status]
        )

        red_side_undo_btn.click(
            fn=handle_side_undo,
            inputs=[red_side_base_image, red_side_box_points_state, red_side_name_state],
            outputs=[red_side_calibration_image, red_side_box_points_state, red_side_status]
        )
        
        def handle_skip():
            return [], "**Calibration skipped** — will use default alignment"
        
        skip_calib_btn.click(
            fn=handle_skip,
            inputs=[],
            outputs=[calibration_points_state, calibration_status]
        )

        ball_highlight_inputs = [
            blue_robot_1,
            blue_robot_2,
            blue_robot_3,
            red_robot_1,
            red_robot_2,
            red_robot_3,
            highlight_ball_robot_dropdown,
        ]
        for robot_input in ball_highlight_inputs[:-1]:
            robot_input.change(
                fn=_update_ball_highlight_dropdown,
                inputs=ball_highlight_inputs,
                outputs=[highlight_ball_robot_dropdown]
            )

        # Connect the processing function
        process_btn.click(
            fn=process_dual_videos,
            inputs=[blue_video_input, red_video_input, center_video_input, composite_video_input, fps_slider, start_seconds_input, end_seconds_input, blue_robot_1, blue_robot_2, blue_robot_3, red_robot_1, red_robot_2, red_robot_3, detect_robots_checkbox, detect_fuel_checkbox, side_ref_image_input, center_ref_image_input, enable_blue_cam, enable_center_cam, enable_red_cam, detect_people_checkbox, calibration_points_state, calibration_image_size_state, blue_side_box_points_state, blue_side_box_image_size_state, red_side_box_points_state, red_side_box_image_size_state, show_unlabeled_checkbox, highlight_ball_robot_dropdown, regional_input],
            outputs=[
                blue_video_output, red_video_output, center_video_output, map_video_output,
                blue1_map, blue1_stats,
                blue2_map, blue2_stats,
                blue3_map, blue3_stats,
                red1_map, red1_stats,
                red2_map, red2_stats,
                red3_map, red3_stats,
            ],
        )
    
    return demo


if __name__ == "__main__":
    _cleanup_old_youtube_downloads()
    demo = create_manual_demo(limited_mode=MANUAL_LIMITED_ROBOT_TRACKING) if MANUAL_ROBOT_TRACKING else create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        head=MANUAL_TRACKER_HEAD if MANUAL_ROBOT_TRACKING else None,
    )
