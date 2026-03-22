# -*- coding: utf-8 -*-
"""
Gradio Web Interface for Spatial Understanding Object Tracking
Based on Google Gemini's Spatial Understanding capabilities.
"""

import os
import json
import shutil
import subprocess
import tempfile
import cv2
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw, ImageFont, ImageColor

from dotenv import load_dotenv
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables
load_dotenv()

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
SAM3_MODEL_PATH = Path(__file__).parent / "sam3.pt"
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
LMSTUDIO_URL = os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:1234/v1/chat/completions")
LMSTUDIO_ENABLED = True  # Set to False to disable LMStudio queries

import requests
import base64


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
    
    def check_needs_llm(self, detections: list) -> tuple:
        """
        Check which detections need LLM queries vs can use tracked labels.
        
        Args:
            detections: List of (x1, y1, x2, y2) bounding boxes in pixels
            
        Returns:
            Tuple of (tracked_labels, needs_llm):
            - tracked_labels: List of labels from tracking (or None if no match)
            - needs_llm: List of booleans indicating if LLM query is needed
        """
        tracked_labels = []
        needs_llm = []
        matched_ids = set()
        
        for bbox in detections:
            center = self._calculate_center(bbox)
            
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
                
                # Use tracked label if confident (identified before)
                if label not in ("robot", "unknown") and confidence >= self.confident_threshold:
                    tracked_labels.append(label)
                    needs_llm.append(False)  # Skip LLM - confident track
                else:
                    # Low confidence or still unknown - need LLM
                    tracked_labels.append(label)
                    needs_llm.append(True)
            else:
                # New robot - no match found, need LLM
                tracked_labels.append(None)
                needs_llm.append(True)
        
        return tracked_labels, needs_llm
    
    def update(self, detections: list, new_labels: list) -> list:
        """
        Update tracking with new detections and labels.
        
        Args:
            detections: List of (x1, y1, x2, y2) bounding boxes in pixels
            new_labels: List of labels from LLM (or "unknown")
            
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
                
                if label == "unknown" or label == "robot":
                    # Keep previous label
                    final_label = old_data['label']
                    confidence = old_data['confidence']
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


def label_robot_bboxes_with_local_llm(raw_bboxes: list,
                                      pil_frame: Image.Image,
                                      available_numbers: list,
                                      robot_label_tracker: RobotLabelTracker = None,
                                      crop_padding: int = 50) -> list:
    """Label robot crops with the local LLM while preserving track identity across frames."""
    if not raw_bboxes:
        return []

    available_numbers = [
        str(n).strip()
        for n in (available_numbers or [])
        if n is not None and str(n).strip()
    ]
    img_width, img_height = pil_frame.size

    if available_numbers and robot_label_tracker:
        tracked_labels, needs_query = robot_label_tracker.check_needs_llm(raw_bboxes)
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

        llm_queries = []
        llm_indices = []
        for i, bbox in enumerate(raw_bboxes):
            if not needs_query[i]:
                continue
            x1, y1, x2, y2 = bbox
            cx1 = max(0, x1 - crop_padding)
            cy1 = max(0, y1 - crop_padding)
            cx2 = min(img_width, x2 + crop_padding)
            cy2 = min(img_height, y2 + crop_padding)
            cropped = pil_frame.crop((cx1, cy1, cx2, cy2))
            llm_queries.append({
                'cropped_image': cropped,
                'available_numbers': available_numbers,
                'previous_label': tracked_labels[i] if tracked_labels[i] else None
            })
            llm_indices.append(i)

        if llm_queries:
            print(f"[Bumper+LLM] Querying {len(llm_queries)} robots via local LLM...")
            parallel_results = query_local_llm_batch(
                llm_queries,
                max_workers=min(50, len(llm_queries))
            )
        else:
            parallel_results = []

        all_labels = []
        parallel_idx = 0
        for i in range(len(raw_bboxes)):
            if needs_query[i]:
                all_labels.append(parallel_results[parallel_idx])
                parallel_idx += 1
            else:
                all_labels.append(tracked_labels[i])

        return robot_label_tracker.update(raw_bboxes, all_labels)

    return ["robot"] * len(raw_bboxes)


def query_side_camera_presence(
    frame_image: Image.Image,
    alliance_robots: list,
    camera_side: str = "blue",
    timeout: float = 60.0
) -> list:
    """
    Query local LMStudio to determine which alliance robots are visible in a side camera frame
    and which of the 3 side-camera position buckets each robot occupies.
    
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
    
    # Filter out empty/None robot numbers
    valid_robots = [str(r).strip() for r in alliance_robots if r and str(r).strip()]
    if not valid_robots:
        return []
    
    # Convert image to base64
    buffered = BytesIO()
    frame_image.save(buffered, format="JPEG", quality=85)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # Build prompt
    numbers_str = ", ".join(valid_robots)
    side_name = str(camera_side).strip().lower()
    if side_name == "red":
        valid_positions = {
            "middle": 1,
            "right": 2,
            "far right": 3,
            "farright": 3,
        }
        bucket_instructions = (
            "Assign each to one position: 'middle', 'right', or 'far right': "
            "Default to 'middle' if between middle and right. "
            "Use 'right' only if clearly in the right lane but not at the edge. "
            "Use 'far right' only if at the extreme right lane or against the wall. "
            "Base decisions on visual evidence (guide-box, lane, wall, center) and compare nearby boxes when helpful. "
            "Each robot may appear only once; prefer at most one per box."
        )
        example_json = (
            "[{\"team\":\"77235\",\"description\":\"This robot sits mainly inside the middle guide box and is not close enough to the right wall to count as right, so I placed it in middle.\",\"position\":\"middle\"},"
            "{\"team\":\"4909\",\"description\":\"This robot is shifted into the right-side lane and aligns better with the right guide box than the middle one, but it is not extreme enough to be far right.\",\"position\":\"right\"},"
            "{\"team\":\"5962\",\"description\":\"This robot is pushed all the way against the far-right edge lane near the wall, clearly beyond the normal right box, so it belongs in far right.\",\"position\":\"far right\"}]"
        )
    else:
        valid_positions = {
            "middle": 1,
            "left": 2,
            "far left": 3,
            "farleft": 3,
        }
        bucket_instructions = (
            "Assign each to one position: 'middle', 'left', or 'far left': "
            "Default to 'middle' if between middle and left. "
            "Use 'left' only if clearly in the left lane but not at the edge. "
            "Use 'far left' only if at the extreme left lane or against the wall. "
            "Base decisions on visual evidence (guide-box, lane, wall, center) and compare nearby boxes when helpful. "
            "Each robot may appear only once; prefer at most one per box."
        )
        example_json = (
            "[{\"team\":\"77235\",\"description\":\"This robot sits mainly inside the middle guide box and is not close enough to the left wall to count as left, so I placed it in middle.\",\"position\":\"middle\"},"
            "{\"team\":\"4909\",\"description\":\"This robot is shifted into the left-side lane and aligns better with the left guide box than the middle one, but it is not extreme enough to be far left.\",\"position\":\"left\"},"
            "{\"team\":\"5962\",\"description\":\"This robot is pushed all the way against the far-left edge lane near the wall, clearly beyond the normal left box, so it belongs in far left.\",\"position\":\"far left\"}]"
        )
    prompt = (
        f"Which robots from {numbers_str} are visible in this image? "
        f"Only include robots you actually see. "
        f"Not all robots in the list must be detected. "
        f"{bucket_instructions} "
        f"Reply with ONLY a JSON array. "
        f"Return ONLY a JSON array of objects with keys in this order: 'team', 'description', 'position'. "
        f"Descriptions should be 1-2 sentences describing where the robot is in the frame. "
        f"Example: "
        f"{example_json}. "
        f"If none are visible, reply with []."
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
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            print(f"[LMStudio Side Presence {side_name}] Raw response: {answer}")
            
            if "none" in answer.lower():
                return []

            try:
                parsed = json.loads(parse_json(answer))
            except Exception:
                print(f"[Side Camera LLM] Could not parse JSON: {answer}")
                return []

            if not isinstance(parsed, list):
                return []

            found = []
            seen = set()
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                team = str(item.get("team", "")).strip()
                if team not in valid_robots or team in seen:
                    continue
                description = " ".join(str(item.get("description", "")).strip().split())
                position = str(item.get("position", "")).strip().lower()
                position = " ".join(position.split())
                x_bucket = valid_positions.get(position)
                if x_bucket is None:
                    continue
                found.append({
                    "team": team,
                    "description": description,
                    "position": position,
                    "x_bucket": x_bucket
                })
                seen.add(team)

            return found
        else:
            print(f"[Side Camera LLM] Error: {response.status_code}")
            return []
            
    except requests.exceptions.Timeout:
        print("[Side Camera LLM] Timeout")
        return []
    except requests.exceptions.ConnectionError:
        print("[Side Camera LLM] Not available")
        return []
    except Exception as e:
        print(f"[Side Camera LLM] Error: {e}")
        return []


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


def inject_hidden_robot_bboxes(base_bboxes_json: str, persistent_hidden_robots: dict,
                               side_camera_visible_robots: dict, frame_count: int,
                               width: int, height: int,
                               edge_persist_frames: int = 60) -> tuple:
    """
    Augment center-camera detections with hidden robots inferred from the side cameras.

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

    updated_hidden = dict(persistent_hidden_robots or {})

    for robot_label in center_detected_labels:
        hidden_meta = dict(updated_hidden.get(robot_label, {}))
        hidden_meta['last_center_seen_frame'] = frame_count
        updated_hidden[robot_label] = hidden_meta

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

            hidden_meta = dict(updated_hidden.get(robot_label, {}))
            hidden_meta.update({
                'side': side_name,
                'x_bucket': x_bucket,
                'last_side_seen_frame': latest_side_frame
            })
            updated_hidden[robot_label] = hidden_meta

    injected_bboxes = list(center_bboxes)
    slot_counts = {}
    ordered_hidden = sorted(
        [
            (robot_label, meta)
            for robot_label, meta in updated_hidden.items()
            if robot_label not in center_detected_labels
            and meta.get('side') in ('blue', 'red')
            and meta.get('x_bucket') in (1, 2, 3)
            and meta.get('last_side_seen_frame') is not None
            and meta.get('last_side_seen_frame') > (
                meta.get('last_center_seen_frame')
                if meta.get('last_center_seen_frame') is not None else -1
            )
        ],
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


def _side_camera_bucket_for_bbox(x1: int, y1: int, x2: int, y2: int,
                                 camera_side: str,
                                 frame_width: int, frame_height: int) -> tuple:
    """Assign a side-camera bbox to the nearest guide box."""
    rects = _get_side_camera_zone_rects(camera_side, frame_width, frame_height)
    if not rects:
        return None, None

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    best = None

    for label, (zx1, zy1, zx2, zy2) in rects:
        overlap_w = max(0, min(x2, zx2) - max(x1, zx1))
        overlap_h = max(0, min(y2, zy2) - max(y1, zy1))
        overlap_area = overlap_w * overlap_h
        center_inside = zx1 <= cx <= zx2 and zy1 <= cy <= zy2
        zone_cx = (zx1 + zx2) / 2.0
        zone_cy = (zy1 + zy2) / 2.0
        center_dist = abs(cx - zone_cx) + (abs(cy - zone_cy) * 0.1)
        candidate = (1 if center_inside else 0, overlap_area, -center_dist, label)
        if best is None or candidate > best:
            best = candidate

    label = best[3] if best else None
    if not label:
        return None, None

    position = label.strip().lower()
    bucket_lookup = {
        "middle": 1,
        "left": 2,
        "right": 2,
        "far left": 3,
        "far right": 3,
    }
    return position, bucket_lookup.get(position)


def build_side_camera_visible_robots(raw_bboxes: list, labels: list,
                                     camera_side: str,
                                     frame_width: int, frame_height: int) -> list:
    """Convert labeled side-camera detections into bucketed visibility entries."""
    best_by_team = {}
    unknown_labels = {"robot", "unknown", "Unknown", ""}

    for bbox, label in zip(raw_bboxes or [], labels or []):
        team = str(label).strip()
        if team in unknown_labels:
            continue
        if not bbox or len(bbox) < 4:
            continue

        x1, y1, x2, y2 = bbox
        position, x_bucket = _side_camera_bucket_for_bbox(
            x1, y1, x2, y2, camera_side, frame_width, frame_height
        )
        if x_bucket not in (1, 2, 3):
            continue

        bbox_area = max(1, (x2 - x1) * (y2 - y1))
        entry = {
            "team": team,
            "position": position,
            "x_bucket": x_bucket,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "bbox_area": bbox_area,
        }

        old = best_by_team.get(team)
        if old is None or bbox_area > old["bbox_area"]:
            best_by_team[team] = entry

    return [
        {
            "team": item["team"],
            "position": item["position"],
            "x_bucket": item["x_bucket"],
        }
        for item in sorted(best_by_team.values(), key=lambda item: (item["x_bucket"], item["team"]))
    ]


def annotate_side_camera_guides(frame: Image.Image, camera_side: str) -> Image.Image:
    """
    Draw lightweight side-camera lane guides for bucketed side-camera robot positions.
    """
    if frame is None:
        return frame

    img = frame.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = get_font(18)
    is_blue = str(camera_side).strip().lower() == "blue"
    line_color = (80, 180, 255, 180) if is_blue else (255, 110, 110, 180)
    fill_color = (80, 180, 255, 30) if is_blue else (255, 110, 110, 30)
    tag_fill = (18, 18, 18, 175)
    text_fill = (255, 255, 255, 235)

    for label, (x1, y1, x2, y2) in _get_side_camera_zone_rects(camera_side, img.width, img.height):
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
        self.fps = fps
        self.shot_label_duration = shot_label_duration
        self.min_upward_pixels = min_upward_pixels
        self.max_matching_distance = max_matching_distance
        self.max_frames_lost = max_frames_lost
        self.camera_side = camera_side
        self.blue_robots = [str(r).strip() for r in (blue_robots or []) if r]
        self.red_robots = [str(r).strip() for r in (red_robots or []) if r]
        self.start_seconds = start_seconds  # Video start time for period calculation
        self.ferry_tracker = ferry_tracker  # Reference to ferry tracker
        self.frame_width = frame_width  # Stored for edge-robot detection
        self.frame_height = frame_height  # Stored for edge-robot detection
        self.possession_memory_frames = max(2, int(round(self.fps * 0.30)))
        
        # Track balls: {ball_id: {'pos': (x, y, r), 'prev_pos': (x, y, r), 'shot_by': robot_label, ...}}
        self.tracked_balls = {}
        
        # Lost balls (temporarily unmatched): {ball_id: {'data': ball_data, 'frames_lost': int, 'predicted_pos': (x, y)}}
        self.lost_balls = {}
        
        self.next_ball_id = 0
        self.current_frame = 0
        
        # Store robot bounding boxes from nearest Gemini detection
        # Format: [(y1, x1, y2, x2, label), ...]
        self.robot_bboxes = []
        
        # Shot statistics: {robot_label: {'attempts': 0, 'made': 0, 'by_period': {...}}}
        self.robot_stats = {}
        
        # Shot event log for cross-camera deduplication: [(elapsed_seconds, robot_label, made_bool), ...]
        self.shot_events = []
        
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
    
    def _record_shot(self, robot_label: str, made: bool):
        """
        Record a shot attempt for a robot, tracking both total and by period.
        
        Args:
            robot_label: The robot's team number
            made: True if shot was made, False if missed
        """
        stats = self._get_robot_stats(robot_label)
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
                    self.robot_bboxes.append((y1, x1, y2, x2, label))
        except Exception as e:
            print(f"Error parsing robot bboxes: {e}")
    
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
        
        for (y1, x1, y2, x2, label) in self.robot_bboxes:
            # Only consider robots in the camera's alliance for shot attribution
            if not self._is_robot_in_camera_alliance(label):
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
    
    def _find_nearest_alliance_robot(self, ball_x: int, ball_y: int, max_dist: float = 150.0) -> str:
        """
        Find the nearest alliance robot to a ball by center-to-center distance.
        More forgiving than bbox overlap — works even when the ball has exited the bbox.
        
        Args:
            ball_x: Ball center x
            ball_y: Ball center y
            max_dist: Maximum distance to consider (pixels)
            
        Returns:
            Robot label of nearest alliance robot, or None if none within max_dist
        """
        best_label = None
        best_dist = max_dist
        
        for (y1, x1, y2, x2, label) in self.robot_bboxes:
            if not self._is_robot_in_camera_alliance(label):
                continue
            robot_cx = (x1 + x2) / 2
            robot_cy = (y1 + y2) / 2
            dist = ((ball_x - robot_cx) ** 2 + (ball_y - robot_cy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_label = label
        
        return best_label

    def _get_shot_origin_robot(self, overlapping_robot: str, last_overlap_robot: str,
                               last_overlap_frame: int, last_near_robot: str) -> str:
        """
        Choose which robot should own a newly detected shot.

        We prefer a recent true overlap/possession signal over nearest-robot fallback
        so airborne balls don't get reassigned to a robot behind the shooter.
        """
        if overlapping_robot:
            return overlapping_robot

        if last_overlap_robot and last_overlap_frame is not None:
            frames_since_overlap = self.current_frame - last_overlap_frame
            if frames_since_overlap <= self.possession_memory_frames:
                return last_overlap_robot

        return last_near_robot
    
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
            List of (x, y, radius, robot_label_or_None) tuples
        """
        self.current_frame += 1
        
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
                vy += 0.5  # Gravity: ~0.5 px/frame² downward acceleration
                pred_x += vx
                pred_y += vy
                lost_data['velocity'] = (vx, vy)
                lost_data['predicted_pos'] = (pred_x, pred_y)
            
            # Check if ball has been lost too long
            if lost_data['frames_lost'] <= self.max_frames_lost:
                updated_lost_balls[ball_id] = lost_data
            else:
                # Ball lost for too long - finalize shot stats if applicable
                shot_by = lost_data['data'].get('shot_by')
                shot_evaluated = lost_data['data'].get('shot_evaluated', False)
                
                # Only count MADE shots (ball disappeared into goal)
                if shot_by and not shot_evaluated:
                    x, y, _ = lost_data['data']['pos']
                    in_goal = lost_data['data'].get('last_seen_in_goal', False) or self._is_in_goal(x, y)
                    
                    if in_goal:
                        self._record_shot(shot_by, made=True)
                        period = get_match_period(self._get_elapsed_seconds())
                        print(f"[SHOT MADE] Robot {shot_by} @ {period}: pos=({x:.0f},{y:.0f})")
                    else:
                        print(f"[SHOT NOT SCORED] Robot {shot_by}: pos=({x:.0f},{y:.0f}) - not in goal")
        
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
                overlapping_robot = old_data.get('overlapping_robot')
                last_near_robot = old_data.get('last_near_robot')
                last_overlap_robot = old_data.get('last_overlap_robot')
                last_overlap_frame = old_data.get('last_overlap_frame')
                was_ever_in_goal = old_data.get('last_seen_in_goal', False)
                
            elif ball_id is None:
                # New ball
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
                    'last_seen_in_goal': False
                }
                robot_label = None
                results.append((x, y, r, robot_label))
                continue
            else:
                # Existing active ball
                old_data = self.tracked_balls[ball_id]
                prev_pos = old_data['pos']
                shot_by = old_data.get('shot_by')
                shot_time = old_data.get('shot_time')
                shot_evaluated = old_data.get('shot_evaluated', False)
                candidate_shot = old_data.get('candidate_shot')
                overlapping_robot = old_data.get('overlapping_robot')
                last_near_robot = old_data.get('last_near_robot')
                last_overlap_robot = old_data.get('last_overlap_robot')
                last_overlap_frame = old_data.get('last_overlap_frame')
                was_ever_in_goal = old_data.get('last_seen_in_goal', False)
            
            # Check for shot initiation (upward movement >= min_upward_pixels)
            y_change = prev_pos[1] - y  # Positive if ball moved up
            

            
            # If not currently a shot/candidate, check if we should start tracking a shot
            # Prefer recent true possession/overlap over nearest-robot fallback.
            if not shot_by and not candidate_shot:
                nearby_robot = self._get_shot_origin_robot(
                    overlapping_robot,
                    last_overlap_robot,
                    last_overlap_frame,
                    last_near_robot
                )
                if y_change >= self.min_upward_pixels and nearby_robot:
                    # Start candidate tracking
                    candidate_shot = {
                        'robot': nearby_robot,
                        'start_pos': prev_pos,
                        'start_frame': self.current_frame
                    }
            
            # Validate candidate shot - based on height gain ONLY
            if candidate_shot:
                frames_since_start = self.current_frame - candidate_shot['start_frame']
                seconds_since_start = frames_since_start / self.fps
                
                if seconds_since_start >= 0.125:
                    # Validation Check: Must be >= 25 pixels higher than start
                    start_y = candidate_shot['start_pos'][1]
                    current_y = y
                    height_gain = start_y - current_y
                    
                    if height_gain >= 25:
                        # Confirmed! Promoting to full shot - label the ball
                        # NOTE: Attempts are NOT counted here - they're counted after 2 seconds
                        shot_by = candidate_shot['robot']
                        shot_time = self.current_frame
                        shot_evaluated = False  # Haven't counted this shot yet
                        candidate_shot = None
                        print(f"[SHOT DETECTED] Ball {ball_id} shot by {shot_by} at pos=({x:.0f},{y:.0f}), height_gain={height_gain:.0f}px")
                    else:
                        # Failed validation - not enough height
                        print(f"[SHOT FAILED] Ball {ball_id} height_gain={height_gain:.0f}px < 25px required")
                        candidate_shot = None
                        
                elif y > candidate_shot['start_pos'][1] + 20:
                    # Early failure check
                    candidate_shot = None
            
            # Check if 2 seconds have passed since shot - time to evaluate!
            # Only count MADE shots (ball in goal)
            if shot_time is not None and not shot_evaluated:
                frames_since_shot = self.current_frame - shot_time
                seconds_since_shot = frames_since_shot / self.fps
                
                if seconds_since_shot >= 2.0:
                    is_in_goal_now = was_ever_in_goal or self._is_in_goal(x, y)
                    
                    if is_in_goal_now:
                        self._record_shot(shot_by, made=True)
                        period = get_match_period(self._get_elapsed_seconds())
                        print(f"[SHOT MADE] Robot {shot_by} @ {period}: 2sec eval, in goal")
                    else:
                        print(f"[SHOT NOT SCORED] Robot {shot_by}: 2sec eval, not in goal")
                    
                    shot_evaluated = True
            
            # Check if shot label should expire (keep the visual label for duration)
            if shot_time is not None:
                frames_since_shot = self.current_frame - shot_time
                seconds_since_shot = frames_since_shot / self.fps
                if seconds_since_shot > self.shot_label_duration:
                    shot_by = None
                    shot_time = None
            
            is_in_goal = self._is_in_goal(x, y)
            
            # Sticky flags: once True, stays True
            ever_in_goal = was_ever_in_goal or is_in_goal
            
            # Debug: track when shot balls enter goal
            if shot_by and is_in_goal:
                print(f"[IN GOAL] Ball {ball_id} (shot by {shot_by}) at pos=({x:.0f},{y:.0f})")
            
            cur_overlap = self._ball_overlaps_robot(x, y, r)
            cur_nearest = self._find_nearest_alliance_robot(x, y)
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
                'last_seen_in_goal': ever_in_goal
            }
            
            # Add to results
            robot_label = new_tracked[ball_id].get('shot_by')
            results.append((x, y, r, robot_label))
        
        self.tracked_balls = new_tracked
        return results
    
    def reset(self):
        """Reset tracker state."""
        self.tracked_balls = {}
        self.lost_balls = {}
        self.robot_stats = {}
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
                in_goal = ball_data.get('last_seen_in_goal', False) or self._is_in_goal(x, y)
                
                if in_goal:
                    self._record_shot(shot_by, made=True)
                    print(f"[FINALIZE MADE] Robot {shot_by}: pos=({x:.0f},{y:.0f})")
                else:
                    print(f"[FINALIZE NOT SCORED] Robot {shot_by}: pos=({x:.0f},{y:.0f})")
        
        # Finalize all balls in the lost pool — only count MADE shots
        for ball_id, lost_data in self.lost_balls.items():
            shot_by = lost_data['data'].get('shot_by')
            shot_evaluated = lost_data['data'].get('shot_evaluated', False)
            
            if shot_by and not shot_evaluated:
                x, y, _ = lost_data['data']['pos']
                in_goal = lost_data['data'].get('last_seen_in_goal', False) or self._is_in_goal(x, y)
                
                if in_goal:
                    self._record_shot(shot_by, made=True)
                    print(f"[FINALIZE LOST MADE] Robot {shot_by}: pos=({x:.0f},{y:.0f})")
                else:
                    print(f"[FINALIZE LOST NOT SCORED] Robot {shot_by}: pos=({x:.0f},{y:.0f})")
        
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
        
        # Crop to center camera portion (matches split_composite_video: crop=1918:709:1:0)
        h, w = frame.shape[:2]
        center_frame = frame[0:min(709, h), 1:min(1919, w)]  # y=0..709, x=1..1919
        
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
NO_SCAN_POINT_LABELS = ["BZ1", "BZ2", "BZ3", "BZ4", "RZ1", "RZ2", "RZ3", "RZ4"]
ALL_CALIBRATION_POINT_LABELS = CALIBRATION_POINT_LABELS + NO_SCAN_POINT_LABELS
CALIBRATION_REQUIRED_POINTS = len(CALIBRATION_POINT_LABELS)
CALIBRATION_TOTAL_POINTS = len(ALL_CALIBRATION_POINT_LABELS)


def _get_calibration_status_text(num_points: int) -> str:
    """Return the next-step instruction for calibration / no-scan clicks."""
    if num_points <= 0:
        return "**Click point B1** (1 of 8)"

    if num_points < CALIBRATION_REQUIRED_POINTS:
        next_label = CALIBRATION_POINT_LABELS[num_points]
        return f"**Click point {next_label}** ({num_points + 1} of 8)"

    if num_points == CALIBRATION_REQUIRED_POINTS:
        return (
            "**Calibration points set!** Optional: click **BZ1** to start the blue no-scan box "
            "(9 of 16), or click 'Process Video' now."
        )

    if num_points < CALIBRATION_TOTAL_POINTS:
        next_label = ALL_CALIBRATION_POINT_LABELS[num_points]
        return f"**Click point {next_label}** ({num_points + 1} of 16)"

    return "**Calibration + no-scan boxes set!** Click 'Process Video' to start."


def _split_calibration_and_exclusion_points(clicked_points: list) -> tuple:
    """Split UI clicks into homography points and optional robot no-scan polygons."""
    points = list(clicked_points or [])
    calibration_points = points[:CALIBRATION_REQUIRED_POINTS]
    extra_points = points[CALIBRATION_REQUIRED_POINTS:CALIBRATION_TOTAL_POINTS]

    polygons = []
    for idx in range(0, len(extra_points), 4):
        polygon = extra_points[idx:idx + 4]
        if len(polygon) == 4:
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
        label = ALL_CALIBRATION_POINT_LABELS[i] if i < len(ALL_CALIBRATION_POINT_LABELS) else f"P{i}"
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
            # Handle 2-tuple (cx, cy), 3-tuple (cx, cy, side), and 4-tuple (cx, cy, side, bbox_area)
            if len(pos) >= 4:
                cx, cy, side, _ = pos  # Ignore bbox_area for drawing
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
                
                # Handle 2-tuple, 3-tuple, and 4-tuple formats
                if len(pos) >= 4:
                    cx, cy, side, _ = pos  # Ignore bbox_area
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
                    # Handle 2-tuple, 3-tuple, and 4-tuple formats
                    if len(pos) >= 4:
                        cx, cy, side, _ = pos  # Ignore bbox_area
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


# Center camera ROI regions (1918x709 frame)
# Only the bottom-left and bottom-right corners contain scoring areas
_CENTER_CAM_ROIS = [
    (0,    129, 730,  709),   # Left side  — 730x580
    (1188, 129, 1918, 709),   # Right side — 730x580
]


def detect_fuel_sam3(frame_bgr: np.ndarray, predictor,
                     min_radius: int = 3, max_radius: int = 30,
                     camera_side: str = "blue") -> list:
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


def draw_fuel_detections(frame: Image.Image, fuel_detections: list, blue_robots: list = None, red_robots: list = None) -> Image.Image:
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
            atm = stats[team_number]['attempts']
            if atm > 0:
                label_text += f" - {made}/{atm}"
        
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

# Minimum contour area / color pixels for bumper detection (reject small noise)
_BUMPER_MIN_AREA = 90
_BUMPER_MIN_COLOR_PIXELS = 65
_BUMPER_MERGE_GAP_X = 90
_BUMPER_MERGE_GAP_Y = 45

# Center camera playing field ROI (x1, y1, x2, y2) — excludes audience areas
_BUMPER_FIELD_ROI = (0, 315, 1918, 705)


def _build_center_blue_mask(field_region_bgr: np.ndarray, hsv_region: np.ndarray) -> np.ndarray:
    """
    Detect blue bumpers, including dark navy shades such as rgb(18, 21, 38),
    while rejecting nearly neutral black structures.
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


def _get_bumper_field_roi(camera_side: str, frame_width: int, frame_height: int) -> tuple:
    """Return the robot-detection ROI for the given camera."""
    if str(camera_side).strip().lower() == "center":
        return _get_calibrated_field_roi()
    return 0, 0, frame_width, frame_height


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



def compute_field_pixel_mask(video_path: str, start_seconds: float = 0,
                             sample_fps: float = 3.0,
                             threshold: float = 0.40,
                             camera_side: str = "center") -> np.ndarray:
    """
    Pre-scan a camera video to build a per-pixel field exclusion mask.

    Any pixel in the field ROI that is red or blue in >= `threshold` fraction of
    sampled frames is considered a static field element and will be excluded from
    robot bumper detection.  The first 3 seconds of the video are always skipped
    (robots may still be settling into position).

    Args:
        video_path: Path to the camera video file.
        start_seconds: Global start offset already applied to the video (so we
                       skip an additional 3 s on top of this).
        sample_fps: How many frames per second to sample (default 3).
        threshold: Fraction of frames a pixel must be red/blue to be excluded.
        camera_side: "center", "blue", or "red".

    Returns:
        Binary mask the same size as the field ROI (h, w) where
        0 = excluded (field element) and 255 = allowed (potential robot).
    """
    roi_x1 = roi_y1 = 0
    roi_x2 = roi_y2 = 1
    roi_h = roi_w = 1

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[FieldMask] Could not open video – returning empty mask")
        return np.ones((roi_h, roi_w), dtype=np.uint8) * 255

    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1
    roi_x1, roi_y1, roi_x2, roi_y2 = _get_bumper_field_roi(camera_side, frame_width, frame_height)
    roi_h = max(1, roi_y2 - roi_y1)
    roi_w = max(1, roi_x2 - roi_x1)

    # Skip the first 3 seconds (on top of any user-specified start offset)
    skip_seconds = 3.0
    first_valid_frame = int((start_seconds + skip_seconds) * original_fps)
    sample_interval = max(1, int(original_fps / sample_fps))

    # ROI already computed above via _get_bumper_field_roi()

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

            # Red (two HSV ranges)
            r1 = cv2.inRange(hsv, lower_red1, upper_red1)
            r2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_pix = cv2.bitwise_or(r1, r2)

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
                                  robot_exclusion_polygons: list = None,
                                  camera_side: str = "center") -> tuple:
    """
    Detect robots by finding red and blue bumper regions using HSV color matching.
    
    Uses two HSV ranges for red (wraps around 0° in HSV) and one for blue.
    Returns bounding boxes in the same JSON format as other detection backends,
    plus raw masks for visual highlighting.
    
    Args:
        frame_bgr: OpenCV BGR image
        person_mask: Optional binary mask of detected people (255 = person, 0 = not)
        field_pixel_mask: Optional per-pixel exclusion mask from compute_field_pixel_mask().
        robot_exclusion_polygons: Optional list of user-drawn no-scan polygons in frame coordinates.
        camera_side: "center", "blue", or "red".
        
    Returns:
        Tuple of (bounding_boxes_json, red_mask, blue_mask):
        - bounding_boxes_json: JSON string with detections in standard format
        - red_mask: Binary mask of red bumper pixels
        - blue_mask: Binary mask of blue bumper pixels
    """
    h_full, w_full = frame_bgr.shape[:2]
    # Crop to the robot-detection ROI (calibration-adjusted for center, full frame for side cameras)
    roi_x1, roi_y1, roi_x2, roi_y2 = _get_bumper_field_roi(camera_side, w_full, h_full)
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
    
    # HSV color ranges
    lower_red1 = np.array([0, 80, 80])
    upper_red1 = np.array([10, 255, 220])
    lower_red2 = np.array([160, 80, 80])
    upper_red2 = np.array([180, 255, 220])
    try:
        # GPU Acceleration Path (using OpenCV T-API / OpenCL)
        umat_roi = cv2.UMat(field_region)
        hsv = cv2.cvtColor(umat_roi, cv2.COLOR_BGR2HSV)
        
        # Red bumper (two ranges for HSV wrap-around)
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Blue bumper
        blue_mask = _build_center_blue_mask(field_region, hsv.get())
        
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
        
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        blue_mask = _build_center_blue_mask(field_region, hsv)
        
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

        for x1, y1, x2, y2 in _merge_bumper_boxes(candidate_boxes):
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


def process_single_video(video_path: str, camera_side: str = "blue", target_fps: int = 3, start_seconds: float = 0, end_seconds: float = 0, blue_robots: list = None, red_robots: list = None, enable_robot_detection: bool = True, enable_fuel_detection: bool = True, progress=gr.Progress(), camera_name: str = "Camera", enable_person_detection: bool = True, calibration_points: list = None, calibration_image_size: tuple = None, side_camera_visible_robots: dict = None, show_unlabeled_robots: bool = True) -> tuple:
    """
    Process a single video, tracking objects at specified FPS.
    Uses bumper color detection for robot identification.
    
    Args:
        video_path: Path to input video
        camera_side: "blue", "red", or "center" for camera perspective
        target_fps: Target FPS for processing (default 3)
        start_seconds: Start processing at this time (0 = from beginning)
        end_seconds: Stop processing at this time (0 = process to end)
        blue_robots: List of blue alliance team numbers [robot1, robot2, robot3]
        red_robots: List of red alliance team numbers [robot1, robot2, robot3]
        enable_robot_detection: Whether to detect robots (default True)
        enable_fuel_detection: Whether to detect yellow fuel balls (default True)
        progress: Gradio progress tracker
        camera_name: Display name for the camera (e.g., "Blue Camera")
        side_camera_visible_robots: Dict of side camera visibility data (for center camera hidden robot injection)
            Format: {'blue': {frame_num: [{'team': str, 'position': str, 'x_bucket': int}]}, ...}
        
    Returns:
        Tuple of (output_video_path, robot_tracks, tracks_by_frame, width, height, robot_stats, ferry_counts, disabled_statuses, shot_events, side_visible_robots)
    """
    if not video_path:
        raise gr.Error("Please upload a video file.")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error("Could not open video file.")
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
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
    
    # Skip to start frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Create output video at 30fps (ball detection rate) - use H264 codec for better compatibility
    output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    output_fps = min(30.0, original_fps)  # Output at 30fps or video fps if lower
    # Try multiple codecs for Windows compatibility
    fourcc_options = ['avc1', 'H264', 'mp4v', 'XVID']
    out = None
    for codec in fourcc_options:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
            if out.isOpened():
                print(f"Using codec: {codec}")
                break
        except:
            continue
    
    if out is None or not out.isOpened():
        # Fallback to avi format
        output_path = tempfile.NamedTemporaryFile(suffix=".avi", delete=False).name
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
    
    if not out.isOpened():
        cap.release()
        raise gr.Error("Could not create output video file.")
    
    # Calculate frame intervals
    # Robot detection at user-specified FPS
    # Side cameras stay at a fixed 3 FPS for bumper detection / OCR
    if camera_side in ("blue", "red"):
        robot_frame_interval = max(1, int(original_fps / 3.0))
    else:
        robot_frame_interval = max(1, int(original_fps / target_fps))
    # Ball detection at 30fps (or video fps if lower)
    ball_fps = min(30.0, original_fps)
    ball_frame_interval = max(1, round(original_fps / ball_fps))
    # Person detection at 6fps (independent of robot FPS)
    person_frame_interval = max(1, int(original_fps / 6))
    
    frame_count = start_frame
    processed_frames = 0
    total_ball_frames = (end_frame - start_frame) // ball_frame_interval
    

    
    # Robot tracking for map visualization
    robot_tracks = {}  # label -> list of (center_x, center_y, camera_side)
    tracks_by_frame = []  # List of dicts {label: (cx, cy, side)} for each frame
    
    # Ferry tracker for counting fuel ferries (cross out, cross back, shoot)
    ferry_tracker = FerryTracker(blue_robots=blue_robots, red_robots=red_robots)
    
    # Persistent hidden robot tracking (label -> {'side': str, 'x_bucket': int})
    persistent_hidden_robots = {}
    
    # Disabled tracker for detecting when robots stop moving
    disabled_tracker = DisabledTracker(fps=target_fps)
    
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

    # Store latest robot detection for use with ball frames
    current_bboxes_json = "[]"
    
    # Pre-compute field pixel mask for bumper detection
    field_pixel_mask = None
    if enable_robot_detection:
        if progress is not None:
            progress(0, desc="Scanning video to build field pixel mask...")
        field_pixel_mask = compute_field_pixel_mask(
            video_path,
            start_seconds=start_seconds,
            camera_side=camera_side
        )
    
    # Bumper detection masks (stored for rendering on ball frames)
    current_bumper_red_mask = None
    current_bumper_blue_mask = None
    
    # Person detection state (stored for rendering and bumper exclusion)
    current_person_mask = None
    
    # Side-camera visibility data for hidden-robot injection
    # Maps frame_number -> list of {'team', 'position', 'x_bucket'}
    side_visible_robots_by_frame = {}
    
    # Robot label tracker for YOLO + LLM mode (maintains identity across frames)
    robot_label_tracker = RobotLabelTracker(max_distance=100.0)
    
    if progress is not None:
        progress(0, desc=f"Processing {camera_name} - Frame 0/{total_ball_frames}")

    # Use threaded reader/writer to overlap I/O with processing
    reader = ThreadedVideoReader(cap, frame_count, end_frame)
    writer = ThreadedVideoWriter(out)
    
    while True:
        ret, frame, frame_count = reader.read()
        if not ret:
            break
        
        # Person detection at 6fps (center camera only)
        if (frame_count % person_frame_interval == 0 and enable_person_detection
                and camera_side == "center"
                and YOLO_PERSON_MODEL is not None):
            current_person_mask, current_person_count = detect_people_yolo(frame)
            if current_person_count > 0:
                print(f"[Person Detection] Found {current_person_count} people at frame {frame_count}")
        
        # Robot detection at user-specified FPS (less frequent)
        if frame_count % robot_frame_interval == 0 and enable_robot_detection:
            # Convert BGR (OpenCV) to RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            
            # Combine robot numbers
            robot_numbers = (blue_robots or []) + (red_robots or [])
            
            # Bumper detection
            if camera_side in ("blue", "red"):
                # Side camera: bumper detection + OCR labeling + side-box bucketing
                alliance_robots = blue_robots if camera_side == "blue" else red_robots
                color_bboxes_json, current_bumper_red_mask, current_bumper_blue_mask, raw_bboxes = detect_robots_by_bumper_color(
                    frame,
                    field_pixel_mask=field_pixel_mask,
                    camera_side=camera_side
                )
                try:
                    color_detections = json.loads(parse_json(color_bboxes_json))
                except Exception:
                    color_detections = []

                alliance_color = str(camera_side).strip().lower()
                alliance_raw_bboxes = [
                    bbox for bbox, det in zip(raw_bboxes, color_detections)
                    if str(det.get("label", "")).strip().lower() == alliance_color
                ]

                final_labels = label_robot_bboxes_with_local_llm(
                    alliance_raw_bboxes,
                    pil_frame,
                    alliance_robots,
                    robot_label_tracker
                )

                detections = []
                for i, (x1, y1, x2, y2) in enumerate(alliance_raw_bboxes):
                    y1_norm = int((y1 / height) * 1000)
                    x1_norm = int((x1 / width) * 1000)
                    y2_norm = int((y2 / height) * 1000)
                    x2_norm = int((x2 / width) * 1000)
                    detections.append({
                        "box_2d": [y1_norm, x1_norm, y2_norm, x2_norm],
                        "label": final_labels[i] if i < len(final_labels) else "robot"
                    })
                bounding_boxes_json = json.dumps(detections)

                visible_robots = build_side_camera_visible_robots(
                    alliance_raw_bboxes,
                    final_labels,
                    camera_side,
                    width,
                    height
                )
                side_visible_robots_by_frame[frame_count] = visible_robots
                if visible_robots:
                    print(f"[Side Camera Bumper] {camera_name} sees robots: {visible_robots}")
            else:
                # Center camera: bumper color detection + LLM labeling
                _, current_bumper_red_mask, current_bumper_blue_mask, raw_bboxes = detect_robots_by_bumper_color(
                    frame,
                    person_mask=current_person_mask,
                    field_pixel_mask=field_pixel_mask,
                    robot_exclusion_polygons=robot_exclusion_polygons,
                    camera_side=camera_side
                )
                
                # Use RobotLabelTracker + local LLM OCR to assign team numbers
                final_labels = label_robot_bboxes_with_local_llm(
                    raw_bboxes,
                    pil_frame,
                    robot_numbers,
                    robot_label_tracker
                )
                
                # Rebuild bounding_boxes_json with team number labels
                detections = []
                img_height, img_width = frame.shape[:2]

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
                    bounding_boxes_json, persistent_hidden_robots = inject_hidden_robot_bboxes(
                        bounding_boxes_json,
                        persistent_hidden_robots,
                        side_camera_visible_robots,
                        frame_count,
                        width,
                        height,
                        edge_persist_frames=max(30, int(round(original_fps * 2.0)))
                    )
            
            # Store for tracking
            current_bboxes_json = bounding_boxes_json
            

            
            # Update ball tracker with robot bounding boxes
            ball_tracker.update_robot_bboxes(bounding_boxes_json, width, height)
            
            # Extract and track robot positions
            bbox_centers = extract_bbox_centers(bounding_boxes_json, width, height)
            frame_tracks = {}
            for label, (cx, cy, bbox_area) in bbox_centers.items():
                if label not in robot_tracks:
                    robot_tracks[label] = []
                robot_tracks[label].append((cx, cy, camera_side, bbox_area))
                frame_tracks[label] = (cx, cy, camera_side, bbox_area)
                
                # Update disabled tracker and ferry tracker with map coordinates
                # Use rotated map dimensions (961x574)
                map_x, map_y = transform_to_map(cx, cy, width, height, 961, 574, camera_side)
                if map_x is not None:
                    disabled_tracker.update_position(label, map_x, map_y)
                    ferry_tracker.update_position(label, map_x, map_y)
            
            tracks_by_frame.append(frame_tracks)
        
        # Ball detection and output at 30fps (or video fps)
        if frame_count % ball_frame_interval == 0:
            # Convert BGR (OpenCV) to RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            
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
            
            # Draw bounding boxes with alliance colors (for robots) - only if robot detection enabled
            annotated_frame = pil_frame.copy()
            if enable_robot_detection:
                # Draw bumper color highlights using the cached masks from the last robot detection frame
                annotated_frame = draw_bumper_highlights(
                    annotated_frame,
                    current_bumper_red_mask,
                    current_bumper_blue_mask,
                    field_pixel_mask=field_pixel_mask
                )
                
                # Draw person segmentation in grey
                if current_person_mask is not None and enable_person_detection and np.any(current_person_mask):
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
            if camera_side in ("blue", "red"):
                annotated_frame = annotate_side_camera_guides(annotated_frame, camera_side)
            
            # Update ball tracker with best available robot bboxes (interpolated on non-keyframes)
            # This ensures shot attribution uses accurate robot positions every ball frame,
            # not just stale keyframe data
            if enable_robot_detection:
                ball_tracker.update_robot_bboxes(render_bboxes_json, width, height)
            
            # Detect and draw fuel using color-based detection if enabled
            if enable_fuel_detection:
                if SAM3_PREDICTOR is not None:
                    fuel_detections = detect_fuel_sam3(frame, SAM3_PREDICTOR,
                                                       min_radius=3, max_radius=30,
                                                       camera_side=camera_side)
                else:
                    fuel_detections = detect_fuel(frame, min_radius=3, max_radius=30,
                                                  tracked_positions=ball_tracker.get_predicted_positions())
                
                # Track balls and detect shots
                tracked_balls = ball_tracker.update(fuel_detections)
                
                # Draw with shot attribution
                annotated_frame = draw_fuel_detections(annotated_frame, tracked_balls, blue_robots, red_robots)
            
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
                    
                # Helper to transform a reference-coords rectangle to current frame polygon
                def transform_rect(x1, y1, x2, y2):
                    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                    return [ref_to_current(cx, cy) for cx, cy in corners]
                    

                
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
            
            
            # Convert back to BGR for OpenCV
            annotated_bgr = cv2.cvtColor(np.array(annotated_frame), cv2.COLOR_RGB2BGR)
            
            # Write frame to output (non-blocking, queued to writer thread)
            writer.write(annotated_bgr)
            processed_frames += 1
        
    
    # Wait for all frames to be written, then release resources
    reader.stop()
    writer.stop()
    cap.release()
    out.release()
    
    # Finalize all remaining tracked balls to ensure all shots are counted
    # This is critical for counting misses that exit the frame
    ball_tracker.finalize_all()
    
    # Get ferry counts from the ferry tracker
    ferry_counts = ferry_tracker.get_all_ferry_counts()
    
    # Get disabled statuses from the disabled tracker
    disabled_statuses = disabled_tracker.get_all_disabled_statuses()
    
    return output_path, robot_tracks, tracks_by_frame, width, height, ball_tracker.robot_stats, ferry_counts, disabled_statuses, ball_tracker.shot_events, side_visible_robots_by_frame


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
    Split a 1920x1080 composite video into 3 separate camera feeds.
    
    Crop regions (from HTML image map coords):
        Center Camera: (1, 0) -> (1919, 709)    = 1918x709
        Blue Side:     (1, 739) -> (941, 1078)   = 940x339
        Red Side:      (979, 739) -> (1919, 1078) = 940x339
    
    Uses FFmpeg subprocess for speed (parallel, hardware-friendly).
    Falls back to OpenCV frame loop if FFmpeg is unavailable.
    
    Args:
        composite_path: Path to the 1920x1080 composite video
        progress: Optional Gradio progress tracker
        
    Returns:
        Tuple of (center_path, blue_path, red_path) temp file paths
    """
    # FFmpeg crop filter format: crop=w:h:x:y
    crops = {
        'center': {'filter': 'crop=1918:709:1:0',   'size': (1918, 709)},
        'blue':   {'filter': 'crop=940:339:1:739',   'size': (940, 339)},
        'red':    {'filter': 'crop=940:339:979:739',  'size': (940, 339)},
    }
    
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
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Crop regions as (x1, y1, x2, y2)
    crop_rects = {
        'center': (1, 0, 1919, 709),
        'blue':   (1, 739, 941, 1078),
        'red':    (979, 739, 1919, 1078),
    }
    
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


def process_dual_videos(blue_video_path: str, red_video_path: str, center_video_path: str = None, composite_video_path: str = None, target_fps: int = 3, start_seconds: float = 0, end_seconds: float = 0, blue_robot_1: str = "", blue_robot_2: str = "", blue_robot_3: str = "", red_robot_1: str = "", red_robot_2: str = "", red_robot_3: str = "", enable_robot_detection: bool = True, enable_fuel_detection: bool = True, side_ref_image: Image.Image = None, center_ref_image: Image.Image = None, enable_blue_camera: bool = True, enable_center_camera: bool = True, enable_red_camera: bool = True, enable_person_detection: bool = True, calibration_points: list = None, calibration_image_size: tuple = None, show_unlabeled_robots: bool = True, progress=gr.Progress()) -> tuple:
    """
    Process blue, red, and center camera videos using bumper detection.
    
    Args:
        blue_video_path: Path to blue side camera video
        red_video_path: Path to red side camera video
        center_video_path: Path to center camera video (2136x836, views both sides)
        target_fps: Target FPS for processing
        start_seconds: Start processing at this time (0 = from beginning)
        end_seconds: Stop processing at this time (0 = process to end)
        blue_robot_1, blue_robot_2, blue_robot_3: Blue alliance team numbers
        red_robot_1, red_robot_2, red_robot_3: Red alliance team numbers
        enable_robot_detection: Whether to detect robots
        enable_fuel_detection: Whether to detect yellow fuel balls
        progress: Gradio progress tracker
        
    Returns:
        Tuple of (blue_output_path, red_output_path, center_output_path, map_video_path, ...)
    """

    
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
    
    results = {}
    
    # Process videos sequentially to allow real-time progress updates
    if blue_video_path and enable_blue_camera:
        progress(0, desc="Starting Blue Camera processing...")
        try:
            output_path, robot_tracks, tracks_by_frame, width, height, robot_stats, ferry_counts, disabled_statuses, shot_events, side_visible_robots = process_single_video(
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
                enable_person_detection=enable_person_detection
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
                'side_visible_robots': side_visible_robots
            }
        except Exception as e:
            import traceback
            print(f"Error processing blue camera: {e}")
            print(traceback.format_exc())
            raise gr.Error(f"Error processing blue camera: {e}")
    
    if red_video_path and enable_red_camera:
        progress(0.5, desc="Starting Red Camera processing...")
        try:
            output_path, robot_tracks, tracks_by_frame, width, height, robot_stats, ferry_counts, disabled_statuses, shot_events, side_visible_robots = process_single_video(
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
                enable_person_detection=enable_person_detection
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
                'side_visible_robots': side_visible_robots
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
            output_path, robot_tracks, tracks_by_frame, width, height, robot_stats, ferry_counts, disabled_statuses, shot_events, _ = process_single_video(
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
                show_unlabeled_robots=show_unlabeled_robots
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
                'shot_events': shot_events
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
    
    progress(1.0, desc="All processing complete!")
    
    # Merge robot stats from all cameras using shot event deduplication
    # Collect shot events from all cameras
    all_shot_events = []  # List of (elapsed_seconds, robot_label, made_bool)
    for camera in ['blue', 'red', 'center']:
        camera_events = results.get(camera, {}).get('shot_events', [])
        all_shot_events.extend(camera_events)
    
    # Deduplicate shots: group by robot, sort by time, merge events within 3-second window
    DEDUP_WINDOW_SECONDS = 3.0
    
    # Group events by robot
    events_by_robot = {}
    for elapsed, robot_label, made in all_shot_events:
        if robot_label not in events_by_robot:
            events_by_robot[robot_label] = []
        events_by_robot[robot_label].append((elapsed, made))
    
    # Deduplicate per robot and rebuild stats
    merged_stats = {}
    for robot_label, events in events_by_robot.items():
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
        total = f"**{robot_data['made']}/{robot_data['attempts']} shots made**"
        
        # Add ferry count if any
        if ferry_count > 0:
            total += f" | **Ferried: {ferry_count}x**"
        
        # Build period table
        rows = ["| Period | Made | Missed |", "|--------|------|--------|"]
        for period_name, _, _ in MATCH_PERIODS:
            p = robot_data['by_period'].get(period_name, {'attempts': 0, 'made': 0})
            missed = p['attempts'] - p['made']
            if p['attempts'] > 0:
                rows.append(f"| {period_name} | {p['made']} | {missed} |")
        
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


def create_demo():
    """Create and return the Gradio interface."""
    
    with gr.Blocks(title="Robot Scouter") as demo:
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("<div class='panel-title'>Input</div>", elem_classes="input-panel")
                
                composite_video_input = gr.Video(
                    label="Match Video (1920×1080 — auto-splits into 3 cameras)",
                    sources=["upload"],
                )
                
                # Hidden placeholders for individual camera paths (not shown in UI)
                blue_video_input = gr.State(None)
                center_video_input = gr.State(None)
                red_video_input = gr.State(None)
                
                # --- Center Camera Calibration ---
                gr.Markdown("### Center Camera Calibration")
                gr.Markdown(
                    "After the 8 field points, you can optionally click 2 rotated no-scan boxes: "
                    "BZ1->BZ4 for the blue-side box, then RZ1->RZ4 for the red-side box."
                )
                gr.Markdown("Click the 8 field landmarks in order (B1→B4, R1→R4) on the frame below.")
                
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

                # Robot number inputs
                gr.Markdown("### Blue Alliance")
                with gr.Row():
                    blue_robot_1 = gr.Textbox(
                        label="Robot 1",
                        value="1768",
                        placeholder="e.g., 1919",
                        max_lines=1
                    )
                    blue_robot_2 = gr.Textbox(
                        label="Robot 2",
                        value="4909",
                        placeholder="e.g., 334",
                        max_lines=1
                    )
                    blue_robot_3 = gr.Textbox(
                        label="Robot 3",
                        value="5962",
                        placeholder="e.g., 254",
                        max_lines=1
                    )
                
                gr.Markdown("### Red Alliance")
                with gr.Row():
                    red_robot_1 = gr.Textbox(
                        label="Robot 1",
                        value="2342",
                        placeholder="e.g., 118",
                        max_lines=1
                    )
                    red_robot_2 = gr.Textbox(
                        label="Robot 2",
                        value="6328",
                        placeholder="e.g., 973",
                        max_lines=1
                    )
                    red_robot_3 = gr.Textbox(
                        label="Robot 3",
                        value="2877",
                        placeholder="e.g., 2056",
                        max_lines=1
                    )
                
                with gr.Row():
                    fps_slider = gr.Slider(
                        minimum=1,
                        maximum=30,
                        value=8,
                        step=1,
                        label="Processing FPS",
                        info="Higher FPS = more API calls & slower processing"
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
        
        # --- Calibration Event Wiring ---
        
        def handle_video_upload(video_path, start_seconds):
            """Extract calibration frame when video is uploaded."""
            if video_path is None:
                return None, None, [], None, "*Upload a video to begin calibration*"
            frame = CenterCameraCalibrator.extract_calibration_frame(video_path, start_seconds or 0)
            if frame is None:
                return None, None, [], None, "Failed to extract frame from video"
            img_size = frame.size  # (width, height)
            return frame, frame, [], img_size, _get_calibration_status_text(0)
        
        composite_video_input.change(
            fn=handle_video_upload,
            inputs=[composite_video_input, start_seconds_input],
            outputs=[calibration_image, calibration_base_image, calibration_points_state, calibration_image_size_state, calibration_status]
        )
        
        def handle_image_click(base_image, clicked_points, evt: gr.SelectData):
            if base_image is None:
                return None, clicked_points, "Upload a video first"
            x, y = evt.index
            n = len(clicked_points)
            if n >= CALIBRATION_TOTAL_POINTS:
                annotated = _redraw_calibration_image(base_image, clicked_points)
                return annotated, clicked_points, _get_calibration_status_text(n)
            label = ALL_CALIBRATION_POINT_LABELS[n] if n < len(ALL_CALIBRATION_POINT_LABELS) else f"P{n}"
            img_w, img_h = base_image.size
            print(f"[Calibration UI] Click #{n+1} ({label}): raw=({x}, {y}), base_image_size=({img_w}x{img_h})")
            clicked_points = list(clicked_points) + [(x, y)]
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
            inputs=[calibration_base_image, calibration_points_state],
            outputs=[calibration_image, calibration_points_state, calibration_status]
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
        
        def handle_skip():
            return [], "**Calibration skipped** — will use default alignment"
        
        skip_calib_btn.click(
            fn=handle_skip,
            inputs=[],
            outputs=[calibration_points_state, calibration_status]
        )
        
        # Connect the processing function
        process_btn.click(
            fn=process_dual_videos,
            inputs=[blue_video_input, red_video_input, center_video_input, composite_video_input, fps_slider, start_seconds_input, end_seconds_input, blue_robot_1, blue_robot_2, blue_robot_3, red_robot_1, red_robot_2, red_robot_3, detect_robots_checkbox, detect_fuel_checkbox, side_ref_image_input, center_ref_image_input, enable_blue_cam, enable_center_cam, enable_red_cam, detect_people_checkbox, calibration_points_state, calibration_image_size_state, show_unlabeled_checkbox],
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
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
