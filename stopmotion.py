#!/usr/bin/env python3
"""
Kid Stop Motion — a simple stop motion webcam app.

Controls:
  SPACE  capture frame
  P      preview animation
  E      export MP4
  Z      delete last frame
  S      save current project frames
  L      load saved project frames
  O      toggle onion skin (ghost of last frame)
  T      toggle auto-capture every 2 seconds
  N      set a frame target and show progress
  Q/ESC  quit

Assumes 12 fps.
"""

import cv2
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from datetime import datetime

# Most tabletop rigs end up with the camera clamped upside down, so the
# picture is rotated by default. Set this to False if yours comes out
# inverted.
ROTATE_180 = True

FPS = 12

WINDOW_NAME = "Kid Stop Motion - SPACE capture | P preview | E export | Z undo | O onion | T auto | N target | Q quit"


def has_ffmpeg():
    return shutil.which("ffmpeg") is not None

def make_export_name():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"stop_motion_export_{timestamp}.mp4"

def draw_ui(frame, captured_count, is_preview=False, onion_frame=None, frame_target=0):
    duration = captured_count / FPS if captured_count else 0

    overlay = frame.copy()
    h, w = frame.shape[:2]

    # Onion skin: blend the last captured frame as a ghost
    if onion_frame is not None and not is_preview:
        frame = cv2.addWeighted(frame, 0.5, onion_frame, 0.5, 0)

    cv2.rectangle(overlay, (0, 0), (w, 95), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

    status = "PREVIEWING" if is_preview else "LIVE CAMERA"
    cv2.putText(frame, status, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    info = f"Frames: {captured_count}   Duration: {duration:.1f}s   FPS: {FPS}"
    cv2.putText(frame, info, (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    controls = "SPACE=capture  P=preview  E=export  Z=undo  S=save  L=load  O=onion  T=auto  N=target  Q=quit"
    cv2.putText(frame, controls, (16, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)

    # Frame target progress bar
    if frame_target > 0:
        progress = min(captured_count / frame_target, 1.0)
        bar_y = 105
        bar_h = 18
        bar_w = w - 32
        bar_x = 16
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        fill_w = int(bar_w * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 200, 0), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)
        label = f"Goal: {captured_count}/{frame_target} frames"
        cv2.putText(frame, label, (bar_x, bar_y + bar_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return frame


def save_frame(frame, frames_dir, frame_number):
    filename = frames_dir / f"frame_{frame_number:05d}.png"
    cv2.imwrite(str(filename), frame)
    return filename


def export_mp4(frames_dir, output_path):
    if not has_ffmpeg():
        print("ffmpeg not found. Install it with: brew install ffmpeg")
        return False

    pattern = str(frames_dir / "frame_%05d.png")

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(FPS),
        "-i", pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-r", str(FPS),
        str(output_path),
    ]

    print("\nExporting MP4...")
    print(" ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
        print(f"\nExport complete: {output_path}")
        return True
    except subprocess.CalledProcessError:
        print("Export failed.")
        return False


def preview_frames(frame_files, frame_target=0):
    if not frame_files:
        print("No frames to preview yet.")
        return

    delay_ms = int(1000 / FPS)

    print("Previewing. Press any key to stop preview.")

    while True:
        for file in frame_files:
            img = cv2.imread(str(file))
            if img is None:
                continue

            img = draw_ui(img, len(frame_files), is_preview=True, frame_target=frame_target)
            cv2.imshow(WINDOW_NAME, img)

            key = cv2.waitKey(delay_ms) & 0xFF
            if key != 255:
                return


def save_project(frames_dir):
    if not any(frames_dir.glob("frame_*.png")):
        print("No frames to save.")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = Path.cwd() / f"stopmotion_project_{timestamp}"
    shutil.copytree(frames_dir, save_dir)
    print(f"Project saved to: {save_dir}")


def load_project(frames_dir):
    folders = sorted(Path.cwd().glob("stopmotion_project_*"))

    if not folders:
        print("No saved stopmotion_project_* folders found in this directory.")
        return []

    latest = folders[-1]
    print(f"Loading latest project: {latest}")

    for f in frames_dir.glob("frame_*.png"):
        f.unlink()

    loaded_files = sorted(latest.glob("frame_*.png"))

    for i, src in enumerate(loaded_files, start=1):
        dst = frames_dir / f"frame_{i:05d}.png"
        shutil.copy(src, dst)

    print(f"Loaded {len(loaded_files)} frames.")
    return sorted(frames_dir.glob("frame_*.png"))


def set_frame_target(current_count):
    """Prompt for a target frame count; blank / 'x' / 0 clears it.

    Returns the new target (int >= 1), 0 to clear, or None if unchanged.
    """
    print(f"\n  Current frames: {current_count}")
    raw = input("  Target frame count (blank to clear): ").strip()
    if raw.lower() in ("", "x", "0"):
        print("  Target cleared.")
        return 0
    try:
        n = int(raw)
        if n < 1:
            raise ValueError
    except ValueError:
        print("  Invalid number, target unchanged.")
        return None
    remaining = max(n - current_count, 0)
    print(f"  Target set to {n}.  {remaining} frame(s) to go.")
    return n


def main():
    if not has_ffmpeg():
        print("Warning: ffmpeg not found.")
        print("Install it with:")
        print("  brew install ffmpeg")
        print()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam.")
        print("On macOS, you may need to allow Terminal or your Python app to access the camera:")
        print("System Settings → Privacy & Security → Camera")
        return

    # Try to set a friendly webcam resolution.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with tempfile.TemporaryDirectory() as tmp:
        frames_dir = Path(tmp)
        frame_files = []
        onion_skin = False
        last_frame = None
        auto_capture = False
        next_capture_time = 0.0
        frame_target = 0

        print("\nStop Motion Camera Ready")
        print("SPACE = capture frame")
        print("P     = preview")
        print("E     = export MP4")
        print("Z     = delete last frame")
        print("S     = save project")
        print("L     = load latest saved project")
        print("O     = toggle onion skin")
        print("T     = toggle auto-capture (2s)")
        print("N     = set frame target")
        print("Q     = quit\n")

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Could not read from webcam.")
                break

            if ROTATE_180:
                frame = cv2.flip(frame, -1)

            display = draw_ui(frame, len(frame_files), onion_frame=last_frame if onion_skin else None, frame_target=frame_target)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" ") or key == 32:
                frame_number = len(frame_files) + 1
                saved = save_frame(frame, frames_dir, frame_number)
                frame_files.append(saved)
                last_frame = frame.copy()
                target_str = f" / {frame_target}" if frame_target > 0 else ""
                print(f"Captured frame {frame_number}{target_str}")

            elif key == ord("z"):
                if frame_files:
                    last = frame_files.pop()
                    last.unlink(missing_ok=True)
                    print(f"Deleted last frame. Frames left: {len(frame_files)}")
                else:
                    print("No frames to delete.")

            elif key == ord("p"):
                preview_frames(frame_files, frame_target=frame_target)

            elif key == ord("e"):
                if not frame_files:
                    print("No frames to export.")
                else:
                    output_name = make_export_name()
                    output_path = Path.cwd() / output_name
                    export_mp4(frames_dir, output_path)

            elif key == ord("s"):
                save_project(frames_dir)

            elif key == ord("l"):
                frame_files = load_project(frames_dir)

            elif key == ord("o"):
                onion_skin = not onion_skin
                if onion_skin:
                    last_frame = frame.copy() if frame_files else None
                    print("Onion skin ON (ghost of last frame)")
                else:
                    last_frame = None
                    print("Onion skin OFF")

            elif key == ord("t"):
                auto_capture = not auto_capture
                if auto_capture:
                    next_capture_time = time.time() + 2.0
                    print("Auto-capture ON (every 2s)")
                else:
                    print("Auto-capture OFF")

            elif key == ord("n"):
                new_target = set_frame_target(len(frame_files))
                if new_target is not None:
                    frame_target = new_target

            elif key == ord("q") or key == 27:
                break

            # Auto-capture: fire a frame every 2 seconds when enabled
            if auto_capture and time.time() >= next_capture_time:
                frame_number = len(frame_files) + 1
                saved = save_frame(frame, frames_dir, frame_number)
                frame_files.append(saved)
                last_frame = frame.copy()
                target_str = f" / {frame_target}" if frame_target > 0 else ""
                print(f"Auto-captured frame {frame_number}{target_str}")
                next_capture_time = time.time() + 2.0

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()