#!/usr/bin/env python3
"""
Kid Stop Motion - a tiny stop motion studio for kids, using any webcam.

Every picture is saved the moment it's taken, into a project folder
(~/Stop Motion Projects/<Movie name>/ by default) that holds a
<Movie name>.stopmo project file you can open again later.

Keys (the buttons on screen do the same things):
  SPACE   take a picture              P    play the movie (with sound)
  O       ghost of the last picture   T    auto-snap every 2 seconds
  C       add / edit a title card     F    transition before a picture
  R       record a voice track        V    speed (pictures per second)
  E       make an MP4 movie           L    projects: open one / new movie
  < >     pick a picture              X    delete it       D   copy it
  + -     show it longer / shorter    [ ]  move it         Z/Y undo / redo
  N       set a picture goal          S    save now        H   help
  K       camera panel: C switch camera, R flip, F / X / W lock focus,
          exposure, white balance, [ ] nudge focus, - + darker / brighter
  Q/ESC   quit (press twice)

Run:  python3 stopmotion.py [path/to/Movie.stopmo]
"""

import argparse
import ctypes
import ctypes.util
import errno
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
import wave
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ------------------------------------------------------------------ settings

# Set this to True if your camera is mounted upside down (tabletop rigs often
# end up that way). In the studio, K then R flips the picture for the camera
# you're using, and that's remembered.
ROTATE_180 = False

# Pictures per second for new movies. Each movie remembers its own speed,
# which you can change in the app with V.
FPS = 12

CAMERA_INDEX = 0             # which camera to start with (0 = the first); K then C switches
CAMERA_SIZE = (1280, 720)    # the resolution we ask the webcam for

# Each press of - or + in the camera panel changes exposure by this many stops.
# On cameras we can't control, the app brightens the picture itself, up to 2 stops.
EXPOSURE_STEP = 1 / 3
MAX_EXPOSURE_STEPS = 6

# Where new movies go. Each movie gets its own folder in here.
PROJECTS_DIR = Path.home() / "Stop Motion Projects"

ONION_OPACITY = 0.45         # how strong the ghost of the last picture is
AUTO_CAPTURE_SECONDS = 2.0
TITLE_SECONDS = 2.0          # how long a new title card stays on screen
TRANSITION_SECONDS = 0.75
SHUTTER_SOUND = True         # a little click when a picture is taken
WINDOW_SCALE = 1.0           # try 0.8 if the window is too big for your screen
WRITE_LAUNCHERS = True       # put an "Open <movie>" double-click file in each project folder

# ----------------------------------------------------------------- constants

PROJECT_EXT = ".stopmo"
PROJECT_FORMAT = "kid-stop-motion"
PROJECT_VERSION = 1
FRAMES_DIR = "frames"
AUDIO_DIR = "audio"
BACKUPS_DIR = "backups"
TRASH_DIR = "trash"
LOCK_FILE = ".in-use"
SETTINGS_FILE = ".settings.json"     # in the projects folder: which cameras are upside down
FRAME_EXT = ".jpg"
JPEG_QUALITY = 95
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
BACKUP_EVERY_SECONDS = 120
KEEP_BACKUPS = 30
KEEP_TRASH = 500
DEFAULT_SIZE = (1280, 720)
SPEEDS = [4, 6, 8, 10, 12, 15, 20, 24]
MAX_HOLD = 120
MAX_TITLE_LINES = 3
MAX_TITLE_CHARS = 32
MAX_RECORD_SECONDS = 600

WINDOW_NAME = "Kid Stop Motion"

# Transitions go *before* a picture (or at the very end of the movie).
TRANSITIONS = ["fade", "dissolve", "wipe", "circle", "slide"]
TRANSITION_NAMES = {
    "fade": "Fade", "dissolve": "Melt", "wipe": "Wipe", "circle": "Circle", "slide": "Slide",
}

TITLE_STYLES = [
    {"name": "Silent movie", "bg": (18, 18, 18), "fg": (240, 240, 240), "shadow": (75, 75, 75),
     "border": (225, 225, 225), "font": cv2.FONT_HERSHEY_TRIPLEX},
    {"name": "Sunshine", "bg": (60, 205, 255), "fg": (110, 30, 90), "shadow": (150, 235, 255),
     "border": (40, 150, 240), "font": cv2.FONT_HERSHEY_DUPLEX},
    {"name": "Ocean", "bg": (150, 90, 20), "fg": (255, 255, 255), "shadow": (95, 50, 0),
     "border": (230, 200, 120), "font": cv2.FONT_HERSHEY_DUPLEX},
    {"name": "Jungle", "bg": (50, 130, 40), "fg": (90, 235, 255), "shadow": (20, 70, 15),
     "border": (120, 200, 110), "font": cv2.FONT_HERSHEY_DUPLEX},
    {"name": "Bubblegum", "bg": (190, 120, 250), "fg": (255, 255, 255), "shadow": (140, 40, 170),
     "border": (255, 230, 250), "font": cv2.FONT_HERSHEY_DUPLEX},
    {"name": "Chalkboard", "bg": (55, 70, 45), "fg": (225, 235, 235), "shadow": (35, 45, 30),
     "border": (40, 90, 140), "font": cv2.FONT_HERSHEY_COMPLEX},
]


# ------------------------------------------------------------------- helpers

def has_ffmpeg():
    return shutil.which("ffmpeg") is not None


def new_id():
    return uuid.uuid4().hex[:10]


def now_stamp():
    return datetime.now().strftime("%Y-%m-%d %H-%M-%S")


def ascii_text(text):
    """OpenCV's fonts only draw plain ASCII."""
    return str(text).encode("ascii", "replace").decode("ascii")


def safe_filename(name):
    cleaned = "".join(c if c.isalnum() or c in " -_'!&(),." else "_" for c in str(name))
    cleaned = " ".join(cleaned.split()).strip(" .")
    return cleaned[:60].strip(" .") or "Movie"


def atomic_write(path, data):
    """Write a file so it's either the old version or the new one, never half."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def even_size(w, h):
    return max(2, int(w) // 2 * 2), max(2, int(h) // 2 * 2)


def plural(n, word):
    return f"{n} {word}" + ("" if n == 1 else "s")


def clamp(value, low, high):
    return max(low, min(high, value))


def fit_image(img, size, color=(0, 0, 0)):
    """Scale img to fit inside size (w, h), adding bars if the shapes differ."""
    w, h = size
    ih, iw = img.shape[:2]
    if (iw, ih) == (w, h):
        return img
    scale = min(w / iw, h / ih)
    nw, nh = max(1, round(iw * scale)), max(1, round(ih * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    if (nw, nh) == (w, h):
        return resized
    out = np.full((h, w, 3), color, np.uint8)
    x, y = (w - nw) // 2, (h - nh) // 2
    out[y:y + nh, x:x + nw] = resized
    return out


def frame_item(rel_file):
    return {"id": new_id(), "kind": "frame", "file": rel_file, "hold": 1, "transition": None}


def title_item(lines, style=0, hold=None, fps=FPS):
    if hold is None:
        hold = max(1, round(TITLE_SECONDS * fps))
    return {"id": new_id(), "kind": "title", "lines": list(lines), "style": style,
            "hold": hold, "transition": None}


def item_key(item):
    """Identifies what an item looks like (frame files never change once written)."""
    if item["kind"] == "frame":
        return ("frame", item["file"])
    return ("title", tuple(item.get("lines", ())), item.get("style", 0))


def make_transition(kind, fps):
    return {"type": kind, "frames": max(2, round(TRANSITION_SECONDS * fps))}


def next_transition(current, fps):
    order = [None] + TRANSITIONS
    kind = current.get("type") if current else None
    nxt = order[(order.index(kind) + 1) % len(order)] if kind in order else None
    return make_transition(nxt, fps) if nxt else None


def valid_transition(tr):
    return (isinstance(tr, dict) and tr.get("type") in TRANSITIONS
            and isinstance(tr.get("frames"), int) and 1 <= tr["frames"] <= 240)


def backup_time(path):
    """When a backup was made, from its name ("Movie 2026-09-29 13-14-05.stopmo")."""
    found = re.search(r"\d{4}-\d\d-\d\d \d\d-\d\d-\d\d", Path(path).stem)
    return found.group(0) if found else Path(path).stem


def file_number(path):
    stem = Path(path).stem.split(" ")[0]
    return int(stem) if stem.isdigit() else 0


# ------------------------------------------------------------------- project

class ProjectError(Exception):
    pass


def new_document(name, fps=FPS):
    stamp = datetime.now().isoformat(timespec="seconds")
    return {
        "format": PROJECT_FORMAT,
        "version": PROJECT_VERSION,
        "name": name,
        "fps": fps,
        "size": list(DEFAULT_SIZE),
        "created": stamp,
        "modified": stamp,
        "next_frame": 1,
        "onion_skin": True,
        "frame_goal": 0,
        "items": [],
        "end_transition": None,
        "audio": None,
    }


class Project:
    """A movie on disk.

        <folder>/<name>.stopmo   the project file (JSON) - open this one
        <folder>/frames/         every picture, one file each
        <folder>/audio/          the voice track
        <folder>/backups/        older copies of the project file
        <folder>/trash/          pictures you deleted (kept, just in case)

    Every change is written straight away and atomically, so a crash or a
    flat battery never loses more than the picture being taken.
    """

    EDIT_KEYS = ("items", "end_transition", "audio", "fps", "size")

    def __init__(self, path, doc):
        self.path = Path(path)
        self.doc = doc
        self.undo_stack = []
        self.redo_stack = []
        self.notes = []            # things to tell the user after opening
        self._last_backup = float("-inf")
        self._lock_file = None

    @property
    def folder(self):
        return self.path.parent

    @property
    def name(self):
        # The file's name, so renaming it in Finder / Explorer renames the movie.
        return self.path.stem

    @property
    def items(self):
        return self.doc["items"]

    @property
    def fps(self):
        return self.doc["fps"]

    @property
    def size(self):
        return tuple(self.doc["size"])

    def file_path(self, rel):
        return self.folder / rel

    def frame_count(self):
        return sum(1 for it in self.items if it["kind"] == "frame")

    def audio_file(self):
        audio = self.doc.get("audio")
        if not audio:
            return None
        path = self.file_path(audio["file"])
        return path if path.is_file() else None

    # -- creating and opening

    @classmethod
    def create(cls, name, parent=None):
        parent = Path(parent or PROJECTS_DIR).expanduser()
        base = safe_filename(name)
        folder, n = parent / base, 2
        while folder.exists():
            folder, n = parent / f"{base} ({n})", n + 1
        (folder / FRAMES_DIR).mkdir(parents=True)
        project = cls(folder / (folder.name + PROJECT_EXT), new_document(folder.name))
        project.save()
        return project

    @classmethod
    def open(cls, path, projects_dir=None, lock=False):
        """Open a .stopmo file - or a project folder, or any file inside one.

        With lock=True the project is locked (see lock()) before anything is
        written, and ProjectError is raised if another window has it open."""
        path = Path(path).expanduser().absolute()
        if path.is_file() and path.suffix.lower() != PROJECT_EXT:
            for folder in (path.parent, path.parent.parent):
                if any(folder.glob("*" + PROJECT_EXT)):
                    path = folder
                    break
        if path.parent.name == BACKUPS_DIR and (path.parent.parent / FRAMES_DIR).is_dir():
            return cls._restore_backup(path, lock)
        if path.suffix.lower() == PROJECT_EXT and not path.exists() and path.parent.is_dir():
            path = path.parent          # e.g. it was renamed: open the project file that's there now
        if path.is_dir():
            found = sorted(path.glob("*" + PROJECT_EXT), key=lambda p: p.stat().st_mtime, reverse=True)
            if found:
                path = found[0]
            elif list(path.glob("frame_*.png")):
                project = cls.import_legacy(path, projects_dir)
                if lock:
                    project.lock()        # a brand new folder: nobody else can have it
                return project
            elif (path / FRAMES_DIR).is_dir():
                path = path / (path.name + PROJECT_EXT)
            else:
                raise ProjectError(f"No stop motion project in {path}")
        elif path.suffix.lower() != PROJECT_EXT:
            raise ProjectError(f"{path.name} is not a {PROJECT_EXT} project file")
        elif not path.exists() and not (path.parent / FRAMES_DIR).is_dir():
            raise ProjectError(f"Can't find {path}")

        project = cls(path, None)
        if lock and not project.lock():
            raise ProjectError("That movie is already open in another window.")
        doc, source = cls._read_newest_good(path)
        project.doc = doc or new_document(path.stem)
        changed = False
        if source != path and path.exists():
            # Keep the damaged file for the grown-ups, then carry on.
            shutil.copy2(path, path.with_name(f"{path.name}.damaged {now_stamp()}"))
        if doc is None:
            project._rebuild_from_frames()
            project.notes.append("The project file was missing or damaged, so it was rebuilt from the pictures.")
            changed = True
        elif source != path:
            project.notes.append(f"The project file was damaged, so its backup ({backup_time(source)}) was used.")
            project._recover_orphans()
            changed = True
        changed = project._repair() or changed
        if changed:
            project.save()
        return project

    @classmethod
    def _restore_backup(cls, backup, lock=False):
        """Opening a file from backups/ goes back to it - as an ordinary change,
        so Z (undo) returns to how things were."""
        doc = cls.load_doc(backup)
        if doc is None:
            raise ProjectError(f"{backup.name} is damaged")
        project = cls.open(backup.parent.parent, lock=lock)
        with project.change():
            project.doc.update({k: doc.get(k, project.doc.get(k)) for k in cls.EDIT_KEYS})
            project._repair()
        project.notes.append(f"Went back to the backup from {backup_time(backup)}. Press Z to undo that.")
        return project

    @classmethod
    def import_legacy(cls, folder, parent=None):
        """Turn an old stopmotion_project_* folder of frame_*.png files into a project."""
        folder = Path(folder).absolute()
        files = sorted(folder.glob("frame_*.png"))
        if not files:
            raise ProjectError(f"No frame_*.png pictures in {folder}")
        label = folder.name.replace("stopmotion_project_", "").replace("_", " ")
        project = cls.create(f"Old save {label}", parent)
        for src in files:
            rel = project._next_frame_rel(src.suffix.lower())
            shutil.copy2(src, project.file_path(rel))
            project.items.append(frame_item(rel))
        project._set_size_from(project.file_path(project.items[0]["file"]))
        project.doc["imported_from"] = str(folder)
        project.save()
        return project

    @staticmethod
    def load_doc(path):
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if isinstance(doc, dict) and doc.get("format") == PROJECT_FORMAT and isinstance(doc.get("items"), list):
            return doc
        return None

    @classmethod
    def _read_newest_good(cls, path):
        backups = sorted((path.parent / BACKUPS_DIR).glob("*" + PROJECT_EXT),
                         key=lambda p: p.stat().st_mtime, reverse=True)
        for candidate in [path] + backups:
            doc = cls.load_doc(candidate)
            if doc is not None:
                return doc, candidate
        return None, None

    def _picture_files(self, sub=FRAMES_DIR):
        folder = self.folder / sub
        if not folder.is_dir():
            return []
        return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

    def _set_size_from(self, image_path):
        img = cv2.imread(str(image_path))
        if img is not None:
            self.doc["size"] = list(even_size(img.shape[1], img.shape[0]))

    def _rebuild_from_frames(self):
        files = self._picture_files()
        self.doc["items"] = [frame_item(f"{FRAMES_DIR}/{p.name}") for p in files]
        if files:
            self._set_size_from(files[0])

    def _recover_orphans(self):
        """After falling back to a backup, add back pictures taken since it."""
        used = {it.get("file") for it in self.items if isinstance(it, dict)}
        newest = max((file_number(f) for f in used if f), default=0)
        orphans = [p for p in self._picture_files()
                   if f"{FRAMES_DIR}/{p.name}" not in used and file_number(p) > newest]
        for p in orphans:
            self.items.append(frame_item(f"{FRAMES_DIR}/{p.name}"))
        if orphans:
            self.notes.append(f"{len(orphans)} picture(s) taken after that backup were put back.")

    def _ensure_file(self, rel):
        """True if rel exists - fishing it back out of trash/ if need be."""
        path = self.file_path(rel)
        if path.is_file():
            return True
        trashed = self.folder / TRASH_DIR / Path(rel).name
        if trashed.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(trashed), str(path))
            return True
        return False

    def _repair(self):
        """Fix anything odd in a loaded project. Returns True if it changed."""
        doc = self.doc
        before = json.dumps(doc, sort_keys=True)
        for key, value in new_document(self.path.stem).items():
            doc.setdefault(key, value)
        if not isinstance(doc["fps"], int) or not 1 <= doc["fps"] <= 60:
            doc["fps"] = FPS
        size = doc["size"]
        if not (isinstance(size, list) and len(size) == 2 and all(isinstance(v, int) and v >= 16 for v in size)):
            doc["size"] = list(DEFAULT_SIZE)
        good, seen, missing = [], set(), 0
        for item in doc["items"]:
            if not isinstance(item, dict):
                continue
            kind = item.get("kind")
            if kind == "frame":
                if not isinstance(item.get("file"), str):
                    continue
                if not self._ensure_file(item["file"]):
                    missing += 1          # kept (shown as a grey card) in case the file turns up again
                default_hold = 1
            elif kind == "title":
                lines = item.get("lines")
                if not isinstance(lines, list):
                    lines = [str(lines or "")]
                item["lines"] = [str(line)[:MAX_TITLE_CHARS] for line in lines][:MAX_TITLE_LINES]
                if not isinstance(item.get("style"), int):
                    item["style"] = 0
                default_hold = max(1, round(TITLE_SECONDS * doc["fps"]))
            else:
                continue
            if not isinstance(item.get("id"), str) or item["id"] in seen:
                item["id"] = new_id()
            seen.add(item["id"])
            if not isinstance(item.get("hold"), int) or item["hold"] < 1:
                item["hold"] = default_hold
            if item.get("transition") is not None and not valid_transition(item["transition"]):
                item["transition"] = None
            good.append(item)
        doc["items"] = good
        if doc["end_transition"] is not None and not valid_transition(doc["end_transition"]):
            doc["end_transition"] = None
        audio = doc.get("audio")
        if audio is not None and not (isinstance(audio, dict) and isinstance(audio.get("file"), str)
                                      and self._ensure_file(audio["file"])):
            doc["audio"] = None
            self.notes.append("The voice track file is missing.")
        highest = max((file_number(p) for p in self._picture_files() + self._picture_files(TRASH_DIR)), default=0)
        if not isinstance(doc["next_frame"], int) or doc["next_frame"] <= highest:
            doc["next_frame"] = highest + 1
        if not isinstance(doc["frame_goal"], int) or doc["frame_goal"] < 0:
            doc["frame_goal"] = 0
        doc["onion_skin"] = bool(doc["onion_skin"])
        if missing:
            self.notes.append(f"{plural(missing, 'picture')} can't be found (shown as grey cards).")
        return json.dumps(doc, sort_keys=True) != before

    # -- saving

    def save(self, backup=False):
        self.doc["modified"] = datetime.now().isoformat(timespec="seconds")
        data = json.dumps(self.doc, indent=1).encode("utf-8")
        if self.path.exists() and (backup or time.monotonic() - self._last_backup >= BACKUP_EVERY_SECONDS):
            try:
                self._backup()
            except OSError as e:
                self._last_backup = time.monotonic()
                print(f"Couldn't make a backup copy ({e}); saving anyway.")
        atomic_write(self.path, data)

    def _backup(self):
        folder = self.folder / BACKUPS_DIR
        folder.mkdir(exist_ok=True)
        name, n = f"{self.path.stem} {now_stamp()}", 2
        while (folder / (name + PROJECT_EXT)).exists():
            name, n = f"{self.path.stem} {now_stamp()} ({n})", n + 1
        shutil.copy2(self.path, folder / (name + PROJECT_EXT))
        self._last_backup = time.monotonic()
        old = sorted(folder.glob("*" + PROJECT_EXT), key=lambda p: p.stat().st_mtime)
        for extra in old[:-KEEP_BACKUPS]:
            extra.unlink(missing_ok=True)

    def set_option(self, key, value):
        """Settings that aren't edits (ghost on/off, goal): saved, not undoable."""
        self.doc[key] = value
        self.save()

    def tidy(self):
        """Move pictures and sounds nothing uses any more into trash/."""
        used = set()
        docs = [self.doc] + [Project.load_doc(p) for p in self.folder.glob("*" + PROJECT_EXT) if p != self.path]
        for doc in docs:              # a copy or renamed project file may share this folder
            if not doc:
                continue
            used |= {it.get("file") for it in doc["items"] if isinstance(it, dict) and it.get("kind") == "frame"}
            if isinstance(doc.get("audio"), dict):
                used.add(doc["audio"].get("file"))
        trash = self.folder / TRASH_DIR
        moved = 0
        for sub in (FRAMES_DIR, AUDIO_DIR):
            folder = self.folder / sub
            if not folder.is_dir():
                continue
            for path in folder.iterdir():
                if not path.is_file() or f"{sub}/{path.name}" in used:
                    continue
                if path.name.endswith(".tmp"):
                    path.unlink(missing_ok=True)     # half-written leftovers from a crash
                    continue
                trash.mkdir(exist_ok=True)
                dest = trash / path.name
                if dest.exists():
                    dest = trash / f"{path.stem} {new_id()[:4]}{path.suffix}"
                shutil.move(str(path), str(dest))
                os.utime(dest)            # so the trash forgets the oldest *deleted* files first
                moved += 1
        if trash.is_dir():
            old = sorted((p for p in trash.iterdir() if p.is_file()), key=lambda p: p.stat().st_mtime)
            for extra in old[:-KEEP_TRASH]:
                extra.unlink(missing_ok=True)
        return moved

    # -- editing (wrap changes in `with project.change():` so they can be undone)

    def edit_state(self):
        return json.dumps({k: self.doc.get(k) for k in self.EDIT_KEYS})

    @contextmanager
    def change(self, ui=None):
        before = self.edit_state()
        yield
        if self.edit_state() != before:
            self.undo_stack.append((before, ui))
            del self.undo_stack[:-100]
            self.redo_stack.clear()
            self.save()

    def undo(self, ui=None):
        return self._swap(self.undo_stack, self.redo_stack, ui)

    def redo(self, ui=None):
        return self._swap(self.redo_stack, self.undo_stack, ui)

    def _swap(self, source, dest, ui):
        if not source:
            return False, None
        state, old_ui = source.pop()
        dest.append((self.edit_state(), ui))
        self.doc.update(json.loads(state))
        self.save()
        return True, old_ui

    def _next_frame_rel(self, ext=FRAME_EXT):
        n = self.doc["next_frame"]
        while any((self.folder / sub / f"{n:06d}{ext}").exists() for sub in (FRAMES_DIR, TRASH_DIR)):
            n += 1                        # never write over a picture that's already there
        self.doc["next_frame"] = n + 1
        return f"{FRAMES_DIR}/{n:06d}{ext}"

    def lock(self):
        """Stop one movie being open in two windows at once (they would write over
        each other). The computer lets go of the lock by itself if we crash.
        Returns False if another window has it."""
        try:
            f = open(self.folder / LOCK_FILE, "a+")
        except OSError:
            return True                   # e.g. a read-only folder: carry on without a lock
        try:
            if os.name == "nt":
                import msvcrt
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            if isinstance(e, (BlockingIOError, PermissionError)) or e.errno in (errno.EACCES, errno.EAGAIN,
                                                                               errno.EDEADLK):
                f.close()
                return False
        self._lock_file = f
        return True

    def unlock(self):
        if self._lock_file is not None:
            self._lock_file.close()
            self._lock_file = None

    def add_frame(self, image, index):
        ok, buf = cv2.imencode(FRAME_EXT, image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise OSError("could not encode the picture")
        rel = self._next_frame_rel()
        atomic_write(self.file_path(rel), buf.tobytes())
        if self.frame_count() == 0:
            self.doc["size"] = list(even_size(image.shape[1], image.shape[0]))
        item = frame_item(rel)
        self.items.insert(index, item)
        return item

    def set_audio(self, samples, samplerate):
        rel = f"{AUDIO_DIR}/voice {now_stamp()}.wav"
        write_wav(self.file_path(rel), samples, samplerate)
        self.doc["audio"] = {"file": rel}


def list_projects(folder):
    """Summaries of the projects in a folder, newest first."""
    folder = Path(folder)
    found = []
    if not folder.is_dir():
        return found
    for sub in folder.iterdir():
        if not sub.is_dir():
            continue
        files = sorted(sub.glob("*" + PROJECT_EXT), key=lambda p: p.stat().st_mtime, reverse=True)
        if files:
            path = files[0]
        elif (sub / FRAMES_DIR).is_dir():
            path = sub              # opening the folder rebuilds the project file
        else:
            continue
        info = {"path": path, "name": sub.name, "frames": 0, "seconds": 0.0, "thumb": None,
                "modified": path.stat().st_mtime, "imported_from": None, "damaged": False}
        doc = Project.load_doc(path) if path.is_file() else None
        if doc is None:
            info["damaged"] = True
        else:
            try:
                project = Project(path, doc)
                info["name"] = project.name
                info["frames"] = project.frame_count()
                info["seconds"] = len(build_plan(project)) / project.fps
                info["imported_from"] = doc.get("imported_from")
                first = next((it for it in project.items if it["kind"] == "frame"), None)
                info["thumb"] = str(project.file_path(first["file"])) if first else None
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                info["damaged"] = True
        found.append(info)
    found.sort(key=lambda info: info["modified"], reverse=True)
    return found


def default_movie_name(folder):
    n = 1
    while (Path(folder) / f"Movie {n}").exists():
        n += 1
    return f"Movie {n}"


# ------------------------------------------------ movie: titles, transitions

def build_plan(project):
    """Every frame of the finished movie, in order.

    Each step is ("item", index) or ("mix", kind, a, b, t), a transition
    frame part-way (t) from item a to item b, where None means black.
    """
    items = project.items
    plan = []
    for i, item in enumerate(items):
        tr = item.get("transition")
        if tr:
            n = tr["frames"]
            prev = i - 1 if i > 0 else None
            for k in range(n):
                t = k / n if prev is None else (k + 1) / (n + 1)
                plan.append(("mix", tr["type"], prev, i, t))
        plan.extend([("item", i)] * item["hold"])
    end = project.doc.get("end_transition")
    if end and items:
        n = end["frames"]
        for k in range(n):
            plan.append(("mix", end["type"], len(items) - 1, None, (k + 1) / n))
    return plan


def plan_item(step):
    """The timeline item a movie frame belongs to."""
    if step[0] == "item":
        return step[1]
    return step[3] if step[3] is not None else step[2]


def smooth(t):
    return t * t * (3 - 2 * t)


def mix_images(kind, a, b, t):
    t = clamp(t, 0.0, 1.0)
    h, w = a.shape[:2]
    if kind == "fade":            # dip to black and back
        if t < 0.5:
            return cv2.convertScaleAbs(a, alpha=1 - 2 * t)
        return cv2.convertScaleAbs(b, alpha=2 * t - 1)
    if kind == "wipe":            # the new picture sweeps in from the left
        soft = max(1, w // 12)
        edge = smooth(t) * (w + soft)
        alpha = np.clip((edge - np.arange(w, dtype=np.float32)) / soft, 0, 1).reshape(1, w, 1)
        return (a * (1 - alpha) + b * alpha).astype(np.uint8)
    if kind == "circle":          # old cartoon iris: close on one, open on the next
        big = np.hypot(w, h) / 2 + 2
        img, r = (a, big * (1 - smooth(t * 2))) if t < 0.5 else (b, big * smooth(t * 2 - 1))
        mask = np.zeros((h, w, 3), np.uint8)
        if r >= 1:
            cv2.circle(mask, (w // 2, h // 2), int(r), (255, 255, 255), -1, cv2.LINE_AA)
        return cv2.multiply(img, mask, scale=1 / 255)
    if kind == "slide":           # the new picture pushes the old one off
        off = int(round(smooth(t) * w))
        out = np.empty_like(a)
        out[:, :w - off] = a[:, off:]
        out[:, w - off:] = b[:, :off]
        return out
    return cv2.addWeighted(a, 1 - t, b, t, 0)     # "dissolve": melt one into the other


def render_title(item, size):
    w, h = size
    style = TITLE_STYLES[item.get("style", 0) % len(TITLE_STYLES)]
    img = np.full((h, w, 3), style["bg"], np.uint8)
    u = min(w, h)
    aa = cv2.LINE_AA
    if style["name"] == "Silent movie":
        m = int(u * 0.05)
        cv2.rectangle(img, (m, m), (w - m, h - m), style["border"], max(2, u // 110), aa)
        m2 = m + int(u * 0.025)
        cv2.rectangle(img, (m2, m2), (w - m2, h - m2), style["border"], max(1, u // 300), aa)
    elif style["name"] == "Chalkboard":
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), style["border"], max(4, int(u * 0.07)))
    else:
        m = int(u * 0.04)
        round_rect(img, m, m, w - m, h - m, style["border"], int(u * 0.06), max(2, u // 90))

    lines = [ascii_text(line) for line in item.get("lines", [])]
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return img
    font = style["font"]
    max_w = w * 0.84
    laid = []
    for i, text in enumerate(lines):
        target = h * (0.16 if i == 0 else 0.09)
        (tw, th), _ = cv2.getTextSize(text or " ", font, 1.0, 2)
        scale = target / max(th, 1)
        if tw * scale > max_w:
            scale = max_w / tw
        thick = max(1, int(round(scale * 1.7)))
        (tw, th), base = cv2.getTextSize(text or " ", font, scale, thick)
        laid.append((text, scale, thick, tw, th, base))
    gap = h * 0.05
    total = sum(th + base for _, _, _, _, th, base in laid) + gap * (len(laid) - 1)
    y = (h - total) / 2
    for text, scale, thick, tw, th, base in laid:
        y += th
        x = (w - tw) // 2
        off = max(2, int(scale * 2.5))
        cv2.putText(img, text, (x + off, int(y) + off), font, scale, style["shadow"], thick, aa)
        cv2.putText(img, text, (x, int(y)), font, scale, style["fg"], thick, aa)
        y += base + gap
    return img


def missing_picture(size):
    w, h = size
    img = np.full((h, w, 3), (70, 70, 70), np.uint8)
    put_text(img, "missing picture", (w // 2, h // 2), max(0.4, w / 900), C_TEXT, 1, align="center", valign="middle")
    return img


class Renderer:
    """Turns timeline items into full-size pictures, with a small cache."""

    def __init__(self, project, size=None, cache_size=48):
        self.project = project
        self.fixed_size = tuple(size) if size else None
        self.cache = OrderedDict()
        self.cache_size = cache_size

    @property
    def size(self):
        return self.fixed_size or self.project.size

    def image(self, item):
        key = (item_key(item), self.size)
        img = self.cache.get(key)
        if img is None:
            img = self._make(item)
            self.cache[key] = img
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(key)
        return img

    def _make(self, item):
        if item["kind"] == "title":
            return render_title(item, self.size)
        img = cv2.imread(str(self.project.file_path(item["file"])))
        if img is None:
            return missing_picture(self.size)
        return fit_image(img, self.size)

    def render(self, step):
        items = self.project.items
        if step[0] == "item":
            return self.image(items[step[1]])
        _, kind, a, b, t = step
        w, h = self.size
        black = np.zeros((h, w, 3), np.uint8)
        img_a = self.image(items[a]) if a is not None else black
        img_b = self.image(items[b]) if b is not None else black
        if kind == "fade" and (a is None or b is None):
            kind = "dissolve"         # fading in from / out to black
        return mix_images(kind, img_a, img_b, t)


# --------------------------------------------------------------------- sound

_SOUND = {"tried": False, "module": None, "error": None}


def sound_module():
    """sounddevice is optional: without it everything but sound still works."""
    if not _SOUND["tried"]:
        _SOUND["tried"] = True
        try:
            import sounddevice
            _SOUND["module"] = sounddevice
        except ImportError:
            _SOUND["error"] = "To use the microphone, install sounddevice:  python3 -m pip install sounddevice"
        except Exception as e:    # usually OSError: the PortAudio library is missing
            _SOUND["error"] = f"Sound isn't working ({e}). On Linux try: sudo apt install libportaudio2"
    return _SOUND["module"]


def sound_problem():
    return None if sound_module() else _SOUND["error"]


def write_wav(path, samples, samplerate):
    samples = np.asarray(samples, dtype=np.int16)
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        with wave.open(f, "wb") as w:
            w.setnchannels(samples.shape[1])
            w.setsampwidth(2)
            w.setframerate(int(samplerate))
            w.writeframes(samples.tobytes())
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        channels, width, rate, count = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        data = w.readframes(count)
    if width != 2:
        raise ValueError("only 16-bit WAV files are supported")
    return np.frombuffer(data, dtype=np.int16).reshape(-1, channels), rate


class MicRecorder:
    def __init__(self):
        self.chunks = []
        self.level = 0.0
        self.stream = None
        self.samplerate = 0

    def start(self):
        sd = sound_module()
        if sd is None:
            raise RuntimeError(sound_problem())
        self.stream = sd.InputStream(channels=1, dtype="int16", callback=self._callback)
        self.samplerate = int(self.stream.samplerate)
        self.stream.start()

    def _callback(self, indata, frames, time_info, status):
        self.chunks.append(indata.copy())
        self.level = float(np.abs(indata).max()) / 32768.0

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if not self.chunks:
            return np.zeros((0, 1), np.int16), self.samplerate
        return np.concatenate(self.chunks), self.samplerate


class Speaker:
    def __init__(self):
        self._click = None

    def play(self, samples, samplerate):
        sd = sound_module()
        if sd is None:
            return False
        try:
            sd.play(samples, samplerate)
            return True
        except Exception as e:
            print(f"Could not play sound: {e}")
            return False

    def stop(self):
        if _SOUND["module"] is not None:
            try:
                _SOUND["module"].stop()
            except Exception:
                pass

    def click(self):
        if not SHUTTER_SOUND or sound_module() is None:
            return
        if self._click is None:
            rate = 44100
            t = np.arange(int(rate * 0.06)) / rate
            noise = np.random.default_rng(1).uniform(-1, 1, t.size) * np.exp(-t * 90) * 0.5
            tone = np.sin(2 * np.pi * 1800 * t) * np.exp(-t * 140) * 0.3
            self._click = ((noise + tone) * 20000).astype(np.int16), rate
        try:
            sound_module().play(*self._click)
        except Exception:
            pass


# -------------------------------------------------------------------- export

def export_movie(project, out_path, progress=None):
    """Write the finished movie. Returns (True, note) or (False, reason).

    progress(done, total) is called as frames are written; return False from
    it to cancel.
    """
    plan = build_plan(project)
    if not plan:
        return False, "Take some pictures first!"
    out_path = Path(out_path)
    size = even_size(*project.size)
    renderer = Renderer(project, size)
    fps = project.fps
    audio = project.audio_file()
    part = out_path.with_name(out_path.stem + ".partial" + out_path.suffix)
    note = ""

    def frames():
        for i, step in enumerate(plan):
            yield renderer.render(step)
            if progress is not None and progress(i + 1, len(plan)) is False:
                raise KeyboardInterrupt

    try:
        if has_ffmpeg():
            cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                   "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}",
                   "-framerate", str(fps), "-i", "-"]
            if audio:
                cmd += ["-i", str(audio), "-map", "0:v", "-map", "1:a",
                        "-c:a", "aac", "-b:a", "160k", "-af", "apad"]
            cmd += ["-t", f"{len(plan) / fps:.3f}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "18", "-r", str(fps), "-movflags", "+faststart", str(part)]
            with tempfile.TemporaryFile() as errors:
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
                try:
                    for img in frames():
                        proc.stdin.write(img.tobytes())
                except OSError as e:
                    # ffmpeg stopped early (Windows calls that EINVAL); its error message says why.
                    if not isinstance(e, BrokenPipeError) and e.errno != errno.EINVAL:
                        proc.kill()
                        raise
                except BaseException:
                    proc.kill()
                    raise
                finally:
                    try:
                        proc.stdin.close()
                    except OSError:
                        pass
                    returncode = proc.wait()
                if returncode != 0:
                    errors.seek(0)
                    detail = errors.read().decode("utf-8", "replace").strip().splitlines()
                    part.unlink(missing_ok=True)
                    return False, "ffmpeg couldn't make the movie: " + (detail[-1] if detail else "unknown error")
        else:
            for codec in ("avc1", "mp4v"):    # H.264 if this OpenCV can, otherwise MPEG-4
                writer = cv2.VideoWriter(str(part), cv2.VideoWriter_fourcc(*codec), fps, size)
                if writer.isOpened():
                    break
            else:
                return False, "Couldn't write a movie file. Installing ffmpeg usually fixes this."
            try:
                for img in frames():
                    writer.write(img)
            finally:
                writer.release()
            note = "No ffmpeg, so the movie has no sound." if audio else ""
    except KeyboardInterrupt:
        part.unlink(missing_ok=True)
        return False, "Stopped making the movie."
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    os.replace(part, out_path)
    return True, note


def reveal_file(path):
    """Show a file in Finder / Explorer / the file manager."""
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(path)])
        elif os.name == "nt":
            subprocess.Popen(["explorer", "/select,", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(Path(path).parent)])
    except OSError as e:
        print(f"Couldn't open the folder: {e}")


# --------------------------------------------- opening files from the desktop

def ask_open_path(initial_dir):
    """Show the computer's own "Open file" window. Returns (path or None, problem or None)."""
    initial_dir = Path(initial_dir)
    if not initial_dir.is_dir():
        initial_dir = Path.home()
    try:
        if sys.platform == "darwin":
            folder = str(initial_dir).replace("\\", "\\\\").replace('"', '\\"')
            script = ('POSIX path of (choose file with prompt "Open a stop motion movie (.stopmo)" '
                      f'default location (POSIX file "{folder}"))')
            cmd = ["osascript", "-e", script]
        elif os.name != "nt" and shutil.which("zenity"):
            cmd = ["zenity", "--file-selection", "--title=Open a stop motion movie",
                   f"--filename={initial_dir}/", "--file-filter=Stop motion movies | *.stopmo",
                   "--file-filter=All files | *"]
        elif os.name != "nt" and shutil.which("kdialog"):
            cmd = ["kdialog", "--getopenfilename", str(initial_dir), "*.stopmo|Stop motion movies"]
        else:
            code = ("import sys, tkinter, tkinter.filedialog as fd\n"
                    "root = tkinter.Tk(); root.withdraw(); root.attributes('-topmost', True)\n"
                    "print(fd.askopenfilename(title='Open a stop motion movie', initialdir=sys.argv[1],"
                    " filetypes=[('Stop motion movies', '*.stopmo'), ('All files', '*.*')]) or '')\n")
            cmd = [sys.executable, "-c", code, str(initial_dir)]
        result = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as e:
        return None, f"Couldn't show the open window ({e})"
    chosen = result.stdout.strip()
    if chosen:
        return Path(chosen), None
    if result.returncode != 0 and "tkinter" in result.stderr:
        return None, "No open window available here. Run:  python3 stopmotion.py path/to/Movie.stopmo"
    return None, None


LAUNCHER_MARK = "in Kid Stop Motion"


def write_launcher(project):
    """Put a double-clickable "Open <movie>" file next to the project file.
    It opens the folder rather than naming the file, so it keeps working if
    the project file is renamed."""
    if not WRITE_LAUNCHERS:
        return None
    script = Path(__file__).resolve()
    if os.name != "nt":
        ext = ".command" if sys.platform == "darwin" else ".sh"
        content = ("#!/bin/sh\n"
                   f"# Double-click to open this movie {LAUNCHER_MARK}.\n"
                   'cd "$(dirname "$0")" || exit 1\n'
                   f"exec {shlex.quote(sys.executable)} {shlex.quote(str(script))} .\n")
    else:
        ext = ".bat"
        python, script = (str(path).replace("%", "%%") for path in (sys.executable, script))
        content = ("@echo off\r\n"
                   "chcp 65001 >nul\r\n"
                   f"rem Double-click to open this movie {LAUNCHER_MARK}.\r\n"
                   'cd /d "%~dp0"\r\n'
                   f'"{python}" "{script}" .\r\n')
    launcher = project.folder / f"Open {project.path.stem}{ext}"
    try:
        for old in project.folder.glob("Open *" + ext):      # left behind by a rename
            if old != launcher and LAUNCHER_MARK in old.read_text(encoding="utf-8", errors="replace"):
                old.unlink()
        if not launcher.exists() or launcher.read_text(encoding="utf-8") != content:
            atomic_write(launcher, content.encode("utf-8"))
            launcher.chmod(0o755)
    except OSError as e:
        print(f"Couldn't write {launcher.name}: {e}")
        return None
    return launcher


def open_in_terminal(project_path=None):
    """macOS helper for the app from --install-mac-app: run us in Terminal,
    which already has permission to use the camera and microphone."""
    script = Path(__file__).resolve()
    args = f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"
    if project_path:
        args += " " + shlex.quote(str(project_path))
    fd, tmp = tempfile.mkstemp(prefix="kid-stop-motion-", suffix=".command")
    with os.fdopen(fd, "w") as f:
        f.write(f'#!/bin/sh\nrm -f "$0"\ncd {shlex.quote(str(script.parent))}\nexec {args}\n')
    os.chmod(tmp, 0o755)
    subprocess.run(["open", "-a", "Terminal", tmp])
    return 0


MAC_APP_SCRIPT = """
on run
    my launchStopMotion("")
end run

on open theFiles
    repeat with oneFile in theFiles
        my launchStopMotion(POSIX path of oneFile)
    end repeat
end open

on launchStopMotion(projectPath)
    set cmd to "COMMAND --open-in-terminal"
    if projectPath is not "" then set cmd to cmd & " " & quoted form of projectPath
    do shell script cmd
end launchStopMotion
"""


def install_mac_app(dest_dir=None):
    """Make ~/Applications/Kid Stop Motion.app so .stopmo files open with a double-click."""
    if sys.platform != "darwin":
        print("--install-mac-app only works on a Mac. Use the 'Open <movie>' file in each project folder instead.")
        return 1
    import plistlib
    dest_dir = Path(dest_dir or Path.home() / "Applications")
    dest_dir.mkdir(parents=True, exist_ok=True)
    app = dest_dir / "Kid Stop Motion.app"
    command = f"{shlex.quote(sys.executable)} {shlex.quote(str(Path(__file__).resolve()))}"
    source = MAC_APP_SCRIPT.replace("COMMAND", command.replace("\\", "\\\\").replace('"', '\\"'))
    with tempfile.TemporaryDirectory() as tmp:
        script_file = Path(tmp) / "launcher.applescript"
        script_file.write_text(source, encoding="utf-8")
        if app.exists():
            shutil.rmtree(app)
        result = subprocess.run(["osacompile", "-o", str(app), str(script_file)], capture_output=True, text=True)
        if result.returncode != 0:
            print("osacompile failed:", result.stderr.strip())
            return 1
    info_path = app / "Contents" / "Info.plist"
    with open(info_path, "rb") as f:
        info = plistlib.load(f)
    type_id = "local.kid-stop-motion.project"
    info["CFBundleIdentifier"] = "local.kid-stop-motion"
    info["CFBundleName"] = "Kid Stop Motion"
    info["CFBundleDocumentTypes"] = [{
        "CFBundleTypeName": "Stop Motion Movie",
        "CFBundleTypeRole": "Editor",
        "LSHandlerRank": "Owner",
        "LSItemContentTypes": [type_id],
        "CFBundleTypeExtensions": [PROJECT_EXT.lstrip(".")],
    }]
    info["UTExportedTypeDeclarations"] = [{
        "UTTypeIdentifier": type_id,
        "UTTypeDescription": "Stop Motion Movie",
        "UTTypeConformsTo": ["public.data"],
        "UTTypeTagSpecification": {"public.filename-extension": [PROJECT_EXT.lstrip(".")]},
    }]
    with open(info_path, "wb") as f:
        plistlib.dump(info, f)
    # Editing Info.plist breaks the app's signature; re-sign it for this Mac only.
    subprocess.run(["codesign", "--force", "--deep", "--sign", "-", str(app)], capture_output=True)
    lsregister = ("/System/Library/Frameworks/CoreServices.framework/Frameworks/"
                  "LaunchServices.framework/Support/lsregister")
    if Path(lsregister).exists():
        subprocess.run([lsregister, "-f", str(app)], capture_output=True)
    print(f"Made {app}")
    print("Double-click any .stopmo file to open it. If a Mac asks which app to use, pick Kid Stop Motion")
    print("(right-click the file > Get Info > Open with > Kid Stop Motion > Change All).")
    print("It runs the studio in Terminal, which already has permission to use the camera.")
    return 0


# ----------------------------------------------------------- camera controls
#
# (From the camera-controls branch.)
# Focus, exposure and white balance. OpenCV can't change these on macOS, and
# Apple's camera APIs only offer an auto/locked switch, with no way to set a
# focus distance. So we talk to USB webcams directly with standard USB Video
# Class (UVC) requests, which almost every USB webcam understands, through
# libusb (pip install libusb-package), called with ctypes. Each camera only
# gets the controls it actually has. Built-in laptop cameras aren't USB, so
# they don't get these controls. macOS only for now.

# AVFoundation tells us which camera OpenCV has open, and its USB IDs.

_objc = None
_avfoundation = None


def _load_avfoundation():
    global _objc, _avfoundation
    if _objc is None:
        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.objc_autoreleasePoolPush.restype = ctypes.c_void_p
        objc.objc_autoreleasePoolPop.argtypes = [ctypes.c_void_p]
        _avfoundation = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/AVFoundation.framework/AVFoundation")
        _objc = objc


def _send(obj, selector, restype=ctypes.c_void_p, *args):
    """Call an Objective-C method. Each extra argument is a (ctype, value) pair."""
    signature = ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p, *[t for t, _ in args])
    method = ctypes.cast(_objc.objc_msgSend, signature)
    return method(obj, _objc.sel_registerName(selector.encode()), *[v for _, v in args])


def _mac_cameras():
    """AVCaptureDevices, in the same order OpenCV numbers cameras on macOS."""
    cameras = []
    device_class = _objc.objc_getClass(b"AVCaptureDevice")
    for media_type in ("AVMediaTypeVideo", "AVMediaTypeMuxed"):
        media_type = ctypes.c_void_p.in_dll(_avfoundation, media_type).value
        found = _send(device_class, "devicesWithMediaType:", ctypes.c_void_p, (ctypes.c_void_p, media_type))
        for i in range(_send(found, "count", ctypes.c_ulong)):
            cameras.append(_send(found, "objectAtIndex:", ctypes.c_void_p, (ctypes.c_ulong, i)))
    return cameras


def mac_camera_info(index):
    """Name and unique ID of the camera OpenCV numbers `index`, or None."""
    _load_avfoundation()
    pool = _objc.objc_autoreleasePoolPush()
    try:
        cameras = _mac_cameras()
        if index >= len(cameras):
            return None

        def text(selector):
            return _send(_send(cameras[index], selector), "UTF8String", ctypes.c_char_p).decode()

        return text("localizedName"), text("uniqueID")
    finally:
        _objc.objc_autoreleasePoolPop(pool)


# libusb sends the UVC requests.

_libusb = None
_libusb_context = ctypes.c_void_p()


def _load_libusb():
    """Load libusb once. Raises OSError if it isn't installed."""
    global _libusb
    if _libusb is None:
        paths = []
        try:
            import libusb_package  # a ready-built libusb from pip, so nothing else to install
            paths.append(libusb_package.get_library_path())
        except ImportError:
            pass
        paths += [ctypes.util.find_library("usb-1.0"),
                  "/opt/homebrew/lib/libusb-1.0.dylib",  # Homebrew on Apple silicon
                  "/usr/local/lib/libusb-1.0.dylib"]     # Homebrew on Intel
        lib = None
        for path in paths:
            if path:
                try:
                    lib = ctypes.cdll.LoadLibrary(str(path))
                    break
                except OSError:
                    pass
        if lib is None:
            raise OSError("libusb isn't installed (python3 -m pip install libusb-package)")
        device_list = ctypes.POINTER(ctypes.c_void_p)
        lib.libusb_get_device_list.argtypes = [ctypes.c_void_p, ctypes.POINTER(device_list)]
        lib.libusb_get_device_list.restype = ctypes.c_ssize_t
        lib.libusb_free_device_list.argtypes = [device_list, ctypes.c_int]
        lib.libusb_get_device_descriptor.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.libusb_get_bus_number.argtypes = [ctypes.c_void_p]
        lib.libusb_get_bus_number.restype = ctypes.c_uint8
        lib.libusb_get_port_numbers.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        lib.libusb_open.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        lib.libusb_control_transfer.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint16,
                                                ctypes.c_uint16, ctypes.c_void_p, ctypes.c_uint16, ctypes.c_uint]
        lib.libusb_error_name.restype = ctypes.c_char_p
        lib.libusb_close.argtypes = [ctypes.c_void_p]
        if lib.libusb_init(ctypes.byref(_libusb_context)) != 0:
            raise OSError("libusb failed to start")
        _libusb = lib
    return _libusb


def _open_usb_device(vendor_id, product_id, location):
    """Open the USB device with these IDs at this location.

    `location` is the macOS location ID without its last byte, as it
    appears in the camera's unique ID. It tells apart two cameras of the
    same model. If we can't tell which one is meant, we'd rather have no
    controls than adjust the wrong camera.
    """
    devices = ctypes.POINTER(ctypes.c_void_p)()
    count = _libusb.libusb_get_device_list(_libusb_context, ctypes.byref(devices))
    try:
        same_model, same_place = [], []
        for i in range(max(count, 0)):
            device = devices[i]
            descriptor = (ctypes.c_uint8 * 18)()
            if _libusb.libusb_get_device_descriptor(device, descriptor) != 0:
                continue
            if (descriptor[8] | descriptor[9] << 8, descriptor[10] | descriptor[11] << 8) != (vendor_id, product_id):
                continue
            same_model.append(device)
            # macOS location IDs are the bus number, then one hex digit per port on the way to the device.
            ports = (ctypes.c_uint8 * 7)()
            depth = _libusb.libusb_get_port_numbers(device, ports, len(ports))
            here = _libusb.libusb_get_bus_number(device) << 24
            for level, port in enumerate(ports[:max(depth, 0)]):
                here |= port << (20 - 4 * level)
            if here >> 8 == location:
                same_place.append(device)
        if same_place:
            device = same_place[0]
        elif len(same_model) == 1:
            device = same_model[0]
        elif same_model:
            raise OSError("couldn't tell it apart from another camera of the same model")
        else:
            raise OSError("couldn't find it on USB")
        handle = ctypes.c_void_p()
        result = _libusb.libusb_open(device, ctypes.byref(handle))
        if result != 0:
            raise OSError(_libusb.libusb_error_name(result).decode())
        return handle.value
    finally:
        if count > 0:
            _libusb.libusb_free_device_list(devices, 1)


# UVC requests, and the controls we use as (unit, selector, size in bytes).
# See the USB Video Class 1.1 spec, sections 4.2.2.1 and 4.2.2.3.
SET_CUR, GET_CUR, GET_MIN, GET_MAX, GET_RES = 0x01, 0x81, 0x82, 0x83, 0x84
AE_MODE = ("terminal", 0x02, 1)
EXPOSURE_TIME = ("terminal", 0x04, 4)  # in 0.1 ms
FOCUS = ("terminal", 0x06, 2)
FOCUS_AUTO = ("terminal", 0x08, 1)
WHITE_BALANCE = ("processing", 0x0A, 2)  # in kelvin
WHITE_BALANCE_AUTO = ("processing", 0x0B, 1)
AE_MANUAL, AE_AUTO, AE_APERTURE_PRIORITY = 1, 2, 8


class UVCCamera:
    """A USB webcam's own controls, reached with UVC requests."""

    def __init__(self, vendor_id, product_id, location):
        _load_libusb()
        self.handle = _open_usb_device(vendor_id, product_id, location)
        try:
            self.interface, self.units = self._find_units()
        except OSError:
            self.close()
            raise

    def _transfer(self, request_type, request, value, index, buffer):
        result = _libusb.libusb_control_transfer(self.handle, request_type, request, value, index,
                                                 buffer, len(buffer), 1000)
        if result < 0:
            raise OSError(_libusb.libusb_error_name(result).decode())
        return bytes(buffer[:result])

    def _find_units(self):
        """Find the VideoControl interface, camera terminal and processing unit."""
        config = self._transfer(0x80, 0x06, 0x0200, 0, (ctypes.c_uint8 * 4096)())  # GET_DESCRIPTOR
        interface, units, in_video_control = None, {}, False
        i = 0
        while i + 6 <= len(config) and config[i] > 0:
            length, kind = config[i], config[i + 1]
            if kind == 0x04:  # interface
                in_video_control = config[i + 5:i + 7] == b"\x0e\x01"
                if in_video_control:
                    interface = config[i + 2]
            elif kind == 0x24 and in_video_control:  # VideoControl class descriptor
                subtype = config[i + 2]
                if subtype == 0x02 and config[i + 4:i + 6] == b"\x01\x02":  # camera input terminal
                    units["terminal"] = config[i + 3]
                elif subtype == 0x05:  # processing unit
                    units["processing"] = config[i + 3]
            i += length
        if interface is None or not units:
            raise OSError("it isn't a USB Video Class camera")
        return interface, units

    def _index(self, unit):
        if unit not in self.units:
            raise OSError(f"no {unit} unit")
        return self.units[unit] << 8 | self.interface

    def get(self, control, request=GET_CUR):
        unit, selector, size = control
        data = self._transfer(0xA1, request, selector << 8, self._index(unit), (ctypes.c_uint8 * size)())
        return int.from_bytes(data, "little")

    def set(self, control, value):
        unit, selector, size = control
        buffer = (ctypes.c_uint8 * size)(*int(value).to_bytes(size, "little"))
        self._transfer(0x21, SET_CUR, selector << 8, self._index(unit), buffer)

    def close(self):
        if self.handle:
            _libusb.libusb_close(self.handle)
            self.handle = None


class LockableSetting:
    """Focus, exposure or white balance: either on auto, or locked at a value."""

    def __init__(self, camera, value_control, auto_control, auto_on, auto_off,
                 log_scale=False, format=str):
        self.camera = camera
        self.value_control = value_control
        self.auto_control = auto_control
        self.auto_on, self.auto_off = auto_on, auto_off
        self.format = format
        self.min = camera.get(value_control, GET_MIN)
        self.max = camera.get(value_control, GET_MAX)
        try:
            self.step = max(camera.get(value_control, GET_RES), 1)
        except OSError:
            self.step = 1  # not every camera says
        if self.max <= self.min:
            raise OSError("no usable range")
        self.log_scale = log_scale and self.min > 0  # slider moves in equal ratios
        self.value = camera.get(value_control)
        self.locked = False

    def refresh(self):
        """Read the value auto has picked (the camera keeps it up to date)."""
        self.value = self.camera.get(self.value_control)

    def lock(self, value=None):
        """Turn auto off and hold `value`, or whatever auto had settled on."""
        if value is None:
            value = self.camera.get(self.value_control)
        value = self.min + round((value - self.min) / self.step) * self.step
        value = max(self.min, min(self.max, value))
        self.camera.set(self.auto_control, self.auto_off)
        self.camera.set(self.value_control, value)
        self.value, self.locked = value, True

    def unlock(self):
        self.camera.set(self.auto_control, self.auto_on)
        self.locked = False

    def to_slider(self, value):
        """Slider position (0-100) for a value."""
        value = max(self.min, min(self.max, value))
        if self.log_scale:
            fraction = math.log(value / self.min) / math.log(self.max / self.min)
        else:
            fraction = (value - self.min) / (self.max - self.min)
        return round(fraction * 100)

    def from_slider(self, position):
        fraction = position / 100
        if self.log_scale:
            return self.min * (self.max / self.min) ** fraction
        return self.min + fraction * (self.max - self.min)


class CameraControls:
    """Focus, exposure and white balance for the camera OpenCV opened at `index`."""

    def __init__(self, index, usb=None, name=None):
        self.name = name or f"Camera {index + 1}"
        self.usb = usb        # tests hand in a pretend camera here
        self.settings = {}  # "focus" / "exposure" / "white balance" -> LockableSetting
        self.problem = None   # why there are no controls, if we know

        if usb is None:
            if sys.platform != "darwin":
                self.problem = "These controls only work on a Mac for now."
                return
            try:
                info = mac_camera_info(index)
            except (OSError, ValueError, AttributeError):   # AttributeError: a camera with no name
                info = None
            if info is None:
                return
            self.name, unique_id = info
            # A USB camera's ID is a hex number: its USB location, then vendor and
            # product IDs, e.g. 0x124000046d08e5. Other cameras have other IDs.
            try:
                usb_id = int(unique_id, 16) if unique_id.startswith("0x") else 0
            except ValueError:
                usb_id = 0
            if usb_id >> 32 == 0:
                self.problem = "It isn't a USB webcam (built-in cameras aren't)."
                return
            try:
                self.usb = UVCCamera(usb_id >> 16 & 0xFFFF, usb_id & 0xFFFF, usb_id >> 32)
            except OSError as err:
                print(f"No focus, exposure or white balance controls for {self.name}: {err}")
                self.problem = f"Couldn't reach its controls: {err}"
                return

        makers = {
            "focus": lambda: LockableSetting(self.usb, FOCUS, FOCUS_AUTO, 1, 0),
            "exposure": self._exposure,
            "white balance": lambda: LockableSetting(self.usb, WHITE_BALANCE, WHITE_BALANCE_AUTO, 1, 0,
                                                     format=lambda v: f"{v}K"),
        }
        for name, make in makers.items():
            try:
                self.settings[name] = make()
                self.settings[name].unlock()  # start on auto
            except OSError:
                self.settings.pop(name, None)  # this camera doesn't have it

    def _exposure(self):
        current = self.usb.get(AE_MODE)
        try:
            modes = self.usb.get(AE_MODE, GET_RES)  # bitmap of the modes it supports
        except OSError:
            modes = AE_MANUAL | current  # not every camera says; assume manual works
        if not modes & AE_MANUAL:
            raise OSError("no manual exposure")
        # Unlocking goes back to the camera's own auto mode.
        if current != AE_MANUAL:
            auto = current
        else:
            auto = AE_APERTURE_PRIORITY if modes & AE_APERTURE_PRIORITY else AE_AUTO
        return LockableSetting(self.usb, EXPOSURE_TIME, AE_MODE, auto, AE_MANUAL,
                               log_scale=True, format=lambda v: f"{v / 10:g}ms")

    def summary(self):
        """What this camera lets us control, for the terminal."""
        if not self.settings:
            return "no focus, exposure or white balance controls"
        return "controls: " + ", ".join(self.settings)

    def describe(self, name):
        """A setting's value, for display."""
        setting = self.settings[name]
        return setting.format(setting.value)

    def toggle(self, name):
        """Lock or unlock a setting, and say what happened."""
        setting = self.settings.get(name)
        if setting is None:
            return f"{self.name} can't lock {name}."
        try:
            if setting.locked:
                setting.unlock()
                return f"{name.capitalize()} on AUTO"
            setting.lock()
            return f"{name.capitalize()} LOCKED at {self.describe(name)}"
        except OSError as err:
            return f"Couldn't change {name}: {err}"

    def set_value(self, name, value):
        """Lock a setting at a particular value. Returns what went wrong, or None."""
        try:
            self.settings[name].lock(value)
        except OSError as err:
            print(f"Couldn't set {name}: {err}")
            return f"Couldn't set {name}: {err}"
        return None

    def refresh(self):
        """Pick up the values the camera's auto modes have chosen."""
        for setting in self.settings.values():
            if not setting.locked:
                try:
                    setting.refresh()
                except OSError:
                    pass

    def close(self):
        """Put the camera back on auto, so other apps don't find it locked."""
        for setting in self.settings.values():
            try:
                setting.unlock()
            except OSError:
                pass
        if self.usb:
            self.usb.close()
            self.usb = None


def linear_light(frame):
    """A sample of the picture's pixels in linear light (0-1)."""
    return (frame[::8, ::8] / 255.0) ** 2.2


class PictureMatch:
    """Lock a setting where auto had it, by matching how the picture looked.

    Webcams like the C920 don't report the exposure or white balance their
    auto modes pick (they report the last value we set), and switching auto
    off jumps to that old value. So we switch to manual and adjust over a few
    frames until the picture matches how it looked on auto. Meanwhile the
    app keeps showing the last picture from auto, so nothing jumps about.

    Some cameras only have coarse steps (a C920's exposure times are about a
    stop apart, however finely you ask), so an exact match may not exist.
    Then we settle on the closest value we saw, and `error` says how far off
    that is, so the app can say so.

    Subclasses say what to measure, and how the setting relates to it:
    roughly, measure(frame) = slope * to_x(value) + a constant.
    """

    FRAMES_PER_STEP = 5  # give each change time to reach the picture
    MAX_STEPS = 12
    name = None
    slope = 1.0             # first guess; refined from what we see
    slope_range = (0.3, 3.0)
    tolerance = 0.05        # close enough, in units of measure()

    def __init__(self, controls, frame, offset=0.0):
        self.controls = controls
        self.setting = controls.settings[self.name]
        self.frame = frame  # shown until we're done
        self.target = self.measure(frame) + offset  # offset: e.g. brighter than it was
        self.wait = self.FRAMES_PER_STEP
        self.steps = 0
        self.previous = None    # (x, measurement) from the last step
        self.best = None        # (distance, value, error): the closest we've been
        self.overshoots = 0
        self.settling = False   # gone back to the best value; waiting for it to show
        self.error = 0.0        # how far off we ended up
        controls.set_value(self.name, self.setting.value)  # manual, from the last value we know

    def update(self, frame):
        """Call once per frame with the live picture. Returns True when done."""
        if not self.setting.locked:
            return True  # locking failed
        self.wait -= 1
        if self.wait > 0:
            return False
        self.wait = self.FRAMES_PER_STEP
        if self.settling:
            return True
        self.steps += 1

        value = self.setting.value
        x, measured = self.to_x(value), self.measure(frame)
        error = self.target - measured
        if self.best is None or abs(error) < self.best[0]:
            self.best = (abs(error), value, error)
        if abs(error) <= self.tolerance:
            self.error = error
            return True
        if self.previous and (self.target - self.previous[1] > 0) != (error > 0):
            self.overshoots += 1
        if self.overshoots >= 2 or self.steps >= self.MAX_STEPS:
            # The camera can't land any closer: go back to the closest we saw.
            self.error = self.best[2]
            self.controls.set_value(self.name, self.best[1])
            self.settling = value != self.best[1]
            return not self.settling
        if self.previous and x != self.previous[0]:
            slope = (measured - self.previous[1]) / (x - self.previous[0])
            if self.slope_range[0] <= abs(slope) <= self.slope_range[1]:
                self.slope = slope
        self.previous = (x, measured)

        self.controls.set_value(self.name, self.from_x(x + error / self.slope))
        if self.setting.value == value:  # at the end of its range
            self.error = error
            return True
        return False

    def shortfall(self):
        """How far off we ended up, in words, or None if we matched."""
        return None if abs(self.error) <= self.tolerance else self.describe_error(self.error)


class ExposureMatch(PictureMatch):
    """Brightness goes up in step with exposure time."""

    name = "exposure"

    def measure(self, frame):
        return math.log(max(float(linear_light(frame).mean()), 1e-6))

    def to_x(self, value):
        return math.log(max(value, 1))           # some cameras' exposure range starts at 0

    def from_x(self, x):
        return math.exp(x)

    def describe_error(self, error):
        stops = error / math.log(2)
        return f"{abs(stops):.1f} stops {'darker' if stops > 0 else 'brighter'}"


class WhiteBalanceMatch(PictureMatch):
    """The blue/red balance shifts evenly with the setting in mireds (1e6 / kelvin)."""

    name = "white balance"
    slope = 0.005
    slope_range = (0.001, 0.05)
    tolerance = 0.02

    def measure(self, frame):
        light = linear_light(frame)
        blue, red = float(light[..., 0].mean()), float(light[..., 2].mean())
        return math.log(max(blue, 1e-6) / max(red, 1e-6))

    def to_x(self, value):
        return 1e6 / value

    def from_x(self, x):
        return 1e6 / x

    def describe_error(self, error):
        return "warmer" if error > 0 else "cooler"



MATCHES = {"exposure": ExposureMatch, "white balance": WhiteBalanceMatch}


def exposure_lut(stops):
    """Lookup table that brightens (or darkens, if negative) by `stops`."""
    linear = (np.arange(256) / 255.0) ** 2.2
    return np.clip(np.round((linear * 2 ** stops) ** (1 / 2.2) * 255), 0, 255).astype(np.uint8)



def fill_image(img, size):
    """Scale and centre-crop img to exactly size (w, h), so a picture from a
    different camera still fits the movie."""
    w, h = size
    ih, iw = img.shape[:2]
    if (iw, ih) == (w, h):
        return img
    scale = max(w / iw, h / ih)
    img = cv2.resize(img, (max(w, round(iw * scale)), max(h, round(ih * scale))))
    top, left = (img.shape[0] - h) // 2, (img.shape[1] - w) // 2
    return img[top:top + h, left:left + w]


class CameraSetup:
    """How the camera sees: which way up (remembered for each camera) and, on a
    Mac with a USB webcam, focus, exposure and white balance."""

    NAMES = {"focus": "Focus", "exposure": "Exposure", "white balance": "White balance"}

    def __init__(self, settings_file):
        self.settings_file = Path(settings_file)
        self.controls = None
        self.generation = None
        self.index = None
        self.matches = {}         # exposure / white balance locks still settling: name -> PictureMatch
        self.brightness = 0       # our own brightening, in EXPOSURE_STEPs, for cameras we can't control
        self._lut = None
        self._refresh_at = 0.0
        self.camera_frame = None  # the newest picture as the camera took it
        self.notes = []           # things to tell the user
        try:
            self.flips = dict(json.loads(self.settings_file.read_text(encoding="utf-8")).get("flip", {}))
        except (OSError, ValueError, AttributeError):
            self.flips = {}

    def make_controls(self, index):
        return CameraControls(index)

    def sync(self, camera):
        """Set up the controls for the camera we have now (it may have been
        switched, or come back after dropping out)."""
        if camera.generation == self.generation:
            return
        # The same camera coming back after dropping out keeps its locked settings.
        same = self.controls is not None and camera.index == self.index
        name, brightness = self.name, self.brightness
        kept = {key: setting.value for key, setting in self.settings.items() if setting.locked} if same else {}
        self.close()
        self.generation, self.index = camera.generation, camera.index
        if not camera.ok:
            return
        self.controls = self.make_controls(camera.index)
        print(f"Camera {camera.index + 1}: {self.controls.name} ({self.controls.summary()})")
        if same and self.controls.name == name and (kept or brightness):
            for key, value in kept.items():
                if key in self.settings:
                    self.controls.set_value(key, value)
            if brightness:
                self.brightness, self._lut = brightness, exposure_lut(brightness * EXPOSURE_STEP)
            self.say("The camera is back, with its settings locked as before.")

    def close(self):
        """Put the camera back on auto, so other apps don't find it locked."""
        if self.controls is not None:
            self.controls.close()
            self.controls = None
        self.matches = {}
        self.brightness = 0
        self.camera_frame = None

    @property
    def name(self):
        return self.controls.name if self.controls else "Camera"

    @property
    def settings(self):
        return self.controls.settings if self.controls else {}

    def flipped(self):
        return bool(self.flips.get(self.name, ROTATE_180))

    def toggle_flip(self):
        self.flips[self.name] = not self.flipped()
        try:
            atomic_write(self.settings_file, json.dumps({"flip": self.flips}, indent=1).encode("utf-8"))
        except OSError as e:
            print(f"Couldn't remember that: {e}")
        return self.flipped()

    def process(self, frame, size=None):
        """The camera's picture as the studio should show and capture it,
        cropped to `size` (the movie's) if given."""
        if self.flipped():
            frame = cv2.flip(frame, -1)
        if size is not None:
            frame = fill_image(frame, size)
        self.camera_frame = frame
        for name, match in list(self.matches.items()):
            if match.update(frame):
                if self.settings[name].locked:
                    shortfall = match.shortfall()
                    note = f" (as close as this camera gets: {shortfall})" if shortfall else ""
                    self.say(f"{name.capitalize()} LOCKED at {self.controls.describe(name)}{note}")
                del self.matches[name]
        if self.matches:
            # Keep showing the picture from auto until the lock has settled.
            frame = next(iter(self.matches.values())).frame
        if self.brightness:
            frame = cv2.LUT(frame, self._lut)
        if self.controls is not None and time.monotonic() >= self._refresh_at:
            self.controls.refresh()       # sliders for settings on auto follow what the camera picks
            self._refresh_at = time.monotonic() + 0.5
        return frame

    def say(self, text):
        print(text)
        self.notes.append(text)

    def take_notes(self):
        notes, self.notes = self.notes, []
        return notes

    def toggle(self, name):
        """F / X / W: lock where auto has settled, or go back to auto."""
        setting = self.settings.get(name)
        if setting is not None and not setting.locked and name in MATCHES:
            if self.camera_frame is None:
                return "The camera isn't ready yet."
            self.matches[name] = MATCHES[name](self.controls, self.camera_frame)
            return f"Locking {name}..."
        self.matches.pop(name, None)
        if self.controls is None:
            return "The camera isn't ready yet."
        return self.controls.toggle(name)

    def nudge_focus(self, direction):
        focus = self.settings.get("focus")
        if focus is None:
            return f"{self.name} can't set focus."
        problem = self.controls.set_value("focus", focus.value + direction * focus.step)
        return problem or f"Focus LOCKED at {self.controls.describe('focus')}"

    def brighter(self, step):
        """- / +: a third of a stop darker or brighter, and locked there."""
        exposure = self.settings.get("exposure")
        if exposure is None:
            # No exposure control on this camera, so brighten the picture ourselves.
            self.brightness = clamp(self.brightness + step, -MAX_EXPOSURE_STEPS, MAX_EXPOSURE_STEPS)
            self._lut = exposure_lut(self.brightness * EXPOSURE_STEP)
            return f"Brightness {self.brightness * EXPOSURE_STEP:+.1f} stops"
        if "exposure" in self.matches:
            self.matches["exposure"].target += step * EXPOSURE_STEP * math.log(2)
            return "Locking exposure..."
        if exposure.locked:
            problem = self.controls.set_value("exposure", exposure.value * 2 ** (step * EXPOSURE_STEP))
            return problem or f"Exposure LOCKED at {exposure.format(exposure.value)}"
        if self.camera_frame is None:
            return "The camera isn't ready yet."
        # On auto: lock a step brighter or darker than auto has it.
        self.matches["exposure"] = ExposureMatch(self.controls, self.camera_frame,
                                                 step * EXPOSURE_STEP * math.log(2))
        return "Locking exposure..."

    def set_slider(self, name, fraction):
        """A slider moved: lock that setting at the slider's value."""
        setting = self.settings.get(name)
        if setting is None:
            return None
        self.matches.pop(name, None)
        return self.controls.set_value(name, setting.from_slider(clamp(fraction, 0.0, 1.0) * 100))

    def status(self, name):
        """(words, state) for a setting; state is "n/a", "auto", "locking" or "locked"."""
        setting = self.settings.get(name)
        if setting is None:
            if name == "exposure" and self.brightness:
                return f"picture {self.brightness * EXPOSURE_STEP:+.1f}", "locked"
            return "n/a", "n/a"
        if name in self.matches:
            return "LOCKING...", "locking"
        if setting.locked:
            return f"LOCKED {self.controls.describe(name)}", "locked"
        return "AUTO", "auto"


# -------------------------------------------------------------------- camera

class Camera:
    """The webcam. If it drops out (unplugged, settings changed, another app
    grabbed it) the studio keeps going and quietly reconnects."""

    RETRY_SECONDS = 3.0

    def __init__(self, index=CAMERA_INDEX):
        self.index = index
        self.cap = None
        self.generation = 0         # goes up each time a camera is (re)connected
        self.failures = 0
        self._retry_at = 0.0
        self._opener = None
        self._opened = None
        self._attempt = 0           # a reconnect that finishes after a switch is thrown away
        self._guard = threading.Lock()

    def _open(self, index=None):
        """Open a camera and check it delivers a picture; None if it doesn't."""
        cap = cv2.VideoCapture(self.index if index is None else index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SIZE[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SIZE[1])
            if cap.read()[0]:
                return cap
        cap.release()
        return None

    def start(self):
        # The first open happens here on the main thread, because that's where
        # macOS asks for camera permission.
        self.cap = self._open()
        self.generation += 1
        return self.cap is not None

    def switch(self):
        """Move to the next camera, wrapping round to the first. False if there's no other one."""
        for candidate in (self.index + 1, 0):
            if candidate != self.index:
                cap = self._open(candidate)
                if cap is not None:
                    self.release()
                    self.index, self.cap, self.failures = candidate, cap, 0
                    self.generation += 1
                    return True
        return False

    def _forget_reconnect(self):
        with self._guard:
            self._attempt += 1
            if self._opened is not None:
                self._opened.release()
                self._opened = None

    @property
    def ok(self):
        return self.cap is not None

    def read(self):
        if self.cap is None:
            self._reconnect()
            return None
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.failures += 1
            if self.failures >= 20:
                print("The camera stopped sending pictures - trying to reconnect...")
                self.cap.release()
                self.cap = None
                self.failures = 0
                self._retry_at = time.monotonic() + 1.0
            return None
        self.failures = 0
        return frame

    def flush(self):
        """Throw away pictures the camera queued up while we weren't looking,
        so the next capture shows the set as it is *now*."""
        if self.cap is None:
            return
        for _ in range(6):
            start = time.monotonic()
            if not self.cap.grab() or time.monotonic() - start > 0.02:
                break

    def _reconnect(self):
        with self._guard:
            opened, self._opened = self._opened, None
        if opened is not None:
            self.cap = opened
            self.generation += 1
            print("Camera reconnected.")
            return
        if (self._opener is not None and self._opener.is_alive()) or time.monotonic() < self._retry_at:
            return
        self._retry_at = time.monotonic() + self.RETRY_SECONDS
        attempt, index = self._attempt, self.index

        def work():
            cap = self._open(index)
            with self._guard:
                if attempt == self._attempt:
                    self._opened = cap
                    return
            if cap is not None:
                cap.release()           # the camera was switched or closed meanwhile

        self._opener = threading.Thread(target=work, daemon=True)
        self._opener.start()

    def release(self):
        self._forget_reconnect()
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# -------------------------------------------------------------------- window

class Display:
    """The OpenCV window: shows pictures, hands back keys and mouse clicks."""

    def __init__(self, title=WINDOW_NAME, scale=WINDOW_SCALE):
        self.title = title
        self.scale = scale
        self.events = []
        # Linux (Qt) windows zoom the picture when you scroll, and there's no
        # way to switch that off - so after a scroll we make a fresh window.
        framework = getattr(cv2, "currentUIFramework", lambda: "")
        self.qt = str(framework()).upper().startswith("QT")
        self._unzoom = False
        self.closed = False
        self._seen_visible = False
        self._create()

    def _create(self):
        # GUI_NORMAL hides the zoom toolbar that Qt windows add.
        cv2.namedWindow(self.title, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(self.title, self._mouse)

    def _mouse(self, event, x, y, flags, param):
        try:
            self._record_mouse(event, int(x / self.scale), int(y / self.scale), flags)
        except Exception as e:           # never let a stray event upset the window
            print(f"Mouse event ignored: {e}")

    def _record_mouse(self, event, x, y, flags):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.events.append(("down", x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            self.events.append(("up", x, y))
        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
            if self.events and self.events[-1][0] == "drag":
                self.events[-1] = ("drag", x, y)
            else:
                self.events.append(("drag", x, y))
        elif event in (cv2.EVENT_MOUSEWHEEL, cv2.EVENT_MOUSEHWHEEL):
            # The wheel amount lives in the top bits of flags (getMouseWheelDelta in C++).
            self.events.append(("wheel", x, y, 1 if (flags >> 16) > 0 else -1))
            self._unzoom = self.qt

    def show(self, canvas):
        if self._unzoom:
            self._unzoom = False
            cv2.destroyWindow(self.title)
            self._create()
        # Once the window has been seen, it disappearing means its close button was
        # clicked. (A minimized window still exists; backends that can't tell never
        # report "visible", so they never trigger this.)
        try:
            visible = cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            visible = -1
        if visible >= 1:
            self._seen_visible = True
        elif visible < 0 and self._seen_visible:
            self.closed = True
            return
        if self.scale != 1.0:
            canvas = cv2.resize(canvas, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        cv2.imshow(self.title, canvas)

    def wait(self, ms):
        return cv2.waitKeyEx(max(1, int(ms)))

    def take_events(self):
        events, self.events = self.events, []
        return events

    def close(self):
        cv2.destroyAllWindows()


# Key codes differ between Windows, macOS and Linux; this turns them into names.
SPECIAL_KEYS = {
    8: "backspace", 127: "backspace", 9: "tab", 10: "enter", 13: "enter", 27: "esc", 32: "space",
    # Windows
    2424832: "left", 2555904: "right", 2490368: "up", 2621440: "down",
    2359296: "home", 2293760: "end", 3014656: "delete",
    # Linux
    65361: "left", 65363: "right", 65362: "up", 65364: "down", 65360: "home", 65367: "end",
    65535: "delete", 65288: "backspace", 65289: "tab", 65293: "enter", 65421: "enter", 65307: "esc",
    # macOS
    63234: "left", 63235: "right", 63232: "up", 63233: "down", 63273: "home", 63275: "end",
    63272: "delete",
}


def key_name(code):
    """'left', 'enter', 'space', ... or the typed character, or None."""
    if code is None or code < 0:
        return None
    if code in SPECIAL_KEYS:
        return SPECIAL_KEYS[code]
    low = code & 0xFFFF        # some Linux setups add NumLock bits
    if low in SPECIAL_KEYS:
        return SPECIAL_KEYS[low]
    if 32 < low < 127:
        return chr(low)
    return None


SHIFT_KEYS = (65505, 65506)
CAPS_LOCK_KEY = 65509


KEY_COMMANDS = {
    "space": "snap", "enter": "enter", "p": "play", "o": "onion", "t": "auto", "c": "title",
    "f": "transition", "r": "record", "v": "speed", "e": "export", "l": "projects",
    "x": "delete", "delete": "delete", "backspace": "delete", "d": "duplicate",
    "+": "longer", "=": "longer", "-": "shorter", "_": "shorter", "[": "move_left", "]": "move_right",
    "z": "undo", "y": "redo", "s": "save", "n": "goal", "h": "help", "?": "help",
    "left": "prev", ",": "prev", "<": "prev", "right": "next", ".": "next", ">": "next",
    "home": "first", "end": "last", "q": "quit", "esc": "escape", "k": "camera",
}

# While the camera panel (K) is open, these keys adjust the camera.
CAMERA_KEYS = {
    "c": "switch", "r": "flip", "f": "focus", "x": "exposure", "w": "white balance",
    "[": "[", "]": "]", "-": "-", "_": "-", "+": "+", "=": "+", "k": "close", "esc": "close",
}

HELP_LINES = [
    ("SPACE", "Take a picture"),
    ("P", "Play the movie (with your voice track)"),
    ("O", "Ghost of the last picture on / off"),
    ("T", "Auto snap: a picture every 2 seconds"),
    ("C", "Add a title card (or edit the picked one)"),
    ("F", "Transition before the picked picture"),
    ("R", "Record a voice track with the microphone"),
    ("V", "Speed: pictures per second"),
    ("E", "Make an MP4 movie"),
    ("L", "Projects: open another movie, or a new one"),
    ("K", "Camera: switch, flip, lock focus and colour"),
    ("< >  (or , .)", "Pick a picture. The camera is a tile too!"),
    ("X", "Delete the picked picture"),
    ("D", "Copy the picked picture"),
    ("+  -", "Show it for longer / shorter"),
    ("[  ]", "Move it left / right (or drag it)"),
    ("Z  Y", "Undo / redo"),
    ("N", "Set a picture goal"),
    ("S", "Save now (it saves by itself too)"),
    ("Q", "Quit (press it twice)"),
]


# ------------------------------------------------------------------- drawing

UI_FONT = cv2.FONT_HERSHEY_SIMPLEX
UI_BOLD = cv2.FONT_HERSHEY_DUPLEX

WHITE = (255, 255, 255)
C_BG = (36, 31, 30)
C_BAR = (54, 47, 45)
C_TILE = (70, 62, 60)
C_BUTTON = (84, 74, 70)
C_ACTIVE = (150, 130, 40)
C_TEXT = (242, 242, 242)
C_DIM = (165, 160, 158)
C_DARK = (30, 30, 30)
C_RED = (70, 70, 225)
C_GREEN = (90, 175, 80)
C_YELLOW = (60, 210, 250)
C_ORANGE = (40, 150, 250)
C_PURPLE = (190, 100, 170)
C_BLUE = (200, 140, 60)

CANVAS_W = 1280
TOP_H = 50
VIEW_X, VIEW_Y, VIEW_W, VIEW_H = 0, TOP_H, 1024, 576
PANEL_X = VIEW_W
TL_Y = TOP_H + VIEW_H
TILE_W, TILE_H, TILE_GAP = 120, 68, 26
TILE_STEP = TILE_W + TILE_GAP
TILES_X0 = 34
TILE_Y = TL_Y + 12
TILES_FIT = (CANVAS_W - 2 * TILES_X0 + TILE_GAP) // TILE_STEP
EDIT_Y = TL_Y + 106
CANVAS_H = TL_Y + 150

PICK_COLS, PICK_ROWS = 5, 3
PICK_W, PICK_H = 228, 128
PICK_STEP_X, PICK_STEP_Y = PICK_W + 20, PICK_H + 68
PICK_X0 = (CANVAS_W - (PICK_COLS * PICK_STEP_X - 20)) // 2
PICK_Y0 = 112


def put_text(img, text, org, scale=0.6, color=C_TEXT, thick=1, font=UI_FONT, align="left", valign="baseline"):
    text = ascii_text(text)
    x, y = int(org[0]), int(org[1])
    (w, h), _ = cv2.getTextSize(text, font, scale, thick)
    if align == "center":
        x -= w // 2
    elif align == "right":
        x -= w
    if valign == "middle":
        y += h // 2
    cv2.putText(img, text, (x, y), font, scale, color, thick, cv2.LINE_AA)
    return w


def fill_rect(img, x0, y0, x1, y1, color, alpha=1.0):
    h, w = img.shape[:2]
    x0, x1 = clamp(int(x0), 0, w), clamp(int(x1), 0, w)
    y0, y1 = clamp(int(y0), 0, h), clamp(int(y1), 0, h)
    if x1 <= x0 or y1 <= y0:
        return
    if alpha >= 1:
        img[y0:y1, x0:x1] = color
    else:
        roi = img[y0:y1, x0:x1]
        roi[:] = cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, color), alpha, 0)


def round_rect(img, x0, y0, x1, y1, color, radius=8, thickness=-1):
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    r = int(max(0, min(radius, (x1 - x0) // 2, (y1 - y0) // 2)))
    aa = cv2.LINE_AA
    if thickness < 0:
        cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), color, -1)
        cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), color, -1)
        if r > 0:
            for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
                cv2.circle(img, (cx, cy), r, color, -1, aa)
        return
    t = thickness
    cv2.line(img, (x0 + r, y0), (x1 - r, y0), color, t, aa)
    cv2.line(img, (x0 + r, y1), (x1 - r, y1), color, t, aa)
    cv2.line(img, (x0, y0 + r), (x0, y1 - r), color, t, aa)
    cv2.line(img, (x1, y0 + r), (x1, y1 - r), color, t, aa)
    if r > 0:
        cv2.ellipse(img, (x0 + r, y0 + r), (r, r), 180, 0, 90, color, t, aa)
        cv2.ellipse(img, (x1 - r, y0 + r), (r, r), 270, 0, 90, color, t, aa)
        cv2.ellipse(img, (x1 - r, y1 - r), (r, r), 0, 0, 90, color, t, aa)
        cv2.ellipse(img, (x0 + r, y1 - r), (r, r), 90, 0, 90, color, t, aa)


def pill(img, text, x, y, bg, fg=WHITE, scale=0.6, align="left"):
    """A rounded label. Returns its rectangle."""
    text = ascii_text(text)
    (tw, th), _ = cv2.getTextSize(text, UI_BOLD, scale, 1)
    h, w = th + 18, tw + 30
    if align == "center":
        x -= w // 2
    elif align == "right":
        x -= w
    round_rect(img, x, y, x + w, y + h, bg, h // 2)
    cv2.putText(img, text, (x + 15, y + (h + th) // 2), UI_BOLD, scale, fg, 1, cv2.LINE_AA)
    return x, y, x + w, y + h


def wrap_text(text, width, scale, font=UI_BOLD):
    words, lines, line = ascii_text(text).split(), [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if line and cv2.getTextSize(trial, font, scale, 1)[0][0] > width:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    return lines or [""]


def darken(img, amount=0.35):
    return cv2.convertScaleAbs(img, alpha=amount)


def draw_icon(img, name, cx, cy, s, color):
    """Little icons drawn with lines and circles (s = half their size)."""
    cx, cy, s = int(cx), int(cy), int(s)
    aa = cv2.LINE_AA
    if name == "snap":
        cv2.circle(img, (cx, cy), s - 2, color, -1, aa)
        cv2.circle(img, (cx, cy), s + 1, WHITE, 2, aa)
    elif name == "play":
        pts = np.array([(cx - s * 2 // 3, cy - s), (cx - s * 2 // 3, cy + s), (cx + s, cy)], np.int32)
        cv2.fillPoly(img, [pts], color, aa)
    elif name == "stop":
        cv2.rectangle(img, (cx - s + 3, cy - s + 3), (cx + s - 3, cy + s - 3), color, -1)
    elif name == "ghost":
        r = s * 4 // 5
        cv2.circle(img, (cx, cy - s // 5), r, color, -1, aa)
        cv2.rectangle(img, (cx - r, cy - s // 5), (cx + r, cy + s * 3 // 5), color, -1)
        for k in (-1, 0, 1):
            cv2.circle(img, (cx + k * r * 2 // 3, cy + s * 3 // 5), r // 3 + 1, color, -1, aa)
        for k in (-1, 1):
            cv2.circle(img, (cx + k * r // 3, cy - s // 4), max(2, s // 6), C_DARK, -1, aa)
    elif name == "auto":
        cv2.circle(img, (cx, cy), s, color, 2, aa)
        cv2.line(img, (cx, cy), (cx, cy - s * 2 // 3), color, 2, aa)
        cv2.line(img, (cx, cy), (cx + s // 2, cy), color, 2, aa)
    elif name == "title":
        cv2.rectangle(img, (cx - s, cy - s * 3 // 4), (cx + s, cy + s * 3 // 4), color, 2, aa)
        put_text(img, "T", (cx, cy), s / 24, color, 2, UI_BOLD, align="center", valign="middle")
    elif name == "transition":
        pts = []
        for k in range(8):
            angle = k * np.pi / 4 - np.pi / 2
            r = s if k % 2 == 0 else s * 0.35
            pts.append((cx + r * np.cos(angle), cy + r * np.sin(angle)))
        cv2.fillPoly(img, [np.array(pts, np.int32)], color, aa)
    elif name == "mic":
        w = max(3, s * 2 // 5)
        round_rect(img, cx - w, cy - s, cx + w, cy + s // 4, color, w)
        cv2.ellipse(img, (cx, cy - s // 6), (s * 2 // 3, s * 2 // 3), 0, 0, 180, color, 2, aa)
        cv2.line(img, (cx, cy + s // 2), (cx, cy + s), color, 2, aa)
        cv2.line(img, (cx - s // 2, cy + s), (cx + s // 2, cy + s), color, 2, aa)
    elif name == "speed":
        cv2.ellipse(img, (cx, cy + s // 3), (s, s), 0, 180, 360, color, 2, aa)
        cv2.line(img, (cx, cy + s // 3), (cx + s * 2 // 3, cy - s // 3), color, 2, aa)
        cv2.circle(img, (cx, cy + s // 3), 3, color, -1, aa)
    elif name == "movie":
        cv2.rectangle(img, (cx - s, cy - s * 3 // 4), (cx + s, cy + s * 3 // 4), color, -1)
        for k in range(-2, 3):
            x = cx + k * s * 2 // 5
            for y in (cy - s * 3 // 4 + 2, cy + s * 3 // 4 - 5):
                cv2.rectangle(img, (x - 2, y), (x + 1, y + 2), C_DARK, -1)
        pts = np.array([(cx - s // 4, cy - s // 3), (cx - s // 4, cy + s // 3), (cx + s // 3, cy)], np.int32)
        cv2.fillPoly(img, [pts], C_DARK, aa)
    elif name == "folder":
        pts = np.array([(cx - s, cy - s * 2 // 3), (cx - s // 4, cy - s * 2 // 3), (cx, cy - s // 3),
                        (cx + s, cy - s // 3), (cx + s, cy + s * 2 // 3), (cx - s, cy + s * 2 // 3)], np.int32)
        cv2.fillPoly(img, [pts], color, aa)
    elif name == "camera":
        round_rect(img, cx - s, cy - s // 2, cx + s, cy + s * 2 // 3, color, max(2, s // 4))
        cv2.rectangle(img, (cx - s // 3, cy - s * 3 // 4), (cx + s // 3, cy - s // 2), color, -1)
        cv2.circle(img, (cx, cy + s // 12), max(2, s // 3), C_DARK, -1, aa)
    elif name == "check":
        pts = np.array([(cx - s, cy), (cx - s // 3, cy + s * 2 // 3), (cx + s, cy - s * 2 // 3)], np.int32)
        cv2.polylines(img, [pts], False, color, 3, aa)
    elif name == "cross":
        cv2.line(img, (cx - s * 2 // 3, cy - s * 2 // 3), (cx + s * 2 // 3, cy + s * 2 // 3), color, 3, aa)
        cv2.line(img, (cx - s * 2 // 3, cy + s * 2 // 3), (cx + s * 2 // 3, cy - s * 2 // 3), color, 3, aa)
    elif name == "plus":
        cv2.line(img, (cx - s, cy), (cx + s, cy), color, max(2, s // 4), aa)
        cv2.line(img, (cx, cy - s), (cx, cy + s), color, max(2, s // 4), aa)


# ---------------------------------------------------------------------- app

class App:
    def __init__(self, display, camera, projects_dir=None):
        self.display = display
        self.camera = camera
        self.projects_dir = Path(projects_dir or PROJECTS_DIR).expanduser()
        self.project = None
        self.renderer = None
        self.running = True
        self.live = None              # the newest camera picture
        self.cursor = None            # picked timeline item, or None when the camera is picked
        self.insert_at = 0            # where the camera tile sits: new pictures go here
        self.onion = True
        self.auto = False
        self.next_auto = 0.0
        self.flash_until = 0.0
        self.quit_armed_until = 0.0
        self.toast_text = ""
        self.toast_kind = "info"
        self.toast_until = 0.0
        self.hotspots = []
        self.thumbs = {}
        self.fitted = OrderedDict()
        self._ghost = (None, None)
        self.tl_first = 0
        self._tl_focus = None
        self.drag = None
        self.last_canvas = None
        self.last_style = 0
        self.picker_thumbs = {}
        self.speaker = Speaker()
        self.recorder_factory = MicRecorder
        self.cam_setup = CameraSetup(self.projects_dir / SETTINGS_FILE)
        self.camera_panel = False
        self.slider_drag = None
        self.slider_rects = {}
        self.countdown_seconds = 3
        self._shift = False
        self._caps = False
        self._audio_len = (None, None)
        self._draw_failed = False
        self._camera_trouble = None

    # -- running

    def run(self, open_path=None):
        if open_path and not self.open_path(open_path):
            print(f"Couldn't open {open_path}: {self.toast_text}")
        if self.project is None and not self.pick_project(startup=True):
            return
        self.studio()

    def shutdown(self):
        self.speaker.stop()
        self.cam_setup.close()
        if self.project is not None:
            try:
                moved = self.project.tidy()
                if moved:
                    print(f"Moved {moved} unused picture(s) into {self.project.folder / TRASH_DIR}")
            except OSError as e:
                print(f"Couldn't tidy the project folder: {e}")
            self.project.unlock()

    def typed(self, code):
        """key_name() for typing words. Linux (Qt) windows only ever report
        lowercase letters, but send Shift and Caps Lock as keys of their own,
        so keep track of those here. (Macs and Windows send capitals directly
        and never send these, so this changes nothing there.)"""
        low = code & 0xFFFF if code is not None and code >= 0 else None
        if low in SHIFT_KEYS:
            self._shift = True
            return None
        if low == CAPS_LOCK_KEY:
            self._caps = not self._caps
            return None
        name = key_name(code)
        if name is not None:
            if len(name) == 1 and name.islower() and self._shift != self._caps:
                name = name.upper()
            self._shift = False
        return name

    def tick(self, canvas, wait_ms=15):
        self.display.show(canvas)
        if getattr(self.display, "closed", False):
            self.running = False
        key = self.display.wait(wait_ms)
        return key, self.display.take_events()

    def safely(self, fn, *args):
        """Run a command; if something goes wrong, say so instead of crashing.
        (Everything is already saved, so nothing is lost.)"""
        try:
            fn(*args)
        except Exception as e:
            traceback.print_exc()
            self.toast(f"Oops: {e}", "error", 6)

    def after_modal(self):
        self.drag = None
        self.slider_drag = None
        # Wait a whole auto-snap gap for a fresh picture: the last one is from
        # before that screen opened, and things may have moved since.
        self.next_auto = time.monotonic() + AUTO_CAPTURE_SECONDS
        self.camera.flush()

    def read_camera(self):
        """Take the newest picture from the camera: the right way up, with the
        camera settings applied, and cropped to fit the movie."""
        try:
            self.cam_setup.sync(self.camera)
        except Exception as e:
            self.camera_trouble(e)
        frame = self.camera.read()
        if frame is None:
            if not self.camera.ok:
                self.live = None
            return
        size = self.project.size if self.project is not None and self.project.frame_count() else None
        try:
            frame = self.cam_setup.process(frame, size)
        except Exception as e:
            self.camera_trouble(e)
            self.cam_setup.matches.clear()
            frame = fill_image(frame, size) if size else frame
        for note in self.cam_setup.take_notes():
            self.toast(note)
        self.live = frame

    def camera_trouble(self, error):
        """Something unexpected from the camera controls: say so, and keep going."""
        text = f"Camera trouble: {error}"
        if text != self._camera_trouble:
            traceback.print_exc()
            self._camera_trouble = text
            self.toast(text, "error", 6)

    def studio(self):
        while self.running:
            self.read_camera()
            try:
                canvas = self.draw_studio()
                self._draw_failed = False
            except Exception as e:
                if not self._draw_failed:
                    traceback.print_exc()
                self._draw_failed = True
                canvas = self.blank()
                self.hotspots = []
                put_text(canvas, "Something went wrong showing this movie.", (CANVAS_W // 2, 330), 0.9, WHITE, 2,
                         UI_BOLD, align="center")
                put_text(canvas, ascii_text(e)[:90], (CANVAS_W // 2, 380), 0.55, C_DIM, align="center")
                put_text(canvas, "Press L to open another movie, or Q twice to quit.", (CANVAS_W // 2, 430), 0.7,
                         C_TEXT, align="center")
            self.last_canvas = canvas
            key, events = self.tick(canvas, 1 if self.camera.ok else 20)
            for event in events:
                self.safely(self.on_mouse, event)
            if key != -1:
                self.safely(self.on_key, key)
            self.safely(self.auto_tick)

    def set_project(self, project):
        if project is not self.project and project._lock_file is None and not project.lock():
            raise ProjectError("That movie is already open in another window.")
        if self.project is not None and self.project is not project:
            if self.project.folder != project.folder:     # (not when it's the same movie, reopened)
                try:
                    self.project.tidy()
                except OSError:
                    pass
            self.project.unlock()
        self.project = project
        self.renderer = Renderer(project)
        self.cursor = None
        self.insert_at = len(project.items)
        self.onion = bool(project.doc.get("onion_skin", True))
        self.auto = False
        self.thumbs.clear()
        self.fitted.clear()
        self._ghost = (None, None)
        self.tl_first = 0
        self._tl_focus = None
        write_launcher(project)
        print(f"Opened {project.path}")
        for note in project.notes:
            print("  " + note)
        if project.notes:
            self.toast(" ".join(project.notes), "error", 8)

    def toast(self, text, kind="info", seconds=3.0):
        self.toast_text = text
        self.toast_kind = kind
        self.toast_until = time.monotonic() + seconds

    # -- the timeline: items with the camera tile sitting at insert_at

    def sequence(self):
        seq = [("item", i) for i in range(len(self.project.items))]
        seq.insert(self.insert_at, ("cam", None))
        return seq

    def cursor_pos(self):
        if self.cursor is None:
            return self.insert_at
        return self.cursor + (1 if self.cursor >= self.insert_at else 0)

    def set_cursor_pos(self, pos):
        seq = self.sequence()
        kind, i = seq[clamp(pos, 0, len(seq) - 1)]
        self.cursor = None if kind == "cam" else i

    def go_live(self, at=None):
        self.insert_at = clamp(len(self.project.items) if at is None else at, 0, len(self.project.items))
        self.cursor = None

    def ui_state(self):
        return {"cursor": self.cursor, "insert_at": self.insert_at}

    def restore_ui(self, ui):
        n = len(self.project.items)
        ui = ui or {}
        self.insert_at = clamp(ui.get("insert_at", n), 0, n)
        cursor = ui.get("cursor")
        self.cursor = cursor if cursor is not None and 0 <= cursor < n else None

    def picture_number(self, i):
        """Pictures are counted without the title cards."""
        return sum(1 for it in self.project.items[:i + 1] if it["kind"] == "frame")

    def describe(self, i):
        if self.project.items[i]["kind"] == "title":
            return "the title card"
        return f"picture {self.picture_number(i)}"

    def target_index(self):
        """The item an edit applies to: the picked one, or the one just before the camera."""
        if self.cursor is not None:
            return self.cursor
        return self.insert_at - 1 if self.insert_at > 0 else None

    def move_tile(self, from_pos, to_pos):
        """Move a tile (a picture, a title, or the camera itself) to a new spot."""
        seq = self.sequence()
        tile = seq.pop(from_pos)
        seq.insert(clamp(to_pos, 0, len(seq)), tile)
        items = self.project.items
        moved = items[tile[1]] if tile[0] == "item" else None
        new_items = [items[i] for kind, i in seq if kind == "item"]
        new_insert = next(p for p, (kind, _) in enumerate(seq) if kind == "cam")
        if [it["id"] for it in new_items] != [it["id"] for it in items]:
            with self.project.change(self.ui_state()):
                items[:] = new_items
        self.insert_at = new_insert
        self.cursor = None if moved is None else next(i for i, it in enumerate(items) if it is moved)

    def ghost_item(self):
        """The ghost is always the last picture before the camera tile, worked out
        fresh every time - so nothing (playing, editing, the camera dropping out)
        can make it disappear."""
        items = self.project.items
        for i in range(min(self.insert_at, len(items)) - 1, -1, -1):
            if items[i]["kind"] == "frame":
                return items[i]
        return None

    def ghost_view(self):
        item = self.ghost_item()
        if item is None:
            return None
        key = item["file"]
        if self._ghost[0] != key:
            img = cv2.imread(str(self.project.file_path(key)))
            self._ghost = (key, fit_image(img, (VIEW_W, VIEW_H)) if img is not None else None)
        return self._ghost[1]

    def fit_view(self, item):
        key = (item_key(item), self.project.size)
        img = self.fitted.get(key)
        if img is None:
            img = fit_image(self.renderer.image(item), (VIEW_W, VIEW_H))
            self.fitted[key] = img
            if len(self.fitted) > 16:
                self.fitted.popitem(last=False)
        return img

    def thumb(self, item):
        key = item_key(item)
        img = self.thumbs.get(key)
        if img is None:
            if item["kind"] == "frame":
                src = cv2.imread(str(self.project.file_path(item["file"])), cv2.IMREAD_REDUCED_COLOR_4)
                if src is None:
                    src = missing_picture((TILE_W * 2, TILE_H * 2))
            else:
                w, h = self.project.size
                src = render_title(item, (320, max(2, 320 * h // w)))
            img = fit_image(src, (TILE_W, TILE_H), C_TILE)
            self.thumbs[key] = img
        return img

    def blank(self):
        return np.full((CANVAS_H, CANVAS_W, 3), C_BG, np.uint8)

    def backdrop(self):
        return darken(self.last_canvas if self.last_canvas is not None else self.blank(), 0.3)

    def hit(self, x, y):
        for (x0, y0, x1, y1), action in reversed(self.hotspots):
            if x0 <= x <= x1 and y0 <= y <= y1:
                return action
        return None

    def clicked(self, events):
        """The action under the first click, if any."""
        for event in events:
            if event[0] == "down":
                action = self.hit(event[1], event[2])
                if action is not None:
                    return action
        return None

    # -- drawing the studio

    def view_image(self, mode):
        if mode == "review":
            return self.fit_view(self.project.items[self.cursor])
        ghost = self.ghost_view() if self.onion else None
        if self.live is None:
            img = darken(ghost, 0.5) if ghost is not None else np.full((VIEW_H, VIEW_W, 3), C_DARK, np.uint8)
            draw_icon(img, "camera", VIEW_W // 2, VIEW_H // 2 - 50, 40, C_DIM)
            put_text(img, "Looking for the camera...", (VIEW_W // 2, VIEW_H // 2 + 30), 0.9, C_TEXT, 2,
                     UI_BOLD, align="center")
            put_text(img, "Check it's plugged in and no other app is using it.",
                     (VIEW_W // 2, VIEW_H // 2 + 70), 0.6, C_DIM, 1, align="center")
            return img
        img = fit_image(self.live, (VIEW_W, VIEW_H))
        if ghost is not None:
            img = cv2.addWeighted(img, 1 - ONION_OPACITY, ghost, ONION_OPACITY, 0)
        if time.monotonic() < self.flash_until:
            img = cv2.addWeighted(img, 0.35, np.full_like(img, 255), 0.65, 0)
        return img

    def draw_studio(self, view=None, mode=None, playing=None, extra=None):
        mode = mode or ("live" if self.cursor is None else "review")
        c = self.blank()
        self.hotspots = []
        if view is None:
            view = self.view_image(mode)
        c[VIEW_Y:VIEW_Y + VIEW_H, VIEW_X:VIEW_X + VIEW_W] = view
        self.draw_top_bar(c)
        self.draw_overlays(c, mode, extra or {})
        self.draw_panel(c, mode)
        self.draw_timeline(c, mode, playing)
        self.draw_toast(c)
        return c

    def draw_top_bar(self, c):
        fill_rect(c, 0, 0, CANVAS_W, TOP_H, C_BAR)
        p = self.project
        seconds = len(build_plan(p)) / p.fps
        stats = f"{plural(p.frame_count(), 'picture')}    {seconds:.1f} seconds    {p.fps} per second"
        if p.audio_file():
            stats += "    + voice"
        stats_w = put_text(c, stats, (VIEW_W - 16, 32), 0.55, C_DIM, 1, align="right")
        put_text(c, fit_text(p.name, VIEW_W - stats_w - 60, 0.8, UI_BOLD), (16, 34), 0.8, C_TEXT, 1, UI_BOLD)
        put_text(c, "H = help", (CANVAS_W - 16, 32), 0.55, C_DIM, 1, align="right")
        self.hotspots.append(((PANEL_X, 0, CANVAS_W, TOP_H), ("cmd", "help")))

    def draw_overlays(self, c, mode, extra):
        p = self.project
        x0, y0 = VIEW_X + 14, VIEW_Y + 14
        if mode == "live":
            if self.live is not None:
                label = "LIVE CAMERA" + ("  +  GHOST" if self.onion and self.ghost_item() is not None else "")
                pill(c, label, x0, y0, C_RED)
        elif mode == "review":
            item = p.items[self.cursor]
            if item["kind"] == "title":
                label = f"TITLE CARD   {item['hold'] / p.fps:.1f} seconds"
            else:
                label = f"PICTURE {self.picture_number(self.cursor)} OF {p.frame_count()}"
                if item["hold"] > 1:
                    label += f"   (shows {item['hold']}x as long)"
            pill(c, label, x0, y0, C_YELLOW, C_DARK)
        elif mode == "preview":
            pill(c, "PLAYING  -  press any key to stop", x0, y0, C_GREEN)
        elif mode == "record":
            pill(c, f"REC  {extra.get('elapsed', 0):.1f} s  -  press any key to stop", x0, y0, C_RED)
            level = min(1.0, extra.get("level", 0.0) ** 0.5 * 1.3)
            mx, my = x0 + 4, y0 + 50
            round_rect(c, mx, my, mx + 220, my + 14, C_DARK, 7)
            if level > 0.03:
                round_rect(c, mx, my, mx + int(220 * level), my + 14, C_GREEN if level < 0.9 else C_ORANGE, 7)
            put_text(c, "microphone", (mx + 232, my + 12), 0.45, WHITE)
        elif mode == "countdown":
            pill(c, "GET READY TO TALK!", x0, y0, C_RED)
            cx, cy = VIEW_X + VIEW_W // 2, VIEW_Y + VIEW_H // 2
            cv2.circle(c, (cx, cy), 90, C_RED, -1, cv2.LINE_AA)
            put_text(c, str(extra.get("count", "")), (cx, cy), 4, WHITE, 8, UI_BOLD, align="center", valign="middle")

        right_y = y0
        if mode == "live" and not self.camera_panel and self.live is not None:
            parts = []
            for name, short in (("focus", "FOCUS"), ("exposure", "EXPOSURE"), ("white balance", "WB")):
                words, state = self.cam_setup.status(name)
                if name in self.cam_setup.settings:
                    parts.append(f"{short} {state.upper()}")
                elif state != "n/a":                        # our own brightening
                    parts.append(f"BRIGHTNESS {words.split()[-1]}")
            if parts:
                rect = pill(c, "   ".join(parts), VIEW_X + VIEW_W - 14, right_y, (40, 40, 40), WHITE, 0.45,
                            align="right")
                self.hotspots.append((rect, ("cmd", "camera")))
                right_y = rect[3] + 8
        if self.auto and mode == "live":
            left = max(0.0, self.next_auto - time.monotonic())
            pill(c, f"AUTO SNAP  {left:.1f}", VIEW_X + VIEW_W - 14, right_y, C_ORANGE, C_DARK, align="right")
        if extra.get("progress") is not None:
            fill_rect(c, VIEW_X, VIEW_Y + VIEW_H - 6, VIEW_X + int(VIEW_W * extra["progress"]), VIEW_Y + VIEW_H, C_GREEN)

        bottom = VIEW_Y + VIEW_H - 16
        goal = p.doc.get("frame_goal") or 0
        if goal > 0 and mode in ("live", "review"):
            done = p.frame_count()
            bx0, bx1, by1 = VIEW_X + 262, VIEW_X + VIEW_W - 262, bottom
            by0 = by1 - 28
            round_rect(c, bx0, by0, bx1, by1, C_DARK, 14)
            if done:
                round_rect(c, bx0, by0, bx0 + max(28, int((bx1 - bx0) * min(1.0, done / goal))), by1, C_GREEN, 14)
            text = f"Goal: {done} / {plural(goal, 'picture')}"
            if done >= goal:
                text = "Goal reached!  " + text[6:]
            put_text(c, text, ((bx0 + bx1) // 2, (by0 + by1) // 2), 0.55, WHITE, 1, UI_BOLD, align="center", valign="middle")
            bottom = by0 - 8
        hint = None
        if mode == "live" and not p.items:
            hint = "Press SPACE (or the red button) to take your first picture!"
        elif mode == "live" and self.insert_at < len(p.items):
            hint = "New pictures go where the camera tile is.  Press END to go back to the end."
        elif mode == "review":
            hint = "SPACE = add pictures after this    X = delete    + - = longer / shorter    ESC = camera"
        if hint:
            pill(c, hint, VIEW_X + VIEW_W // 2, bottom - 32, C_DARK, WHITE, 0.5, align="center")

    def draw_toast(self, c):
        if not self.toast_text or time.monotonic() >= self.toast_until:
            return
        color = {"good": (60, 140, 50), "error": (50, 50, 190)}.get(self.toast_kind, (40, 40, 40))
        y = VIEW_Y + 64
        for line in wrap_text(self.toast_text, VIEW_W - 140, 0.62)[:4]:
            y = pill(c, line, VIEW_X + VIEW_W // 2, y, color, WHITE, 0.62, align="center")[3] + 4

    def draw_button(self, c, rect, label, icon=None, key=None, color=C_BUTTON, active=False, action=None,
                    sub=None, dim=False, scale=0.58, align="left"):
        x0, y0, x1, y1 = rect
        fill = C_ACTIVE if active else color
        if dim:
            fill = tuple(int(v * 0.55) for v in fill)
        round_rect(c, x0, y0, x1, y1, fill, 10)
        text_color = WHITE if not dim else C_DIM
        tx = x0 + 12
        if icon:
            draw_icon(c, icon, x0 + 26, (y0 + y1) // 2, 12, text_color)
            tx = x0 + 48
        if key:
            put_text(c, key, (x1 - 10, y0 + 20), 0.42, text_color, 1, align="right")
        if align == "center":
            put_text(c, label, ((x0 + x1) // 2, (y0 + y1) // 2), scale, text_color, 1, UI_BOLD, "center", "middle")
        elif sub:
            put_text(c, label, (tx, y0 + 21), scale, text_color, 1, UI_BOLD)
            put_text(c, sub, (tx, y0 + 39), 0.42, (225, 225, 225) if not dim else C_DIM, 1)
        else:
            put_text(c, label, (tx, (y0 + y1) // 2), scale, text_color, 1, UI_BOLD, valign="middle")
        if action is not None:
            self.hotspots.append(((x0, y0, x1, y1), action))

    def audio_seconds(self):
        path = self.project.audio_file()
        if path is None:
            return None
        key = (str(path), path.stat().st_mtime)
        if self._audio_len[0] != key:
            try:
                with wave.open(str(path), "rb") as w:
                    self._audio_len = (key, w.getnframes() / w.getframerate())
            except Exception:             # empty or damaged file
                self._audio_len = (key, None)
        return self._audio_len[1]

    def draw_panel(self, c, mode):
        if self.camera_panel and self.cursor is not None:
            self.camera_panel = False          # picked a picture: back to the usual buttons
        if self.camera_panel and mode == "live":
            self.draw_camera_panel(c)
            return
        fill_rect(c, PANEL_X, VIEW_Y, CANVAS_W, VIEW_Y + VIEW_H, C_BAR)
        p = self.project
        busy = mode in ("preview", "record", "countdown")
        item = p.items[self.cursor] if self.cursor is not None else None
        target = self.transition_target()
        if target is None:
            tr_sub = "pick a picture first"
        else:
            tr = p.doc.get("end_transition") if target == "end" else p.items[target].get("transition")
            name = TRANSITION_NAMES[tr["type"]] if tr else "none"
            if target == "end":
                where = "at the end"
            elif target == 0:
                where = "at the start"
            elif p.items[target]["kind"] == "title":
                where = "before the title"
            else:
                where = f"before #{self.picture_number(target)}"
            tr_sub = f"{where}: {name}"
        voice = self.audio_seconds()
        buttons = [
            ("snap", "Camera here" if item else "Take picture", "add pictures after this" if item else None,
             "SPACE", "snap", C_RED, False),
            ("stop", "Stop", None, "P", "play", C_GREEN, False) if mode == "preview" else
            ("play", "Play movie", "with sound" if voice else None, "P", "play", C_GREEN, False),
            ("ghost", "Ghost", "on: shows last picture" if self.onion else "off", "O", "onion", C_BUTTON, self.onion),
            ("auto", "Auto snap", f"on: every {AUTO_CAPTURE_SECONDS:g} seconds" if self.auto else "off", "T", "auto",
             C_BUTTON, self.auto),
            ("title", "Edit title" if item and item["kind"] == "title" else "Title card",
             "change the words" if item and item["kind"] == "title" else "add some words", "C", "title", C_BUTTON, False),
            ("transition", "Transition", tr_sub, "F", "transition", C_BUTTON, False),
            ("stop", "Stop" if mode == "record" else "Cancel", None, "R", "record", C_RED, False) if mode in
            ("record", "countdown") else
            ("mic", "Record voice", f"voice track: {voice:.1f} s" if voice else "use the microphone", "R", "record",
             C_BUTTON, False),
            ("speed", "Speed", f"{p.fps} pictures a second", "V", "speed", C_BUTTON, False),
            ("movie", "Make movie", "save an MP4 file", "E", "export", C_BUTTON, False),
            ("folder", "Projects", "open or start a movie", "L", "projects", C_BUTTON, False),
        ]
        slot = VIEW_H // len(buttons)
        for k, (icon, label, sub, key, cmd, color, active) in enumerate(buttons):
            y0 = VIEW_Y + k * slot + 4
            self.draw_button(c, (PANEL_X + 10, y0, CANVAS_W - 10, y0 + slot - 7), label, icon, key, color, active,
                             ("cmd", cmd), sub, dim=busy and icon != "stop")

    def draw_timeline(self, c, mode, playing):
        p = self.project
        seq = self.sequence()
        if playing is not None:
            focus = playing + (1 if playing >= self.insert_at else 0)
        else:
            focus = self.cursor_pos()
        if focus != self._tl_focus:          # follow the cursor, but let the scroll arrows work too
            self._tl_focus = focus
            if focus < self.tl_first:
                self.tl_first = focus
            elif focus >= self.tl_first + TILES_FIT:
                self.tl_first = focus - TILES_FIT + 1
        self.tl_first = clamp(self.tl_first, 0, max(0, len(seq) - TILES_FIT))
        first = self.tl_first
        visible = seq[first:first + TILES_FIT]
        dragging = self.drag is not None and self.drag["moved"]
        drag_pos = self.drag_pos() if dragging else None
        numbers, count = [], 0
        for item in p.items:
            count += item["kind"] == "frame"
            numbers.append(count)
        for slot, (kind, i) in enumerate(visible):
            pos = first + slot
            x, y = TILES_X0 + slot * TILE_STEP, TILE_Y
            selected = pos == focus
            if kind == "cam":
                self.draw_camera_tile(c, x, y, mode)
                label = "camera"
                border = C_RED if selected else (90, 82, 80)
            else:
                item = p.items[i]
                c[y:y + TILE_H, x:x + TILE_W] = self.thumb(item)
                if item["kind"] == "title":
                    label, badge = "title", f"{item['hold'] / p.fps:.1f}s"
                else:
                    label, badge = str(numbers[i]), (f"x{item['hold']}" if item["hold"] > 1 else None)
                if badge:
                    pill(c, badge, x + TILE_W - 2, y + TILE_H - 24, C_ORANGE, C_DARK, 0.38, align="right")
                border = C_YELLOW if selected else (90, 82, 80)
                if pos == drag_pos:
                    fill_rect(c, x, y, x + TILE_W, y + TILE_H, C_BG, 0.7)
            cv2.rectangle(c, (x - 2, y - 2), (x + TILE_W + 1, y + TILE_H + 1), border, 3 if selected else 1)
            put_text(c, label, (x + TILE_W // 2, y + TILE_H + 18), 0.45, C_TEXT if selected else C_DIM, 1,
                     align="center")
            self.hotspots.append(((x, y, x + TILE_W, y + TILE_H), ("tile", pos)))
        # Transitions sit in the gaps, on top of the tiles.
        for slot, (kind, i) in enumerate(visible):
            if kind != "item":
                continue
            x = TILES_X0 + slot * TILE_STEP
            if p.items[i].get("transition"):
                self.draw_transition_pill(c, x - TILE_GAP // 2, p.items[i]["transition"], ("trans", i))
            if i == len(p.items) - 1 and p.doc.get("end_transition"):
                self.draw_transition_pill(c, x + TILE_W + TILE_GAP // 2, p.doc["end_transition"], ("trans", "end"))
        if first > 0:
            self.draw_button(c, (4, TILE_Y, 28, TILE_Y + TILE_H), "<", action=("scroll", -1), align="center")
        if first + TILES_FIT < len(seq):
            self.draw_button(c, (CANVAS_W - 28, TILE_Y, CANVAS_W - 4, TILE_Y + TILE_H), ">", action=("scroll", 1),
                             align="center")
        if dragging:
            b = self.drop_boundary(self.drag["x"])
            bx = TILES_X0 + (b - first) * TILE_STEP - TILE_GAP // 2
            cv2.line(c, (bx, TILE_Y - 8), (bx, TILE_Y + TILE_H + 8), C_YELLOW, 5)
        self.draw_edit_row(c, mode)

    def draw_camera_tile(self, c, x, y, mode):
        if self.live is not None and mode in ("live", "review"):
            c[y:y + TILE_H, x:x + TILE_W] = darken(fit_image(self.live, (TILE_W, TILE_H), C_DARK), 0.6)
        else:
            fill_rect(c, x, y, x + TILE_W, y + TILE_H, C_DARK)
        draw_icon(c, "camera", x + TILE_W // 2, y + TILE_H // 2, 18, WHITE)
        if self.live is not None:
            cv2.circle(c, (x + 12, y + 12), 5, C_RED, -1, cv2.LINE_AA)

    def draw_transition_pill(self, c, cx, tr, action):
        rect = pill(c, TRANSITION_NAMES[tr["type"]], cx, TILE_Y + 4, C_PURPLE, WHITE, 0.4, align="center")
        self.hotspots.append((rect, action))

    CAMERA_ROWS = [("focus", "F", ("[", "]")), ("exposure", "X", ("-", "+")), ("white balance", "W", None)]

    def draw_camera_panel(self, c):
        fill_rect(c, PANEL_X, VIEW_Y, CANVAS_W, VIEW_Y + VIEW_H, C_BAR)
        cam = self.cam_setup
        x0, x1 = PANEL_X + 10, CANVAS_W - 10
        y = VIEW_Y + 8
        draw_icon(c, "camera", x0 + 16, y + 20, 13, WHITE)
        put_text(c, fit_text(cam.name, x1 - x0 - 40, 0.58, UI_BOLD), (x0 + 38, y + 27), 0.58, WHITE, 1, UI_BOLD)
        y += 46
        half = (x1 - x0 - 8) // 2
        self.draw_button(c, (x0, y, x0 + half, y + 40), "Switch", None, "C", action=("camera", "switch"), scale=0.52)
        self.draw_button(c, (x1 - half, y, x1, y + 40), "Flip", None, "R", active=cam.flipped(),
                         action=("camera", "flip"), scale=0.52)
        y += 54
        self.slider_rects = {}
        for name, key, nudge in self.CAMERA_ROWS:
            words, state = cam.status(name)
            setting = cam.settings.get(name)
            put_text(c, CameraSetup.NAMES[name], (x0 + 2, y + 16), 0.56, WHITE, 1, UI_BOLD)
            color = {"locked": C_YELLOW, "locking": C_YELLOW, "auto": C_TEXT}.get(state, C_DIM)
            put_text(c, fit_text(words, 120, 0.42), (x1, y + 16), 0.42, color, 1, align="right")
            row = y + 26
            if setting is not None:
                locked = state in ("locked", "locking")
                self.draw_button(c, (x0, row, x0 + 110, row + 34), "Auto" if locked else "Lock", None, key,
                                 C_ACTIVE if locked else C_BUTTON, action=("camera", name), scale=0.5)
                fraction = setting.to_slider(setting.value) / 100
                self.draw_slider(c, name, x0 + 10, x1 - 10, row + 56, fraction, locked)
            else:
                put_text(c, "not on this camera", (x0 + 2, row + 22), 0.42, C_DIM)
            if nudge and (setting is not None or name == "exposure"):
                for k, symbol in enumerate(nudge):
                    bx = x1 - 94 + k * 50
                    self.draw_button(c, (bx, row, bx + 44, row + 34), symbol, action=("camera", symbol),
                                     align="center", scale=0.6)
            y += 122
        problem = self.cam_setup.controls.problem if self.cam_setup.controls else None
        if problem and not cam.settings:
            for k, line in enumerate(wrap_text(problem, x1 - x0, 0.42, UI_FONT)[:2]):
                put_text(c, line, (x0 + 2, y + 4 + k * 18), 0.42, C_DIM)
        self.draw_button(c, (x0, VIEW_Y + VIEW_H - 52, x1, VIEW_Y + VIEW_H - 8), "Done", "check", "K", C_GREEN,
                         action=("cmd", "camera"))

    def draw_slider(self, c, name, x0, x1, y, fraction, locked):
        knob = int(x0 + (x1 - x0) * clamp(fraction, 0.0, 1.0))
        round_rect(c, x0, y - 4, x1, y + 4, C_DARK, 4)
        round_rect(c, x0, y - 4, max(x0 + 8, knob), y + 4, C_YELLOW if locked else C_DIM, 4)
        cv2.circle(c, (knob, y), 10, WHITE, -1, cv2.LINE_AA)
        self.slider_rects[name] = (x0, x1)
        self.hotspots.append(((x0 - 10, y - 16, x1 + 10, y + 16), ("slider", name)))

    EDIT_BUTTONS = [("< Prev", "prev"), ("Next >", "next"), ("Delete X", "delete"), ("Copy D", "duplicate"),
                    ("Longer +", "longer"), ("Shorter -", "shorter"), ("Move [", "move_left"),
                    ("Move ]", "move_right"), ("Undo Z", "undo"), ("Redo Y", "redo"), ("Goal N", "goal"),
                    ("Save S", "save"), ("Camera K", "camera")]

    def draw_edit_row(self, c, mode):
        busy = mode in ("preview", "record", "countdown")
        n, gap = len(self.EDIT_BUTTONS), 6
        width = (CANVAS_W - 20 - gap * (n - 1)) / n
        for k, (label, cmd) in enumerate(self.EDIT_BUTTONS):
            x0 = int(10 + k * (width + gap))
            self.draw_button(c, (x0, EDIT_Y, int(x0 + width), EDIT_Y + 32), label, action=("cmd", cmd),
                             scale=0.45, align="center", dim=busy)

    def tile_id(self, pos):
        kind, i = self.sequence()[pos]
        return "cam" if kind == "cam" else self.project.items[i]["id"]

    def drag_pos(self):
        """Where the tile being dragged is now (things may have moved since)."""
        if self.drag is None:
            return None
        for pos in range(len(self.sequence())):
            if self.tile_id(pos) == self.drag["tile"]:
                return pos
        return None

    def drop_boundary(self, x):
        b = int(round((x - TILES_X0 + TILE_GAP / 2) / TILE_STEP))
        return clamp(self.tl_first + b, 0, len(self.sequence()))

    # -- keys and clicks

    def on_key(self, code):
        name = key_name(code)
        if name is None:
            return
        key = name if len(name) > 1 else name.lower()
        if self.camera_panel and key in CAMERA_KEYS:
            self.run_camera(CAMERA_KEYS[key])
            return
        command = KEY_COMMANDS.get(key)
        if command:
            self.run_command(command)

    def on_mouse(self, event):
        kind, x, y = event[0], event[1], event[2]
        if kind == "down":
            self.drag = None              # in case a button-up got lost
            self.slider_drag = None
            action = self.hit(x, y)
            if action is None:
                return
            if action[0] == "slider":
                self.slider_drag = action[1]
                self.slide(action[1], x)
            elif action[0] == "camera":
                self.run_camera(action[1])
            elif action[0] == "tile":
                self.set_cursor_pos(action[1])
                self.drag = {"tile": self.tile_id(action[1]), "x0": x, "x": x, "moved": False}
            elif action[0] == "cmd":
                self.run_command(action[1])
            elif action[0] == "trans":
                self.cycle_transition(action[1])
            elif action[0] == "scroll":
                self.tl_first += action[1] * (TILES_FIT - 1)
        elif kind == "drag" and self.slider_drag is not None and self.camera_panel:
            self.slide(self.slider_drag, x)
        elif kind == "up" and self.slider_drag is not None:
            self.slider_drag = None
        elif kind == "drag" and self.drag is not None:
            self.drag["x"] = x
            if abs(x - self.drag["x0"]) > 12:
                self.drag["moved"] = True
        elif kind == "up" and self.drag is not None:
            pos = self.drag_pos()
            moved, self.drag = self.drag["moved"], None
            if moved and pos is not None:
                b = self.drop_boundary(x)
                self.move_tile(pos, b if b <= pos else b - 1)
        elif kind == "wheel":
            self.tl_first -= event[3]

    def run_command(self, name):
        getattr(self, "cmd_" + name)()

    def slide(self, name, x):
        x0, x1 = self.slider_rects.get(name, (0, 1))
        problem = self.cam_setup.set_slider(name, (x - x0) / max(1, x1 - x0))
        if problem:
            self.toast(problem, "error")

    def run_camera(self, action):
        """The camera panel's buttons and keys."""
        cam = self.cam_setup
        if action == "close":
            self.camera_panel = False
            self.slider_drag = None
        elif action == "switch":
            if self.camera.switch():
                cam.sync(self.camera)
                summary = cam.controls.summary() if cam.controls else ""
                self.toast(f"Camera {self.camera.index + 1}: {cam.name} ({summary})")
            else:
                self.toast("No other camera found.")
        elif action == "flip":
            cam.toggle_flip()
            self.toast(f"Flipped the picture from {cam.name}. (It'll remember.)")
        elif action in ("focus", "exposure", "white balance"):
            self.toast(cam.toggle(action))
        elif action in ("[", "]"):
            self.toast(cam.nudge_focus(1 if action == "]" else -1))
        elif action in ("-", "+"):
            self.toast(cam.brighter(1 if action == "+" else -1))

    def auto_tick(self):
        if not self.auto:
            return
        now = time.monotonic()
        if self.cursor is not None or self.live is None or self.drag is not None:
            self.next_auto = now + AUTO_CAPTURE_SECONDS       # paused while looking at a picture
        elif now >= self.next_auto:
            self.cmd_snap()
            self.next_auto = now + AUTO_CAPTURE_SECONDS

    # -- commands (keys and buttons both end up here)

    def cmd_snap(self):
        if self.cursor is not None:
            self.go_live(self.cursor + 1)
            self.toast("Camera on! New pictures will go here.")
            return
        if self.live is None:
            self.toast("The camera isn't ready yet.", "error")
            return
        frame = self.live
        with self.project.change(self.ui_state()):
            item = self.project.add_frame(frame, self.insert_at)
        self.insert_at += 1
        self._ghost = (item["file"], fit_image(frame, (VIEW_W, VIEW_H)))
        self.thumbs[item_key(item)] = fit_image(frame, (TILE_W, TILE_H), C_TILE)
        self.flash_until = time.monotonic() + 0.12
        self.speaker.click()
        count, goal = self.project.frame_count(), self.project.doc.get("frame_goal") or 0
        print(f"Captured picture {count}" + (f" / {goal}" if goal else ""))
        if goal and count == goal:
            self.toast(f"Goal reached! {plural(goal, 'picture')}!", "good", 4)

    def cmd_enter(self):
        if self.cursor is not None and self.project.items[self.cursor]["kind"] == "title":
            self.edit_title(self.cursor)
        else:
            self.cmd_snap()

    def cmd_prev(self):
        self.set_cursor_pos(self.cursor_pos() - 1)

    def cmd_next(self):
        self.set_cursor_pos(self.cursor_pos() + 1)

    def cmd_first(self):
        self.set_cursor_pos(0)

    def cmd_last(self):
        self.go_live()

    def cmd_escape(self):
        if self.cursor is not None:
            self.cursor = None
        else:
            self.cmd_quit()

    def cmd_quit(self):
        now = time.monotonic()
        if now < self.quit_armed_until:
            self.running = False
            return
        self.quit_armed_until = now + 3
        self.toast("Press Q again to quit. (Your movie is already saved.)")

    def cmd_delete(self):
        i = self.target_index()
        if i is None:
            self.toast("Nothing to delete.")
            return
        items = self.project.items
        reviewing = self.cursor is not None
        what = self.describe(i)
        with self.project.change(self.ui_state()):
            items.pop(i)
        if i < self.insert_at:
            self.insert_at -= 1
        if reviewing:
            self.cursor = min(i, len(items) - 1) if items else None
        self.toast(f"Deleted {what}. Press Z to undo.")

    def cmd_duplicate(self):
        i = self.target_index()
        if i is None:
            self.toast("Pick a picture to copy.")
            return
        items = self.project.items
        copy = json.loads(json.dumps(items[i]))
        copy["id"] = new_id()
        copy["transition"] = None
        with self.project.change(self.ui_state()):
            items.insert(i + 1, copy)
        if i < self.insert_at:
            self.insert_at += 1
        if self.cursor is not None:
            self.cursor = i + 1
        self.toast("Copied!")

    def cmd_longer(self):
        self.change_hold(1)

    def cmd_shorter(self):
        self.change_hold(-1)

    def change_hold(self, delta):
        i = self.target_index()
        if i is None:
            self.toast("Pick a picture first.")
            return
        item, fps = self.project.items[i], self.project.fps
        if item["kind"] == "title":
            step = max(1, fps // 2)
            new = clamp(item["hold"] + delta * step, step, fps * 30)
        else:
            new = clamp(item["hold"] + delta, 1, MAX_HOLD)
        if new == item["hold"]:
            self.toast("It can't go any " + ("longer." if delta > 0 else "shorter."))
            return
        with self.project.change(self.ui_state()):
            item["hold"] = new
        if item["kind"] == "title":
            self.toast(f"The title card shows for {new / fps:.1f} seconds.")
        elif new == 1:
            self.toast(f"Picture {self.picture_number(i)} is back to normal.")
        else:
            self.toast(f"Picture {self.picture_number(i)} now stays {new}x as long.")

    def cmd_move_left(self):
        self.move_picked(-1)

    def cmd_move_right(self):
        self.move_picked(1)

    def move_picked(self, delta):
        if self.cursor is None:
            self.toast("Pick a picture first (the < and > keys).")
            return
        pos = self.cursor_pos()
        if 0 <= pos + delta < len(self.sequence()):
            self.move_tile(pos, pos + delta)

    def cmd_undo(self):
        ok, ui = self.project.undo(self.ui_state())
        if ok:
            self.restore_ui(ui)
        self.toast("Undone." if ok else "Nothing to undo.")

    def cmd_redo(self):
        ok, ui = self.project.redo(self.ui_state())
        if ok:
            self.restore_ui(ui)
        self.toast("Redone." if ok else "Nothing to redo.")

    def cmd_onion(self):
        self.onion = not self.onion
        self.project.set_option("onion_skin", self.onion)
        self.toast("Ghost ON: you'll see the last picture faintly." if self.onion else "Ghost OFF.")

    def cmd_auto(self):
        self.auto = not self.auto
        if self.auto:
            self.cursor = None
            self.next_auto = time.monotonic() + AUTO_CAPTURE_SECONDS
            self.toast(f"Auto snap ON: a picture every {AUTO_CAPTURE_SECONDS:g} seconds.")
        else:
            self.toast("Auto snap OFF.")

    def cmd_title(self):
        if self.cursor is not None and self.project.items[self.cursor]["kind"] == "title":
            self.edit_title(self.cursor)
        else:
            self.edit_title(None)

    def transition_target(self):
        """Where F puts a transition: before the picked item, or at the very end."""
        items = self.project.items
        if self.cursor is not None:
            return self.cursor
        if items and self.insert_at >= len(items):
            return "end"
        return None

    def cmd_transition(self):
        target = self.transition_target()
        if target is None:
            self.toast("Pick a picture (< >) to put a transition before it." if self.project.items
                       else "Take some pictures first!")
            return
        self.cycle_transition(target)

    def cycle_transition(self, target):
        p = self.project
        with p.change(self.ui_state()):
            if target == "end":
                p.doc["end_transition"] = tr = next_transition(p.doc.get("end_transition"), p.fps)
            else:
                item = p.items[target]
                item["transition"] = tr = next_transition(item.get("transition"), p.fps)
        name = TRANSITION_NAMES[tr["type"]] if tr else None
        if target == "end":
            msg = f"The movie ends with: {name}." if tr else "No transition at the end."
        elif target == 0:
            msg = f"The movie starts with: {name}." if tr else "No transition at the start."
        else:
            msg = f"{name} before {self.describe(target)}." if tr else f"No transition before {self.describe(target)}."
        self.toast(msg + ("  (Press F again for a different one.)" if tr else ""))

    def cmd_speed(self):
        p = self.project
        faster = [s for s in SPEEDS if s > p.fps]
        new = faster[0] if faster else SPEEDS[0]
        with p.change(self.ui_state()):
            p.doc["fps"] = new
        self.toast(f"Speed: {new} pictures a second." + ("  (Press V again to go faster.)" if faster[1:] else ""))

    def cmd_goal(self):
        p = self.project
        current = p.doc.get("frame_goal") or 0
        text = self.prompt("Picture goal", str(current) if current else "",
                           hint=f"How many pictures? ({p.fps * 5} pictures = 5 seconds of movie.)  Empty = no goal.",
                           digits=True, max_len=5)
        self.after_modal()
        if text is None:
            return
        goal = int(text) if text.isdigit() else 0
        p.set_option("frame_goal", goal)
        if goal:
            self.toast(f"Goal: {plural(goal, 'picture')}. {max(goal - p.frame_count(), 0)} to go!", "good")
        else:
            self.toast("Goal cleared.")

    def cmd_save(self):
        self.project.save(backup=True)
        print(f"Saved {self.project.path}")
        self.toast("Saved!  (It also saves by itself after every change.)", "good")

    def cmd_camera(self):
        self.camera_panel = not self.camera_panel
        self.slider_drag = None
        if self.camera_panel:
            self.cursor = None            # the panel is for the live camera

    def cmd_help(self):
        c = self.backdrop()
        round_rect(c, 30, 14, CANVAS_W - 30, CANVAS_H - 14, C_BG, 18)
        put_text(c, "How to use Kid Stop Motion", (CANVAS_W // 2, 60), 1.0, WHITE, 2, UI_BOLD, align="center")
        half = (len(HELP_LINES) + 1) // 2
        for k, (key, what) in enumerate(HELP_LINES):
            col, row = divmod(k, half)
            x, y = 60 + col * 610, 118 + row * 54
            round_rect(c, x, y - 30, x + 170, y + 8, C_BUTTON, 8)
            put_text(c, key, (x + 85, y - 11), 0.55, WHITE, 1, UI_BOLD, align="center", valign="middle")
            put_text(c, what, (x + 186, y - 4), 0.6, C_TEXT)
        put_text(c, "Everything saves by itself.   Press any key to go back.", (CANVAS_W // 2, CANVAS_H - 36), 0.6,
                 C_DIM, align="center")
        while self.running:
            key, events = self.tick(c, 30)
            if key != -1 or any(e[0] == "down" for e in events):
                break
        self.after_modal()

    def cmd_play(self):
        self.run_preview()

    def cmd_record(self):
        self.run_record()

    def cmd_export(self):
        self.run_export()

    def cmd_projects(self):
        self.pick_project()

    # -- playing the movie

    def load_audio(self):
        path = self.project.audio_file()
        if path is None or sound_module() is None:
            return None
        try:
            return read_wav(path)
        except Exception as e:            # empty or damaged file
            print(f"Couldn't read {path}: {e!r}")
            self.toast(f"Couldn't read the voice track: {e}", "error")
            return None

    def start_playback(self, audio):
        if audio is not None:
            self.speaker.play(*audio)
        return time.monotonic()

    def run_preview(self):
        p = self.project
        plan = build_plan(p)
        if not plan:
            self.toast("Take some pictures first!")
            return
        audio = self.load_audio()
        shown, view = -1, None
        start = self.start_playback(audio)
        try:
            while self.running:
                idx = int((time.monotonic() - start) * p.fps)
                if idx >= len(plan):          # go round again, like a flip book
                    self.speaker.stop()
                    start = self.start_playback(audio)
                    idx = 0
                if idx != shown:
                    view = fit_image(self.renderer.render(plan[idx]), (VIEW_W, VIEW_H))
                    shown = idx
                c = self.draw_studio(view, "preview", plan_item(plan[idx]), {"progress": (idx + 1) / len(plan)})
                key, events = self.tick(c, 5)
                if key != -1 or any(e[0] == "down" for e in events):
                    break
        finally:
            self.speaker.stop()
        self.after_modal()

    # -- recording a voice track

    def run_record(self):
        problem = sound_problem()
        if problem:
            print(problem)
            self.toast(problem, "error", 8)
            return
        p = self.project
        if p.audio_file():
            choice = self.ask_choice("You already have a voice track.",
                                     [("r", "Record new (R)"), ("x", "Delete it (X)"), ("esc", "Keep it (ESC)")])
            if choice == "x":
                with p.change(self.ui_state()):
                    p.doc["audio"] = None
                self.toast("Voice track deleted. Press Z to undo.")
            if choice != "r":
                self.after_modal()
                return
        plan = build_plan(p)
        first_view = fit_image(self.renderer.render(plan[0]), (VIEW_W, VIEW_H)) if plan else None
        start = time.monotonic()
        while self.running:               # 3, 2, 1...
            left = self.countdown_seconds - (time.monotonic() - start)
            if left <= 0:
                break
            if first_view is None:
                self.read_camera()
            view = first_view if first_view is not None else self.view_image("live")
            c = self.draw_studio(view, "countdown", extra={"count": int(np.ceil(left))})
            key, events = self.tick(c, 20)
            if key != -1 or any(e[0] == "down" for e in events):
                self.toast("Recording cancelled.")
                self.after_modal()
                return
        recorder = self.recorder_factory()
        try:
            recorder.start()
        except Exception as e:
            print(f"Couldn't start the microphone: {e}")
            self.toast(f"Couldn't start the microphone: {e}", "error", 6)
            return
        start, shown, view = time.monotonic(), -1, None
        try:
            while self.running:
                elapsed = time.monotonic() - start
                extra = {"elapsed": elapsed, "level": recorder.level}
                playing = None
                if plan:                  # play the movie (silently) while you talk
                    idx = int(elapsed * p.fps)
                    if idx >= len(plan):
                        break
                    if idx != shown:
                        view = fit_image(self.renderer.render(plan[idx]), (VIEW_W, VIEW_H))
                        shown = idx
                    playing = plan_item(plan[idx])
                    extra["progress"] = (idx + 1) / len(plan)
                else:                     # no pictures yet: record over the live camera
                    if elapsed > MAX_RECORD_SECONDS:
                        break
                    self.read_camera()
                    view = self.view_image("live")
                c = self.draw_studio(view, "record", playing, extra)
                key, events = self.tick(c, 5)
                if key != -1 or any(e[0] == "down" for e in events):
                    break
        finally:
            samples, rate = recorder.stop()
        self.after_modal()
        if len(samples) == 0 or not rate:
            self.toast("Nothing was recorded.", "error")
            return
        peak = int(np.abs(samples.astype(np.int32)).max())
        if peak == 0:
            msg = ("The recording was completely silent. Is the microphone allowed? On a Mac: "
                   "System Settings > Privacy & Security > Microphone > Terminal.")
            print(msg)
            self.toast(msg, "error", 10)
            return
        with p.change(self.ui_state()):
            p.set_audio(samples, rate)
        seconds = len(samples) / rate
        quiet = "  It was very quiet - try talking closer to the computer." if peak < 1000 else ""
        print(f"Recorded {seconds:.1f} s of voice: {p.audio_file()}")
        self.toast(f"Voice recorded ({seconds:.1f} s)! Press P to hear it with the movie.{quiet}", "good", 6)

    # -- title cards

    def edit_title(self, index=None):
        p = self.project
        item = p.items[index] if index is not None else None
        lines = list(item["lines"]) if item and item["lines"] else [""]
        style = item.get("style", 0) if item else self.last_style
        line = len(lines) - 1
        while self.running:
            shown = list(lines)
            shown[line] += "_"
            view = fit_image(render_title({"lines": shown, "style": style}, p.size), (VIEW_W, VIEW_H))
            key, events = self.tick(self.draw_title_editor(view, style, item is None), 30)
            action = self.clicked(events)
            name = self.typed(key)
            if action is not None and action[0] == "style":
                style = action[1]
            elif action == ("cmd", "done") or name == "enter":
                break
            elif action == ("cmd", "cancel") or name == "esc":
                self.after_modal()
                return
            elif name in ("backspace", "delete"):
                if lines[line]:
                    lines[line] = lines[line][:-1]
                elif line > 0:
                    del lines[line]
                    line -= 1
            elif name in ("tab", "down"):
                if line < MAX_TITLE_LINES - 1:
                    line += 1
                    if line == len(lines):
                        lines.append("")
            elif name == "up":
                line = max(0, line - 1)
            elif name in ("left", "right"):
                style = (style + (1 if name == "right" else -1)) % len(TITLE_STYLES)
            elif name == "space" or (name is not None and len(name) == 1):
                if len(lines[line]) < MAX_TITLE_CHARS:
                    lines[line] += " " if name == "space" else name
        else:
            return
        lines = [text.strip() for text in lines]
        while len(lines) > 1 and not lines[-1]:
            lines.pop()
        self.last_style = style
        with p.change(self.ui_state()):
            if item is not None:
                item["lines"], item["style"] = lines, style
            else:
                where = self.cursor + 1 if self.cursor is not None else self.insert_at
                p.items.insert(where, title_item(lines, style, fps=p.fps))
        if item is None:
            if where <= self.insert_at:
                self.insert_at += 1
            if self.cursor is not None:
                self.cursor = where
            self.toast("Title card added! Pick it and press + or - to change how long it shows.", "good", 5)
        else:
            self.toast("Title card changed.", "good")
        self.after_modal()

    def draw_title_editor(self, view, style, is_new):
        c = self.blank()
        self.hotspots = []
        fill_rect(c, 0, 0, CANVAS_W, TOP_H, C_BAR)
        put_text(c, "New title card" if is_new else "Edit title card", (16, 34), 0.8, C_TEXT, 1, UI_BOLD)
        put_text(c, "Type your words!", (VIEW_W - 16, 32), 0.6, C_DIM, align="right")
        c[VIEW_Y:VIEW_Y + VIEW_H, VIEW_X:VIEW_X + VIEW_W] = view
        fill_rect(c, PANEL_X, VIEW_Y, CANVAS_W, VIEW_Y + VIEW_H, C_BAR)
        put_text(c, "Colors  (< >)", (PANEL_X + 16, VIEW_Y + 30), 0.6, C_TEXT, 1, UI_BOLD)
        if not hasattr(self, "_swatches"):
            self._swatches = [render_title({"lines": ["Aa"], "style": k}, (108, 61)) for k in range(len(TITLE_STYLES))]
        for k, swatch in enumerate(self._swatches):
            x0, y0 = PANEL_X + 12 + (k % 2) * 120, VIEW_Y + 46 + (k // 2) * 84
            c[y0:y0 + 61, x0:x0 + 108] = swatch
            put_text(c, TITLE_STYLES[k]["name"], (x0 + 54, y0 + 77), 0.42, C_TEXT if k == style else C_DIM,
                     align="center")
            if k == style:
                cv2.rectangle(c, (x0 - 3, y0 - 3), (x0 + 110, y0 + 63), C_YELLOW, 3)
            self.hotspots.append(((x0, y0, x0 + 108, y0 + 61), ("style", k)))
        self.draw_button(c, (PANEL_X + 12, VIEW_Y + VIEW_H - 122, CANVAS_W - 12, VIEW_Y + VIEW_H - 70), "Done",
                         "check", "ENTER", C_GREEN, action=("cmd", "done"))
        self.draw_button(c, (PANEL_X + 12, VIEW_Y + VIEW_H - 60, CANVAS_W - 12, VIEW_Y + VIEW_H - 8), "Cancel",
                         "cross", "ESC", action=("cmd", "cancel"))
        tips = [("Type your words. The first line is big, the next lines are smaller.", 0.6, C_TEXT),
                ("TAB or DOWN = next line     UP = line above     LEFT / RIGHT = colors     ENTER = done", 0.6, C_TEXT),
                ("Afterwards, pick the title card and press + or - to make it stay longer or shorter.", 0.5, C_DIM)]
        for k, (tip, scale, color) in enumerate(tips):
            put_text(c, tip, (CANVAS_W // 2, TL_Y + 42 + k * 36), scale, color, 1, align="center")
        return c

    # -- little dialogs

    def prompt(self, title, text="", hint="", digits=False, max_len=30):
        """Ask for a bit of typing. The starting text is replaced as soon as you type."""
        bg = self.backdrop()
        fresh = bool(text)
        while self.running:
            c = bg.copy()
            self.hotspots = []
            bx0, by0, bx1, by1 = 240, 220, CANVAS_W - 240, 500
            round_rect(c, bx0, by0, bx1, by1, C_BAR, 16)
            put_text(c, title, (CANVAS_W // 2, by0 + 52), 0.95, WHITE, 2, UI_BOLD, align="center")
            fx0, fy0, fx1, fy1 = bx0 + 40, by0 + 80, bx1 - 40, by0 + 140
            round_rect(c, fx0, fy0, fx1, fy1, (250, 250, 250), 10)
            if fresh:
                tw = cv2.getTextSize(ascii_text(text), UI_BOLD, 0.9, 2)[0][0]
                fill_rect(c, fx0 + 10, fy0 + 12, fx0 + 22 + tw, fy1 - 12, (250, 215, 170))
            put_text(c, text + ("" if fresh else "_"), (fx0 + 16, (fy0 + fy1) // 2), 0.9, C_DARK, 2, UI_BOLD,
                     valign="middle")
            if hint:
                put_text(c, hint, (CANVAS_W // 2, by0 + 175), 0.5, C_DIM, align="center")
            mid = CANVAS_W // 2
            self.draw_button(c, (mid - 200, by1 - 72, mid - 10, by1 - 26), "OK  (ENTER)", color=C_GREEN,
                             action=("ok",), align="center")
            self.draw_button(c, (mid + 10, by1 - 72, mid + 200, by1 - 26), "Cancel  (ESC)", action=("cancel",),
                             align="center")
            key, events = self.tick(c, 30)
            action = self.clicked(events)
            name = self.typed(key)
            if action == ("ok",) or name == "enter":
                return text.strip()
            if action == ("cancel",) or name == "esc":
                return None
            if name in ("backspace", "delete"):
                text = "" if fresh else text[:-1]
                fresh = False
                continue
            ch = " " if name == "space" else name if name is not None and len(name) == 1 else None
            if ch is None or (digits and not ch.isdigit()):
                continue
            if fresh:
                text, fresh = "", False
            if len(text) < max_len:
                text += ch
        return None

    def ask_choice(self, message, options, detail=None):
        """options: [(key name, label)]. Returns the chosen key name, or None."""
        c = self.backdrop()
        self.hotspots = []
        bx0, bx1 = 220, CANVAS_W - 220
        lines = wrap_text(message, bx1 - bx0 - 80, 0.9)
        details = wrap_text(detail, bx1 - bx0 - 80, 0.55)[:3] if detail else []
        height = 58 + 42 * len(lines) + 28 * len(details) + 100
        by0 = (CANVAS_H - height) // 2
        by1 = by0 + height
        round_rect(c, bx0, by0, bx1, by1, C_BAR, 16)
        y = by0 + 58
        for line in lines:
            put_text(c, line, (CANVAS_W // 2, y), 0.9, WHITE, 2, UI_BOLD, align="center")
            y += 42
        for line in details:
            put_text(c, line, (CANVAS_W // 2, y), 0.55, C_DIM, 1, UI_BOLD, align="center")
            y += 28
        n = len(options)
        width = min(240, (bx1 - bx0 - 60 - 20 * (n - 1)) // n)
        x = CANVAS_W // 2 - (n * width + (n - 1) * 20) // 2
        for k, (key, label) in enumerate(options):
            self.draw_button(c, (x, by1 - 80, x + width, by1 - 30), label, color=C_GREEN if k == 0 else C_BUTTON,
                             action=("choice", key), align="center", scale=0.55)
            x += width + 20
        keys = [key for key, _ in options]
        while self.running:
            key, events = self.tick(c, 30)
            action = self.clicked(events)
            if action is not None and action[0] == "choice":
                return action[1]
            name = key_name(key)
            name = name.lower() if name is not None and len(name) == 1 else name
            if name in keys:
                return name
            if name == "esc":
                return None
        return None

    # -- making the MP4

    def run_export(self):
        p = self.project
        if not p.items:
            self.toast("Take some pictures first!")
            return
        out = p.folder / f"{safe_filename(p.name)} {now_stamp()}.mp4"
        bg = self.backdrop()
        state = {"last": 0.0}

        def progress(done, total):
            now = time.monotonic()
            if now - state["last"] < 0.05 and done < total:
                return True
            state["last"] = now
            c = bg.copy()
            bx0, by0, bx1 = 290, 280, CANVAS_W - 290
            round_rect(c, bx0, by0, bx1, by0 + 190, C_BAR, 16)
            put_text(c, "Making your movie...", (CANVAS_W // 2, by0 + 55), 0.95, WHITE, 2, UI_BOLD, align="center")
            round_rect(c, bx0 + 40, by0 + 85, bx1 - 40, by0 + 115, C_DARK, 15)
            round_rect(c, bx0 + 40, by0 + 85, bx0 + 40 + max(30, int((bx1 - bx0 - 80) * done / total)), by0 + 115,
                       C_GREEN, 15)
            put_text(c, f"{done * 100 // total}%     (ESC = stop)", (CANVAS_W // 2, by0 + 155), 0.6, C_DIM,
                     align="center")
            key, _ = self.tick(c, 1)
            return key_name(key) != "esc"

        print(f"Making {out} ...")
        ok, note = export_movie(p, out, progress)
        self.after_modal()
        if not ok:
            print(note)
            self.toast(note, "error", 6)
            return
        print(f"Movie saved: {out}" + (f"  ({note})" if note else ""))
        choice = self.ask_choice("Your movie is ready!", [("enter", "Show me (ENTER)"), ("esc", "Back (ESC)")],
                                 detail=f"{out.name}  in the  {p.folder.name}  folder.  {note}".strip())
        if choice == "enter":
            reveal_file(out)
        self.after_modal()

    # -- projects

    def picker_entries(self):
        entries = [{"kind": "new"}]
        projects = list_projects(self.projects_dir)
        entries += [dict(info, kind="project") for info in projects]
        imported = {info["imported_from"] for info in projects if info["imported_from"]}
        for folder in sorted(Path.cwd().glob("stopmotion_project_*"), reverse=True):
            frames = sorted(folder.glob("frame_*.png")) if folder.is_dir() else []
            if frames and str(folder.absolute()) not in imported:
                entries.append({"kind": "legacy", "path": folder, "frames": len(frames), "seconds": len(frames) / FPS,
                                "name": "Old save " + folder.name.replace("stopmotion_project_", ""),
                                "thumb": str(frames[0]), "modified": folder.stat().st_mtime, "damaged": False})
        return entries

    def picker_thumb(self, entry):
        key = (entry.get("thumb"), entry.get("modified"))
        img = self.picker_thumbs.get(key)
        if img is None:
            src = cv2.imread(entry["thumb"], cv2.IMREAD_REDUCED_COLOR_2) if entry.get("thumb") else None
            if src is None:
                src = np.full((PICK_H, PICK_W, 3), C_TILE, np.uint8)
                draw_icon(src, "movie", PICK_W // 2, PICK_H // 2, 26, C_DIM)
            img = fit_image(src, (PICK_W, PICK_H), C_TILE)
            self.picker_thumbs[key] = img
        return img

    def draw_picker(self, entries, sel, first_row, startup):
        c = self.blank()
        self.hotspots = []
        put_text(c, "Kid Stop Motion", (PICK_X0, 58), 1.3, WHITE, 2, UI_BOLD)
        put_text(c, "Pick a movie to keep working on, or start a new one.", (PICK_X0, 92), 0.6, C_DIM)
        start = first_row * PICK_COLS
        for k, entry in enumerate(entries[start:start + PICK_COLS * PICK_ROWS]):
            idx = start + k
            x, y = PICK_X0 + (k % PICK_COLS) * PICK_STEP_X, PICK_Y0 + (k // PICK_COLS) * PICK_STEP_Y
            rect = (x - 6, y - 6, x + PICK_W + 6, y + PICK_H + 56)
            round_rect(c, *rect, C_BAR, 12)
            if idx == sel:
                round_rect(c, *rect, C_YELLOW, 12, 3)
            if entry["kind"] == "new":
                round_rect(c, x, y, x + PICK_W, y + PICK_H, C_GREEN, 8)
                draw_icon(c, "plus", x + PICK_W // 2, y + PICK_H // 2, 30, WHITE)
                put_text(c, "New movie", (x + 4, y + PICK_H + 26), 0.65, WHITE, 1, UI_BOLD)
                put_text(c, "start a fresh one  (N)", (x + 4, y + PICK_H + 47), 0.45, C_DIM)
            else:
                c[y:y + PICK_H, x:x + PICK_W] = self.picker_thumb(entry)
                put_text(c, fit_text(entry["name"], PICK_W - 8, 0.6, UI_BOLD), (x + 4, y + PICK_H + 26), 0.6, WHITE,
                         1, UI_BOLD)
                if entry.get("damaged"):
                    info = "needs fixing - open it to repair"
                else:
                    when = datetime.fromtimestamp(entry["modified"]).strftime("%b %d")
                    info = f"{plural(entry['frames'], 'picture')}   {entry['seconds']:.1f}s   {when}"
                put_text(c, info, (x + 4, y + PICK_H + 47), 0.45, C_DIM)
                if entry["kind"] == "legacy":
                    pill(c, "OLD SAVE", x + 6, y + 6, C_ORANGE, C_DARK, 0.45)
            self.hotspots.append((rect, ("pick", idx)))
        total_rows = (len(entries) + PICK_COLS - 1) // PICK_COLS
        if first_row > 0:
            put_text(c, "more above (UP key)", (CANVAS_W - PICK_X0, 92), 0.5, C_DIM, align="right")
        if first_row + PICK_ROWS < total_rows:
            put_text(c, "more below (DOWN key)", (CANVAS_W // 2, PICK_Y0 + PICK_ROWS * PICK_STEP_Y - 2), 0.5, C_DIM,
                     align="center")
        by = CANVAS_H - 62
        self.draw_button(c, (PICK_X0, by, PICK_X0 + 290, by + 46), "Open a file...", "folder", "B",
                         action=("cmd", "browse"), sub="find a .stopmo anywhere")
        if startup:
            self.draw_button(c, (CANVAS_W - PICK_X0 - 200, by, CANVAS_W - PICK_X0, by + 46), "Quit", "cross", "ESC",
                             action=("cmd", "quit"))
        else:
            self.draw_button(c, (CANVAS_W - PICK_X0 - 200, by, CANVAS_W - PICK_X0, by + 46), "Back", "camera", "ESC",
                             action=("cmd", "back"))
        put_text(c, f"Movies are saved in  {self.projects_dir}", (CANVAS_W // 2, by + 20), 0.45, C_DIM,
                 align="center")
        put_text(c, "Arrow keys + ENTER, or click", (CANVAS_W // 2, by + 42), 0.45, C_DIM, align="center")
        self.draw_toast(c)
        return c

    def pick_project(self, startup=False):
        """The project chooser. Returns True if a project is open when it closes."""
        entries = self.picker_entries()
        if self.toast_kind != "error":
            self.toast_until = 0
        sel = 0
        if self.project is not None:
            here = {self.project.path.resolve(), self.project.folder.resolve()}
            sel = next((k for k, e in enumerate(entries) if e.get("path") and Path(e["path"]).resolve() in here), 0)
        elif len(entries) > 1 and entries[1]["kind"] == "project":
            sel = 1                     # the movie you worked on last
        first_row = 0
        while self.running:
            row = sel // PICK_COLS
            if row < first_row:
                first_row = row
            elif row >= first_row + PICK_ROWS:
                first_row = row - PICK_ROWS + 1
            c = self.draw_picker(entries, sel, first_row, startup)
            self.last_canvas = c
            key, events = self.tick(c, 30)
            action = self.clicked(events)
            name = key_name(key)
            name = name.lower() if name is not None and len(name) == 1 else name
            if action is not None:
                if action[0] == "pick":
                    sel, name = action[1], "enter"
                else:
                    name = {"browse": "b", "back": "esc", "quit": "esc"}[action[1]]
            if name == "left":
                sel = max(0, sel - 1)
            elif name == "right":
                sel = min(len(entries) - 1, sel + 1)
            elif name == "up" and sel >= PICK_COLS:
                sel -= PICK_COLS
            elif name == "down":
                sel = min(len(entries) - 1, sel + PICK_COLS)
            elif name in ("enter", "space", "n"):
                if self.activate_entry(entries[0] if name == "n" else entries[sel]):
                    self.after_modal()
                    return True
                entries = self.picker_entries()
                sel = min(sel, len(entries) - 1)
            elif name == "b":
                path = self.browse()
                if path is not None and self.open_path(path):
                    self.after_modal()
                    return True
            elif name in ("esc", "q"):
                if startup:
                    self.running = False
                self.after_modal()
                return self.project is not None and not startup
        return False

    def activate_entry(self, entry):
        if entry["kind"] == "new":
            default = default_movie_name(self.projects_dir)
            name = self.prompt("Name your new movie", default, hint="Type a name, or just press ENTER.")
            if name is None:
                return False
            try:
                project = Project.create(name or default, self.projects_dir)
                self.set_project(project)
            except (ProjectError, OSError) as e:
                self.toast(f"Couldn't make the movie folder: {e}", "error", 6)
                return False
            self.toast("New movie! Press SPACE to take your first picture.", "good", 4)
            return True
        if entry["kind"] == "legacy":
            try:
                project = Project.import_legacy(entry["path"], self.projects_dir)
                self.set_project(project)
            except (ProjectError, OSError) as e:
                self.toast(f"Couldn't bring in that old save: {e}", "error", 6)
                return False
            self.toast(f"Brought in your old save as '{project.name}'.", "good", 5)
            return True
        if self.project is not None:
            here = {self.project.path.resolve(), self.project.folder.resolve()}
            if Path(entry["path"]).resolve() in here:
                return True
        return self.open_path(entry["path"])

    def open_path(self, path):
        # Let go of our own lock for a moment, so this window can reopen (or go
        # back to a backup of) the movie it already has open.
        current = self.project
        if current is not None:
            current.unlock()
        try:
            self.set_project(Project.open(path, self.projects_dir, lock=True))
        except (ProjectError, OSError) as e:
            if current is not None and self.project is current:
                current.lock()
            self.toast(f"Couldn't open that: {e}", "error", 6)
            return False
        return True

    def browse(self):
        """The computer's own Open window, run on the side so ours keeps drawing."""
        result = {}
        worker = threading.Thread(target=lambda: result.update(value=ask_open_path(self.projects_dir)), daemon=True)
        worker.start()
        c = self.backdrop()
        round_rect(c, 240, 300, CANVAS_W - 240, 440, C_BAR, 16)
        put_text(c, "Pick your movie's .stopmo file", (CANVAS_W // 2, 360), 0.9, WHITE, 2, UI_BOLD, align="center")
        put_text(c, "in the window that just opened.", (CANVAS_W // 2, 400), 0.6, C_DIM, align="center")
        while worker.is_alive() and self.running:
            self.tick(c, 50)
        path, problem = result.get("value", (None, None))
        if problem:
            print(problem)
            self.toast(problem, "error", 6)
        return path


def fit_text(text, width, scale, font=UI_FONT):
    text = ascii_text(text)
    if cv2.getTextSize(text, font, scale, 1)[0][0] <= width:
        return text
    while len(text) > 1 and cv2.getTextSize(text + "...", font, scale, 1)[0][0] > width:
        text = text[:-1]
    return text.rstrip() + "..."


# ---------------------------------------------------------------------- main

def export_cli(path):
    try:
        project = Project.open(path)
    except (ProjectError, OSError) as e:
        print(f"Couldn't open {path}: {e}")
        return 1
    for note in project.notes:
        print(note)
    out = project.folder / f"{safe_filename(project.name)} {now_stamp()}.mp4"
    shown = [-1]

    def progress(done, total):
        tenth = done * 10 // total
        if tenth != shown[0]:
            shown[0] = tenth
            print(f"  {tenth * 10}%")
        return True

    ok, note = export_movie(project, out, progress)
    print(f"Movie saved: {out}" if ok else note)
    if ok and note:
        print(note)
    return 0 if ok else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="Kid Stop Motion - a stop motion studio for kids.")
    parser.add_argument("project", nargs="?", help="a .stopmo project file (or a project folder) to open")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="which webcam to use: 0, 1, ...")
    parser.add_argument("--projects-dir", default=str(PROJECTS_DIR), help="where new movies are saved")
    parser.add_argument("--scale", type=float, default=WINDOW_SCALE,
                        help="window size, e.g. 0.8 if the window doesn't fit your screen")
    parser.add_argument("--export", metavar="PROJECT", help="make an MP4 from a project and exit (no camera needed)")
    parser.add_argument("--install-mac-app", action="store_true",
                        help="macOS: make .stopmo files open with a double-click")
    parser.add_argument("--open-in-terminal", nargs="?", const="", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.open_in_terminal is not None:
        return open_in_terminal(args.open_in_terminal or None)
    if args.install_mac_app:
        return install_mac_app()
    if args.export:
        return export_cli(args.export)

    if not has_ffmpeg():
        print("Note: ffmpeg not found, so movies will be made without sound.")
        print("Install it with:  brew install ffmpeg   (or: sudo apt install ffmpeg)\n")
    if sound_problem():
        print(f"Note: {sound_problem()}\n")

    camera = Camera(args.camera)
    if not camera.start():
        print("Could not open the webcam. You can still edit, add titles and make movies,")
        print("and the studio will keep trying to find the camera.")
        print("On macOS, you may need to allow Terminal (or your Python app) to use the camera:")
        print("System Settings -> Privacy & Security -> Camera\n")

    display = Display(scale=clamp(args.scale, 0.3, 3.0))
    app = App(display, camera, projects_dir=args.projects_dir)
    try:
        app.run(args.project)
    except KeyboardInterrupt:
        pass
    finally:
        app.shutdown()
        camera.release()
        display.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
