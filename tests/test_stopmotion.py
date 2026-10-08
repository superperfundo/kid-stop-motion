import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import stopmotion as sm  # noqa: E402
from fakes import (FakeCamera, FakeDirectShow, FakeDisplay, FakeRecorder, FakeSpeaker,  # noqa: E402
                   FakeUVC, FakeV4L2, FakeVideoCapture, make_app, run)

RED, BLUE, GREEN = (0, 0, 255), (255, 0, 0), (0, 255, 0)


def solid(color, size=(320, 180)):
    return np.full((size[1], size[0], 3), color, np.uint8)


def view_pixel(display):
    return display.last[sm.VIEW_Y + sm.VIEW_H // 2, sm.VIEW_X + sm.VIEW_W // 2].astype(int)


class TempDirTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)


# ------------------------------------------------------------------ project

class SoundTrackTests(TempDirTest):
    """Sound on the three tracks: clips, mixing, importing, and live recordings."""

    def make_project(self, pictures=4):
        p = sm.Project.create("Movie", self.tmp)
        for k in range(pictures):
            with p.change():
                p.add_frame(solid(RED, (320, 180)), k)
        return p

    def tone(self, seconds=1.0, rate=16000):
        t = np.arange(int(seconds * rate)) / rate
        return (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16).reshape(-1, 1)

    def clip_on(self, p, track, seconds=1.0, start=0.0, name="tone"):
        rel = p.save_recording(self.tone(seconds), 16000, stem=name)
        with p.change():
            return p.add_sound(track, rel, start, seconds, name)

    def lanes_app(self, p):
        app = sm.App(FakeDisplay(), FakeCamera(), projects_dir=self.tmp)
        app.speaker = FakeSpeaker()
        app.set_project(p)
        self.addCleanup(app.shutdown)
        app.draw_studio()                  # lays out the lanes (and their buttons)
        return app

    def at(self, app, action):
        """Click the first thing with this action."""
        for rect, found in app.hotspots:
            if found == action:
                x0, y0, x1, y1 = rect
                return (x0 + x1) // 2, (y0 + y1) // 2
        self.fail(f"no {action} on the timeline")

    def click(self, app, x, y):
        app.on_mouse(("down", x, y))
        app.on_mouse(("up", x, y))

    def test_an_old_voice_track_becomes_a_clip_on_dlg(self):
        p = self.make_project()
        rel = p.save_recording(self.tone(0.5), 16000, stem="voice")
        write_legacy_voice(p, rel)
        reopened = sm.Project.open(p.path)
        (clip,) = reopened.tracks[0]["clips"]
        self.assertEqual((clip["file"], clip["start"], clip["name"]), (rel, 0.0, "voice"))
        self.assertAlmostEqual(clip["length"], 0.5, places=2)
        self.assertNotIn("audio", reopened.doc)
        self.assertEqual(sm.Project.open(p.path).tracks[0]["clips"][0]["id"], clip["id"])   # kept, not redone

    def test_the_mix_places_trims_and_scales_each_clip(self):
        data = np.arange(200000, dtype=np.float32).reshape(-1, 1).repeat(2, axis=1) / 1e6
        mix = sm.SoundMix([(10, 88200, 5, data, 0.5)])
        out = mix.render(0, 44100)
        self.assertTrue(np.all(out[:10] == 0))                       # nothing before the clip starts
        self.assertAlmostEqual(out[10, 0], 0.0, places=6)            # the fade in starts from silence
        self.assertAlmostEqual(out[30000, 0], 0.5 * data[30000 - 10 + 5, 0], places=6)
        later = mix.render(44100, 1000)                              # the same clip, further on
        self.assertAlmostEqual(later[0, 0], 0.5 * data[5 + 44100 - 10, 0], places=6)
        self.assertTrue(np.all(mix.render(100000, 10) == 0))         # and after it has ended

    def test_clips_on_a_muted_track_or_under_a_volume(self):
        p = self.make_project()
        self.clip_on(p, 0, 1.0)
        self.clip_on(p, 2, 1.0)
        with p.change():
            p.tracks[2]["muted"] = True
        self.assertEqual(len(sm.SoundMix.from_project(p).spans), 1)
        with p.change():
            p.tracks[0]["volume"] = 50
        (span,) = sm.SoundMix.from_project(p).spans
        self.assertEqual(span[4], 0.5)

    def test_a_clip_whose_file_cannot_be_read_is_named(self):
        p = self.make_project()
        bad = p.folder / "audio" / "bad.wav"
        bad.parent.mkdir(exist_ok=True)
        bad.write_bytes(b"not a sound")
        with p.change():
            p.tracks[1]["clips"].append({"id": "x", "file": "audio/bad.wav", "name": "bad", "start": 0.0,
                                         "trim": 0.0, "length": 1.0, "source": 1.0})
        mix = sm.SoundMix.from_project(p)
        self.assertEqual(mix.spans, [])
        self.assertEqual(mix.missing, ["audio/bad.wav"])

    def test_importing_any_wav_gives_a_stereo_clip_at_the_mix_rate(self):
        src = self.tmp / "door.wav"
        sm.write_wav(src, self.tone(1.5), 16000)
        p = self.make_project()
        with mock.patch.object(sm, "has_ffmpeg", return_value=False):
            clip = p.import_sound_file(src, 1, 2.0)
        self.assertEqual((clip["start"], clip["name"]), (2.0, "door"))
        self.assertAlmostEqual(clip["length"], 1.5, places=2)
        samples, rate = sm.read_wav(p.file_path(clip["file"]))
        self.assertEqual((rate, samples.shape[1]), (sm.MIX_RATE, 2))
        self.assertIs(p.tracks[1]["clips"][0], clip)

    @unittest.skipUnless(shutil.which("ffmpeg"), "needs ffmpeg")
    def test_importing_with_ffmpeg_reads_more_kinds_of_sound(self):
        src = self.tmp / "song.mp3"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        "sine=frequency=330:duration=1", str(src)], check=True)
        p = self.make_project()
        clip = p.import_sound_file(src, 2, 0.0)
        self.assertAlmostEqual(clip["length"], 1.0, delta=0.1)
        samples, rate = sm.read_wav(p.file_path(clip["file"]))
        self.assertEqual((rate, samples.shape[1]), (sm.MIX_RATE, 2))

    def test_other_formats_need_ffmpeg(self):
        src = self.tmp / "song.mp3"
        src.write_bytes(b"ID3")
        p = self.make_project()
        with mock.patch.object(sm, "has_ffmpeg", return_value=False):
            with self.assertRaises(ValueError) as caught:
                p.import_sound_file(src, 0, 0.0)
        self.assertIn("ffmpeg", str(caught.exception))
        self.assertEqual(p.tracks[0]["clips"], [])

    def test_dragging_a_clip_moves_it_and_trims_its_ends(self):
        p = self.make_project()
        clip = self.clip_on(p, 0, 1.0, start=1.0)
        app = self.lanes_app(p)
        scale = app.sound_scale()
        x = app.x_of(1.5)
        y = app.lane_top(0) + sm.LANE_H // 2
        app.on_mouse(("down", x, y))                      # the clip starts at 1.0 s; grab it at 1.5 s
        app.on_mouse(("drag", x + int(0.5 * scale), y))
        app.on_mouse(("up", x + int(0.5 * scale), y))
        self.assertAlmostEqual(p.tracks[0]["clips"][0]["start"], 1.5, delta=0.05)
        app.draw_studio()
        # Trim the right end in by 0.4 s: it now ends at 2.1 s, so the clip is 0.6 s long.
        end = app.x_of(1.5 + 1.0)
        app.on_mouse(("down", end - 3, y))
        app.on_mouse(("drag", end - 3 - int(0.4 * scale), y))
        app.on_mouse(("up", end - 3 - int(0.4 * scale), y))
        self.assertAlmostEqual(p.tracks[0]["clips"][0]["length"], 0.6, delta=0.05)
        app.draw_studio()
        # Trim the left end in by 0.2 s: the sound starts 0.2 s further into the clip, and the clip starts later.
        start = app.x_of(1.5)
        app.on_mouse(("down", start + 3, y))
        app.on_mouse(("drag", start + 3 + int(0.2 * scale), y))
        app.on_mouse(("up", start + 3 + int(0.2 * scale), y))
        moved = p.tracks[0]["clips"][0]
        self.assertAlmostEqual(moved["trim"], 0.2, delta=0.05)
        self.assertAlmostEqual(moved["start"], 1.7, delta=0.05)
        self.assertEqual(moved["id"], clip["id"])

    def test_a_clip_can_be_dragged_onto_another_track(self):
        p = self.make_project()
        self.clip_on(p, 0, 1.0, start=1.0)
        app = self.lanes_app(p)
        x, y = app.x_of(1.5), app.lane_top(0) + sm.LANE_H // 2
        app.on_mouse(("down", x, y))
        app.on_mouse(("drag", x, y + 40))
        app.on_mouse(("up", x, y + 40))
        self.assertEqual(p.tracks[0]["clips"], [])
        self.assertEqual(p.tracks[1]["clips"][0]["name"], "tone")

    def test_keys_copy_and_delete_the_picked_sound(self):
        p = self.make_project()
        self.clip_on(p, 0, 1.0, start=1.0)
        app = self.lanes_app(p)
        x, y = app.x_of(1.5), app.lane_top(0) + sm.LANE_H // 2
        self.click(app, x, y)
        app.on_key(ord("d"))                          # copies the sound, just after it
        self.assertEqual(len(p.tracks[0]["clips"]), 2)
        self.assertAlmostEqual(p.tracks[0]["clips"][1]["start"], 2.0, places=2)
        app.on_key(ord("x"))                          # deletes the copy, which is picked
        self.assertEqual(len(p.tracks[0]["clips"]), 1)
        self.assertEqual(p.frame_count(), 4)          # the pictures were left alone

    def test_track_buttons_mute_and_change_volume(self):
        p = self.make_project()
        app = self.lanes_app(p)
        self.click(app, *self.at(app, ("track_mute", 1)))
        self.assertTrue(p.tracks[1]["muted"])
        self.click(app, *self.at(app, ("track_volume", 1)))
        self.assertEqual(p.tracks[1]["volume"], 50)

    def test_import_puts_the_files_on_the_track_picked(self):
        src = self.tmp / "boom.wav"
        sm.write_wav(src, self.tone(0.5), 16000)
        p = self.make_project()
        app, display = make_app(self.tmp, ["i", "2", None])
        app.file_picker = lambda: [str(src)]
        with mock.patch("builtins.print"):
            run(app, p.path)
        (clip,) = sm.Project.open(p.path).tracks[1]["clips"]
        self.assertEqual(clip["name"], "boom")
        self.assertIn("Added 1 sound to SFX", app.toast_text)

    def test_the_file_window_failing_says_why(self):
        p = self.make_project()
        app, display = make_app(self.tmp, ["i", None])

        def no_window():
            raise RuntimeError("The file window needs tkinter.")

        app.file_picker = no_window
        with mock.patch("builtins.print"):
            run(app, p.path)
        self.assertIn("tkinter", app.toast_text)

    def test_live_recording_is_a_clip_with_its_sound(self):
        def wait():
            time.sleep(0.25)

        with mock.patch.object(sm, "sound_problem", return_value=None):
            app, display = make_app(self.tmp, ["g", wait, wait, "g", None])
            p = self.make_project(pictures=2)
            run(app, p.path)
        (item,) = [it for it in sm.Project.open(p.path).items if it["kind"] == "live"]
        self.assertGreater(item["seconds"], 0.3)
        self.assertTrue(p.file_path(item["file"]).is_file())
        self.assertTrue(p.file_path(item["audio"]).is_file())
        plan = sm.build_plan(sm.Project.open(p.path))
        self.assertEqual(sum(1 for step in plan if step[0] == "video"), round(item["seconds"] * p.fps))

    def test_a_live_clip_shows_its_own_pictures(self):
        size = (160, 120)
        writer = sm.open_video_writer(self.tmp / "clip.mp4", sm.LIVE_FPS, size)
        self.assertIsNotNone(writer)
        for k in range(30):
            writer.write(solid(BLUE, size))
        writer.release()
        clip = sm.LiveClip(self.tmp / "clip.mp4", sm.LIVE_FPS)
        picture = clip.frame_at(0.5)
        self.assertEqual(picture.shape[:2], (120, 160))
        self.assertLess(np.abs(picture.astype(int) - np.array(BLUE)).mean(), 40)
        clip.close()

    def test_a_live_clip_exports_with_its_sound(self):
        p = self.make_project(pictures=2)
        size = even_size_for(p)
        rel_video, rel_audio = f"{sm.LIVE_DIR}/live.mp4", f"{sm.AUDIO_DIR}/live sound.wav"
        p.file_path(sm.LIVE_DIR).mkdir(exist_ok=True)
        writer = sm.open_video_writer(p.file_path(rel_video), sm.LIVE_FPS, size)
        for _ in range(sm.LIVE_FPS):
            writer.write(solid(GREEN, size))
        writer.release()
        sm.write_wav(p.file_path(rel_audio), self.tone(1.0), 16000)
        with p.change():
            p.add_live(1, rel_video, rel_audio, 1.0)
        out = self.tmp / "movie.mp4"
        ok, note = sm.export_movie(p, out)
        self.assertTrue(ok, note)
        self.assertTrue(out.is_file())
        self.assertFalse(any(".partial" in f.name for f in self.tmp.iterdir()))

    def test_tidying_keeps_the_sounds_and_live_video_a_project_uses(self):
        p = self.make_project()
        used = self.clip_on(p, 0, 1.0)
        stray = p.folder / "audio" / "stray.wav"
        sm.write_wav(stray, self.tone(0.2), 16000)
        (p.folder / sm.LIVE_DIR).mkdir(exist_ok=True)
        old_video = p.folder / sm.LIVE_DIR / "old.mp4"
        old_video.write_bytes(b"x")
        p.tidy()
        self.assertTrue(p.file_path(used["file"]).is_file())
        self.assertFalse(stray.exists())
        self.assertFalse(old_video.exists())


def even_size_for(p):
    return sm.even_size(*p.size)


def write_legacy_voice(p, rel):
    """Write a project the way the last version did: one voice track, under 'audio'."""
    doc = json.loads(p.path.read_text(encoding="utf-8"))
    doc["audio"] = {"file": rel}
    p.path.write_text(json.dumps(doc), encoding="utf-8")


class ProjectFileTests(TempDirTest):
    def test_new_project_layout(self):
        p = sm.Project.create("Dragon Race", self.tmp)
        self.assertEqual(p.path, self.tmp / "Dragon Race" / "Dragon Race.stopmo")
        self.assertTrue((p.folder / "frames").is_dir())
        doc = json.loads(p.path.read_text())
        self.assertEqual(doc["format"], "kid-stop-motion")
        self.assertEqual(doc["items"], [])
        # A second movie with the same name gets its own folder.
        self.assertEqual(sm.Project.create("Dragon Race", self.tmp).folder.name, "Dragon Race (2)")

    def test_names_are_made_safe_for_files(self):
        p = sm.Project.create('Cats/Dogs: "Round 2"', self.tmp)
        self.assertEqual(p.folder.name, "Cats_Dogs_ _Round 2_")

    def test_every_change_is_on_disk_straight_away(self):
        p = sm.Project.create("Movie", self.tmp)
        with p.change():
            p.add_frame(solid(RED), 0)
            p.items.append(sm.title_item(["Hello"]))
        # Reading the file fresh (as if the computer had crashed) sees both.
        again = sm.Project.open(p.path)
        self.assertEqual([it["kind"] for it in again.items], ["frame", "title"])
        img = cv2.imread(str(again.file_path(again.items[0]["file"])))
        self.assertEqual(img.shape, (180, 320, 3))
        self.assertEqual(again.size, (320, 180))

    def test_open_accepts_the_folder_too(self):
        p = sm.Project.create("Movie", self.tmp)
        self.assertEqual(sm.Project.open(p.folder).path, p.path)

    def test_open_from_any_file_in_the_project(self):
        p = sm.Project.create("Movie", self.tmp)
        with p.change():
            p.add_frame(solid(RED), 0)
        self.assertEqual(sm.Project.open(p.file_path(p.items[0]["file"])).path, p.path)
        launcher = sm.write_launcher(p)
        self.assertEqual(sm.Project.open(launcher).path, p.path)

    def test_renaming_the_file_renames_the_movie(self):
        p = sm.Project.create("Movie", self.tmp)
        renamed = p.path.with_name("Dragon.stopmo")
        p.path.rename(renamed)
        self.assertEqual(sm.Project.open(renamed).name, "Dragon")

    def test_opening_a_backup_goes_back_to_it_and_can_be_undone(self):
        p = sm.Project.create("Movie", self.tmp)
        with p.change():
            p.add_frame(solid(RED), 0)
        p.save(backup=True)
        with p.change():
            p.add_frame(solid(BLUE), 1)
        p.save(backup=True)                    # backups: [1 picture], [2 pictures]
        backups = list((p.folder / "backups").glob("*.stopmo"))
        self.assertEqual(len(backups), 3)           # empty, 1 picture, 2 pictures (same second, all kept)
        oldest = next(f for f in backups if len(sm.Project.load_doc(f)["items"]) == 1)
        restored = sm.Project.open(oldest)
        self.assertTrue(any("Went back to the backup from 20" in note for note in restored.notes))
        self.assertEqual(restored.path, p.path)
        self.assertEqual(len(restored.items), 1)
        self.assertEqual(len(sm.Project.load_doc(oldest)["items"]), 1)       # the backup itself untouched
        self.assertFalse((p.folder / "backups" / "frames").exists())
        restored.undo()
        self.assertEqual(len(sm.Project.open(p.path).items), 2)

    def test_missing_pictures_are_kept_not_dropped(self):
        p = sm.Project.create("Movie", self.tmp)
        with p.change():
            p.add_frame(solid(RED), 0)
            p.add_frame(solid(BLUE), 1)
        moved_away = self.tmp / "elsewhere.jpg"
        shutil.move(str(p.file_path(p.items[0]["file"])), str(moved_away))
        again = sm.Project.open(p.path)
        self.assertEqual(len(again.items), 2)
        self.assertTrue(any("can't be found" in note for note in again.notes))
        self.assertEqual(len(sm.Project.load_doc(p.path)["items"]), 2)
        self.assertEqual(sm.Renderer(again).image(again.items[0]).shape, (180, 320, 3))  # a grey card
        shutil.move(str(moved_away), str(p.file_path(p.items[0]["file"])))
        self.assertEqual(len(sm.Project.open(p.path).items), 2)                # and it's back

    def test_a_missing_file_name_opens_the_project_that_is_there(self):
        p = sm.Project.create("Movie", self.tmp)
        with p.change():
            p.add_frame(solid(RED), 0)
        p.path.rename(p.path.with_name("Dragon.stopmo"))
        again = sm.Project.open(p.path)            # the old name
        self.assertEqual(again.name, "Dragon")
        self.assertEqual(len(again.items), 1)
        self.assertEqual(sorted(f.name for f in p.folder.glob("*.stopmo")), ["Dragon.stopmo"])

    def test_tidy_keeps_pictures_another_project_file_uses(self):
        p = sm.Project.create("Movie", self.tmp)
        with p.change():
            p.add_frame(solid(RED), 0)
        shutil.copy2(p.path, p.folder / "Copy.stopmo")
        with p.change():
            p.items.clear()
        self.assertEqual(p.tidy(), 0)

    def test_a_failed_backup_doesnt_stop_saving(self):
        p = sm.Project.create("Movie", self.tmp)
        with mock.patch.object(p, "_backup", side_effect=PermissionError("locked")), mock.patch("builtins.print"):
            with p.change():
                p.add_frame(solid(RED), 0)
        self.assertEqual(len(sm.Project.load_doc(p.path)["items"]), 1)

    def test_picture_files_are_never_overwritten(self):
        p = sm.Project.create("Movie", self.tmp)
        with p.change():
            first = p.add_frame(solid(RED), 0)
        p.doc["next_frame"] = 1                    # e.g. another window got there first
        with p.change():
            second = p.add_frame(solid(BLUE), 1)
        self.assertNotEqual(first["file"], second["file"])
        self.assertGreater(cv2.imread(str(p.file_path(first["file"])))[0, 0, 2], 200)

    def test_one_window_per_movie(self):
        p = sm.Project.create("Movie", self.tmp)
        a, b = sm.Project.open(p.path), sm.Project.open(p.path)
        self.assertTrue(a.lock())
        self.assertFalse(b.lock())
        a.unlock()
        self.assertTrue(b.lock())
        b.unlock()

    def test_trash_forgets_the_oldest_deleted_first(self):
        p = sm.Project.create("Movie", self.tmp)
        with p.change():
            p.add_frame(solid(RED), 0)
        old = time.time() - 10 * 24 * 3600
        os.utime(p.file_path(p.items[0]["file"]), (old, old))
        with p.change():
            p.items.clear()
        p.tidy()
        trashed = next((p.folder / "trash").iterdir())
        self.assertGreater(trashed.stat().st_mtime, time.time() - 60)

    def test_an_empty_voice_file_doesnt_crash(self):
        p = sm.Project.create("Movie", self.tmp)
        (p.folder / "audio").mkdir()
        (p.folder / "audio" / "voice.wav").write_bytes(b"")
        write_legacy_voice(p, "audio/voice.wav")
        first = sm.Project.open(p.path)            # the first open repairs it, and says so
        self.assertFalse(first.has_sound())
        self.assertIn("can't be played", " ".join(first.notes))
        app, display = make_app(self.tmp, ["p", None, "x", None])
        with mock.patch("builtins.print"), mock.patch.object(sm, "sound_module", return_value=object()):
            run(app, p.path)
        self.assertIsNotNone(display.last)

    def test_undo_brings_back_the_movie_size(self):
        p = sm.Project.create("Movie", self.tmp)
        with p.change():
            p.add_frame(solid(RED, (320, 180)), 0)
        with p.change():
            p.items.clear()
        with p.change():
            p.add_frame(solid(BLUE, (160, 120)), 0)          # a different camera
        self.assertEqual(p.size, (160, 120))
        p.undo()
        p.undo()
        self.assertEqual(p.size, (320, 180))

    def test_open_rejects_other_files(self):
        other = self.tmp / "notes.txt"
        other.write_text("hi")
        with self.assertRaises(sm.ProjectError):
            sm.Project.open(other)
        with self.assertRaises(sm.ProjectError):
            sm.Project.open(self.tmp / "nothing.stopmo")

    def test_damaged_file_falls_back_to_backup_and_keeps_later_pictures(self):
        p = sm.Project.create("Movie", self.tmp)
        with p.change():
            p.add_frame(solid(RED), 0)
        p.save(backup=True)                              # backup has 1 picture
        with p.change():
            p.add_frame(solid(BLUE), 1)                  # taken after the backup
        p.path.write_text("{ this is not json")        # e.g. disk trouble
        again = sm.Project.open(p.path)
        self.assertEqual(len(again.items), 2)
        self.assertTrue(any("backup" in note for note in again.notes))
        self.assertTrue(any(f.name.startswith("Movie.stopmo.damaged") for f in p.folder.iterdir()))
        # And it's healthy on disk again.
        self.assertIsNotNone(sm.Project.load_doc(p.path))

    def test_missing_project_file_is_rebuilt_from_pictures(self):
        p = sm.Project.create("Movie", self.tmp)
        with p.change():
            for color in (RED, GREEN, BLUE):
                p.add_frame(solid(color), len(p.items))
        shutil.rmtree(p.folder / "backups", ignore_errors=True)
        p.path.unlink()
        again = sm.Project.open(p.folder)
        self.assertEqual(len(again.items), 3)
        first = cv2.imread(str(again.file_path(again.items[0]["file"])))
        self.assertGreater(first[0, 0, 2], 200)          # still in the right order: red first
        self.assertTrue(again.path.exists())

    def test_backups_are_kept_and_pruned(self):
        p = sm.Project.create("Movie", self.tmp)
        with mock.patch.object(sm, "KEEP_BACKUPS", 3):
            for n in range(6):
                p.doc["frame_goal"] = n
                p.save(backup=True)
                os.utime(p.folder / "backups", None)
                time.sleep(0.01)
        self.assertLessEqual(len(list((p.folder / "backups").glob("*.stopmo"))), 3)

    def test_old_saves_are_imported(self):
        legacy = self.tmp / "stopmotion_project_2026-05-01_10-30-00"
        legacy.mkdir()
        for n, color in enumerate((RED, GREEN), start=1):
            cv2.imwrite(str(legacy / f"frame_{n:05d}.png"), solid(color))
        projects = self.tmp / "projects"
        p = sm.Project.open(legacy, projects)       # opening an old save imports it
        self.assertEqual(p.folder.parent, projects)
        self.assertEqual(len(p.items), 2)
        self.assertEqual(p.doc["imported_from"], str(legacy))
        self.assertTrue(p.name.startswith("Old save 2026-05-01"))

    def test_undo_and_redo(self):
        p = sm.Project.create("Movie", self.tmp)
        with p.change({"cursor": None, "insert_at": 0}):
            p.add_frame(solid(RED), 0)
        with p.change({"cursor": None, "insert_at": 1}):
            p.items[0]["hold"] = 3
        self.assertEqual(p.undo({"x": 1}), (True, {"cursor": None, "insert_at": 1}))
        self.assertEqual(p.items[0]["hold"], 1)
        p.undo()
        self.assertEqual(p.items, [])
        self.assertEqual(p.undo(), (False, None))
        p.redo()
        p.redo()
        self.assertEqual(p.items[0]["hold"], 3)
        # Undo survives on disk too.
        self.assertEqual(sm.Project.open(p.path).items[0]["hold"], 3)

    def test_unused_pictures_go_to_trash_and_come_back_if_needed(self):
        p = sm.Project.create("Movie", self.tmp)
        with p.change():
            p.add_frame(solid(RED), 0)
            p.add_frame(solid(BLUE), 1)
        p.save(backup=True)
        with p.change():
            p.items.pop()
        self.assertEqual(p.tidy(), 1)
        self.assertEqual(len(list((p.folder / "trash").iterdir())), 1)
        # If the project has to come back from the backup, the picture is fished out of the trash.
        p.path.write_text("garbage")
        again = sm.Project.open(p.path)
        self.assertEqual(len(again.items), 2)
        self.assertTrue(again.file_path(again.items[1]["file"]).exists())

    def test_repair_handles_odd_values(self):
        p = sm.Project.create("Movie", self.tmp)
        with p.change():
            p.add_frame(solid(RED), 0)
        doc = json.loads(p.path.read_text())
        doc["fps"] = 0
        doc["items"] += [
            {"kind": "frame", "file": "frames/nope.jpg"},
            {"kind": "title", "lines": "Hi", "hold": -4, "transition": {"type": "zoom"}},
            "junk",
            dict(doc["items"][0]),               # duplicate id
        ]
        doc["items"][0]["transition"] = {"type": "fade", "frames": 9}
        doc["next_frame"] = 1
        del doc["end_transition"]
        p.path.write_text(json.dumps(doc))
        again = sm.Project.open(p.path)
        self.assertEqual(again.fps, sm.FPS)
        # The missing picture is kept (it may turn up again); junk is dropped.
        self.assertEqual([it["kind"] for it in again.items], ["frame", "frame", "title", "frame"])
        self.assertEqual(again.items[2]["lines"], ["Hi"])
        self.assertGreater(again.items[2]["hold"], 0)
        self.assertIsNone(again.items[2]["transition"])
        self.assertEqual(again.items[0]["transition"]["type"], "fade")
        self.assertNotEqual(again.items[0]["id"], again.items[3]["id"])
        self.assertGreater(again.doc["next_frame"], 1)      # never overwrite a picture
        self.assertTrue(any("can't be found" in note for note in again.notes))

    def test_list_projects(self):
        a = sm.Project.create("A", self.tmp)
        with a.change():
            a.add_frame(solid(RED), 0)
        time.sleep(0.02)
        b = sm.Project.create("B", self.tmp)
        (self.tmp / "C" / "frames").mkdir(parents=True)     # lost its project file
        (self.tmp / "random").mkdir()
        found = sm.list_projects(self.tmp)
        names = [info["name"] for info in found]
        self.assertEqual(set(names), {"A", "B", "C"})
        info_a = next(info for info in found if info["name"] == "A")
        self.assertEqual(info_a["frames"], 1)
        self.assertTrue(info_a["thumb"].endswith(".jpg"))
        self.assertEqual(sm.default_movie_name(self.tmp), "Movie 1")
        self.assertIsNotNone(b)

    @unittest.skipIf(os.name == "nt", "shell launcher")
    def test_launcher(self):
        p = sm.Project.create("-It's a Movie!", self.tmp)
        launcher = sm.write_launcher(p)
        self.assertTrue(os.access(launcher, os.X_OK))
        self.assertIn("It's a Movie!", launcher.name)
        subprocess.run(["sh", "-n", str(launcher)], check=True)
        self.assertTrue(launcher.read_text().endswith(" .\n"))   # opens its folder, whatever the file is called
        # After a rename there's one launcher, with the new name.
        p.path.rename(p.path.with_name("Dragon.stopmo"))
        renamed = sm.Project.open(p.folder)
        new_launcher = sm.write_launcher(renamed)
        self.assertEqual([f.name for f in p.folder.glob("Open *")], [new_launcher.name])


# -------------------------------------------------------------------- movie

class MovieTests(TempDirTest):
    def make(self):
        p = sm.Project.create("Movie", self.tmp)
        with p.change():
            p.add_frame(solid(RED), 0)
            p.add_frame(solid(BLUE), 1)
            p.items.append(sm.title_item(["The End"], style=2, hold=6))
        return p

    def test_plan(self):
        p = self.make()
        self.assertEqual(len(sm.build_plan(p)), 1 + 1 + 6)
        p.items[1]["hold"] = 3
        p.items[1]["transition"] = sm.make_transition("dissolve", 12)
        p.items[0]["transition"] = sm.make_transition("fade", 12)
        p.doc["end_transition"] = sm.make_transition("circle", 12)
        plan = sm.build_plan(p)
        n = sm.make_transition("fade", 12)["frames"]
        self.assertEqual(len(plan), n + 1 + n + 3 + 6 + n)
        self.assertEqual(plan[0], ("mix", "fade", None, 0, 0.0))    # starts from black
        self.assertEqual(plan[-1][:4], ("mix", "circle", 2, None))
        self.assertEqual(plan[-1][4], 1.0)                          # ends on black
        self.assertEqual(sm.plan_item(plan[0]), 0)
        self.assertEqual(sm.plan_item(plan[-1]), 2)

    def test_transition_cycle(self):
        tr, seen = None, []
        for _ in range(len(sm.TRANSITIONS) + 1):
            tr = sm.next_transition(tr, 12)
            seen.append(tr["type"] if tr else None)
        self.assertEqual(seen, sm.TRANSITIONS + [None])

    def test_every_transition_renders(self):
        a, b = solid(RED, (64, 36)), solid(BLUE, (64, 36))
        for kind in sm.TRANSITIONS:
            start, mid, end = (sm.mix_images(kind, a, b, t) for t in (0.0, 0.5, 1.0))
            for img in (start, mid, end):
                self.assertEqual(img.shape, a.shape)
                self.assertEqual(img.dtype, np.uint8)
            self.assertTrue(np.array_equal(end, b) or kind in ("fade", "circle") and end.max() >= 250, kind)
            self.assertGreaterEqual(int(start[18, 32, 2]), 250, kind)       # starts on the old picture

    def test_renderer_handles_titles_black_and_mixed_sizes(self):
        p = self.make()
        with p.change():
            p.add_frame(solid(GREEN, (100, 100)), 1)         # a square picture in a wide movie
            p.items[0]["transition"] = sm.make_transition("wipe", 12)
        r = sm.Renderer(p)
        for step in sm.build_plan(p):
            self.assertEqual(r.render(step).shape, (180, 320, 3))
        title = r.image(p.items[-1])
        self.assertEqual(tuple(int(v) for v in title[2, 160]), sm.TITLE_STYLES[2]["bg"])
        missing = {"id": "x", "kind": "frame", "file": "frames/gone.jpg", "hold": 1}
        self.assertEqual(r.image(missing).shape, (180, 320, 3))

    def test_title_cards_fit(self):
        for style in range(len(sm.TITLE_STYLES)):
            card = sm.render_title({"lines": ["A really very long title for a movie", "by me"], "style": style},
                                   (640, 360))
            self.assertEqual(card.shape, (360, 640, 3))
        self.assertEqual(sm.render_title({"lines": [], "style": 0}, (64, 36)).shape, (36, 64, 3))

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs ffmpeg")
    def test_export_with_sound(self):
        p = self.make()
        rate = 16000
        t = np.arange(rate * 5) / rate                 # 5 s of sound for a shorter movie
        with p.change():
            rel = p.save_recording((np.sin(2 * np.pi * 330 * t) * 9000).astype(np.int16), rate)
            p.add_sound(0, rel, 0.0, 5.0, "tone")
            p.doc["end_transition"] = sm.make_transition("fade", p.fps)
        out = self.tmp / "movie.mp4"
        calls = []
        ok, note = sm.export_movie(p, out, lambda done, total: calls.append(done) or True)
        self.assertTrue(ok, note)
        frames = len(sm.build_plan(p))
        self.assertEqual(calls[-1], frames)
        probe = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height,nb_frames:format=duration",
             "-of", "json", str(out)], capture_output=True, text=True, check=True).stdout)
        kinds = {s["codec_type"]: s for s in probe["streams"]}
        self.assertEqual(set(kinds), {"video", "audio"})
        self.assertEqual((kinds["video"]["width"], kinds["video"]["height"]), (320, 180))
        self.assertEqual(int(kinds["video"]["nb_frames"]), frames)
        self.assertAlmostEqual(float(probe["format"]["duration"]), frames / p.fps, delta=0.15)
        self.assertFalse(any(".partial" in f.name for f in self.tmp.iterdir()))

    def test_export_without_ffmpeg(self):
        p = self.make()
        out = self.tmp / "movie.mp4"
        with mock.patch.object(sm, "has_ffmpeg", return_value=False):
            ok, note = sm.export_movie(p, out)
        self.assertTrue(ok, note)
        cap = cv2.VideoCapture(str(out))
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(sm.build_plan(p)))
        cap.release()

    def test_export_cleans_up_after_errors(self):
        p = self.make()
        out = self.tmp / "movie.mp4"

        def boom(done, total):
            if done > 2:
                raise RuntimeError("disk full")
            return True

        for ffmpeg in (True, False):
            with mock.patch.object(sm, "has_ffmpeg", return_value=ffmpeg), self.assertRaises(RuntimeError):
                sm.export_movie(p, out, boom)
            self.assertEqual(list(self.tmp.glob("*.mp4")), [])

    def test_export_can_be_cancelled(self):
        p = self.make()
        out = self.tmp / "movie.mp4"
        ok, note = sm.export_movie(p, out, lambda done, total: done < 3)
        self.assertFalse(ok)
        self.assertFalse(out.exists())
        self.assertEqual(list(self.tmp.glob("*.mp4")), [])

    def test_export_command_line(self):
        p = self.make()
        with mock.patch("builtins.print"):
            self.assertEqual(sm.main(["--export", str(p.path)]), 0)
        self.assertEqual(len(list(p.folder.glob("Movie *.mp4"))), 1)

    def test_wav_roundtrip(self):
        samples = (np.arange(-500, 500) * 30).astype(np.int16)
        sm.write_wav(self.tmp / "a.wav", samples, 22050)
        back, rate = sm.read_wav(self.tmp / "a.wav")
        self.assertEqual(rate, 22050)
        self.assertTrue(np.array_equal(back[:, 0], samples))


class KeyTests(unittest.TestCase):
    def test_key_names(self):
        self.assertEqual(sm.key_name(-1), None)
        self.assertEqual(sm.key_name(ord("p")), "p")
        self.assertEqual(sm.key_name(ord("P")), "P")
        self.assertEqual(sm.key_name(32), "space")
        for code in (65361, 63234, 2424832, 0x100000 | 65361):   # Linux, Mac, Windows, Linux+NumLock
            self.assertEqual(sm.key_name(code), "left")
        self.assertEqual(sm.key_name(0x100000 | ord("q")), "q")
        self.assertEqual(sm.key_name(127), "backspace")
        self.assertEqual(sm.key_name(65505), None)                 # shift on its own


# ----------------------------------------------------------------------- app

class StudioTests(TempDirTest):
    def open_app(self, script, camera=None):
        project = sm.Project.create("Movie", self.tmp)
        app, display = make_app(self.tmp, script, camera)
        self.app, self.display = app, display
        with mock.patch("builtins.print"):
            run(app, project.path)
        return app, display

    def no_flash(self):
        return lambda: setattr(self.app, "flash_until", 0)

    def test_ghost_is_last_picture_and_survives_everything(self):
        """The onion skin must show the last picture taken - after playing, after
        turning it off and on, and after the camera drops out and comes back."""
        cam = FakeCamera(RED)
        seen = {}

        def look(name):
            return lambda: seen.__setitem__(name, view_pixel(self.display))

        def camera(color=None, size=None, gone=False):
            def change():
                cam.cap = None if gone else object()
                if color:
                    cam.color = color
                if size:
                    cam.size = size
            return change

        script = ["space", camera(BLUE), self.no_flash(), None, look("ghost"),
                  "p", None, None, "x", None, look("after_play"),
                  "o", None, look("off"), "o", None, look("on_again"),
                  camera(gone=True), None, None, look("no_camera"),
                  camera(size=(640, 480)), None, look("camera_back"),
                  "c", "H", "i", "enter", None, look("after_title"),
                  "l", "esc", None, look("after_projects")]
        self.open_app(script, cam)
        blend = np.array([255 * (1 - sm.ONION_OPACITY), 0, 255 * sm.ONION_OPACITY])
        for name in ("ghost", "after_play", "on_again", "camera_back", "after_title", "after_projects"):
            np.testing.assert_allclose(seen[name], blend, atol=3, err_msg=name)
        np.testing.assert_allclose(seen["off"], [255, 0, 0], atol=2)
        self.assertLess(seen["no_camera"].max(), 200)         # dimmed ghost + message, no crash
        self.assertGreater(cam.flushes, 0)                    # stale camera pictures thrown away

    def test_ghost_follows_the_camera_tile(self):
        cam = FakeCamera(RED)

        def color(c):
            return lambda: setattr(cam, "color", c)

        script = ["space", color(GREEN), None, "space", color(BLUE), None,
                  "left", "left", None,             # pick picture 1 (red)
                  "space", None,                    # camera goes after it
                  self.no_flash(), None]
        app, display = self.open_app(script, cam)
        self.assertEqual(app.insert_at, 1)
        self.assertIs(app.ghost_item(), app.project.items[0])
        self.assertGreater(view_pixel(display)[2], 100)       # the red picture's ghost

    def test_insert_in_the_middle(self):
        cam = FakeCamera(RED)
        script = ["space", "space", lambda: setattr(cam, "color", BLUE), None,
                  "home", None, "space", "space", None]
        app, _ = self.open_app(script, cam)
        colors = [cv2.imread(str(app.project.file_path(it["file"])))[360, 640] for it in app.project.items]
        self.assertEqual([int(c[2]) > 200 for c in colors], [True, False, True])

    def test_editing_keys(self):
        cam = FakeCamera(RED)
        script = ["space", "space", "space", None,
                  "left", "+", "+", None,           # picture 3 shows 3x
                  "left", "x", None,                # delete picture 2: picture 3 is picked now
                  "d", None,                        # copy it
                  "]", None,                        # move the copy right, past the camera tile
                  "f", "f", None,                   # transition: Fade, then Melt
                  "z", None]                        # undo back to Fade
        app, _ = self.open_app(script, cam)
        items = app.project.items
        self.assertEqual([it["hold"] for it in items], [1, 3, 3])
        self.assertEqual((app.cursor, app.insert_at), (2, 2))
        self.assertEqual(items[2]["transition"]["type"], "fade")
        disk = sm.Project.open(app.project.path)
        self.assertEqual([it["hold"] for it in disk.items], [1, 3, 3])

    def test_drag_to_reorder(self):
        cam = FakeCamera(RED)
        x = [sm.TILES_X0 + k * sm.TILE_STEP + sm.TILE_W // 2 for k in range(4)]
        y = sm.TILE_Y + sm.TILE_H // 2
        script = ["space", lambda: setattr(cam, "color", BLUE), None, "space", None,
                  ("drag", x[0], y, x[2] - sm.TILE_W // 2 - 5, y), None]
        app, _ = self.open_app(script, cam)
        first = cv2.imread(str(app.project.file_path(app.project.items[0]["file"])))
        self.assertGreater(first[360, 640, 0], 200)           # blue is first now
        self.assertEqual(app.cursor, 1)

    def test_dragging_the_camera_while_auto_snap_fires(self):
        cam = FakeCamera(RED)
        x = [sm.TILES_X0 + k * sm.TILE_STEP + sm.TILE_W // 2 for k in range(5)]
        y = sm.TILE_Y + sm.TILE_H // 2

        def auto_snap_now():
            self.app.cmd_snap()                # a picture arrives mid-drag

        script = ["space", "space", None,
                  lambda: self.display.events.extend([("down", x[2], y), ("drag", x[1], y)]), None,
                  auto_snap_now, None,
                  lambda: self.display.events.append(("up", x[0] - sm.TILE_W // 2 - 5, y)), None]
        app, _ = self.open_app(script, cam)
        self.assertEqual(app.insert_at, 0)                     # the camera went to the start
        self.assertEqual(len(app.project.items), 3)

    def test_lost_button_release_doesnt_move_things_later(self):
        x0 = sm.TILES_X0 + sm.TILE_W // 2
        y = sm.TILE_Y + sm.TILE_H // 2
        script = ["space", "space", None,
                  lambda: self.display.events.extend([("down", x0, y), ("drag", x0 + 200, y)]), None,
                  ("click", sm.PANEL_X + 100, sm.VIEW_Y + 400), None]      # the release never came
        app, _ = self.open_app(script)
        self.assertIsNone(app.drag)
        self.assertEqual(app.insert_at, 2)

    def test_auto_snap_waits_for_a_fresh_picture_after_another_screen(self):
        cam = FakeCamera(RED)

        def time_passes_and_things_move():
            self.app.next_auto = 0                      # the gap is over while help is showing...
            cam.color = BLUE                            # ...and the puppet has moved

        script = ["t", None, "h", time_passes_and_things_move, "x", None, None]
        app, _ = self.open_app(script, cam)
        self.assertEqual(app.project.items, [])         # no stale red picture snapped

    def test_title_card(self):
        script = ["space", None, "c"] + list("Hi Mom") + ["tab", "b", "y", " ", "S", "a", "m", "backspace",
                                                        "right", "enter", None]
        app, _ = self.open_app(script)
        title = app.project.items[1]
        self.assertEqual(title["kind"], "title")
        self.assertEqual(title["lines"], ["Hi Mom", "by Sa"])
        self.assertEqual(title["style"], 1)
        self.assertEqual(title["hold"], round(sm.TITLE_SECONDS * sm.FPS))
        self.assertEqual(app.insert_at, 2)                    # the camera moved past the title

    def test_capitals_on_linux_windows(self):
        # OpenCV's Qt windows send Shift / Caps Lock as keys and letters in lowercase.
        shift, caps = 65505, 65509
        script = ["c", shift, "h", "i", " ", caps, "m", "o", "m", caps, "!", "enter", None,
                  "n", "5", "enter", None]
        app, _ = self.open_app(script)
        self.assertEqual(app.project.items[0]["lines"], ["Hi MOM!"])
        self.assertEqual(app.project.doc["frame_goal"], 5)

    def test_record_voice_and_play_it(self):
        with mock.patch.object(sm, "sound_problem", return_value=None), \
                mock.patch.object(sm, "sound_module", return_value=object()):
            script = ["space", "space", None, "r", None, None, "x", None, "p", None, "x", None]
            app, _ = self.open_app(script)
        (clip,) = app.project.tracks[0]["clips"]          # the voice goes on DLG, from the start
        self.assertEqual(clip["name"], "voice")
        self.assertEqual((clip["start"], clip["trim"]), (0.0, 0.0))
        self.assertAlmostEqual(clip["length"], 1.0, places=2)
        self.assertEqual(app.speaker.played, [0.0])

    def test_silent_recording_is_not_kept(self):
        FakeRecorder.samples = np.zeros((1000, 1), np.int16)
        self.addCleanup(setattr, FakeRecorder, "samples", None)
        with mock.patch.object(sm, "sound_problem", return_value=None):
            app, _ = self.open_app(["r", None, "x", None])
        self.assertFalse(app.project.has_sound())
        self.assertIn("silent", app.toast_text)

    def test_quit_needs_two_presses(self):
        app, _ = self.open_app(["q", None, None])
        self.assertTrue(app.running)
        app, _ = self.open_app(["q", "q", None])
        self.assertFalse(app.running)

    def test_caps_lock_still_works(self):
        app, _ = self.open_app(["space", "O", None])
        self.assertFalse(app.onion)
        self.assertFalse(sm.Project.open(app.project.path).doc["onion_skin"])

    def test_speed_and_goal(self):
        app, _ = self.open_app(["v", None, "n", "4", "0", "enter", None])
        self.assertEqual(app.project.fps, 15)
        self.assertEqual(app.project.doc["frame_goal"], 40)

    def test_no_camera_at_all(self):
        cam = FakeCamera()
        cam.cap = None
        app, display = self.open_app(["space", None, "c", "H", "enter", None, "p", None, "x", None], cam)
        self.assertEqual([it["kind"] for it in app.project.items], ["title"])


class CameraControlTests(TempDirTest):
    """Focus / exposure / white balance, with a pretend USB webcam."""

    def controls(self, **kwargs):
        self.uvc = FakeUVC(**kwargs)
        return sm.CameraControls(0, usb=self.uvc, name="Fake C920")

    def test_controls_start_on_auto_and_go_back_to_auto(self):
        controls = self.controls()
        self.assertEqual(set(controls.settings), {"focus", "exposure", "white balance"})
        self.assertEqual(self.uvc.values[sm.FOCUS_AUTO], 1)
        self.assertEqual(controls.toggle("focus"), "Focus LOCKED at 100")
        self.assertEqual(self.uvc.values[sm.FOCUS_AUTO], 0)
        self.assertEqual(controls.toggle("focus"), "Focus on AUTO")
        controls.toggle("focus")
        controls.close()
        self.assertEqual(self.uvc.values[sm.FOCUS_AUTO], 1)
        self.assertTrue(self.uvc.closed)

    def test_a_camera_only_gets_the_controls_it_has(self):
        controls = self.controls(has_focus=False)
        self.assertEqual(set(controls.settings), {"exposure", "white balance"})
        self.assertIn("can't lock focus", controls.toggle("focus"))
        none = sm.CameraControls(0)
        self.assertEqual(none.settings, {})
        self.assertIn("no focus", none.summary())

    def test_exposure_slider_is_even_in_stops(self):
        exposure = self.controls().settings["exposure"]
        self.assertEqual(exposure.to_slider(3), 0)
        self.assertEqual(exposure.to_slider(2047), 100)
        self.assertAlmostEqual(exposure.from_slider(50), (3 * 2047) ** 0.5, delta=1)

    def open_studio(self, script, **uvc):
        self.uvc = FakeUVC(**uvc)
        self.cam = FakeCamera(uvc=self.uvc)
        project = sm.Project.create("Movie", self.tmp)
        app, display = make_app(self.tmp, script, self.cam)
        self.app = app
        with mock.patch("builtins.print"):
            run(app, project.path)
        return app, display

    def test_locking_exposure_matches_how_auto_had_it(self):
        seen = {}
        script = ["k", None, "x"] + [None] * 80 + [lambda: seen.update(ae=self.uvc.values[sm.AE_MODE],
                                                                           exposure=self.uvc.exposure())]
        app, _ = self.open_studio(script, auto_exposure=600)
        self.assertEqual(seen["ae"], sm.AE_MANUAL)
        self.assertAlmostEqual(seen["exposure"], 600, delta=30)
        self.assertIn("Exposure LOCKED", app.toast_text)
        self.assertNotEqual(self.uvc.values[sm.AE_MODE], sm.AE_MANUAL)     # back on auto after quitting

    def test_coarse_cameras_lock_to_the_nearest_step_and_say_so(self):
        seen = {}
        script = ["k", None, "x"] + [None] * 100 + [lambda: seen.update(exposure=self.uvc.exposure())]
        app, _ = self.open_studio(script, auto_exposure=450, exposure_steps=[75, 150, 300, 625, 1250])
        self.assertIn(seen["exposure"], (300, 625))
        self.assertIn("as close as this camera gets", app.toast_text)

    def test_locking_white_balance(self):
        seen = {}
        script = ["k", None, "w"] + [None] * 80 + [lambda: seen.update(wb=self.uvc.white_balance(),
                                                                           auto=self.uvc.values[sm.WHITE_BALANCE_AUTO])]
        app, _ = self.open_studio(script, auto_white_balance=5200)
        self.assertEqual(seen["auto"], 0)
        self.assertAlmostEqual(seen["wb"], 5200, delta=250)

    def test_focus_nudges_sliders_and_brighter(self):
        seen = {}
        script = ["k", None, "f", "]", "]", None, lambda: seen.update(focus=self.uvc.values[sm.FOCUS])]
        slider_x = sm.CANVAS_W - 20           # the far right end of a slider
        script += [lambda: self.display_click_slider("focus", slider_x), None,
                   lambda: seen.update(slid=self.uvc.values[sm.FOCUS])]
        self.open_studio(script)
        self.assertEqual(seen["focus"], 110)
        self.assertEqual(seen["slid"], 250)

    def display_click_slider(self, name, x):
        x0, x1 = self.app.slider_rects[name]
        rect = next(r for r, action in self.app.hotspots if action == ("slider", name))
        self.app.display.events += [("down", x0 + 2, (rect[1] + rect[3]) // 2), ("drag", x, (rect[1] + rect[3]) // 2),
                                    ("up", x, (rect[1] + rect[3]) // 2)]

    def test_brighter_and_darker_when_locked(self):
        seen = {}
        script = ["k", None, "x"] + [None] * 80 + [lambda: seen.update(before=self.uvc.values[sm.EXPOSURE_TIME]),
                                                   "+", "+", "+", None,
                                                   lambda: seen.update(after=self.uvc.values[sm.EXPOSURE_TIME])]
        self.open_studio(script)
        self.assertAlmostEqual(seen["after"] / seen["before"], 2.0, delta=0.1)     # 3 thirds of a stop

    def test_panel_keys_only_work_while_it_is_open(self):
        cam = FakeCamera(RED)
        script = ["space", "space", None, "k", "x", "c", "esc", None,
                  lambda: self.assertFalse(self.app.camera_panel), "x", None]
        project = sm.Project.create("Movie", self.tmp)
        app, _ = make_app(self.tmp, script, cam)
        self.app = app
        with mock.patch("builtins.print"):
            run(app, project.path)
        self.assertEqual(len(app.project.items), 1)          # only the X after closing deleted anything
        self.assertTrue(app.running)                         # ESC closed the panel, didn't start quitting

    def test_software_brightness_on_cameras_without_controls(self):
        seen = {}
        cam = FakeCamera((100, 100, 100))
        script = [None, lambda: seen.update(before=int(self.app.live[400, 600, 0])), "k", "+", "+", "+", None,
                  lambda: seen.update(after=int(self.app.live[400, 600, 0]))]
        project = sm.Project.create("Movie", self.tmp)
        self.app, _ = make_app(self.tmp, script, cam)
        run(self.app, project.path)
        self.assertGreater(seen["after"], seen["before"] * 1.25)

    def test_flip_is_remembered_for_each_camera(self):
        cam = FakeCamera(RED)
        project = sm.Project.create("Movie", self.tmp)
        self.app, _ = make_app(self.tmp, [None, "k", "r", None, "space", None], cam)
        with mock.patch("builtins.print"):
            run(self.app, project.path)
        saved = cv2.imread(str(self.app.project.file_path(self.app.project.items[0]["file"])))
        self.assertEqual(int(saved[-5, -5, 1]), 255)               # the white corner is bottom-right now
        again, _ = make_app(self.tmp, [None], FakeCamera(RED))
        with mock.patch("builtins.print"):
            run(again, project.path)
        self.assertEqual(json.loads((self.tmp / sm.SETTINGS_FILE).read_text()), {"flip": {"Camera 1": True}})
        self.assertEqual(int(again.live[-5, -5, 1]), 255)             # still flipped next time

    def test_switching_cameras(self):
        cam = FakeCamera(RED, cameras=2)
        project = sm.Project.create("Movie", self.tmp)
        self.app, _ = make_app(self.tmp, ["k", "c", None], cam)
        with mock.patch("builtins.print"):
            run(self.app, project.path)
        self.assertEqual(cam.index, 1)
        self.assertIn("Camera 2", self.app.toast_text)
        lonely = FakeCamera(RED)
        self.app, _ = make_app(self.tmp, ["k", "c", None], lonely)
        with mock.patch("builtins.print"):
            run(self.app, project.path)
        self.assertIn("No other camera", self.app.toast_text)

    def test_pictures_from_another_camera_fit_the_movie(self):
        cam = FakeCamera(RED)
        project = sm.Project.create("Movie", self.tmp)
        script = ["space", lambda: setattr(cam, "size", (640, 480)), None, "space", None]
        self.app, _ = make_app(self.tmp, script, cam)
        with mock.patch("builtins.print"):
            run(self.app, project.path)
        sizes = {cv2.imread(str(self.app.project.file_path(it["file"]))).shape for it in self.app.project.items}
        self.assertEqual(sizes, {(720, 1280, 3)})

    def test_the_same_camera_coming_back_keeps_its_locks(self):
        seen = {}

        def reconnect():
            self.cam.generation += 1                    # dropped out and came back
            self.uvc.values[sm.FOCUS_AUTO] = 1          # a replugged camera starts on auto

        script = ["k", None, "f", "]", None, reconnect, None, None,
                  lambda: seen.update(auto=self.uvc.values[sm.FOCUS_AUTO], focus=self.uvc.values[sm.FOCUS])]
        app, _ = self.open_studio(script)
        self.assertEqual(seen, {"auto": 0, "focus": 105})
        self.assertIn("camera is back", app.toast_text)

    def test_a_failed_change_is_reported(self):
        def broken(control, value):
            raise OSError("LIBUSB_ERROR_PIPE")

        script = ["k", None, "f", lambda: setattr(self.uvc, "set", broken), "]", None]
        app, _ = self.open_studio(script)
        self.assertIn("Couldn't set focus", app.toast_text)

    def test_slider_drags_end_when_the_panel_closes(self):
        seen = {}

        def press_on_focus_slider():
            x0, x1 = self.app.slider_rects["focus"]
            rect = next(r for r, action in self.app.hotspots if action == ("slider", "focus"))
            self.y = (rect[1] + rect[3]) // 2
            self.app.display.events.append(("down", x0, self.y))

        script = ["k", None, press_on_focus_slider, None, lambda: seen.update(before=self.uvc.values[sm.FOCUS]),
                  "k", None, lambda: self.app.display.events.append(("drag", sm.CANVAS_W - 20, self.y)), None,
                  lambda: seen.update(after=self.uvc.values[sm.FOCUS])]
        self.open_studio(script)
        self.assertEqual(seen["before"], seen["after"])

    def test_camera_trouble_doesnt_end_the_session(self):
        cam = FakeCamera(RED)
        project = sm.Project.create("Movie", self.tmp)
        self.app, _ = make_app(self.tmp, [None, "space", None], cam)
        self.app.cam_setup.make_controls = mock.Mock(side_effect=RuntimeError("no name"))
        with mock.patch("builtins.print"), mock.patch("traceback.print_exc"):
            run(self.app, project.path)
        self.assertEqual(len(self.app.project.items), 1)             # still took the picture
        self.assertIn("Camera trouble", self.app.toast_text)

    def test_clicking_the_status_opens_the_panel(self):
        def click_status():
            rect = next(r for r, action in self.app.hotspots if action == ("cmd", "camera") and r[1] < sm.VIEW_Y + 60)
            self.app.display.events += [("down", rect[0] + 5, rect[1] + 5), ("up", rect[0] + 5, rect[1] + 5)]

        self.open_studio([None, click_status, None, lambda: self.assertTrue(self.app.camera_panel)])


class CameraReconnectTests(unittest.TestCase):
    def test_a_reconnect_that_finishes_after_a_switch_is_thrown_away(self):
        import threading
        release_camera_0 = threading.Event()
        opened = []

        class Cap:
            def __init__(self, index):
                self.index, self.released = index, False
                opened.append(self)

            def release(self):
                self.released = True

        def fake_open(index=None):
            if index == 0:
                release_camera_0.wait(5)            # camera 0 is slow to come back
            return Cap(index)

        cam = sm.Camera(0)
        cam._open = fake_open
        cam.read()                                  # no camera yet: starts reconnecting to camera 0
        self.assertTrue(cam.switch())               # meanwhile, switch to camera 1
        release_camera_0.set()
        cam._opener.join(5)
        cam._reconnect()                            # nothing stale gets adopted
        self.assertEqual(cam.cap.index, 1)
        stale = [c for c in opened if c.index == 0]
        self.assertTrue(stale and all(c.released for c in stale))


class CameraSwitchTests(unittest.TestCase):
    def test_the_old_camera_is_put_back_on_auto_before_it_is_let_go(self):
        events = []

        class Cap(FakeVideoCapture):
            def release(self):
                events.append(("release", self.name))

        def opener(index=None):
            cap = Cap({(1280, 720): 30})
            cap.name = index
            return cap

        cam = sm.Camera(0)
        with mock.patch.object(cam, "_open", opener):
            cam.start()
            self.assertTrue(cam.switch(on_change=lambda: events.append(("auto", cam.cap.name))))
        self.assertEqual(events, [("auto", None), ("release", None)])
        self.assertEqual((cam.index, cam.cap.name), (1, 1))


class CameraSizeTests(unittest.TestCase):
    """Asking each camera for its sharpest picture."""

    def choose(self, cap, platform="mac"):
        with mock.patch.object(sm, "camera_platform", return_value=platform):
            return sm.choose_camera_size(cap, clock=cap.clock), cap.size

    def test_a_1080p_camera_gets_1080p(self):
        cap = FakeVideoCapture({(1920, 1080): 30, (1280, 720): 30, (640, 480): 30})
        self.assertEqual(self.choose(cap), (True, (1920, 1080)))
        self.assertEqual(cap.asked[0], (3840, 2160))           # asked for the most first

    def test_a_4k_camera_gets_4k(self):
        cap = FakeVideoCapture({(3840, 2160): 30, (1920, 1080): 30})
        self.assertEqual(self.choose(cap), (True, (3840, 2160)))

    def test_too_slow_sizes_are_skipped(self):
        cap = FakeVideoCapture({(3840, 2160): 5, (1920, 1080): 30})
        self.assertEqual(self.choose(cap), (True, (1920, 1080)))

    def test_the_setting_caps_the_size(self):
        cap = FakeVideoCapture({(3840, 2160): 30, (1920, 1080): 30})
        with mock.patch.object(sm, "CAMERA_MAX_SIZE", (1920, 1080)):
            self.assertEqual(self.choose(cap), (True, (1920, 1080)))
        self.assertNotIn((3840, 2160), cap.asked)

    def test_cameras_without_any_of_our_sizes_keep_their_best(self):
        cap = FakeVideoCapture({(1600, 1200): 15, (640, 480): 30}, style="nearest")
        self.assertEqual(self.choose(cap, "linux"), (True, (1600, 1200)))
        self.assertIsNotNone(cap.fourcc)                       # asked for compressed pictures on Linux
        refuses = FakeVideoCapture({(800, 600): 30}, style="fail", start=(800, 600))
        self.assertEqual(self.choose(refuses, "windows"), (True, (800, 600)))
        stubborn = FakeVideoCapture({(640, 480): 30}, style="default")
        self.assertEqual(self.choose(stubborn), (True, (640, 480)))

    def test_no_pictures_at_all(self):
        cap = FakeVideoCapture({(1280, 720): 30}, style="fail", start=None)
        cap.modes = {}
        self.assertFalse(self.choose(cap)[0])

    def test_frame_rates_are_timed_not_taken_on_trust(self):
        cap = FakeVideoCapture({(1920, 1080): 5, (1280, 720): 30})    # says 30 at 1080p, sends 5
        self.assertEqual(self.choose(cap), (True, (1280, 720)))

    def test_slow_big_sizes_give_way_to_a_quick_smaller_one(self):
        cap = FakeVideoCapture({(1920, 1080): 5, (800, 600): 30}, start=(800, 600))
        self.assertEqual(self.choose(cap), (True, (800, 600)))

    def test_four_by_three_cameras_keep_their_biggest_quick_picture(self):
        cap = FakeVideoCapture({(2592, 1944): 15, (1600, 1200): 30, (640, 480): 30})
        self.assertEqual(self.choose(cap), (True, (2592, 1944)))

    def test_a_camera_that_ignores_a_size_is_still_asked_for_the_next(self):
        # macOS keeps its current picture for a size it hasn't got; that says nothing about smaller sizes.
        cap = FakeVideoCapture({(1920, 1080): 30, (1280, 720): 30}, style="default", start=(1280, 720))
        self.assertEqual(self.choose(cap), (True, (1920, 1080)))

    def test_each_platform_asks_in_the_order_its_driver_needs(self):
        # DirectShow takes the frame rate for every size it sets up, but its
        # compression only for the size already set, so that comes after.
        windows = FakeVideoCapture({(1920, 1080): 30, (1280, 720): 30}, style="fail",
                                   raw={(1920, 1080): 5}, dshow=True, backend="DSHOW")
        self.assertEqual(self.choose(windows, "windows"), (True, (1920, 1080)))
        self.assertEqual(windows.log[0], "fps")
        self.assertEqual(windows.log[-3:], ["width", "height", "fourcc"])
        # V4L2 keeps its compression, but its frame rate needs asking again after each size.
        linux = FakeVideoCapture({(1920, 1080): 30}, raw={(1920, 1080): 5}, backend="V4L2")
        self.assertEqual(self.choose(linux, "linux"), (True, (1920, 1080)))
        self.assertEqual(linux.log[0], "fourcc")
        self.assertEqual(linux.log[-3:], ["width", "height", "fps"])
        mac = FakeVideoCapture({(1920, 1080): 30})
        self.choose(mac)
        self.assertNotIn("fourcc", mac.log)

    def test_windows_opens_through_directshow_first(self):
        calls = []

        class Refuses(FakeVideoCapture):
            def isOpened(self):
                return False

        def video_capture(index, api=None):
            calls.append(api)
            return Refuses({}) if api == cv2.CAP_DSHOW else FakeVideoCapture({(1920, 1080): 30})

        with mock.patch.object(sm, "camera_platform", return_value="windows"), \
                mock.patch.object(sm.cv2, "VideoCapture", video_capture):
            cap = sm.Camera(0)._open()
        self.assertEqual(calls, [cv2.CAP_DSHOW, None])          # fell back when DirectShow couldn't
        self.assertEqual(cap.size, (1920, 1080))


C920_V4L2 = {
    0x009A0901: [0, 3, 1, 3],            # exposure mode: menu, aperture priority (auto)
    0x009A0902: [3, 2047, 1, 250],       # exposure time, 0.1 ms
    0x009A090A: [0, 250, 5, 0],          # focus
    0x009A090C: [0, 1, 1, 1],            # autofocus
    0x0098091A: [2000, 6500, 1, 4000],   # white balance
    0x0098090C: [0, 1, 1, 1],            # auto white balance
}


class LinuxControlTests(unittest.TestCase):
    def device(self, **kwargs):
        controls = kwargs.pop("controls", C920_V4L2)
        self.v4l2 = FakeV4L2(controls, menus={0x009A0901: {1, 3}}, **kwargs)
        real = sm.V4L2Camera
        return mock.patch.multiple(sm, camera_platform=mock.Mock(return_value="linux"),
                                   V4L2Camera=lambda index: real(index, ioctl=self.v4l2.ioctl, opener=self.v4l2.open))

    def test_a_linux_webcam_gets_its_controls_with_real_ranges(self):
        with self.device():
            controls = sm.CameraControls(1)
        self.assertEqual(self.v4l2.opened, ["/dev/video1"])
        self.assertEqual(controls.name, "HD Pro Webcam C920")
        self.assertEqual(set(controls.settings), {"focus", "exposure", "white balance"})
        focus = controls.settings["focus"]
        self.assertEqual((focus.min, focus.max, focus.step), (0, 250, 5))
        self.assertEqual(controls.toggle("focus"), "Focus LOCKED at 0")
        self.assertEqual(self.v4l2.value(0x009A090C), 0)        # autofocus off
        controls.set_value("focus", 1000)
        self.assertEqual(self.v4l2.value(0x009A090A), 250)      # kept within its range

    def test_exposure_modes_translate_both_ways(self):
        with self.device():
            controls = sm.CameraControls(0)
        controls.set_value("exposure", 500)
        self.assertEqual(self.v4l2.value(0x009A0901), 1)        # V4L2's "manual"
        self.assertEqual(self.v4l2.value(0x009A0902), 500)
        controls.toggle("exposure")
        self.assertEqual(self.v4l2.value(0x009A0901), 3)        # back to its own auto (aperture priority)
        controls.close()
        self.assertTrue(all(self.v4l2.value(cid) in (1, 3) for cid in (0x009A090C, 0x0098090C, 0x009A0901)))

    def test_missing_or_disabled_controls_are_left_out(self):
        no_focus = {cid: c for cid, c in C920_V4L2.items() if cid not in (0x009A090A, 0x009A090C)}
        with self.device(controls=no_focus, disabled={0x0098091A}):
            controls = sm.CameraControls(0)
        self.assertEqual(set(controls.settings), {"exposure"})

    def test_request_codes_match_the_kernels_struct_sizes(self):
        def size(code):
            return (code >> 16) & 0x3FFF            # _IOC: the struct size sits in bits 16-29

        self.assertEqual(size(sm.VIDIOC_QUERYCTRL), sm.V4L2Camera.QUERYCTRL.size)
        self.assertEqual(sm.V4L2Camera.QUERYCTRL.size, 68)
        self.assertEqual(size(sm.VIDIOC_QUERYMENU), struct.calcsize("<II32sI"))
        self.assertEqual(size(sm.VIDIOC_G_CTRL), struct.calcsize("<Ii"))
        self.assertEqual(size(sm.VIDIOC_QUERYCAP), 104)
        for code, number in ((sm.VIDIOC_QUERYCAP, 0), (sm.VIDIOC_G_CTRL, 27), (sm.VIDIOC_S_CTRL, 28),
                             (sm.VIDIOC_QUERYCTRL, 36), (sm.VIDIOC_QUERYMENU, 37)):
            self.assertEqual((code >> 8) & 0xFF, ord("V"))
            self.assertEqual(code & 0xFF, number)

    def test_no_such_device(self):
        with mock.patch.object(sm, "camera_platform", return_value="linux"):
            controls = sm.CameraControls(57)
        self.assertEqual(controls.settings, {})
        self.assertIn("/dev/video57", controls.problem)


class WindowsControlTests(TempDirTest):
    def controls(self, **kwargs):
        self.ds = FakeDirectShow(**kwargs)
        with mock.patch.object(sm, "camera_platform", return_value="windows"):
            return sm.CameraControls(0, capture=self.ds)

    def test_a_windows_webcam_gets_its_controls(self):
        controls = self.controls()
        self.assertEqual(set(controls.settings), {"focus", "exposure", "white balance"})
        self.assertAlmostEqual(controls.settings["exposure"].value, 10000 / 16)        # 2^-4 s, in 0.1 ms
        controls.toggle("white balance")
        self.assertEqual(self.ds.values[cv2.CAP_PROP_AUTO_WB], 0)
        controls.toggle("white balance")
        self.assertEqual(self.ds.values[cv2.CAP_PROP_AUTO_WB], 1)
        controls.toggle("focus")
        self.assertEqual(self.ds.values[cv2.CAP_PROP_AUTOFOCUS], 2)                   # DirectShow's "manual"

    def test_values_it_refuses_find_the_nearest_it_takes(self):
        controls = self.controls()
        exposure = controls.settings["exposure"]
        controls.set_value("exposure", exposure.from_slider(100))    # half a second: too long for it
        self.assertEqual(self.ds.values[cv2.CAP_PROP_EXPOSURE], -2)
        self.assertEqual(self.ds.values[cv2.CAP_PROP_AUTO_EXPOSURE], 0)
        controls.set_value("exposure", exposure.from_slider(0))
        self.assertEqual(self.ds.values[cv2.CAP_PROP_EXPOSURE], -11)
        self.assertEqual(controls.usb.ranges[sm.EXPOSURE_TIME], [-11, -2, 1])        # learned its range
        controls.set_value("white balance", 10000)
        self.assertEqual(self.ds.values[cv2.CAP_PROP_WB_TEMPERATURE], 6500)

    def test_a_camera_without_focus(self):
        controls = self.controls(missing={cv2.CAP_PROP_FOCUS, cv2.CAP_PROP_AUTOFOCUS})
        self.assertEqual(set(controls.settings), {"exposure", "white balance"})

    def test_only_directshow_hands_over_the_controls(self):
        msmf = FakeVideoCapture({}, backend="MSMF")
        with mock.patch.object(sm, "camera_platform", return_value="windows"):
            controls = sm.CameraControls(0, capture=msmf)
        self.assertEqual(controls.settings, {})
        self.assertIn("DirectShow", controls.problem)

    def test_a_focus_that_moves_in_fives(self):
        controls = self.controls(steps={cv2.CAP_PROP_FOCUS: 5})
        focus = controls.settings["focus"]
        controls.set_value("focus", 3)
        self.assertEqual((self.ds.values[cv2.CAP_PROP_FOCUS], focus.value), (5, 5))  # the next one it takes
        controls.set_value("focus", focus.value + focus.step)                       # what ] does
        self.assertEqual(focus.value, 10)
        controls.set_value("focus", 255)
        self.assertEqual((focus.value, focus.max), (250, 250))  # the real end, not a step it skipped
        controls.set_value("focus", 0)
        self.assertEqual((focus.value, focus.min), (0, 0))

    def test_a_white_balance_that_moves_in_fifties(self):
        controls = self.controls(steps={cv2.CAP_PROP_WB_TEMPERATURE: 50})
        controls.set_value("white balance", 4010)
        self.assertEqual(controls.settings["white balance"].value, 4050)
        controls.set_value("white balance", 10000)
        self.assertEqual(controls.settings["white balance"].max, 6500)
        controls.set_value("white balance", 0)
        self.assertEqual(controls.settings["white balance"].min, 2800)

    def test_the_panel_shows_what_the_camera_took(self):
        controls = self.controls()
        controls.set_value("exposure", 300)                    # 30 ms: between 1/64 s and 1/32 s
        self.assertEqual(self.ds.values[cv2.CAP_PROP_EXPOSURE], -5)
        self.assertEqual(controls.describe("exposure"), "31.25ms")

    def test_minus_and_plus_move_a_whole_stop(self):
        setup = sm.CameraSetup(self.tmp / "settings.json")
        setup.controls = self.controls()
        setup.controls.set_value("exposure", 10000 / 64)       # 2^-6 s
        setup.brighter(1)
        self.assertEqual(self.ds.values[cv2.CAP_PROP_EXPOSURE], -5)
        setup.brighter(-1)
        setup.brighter(-1)
        self.assertEqual(self.ds.values[cv2.CAP_PROP_EXPOSURE], -7)

    def test_half_a_second_is_an_exposure_not_a_missing_control(self):
        controls = self.controls()
        self.ds.values[cv2.CAP_PROP_EXPOSURE] = -1             # auto, in the dark (OpenCV's "missing" too)
        controls.refresh()
        self.assertEqual(controls.settings["exposure"].value, 5000)

    def test_locking_exposure_by_matching_the_picture(self):
        ds = FakeDirectShow(auto_exposure=-6)
        cam = FakeCamera()
        cam.frame = lambda: ds.picture(cam.size)
        project = sm.Project.create("Movie", self.tmp)
        app, _ = make_app(self.tmp, ["k", None, "x"] + [None] * 80, cam)
        app.cam_setup.make_controls = lambda camera: sm.CameraControls(0, usb=sm.DirectShowCamera(ds))
        seen = {}
        app.display.script.append(lambda: seen.update(mode=ds.values[cv2.CAP_PROP_AUTO_EXPOSURE],
                                                      exposure=ds.values[cv2.CAP_PROP_EXPOSURE]))
        with mock.patch("builtins.print"):
            run(app, project.path)
        self.assertEqual(seen, {"mode": 0, "exposure": -6})    # manual, where auto had it
        self.assertIn("Exposure LOCKED", app.toast_text)


class CaptureBoxTests(TempDirTest):
    def test_names(self):
        for name in ("Cam Link 4K", "USB Video", "Game Capture HD60 S+", "Live Gamer Portable 2",
                     "UltraStudio Recorder 3G"):
            self.assertTrue(sm.looks_like_capture_box(name), name)
        for name in ("HD Pro Webcam C920", "FaceTime HD Camera", "Elgato Facecam", "Camera 1", None):
            self.assertFalse(sm.looks_like_capture_box(name), name)

    def test_a_usb_video_device_with_no_camera_inside(self):
        class Dongle(FakeUVC):
            units = {"processing": 2}

            def get(self, control, request=sm.GET_CUR):
                if control[0] == "terminal":
                    raise OSError("no terminal unit")
                return super().get(control, request)

        controls = sm.CameraControls(0, usb=Dongle(), name="Some Grabber")
        self.assertTrue(controls.capture_box)
        self.assertIn("on the camera itself", controls.problem)

    def test_a_real_webcam_is_not_a_capture_box(self):
        self.assertFalse(sm.CameraControls(0, usb=FakeUVC(), name="Elgato Facecam").capture_box)
        self.assertFalse(sm.CameraControls(0, usb=FakeUVC(has_focus=False), name="USB Video").capture_box)

    def test_windows_and_linux_devices_with_neither_focus_nor_exposure(self):
        props = {cv2.CAP_PROP_FOCUS, cv2.CAP_PROP_AUTOFOCUS, cv2.CAP_PROP_EXPOSURE}
        with mock.patch.object(sm, "camera_platform", return_value="windows"):
            dongle = sm.CameraControls(0, capture=FakeDirectShow(missing=props))
            webcam = sm.CameraControls(0, capture=FakeDirectShow(missing=props - {cv2.CAP_PROP_EXPOSURE}))
        self.assertTrue(dongle.capture_box)
        self.assertEqual(set(dongle.settings), {"white balance"})
        self.assertFalse(webcam.capture_box)

    def test_the_panel_says_set_it_on_the_camera(self):
        class Nothing:
            def get(self, control, request=sm.GET_CUR):
                raise OSError("no")

            def close(self):
                pass

        project = sm.Project.create("Movie", self.tmp)
        app, display = make_app(self.tmp, [None, "k", None, "+", None])
        app.cam_setup.make_controls = lambda camera: sm.CameraControls(0, usb=Nothing(), name="Cam Link 4K")
        with mock.patch("builtins.print"):
            run(app, project.path)
        self.assertTrue(app.cam_setup.controls is None or app.cam_setup.controls.capture_box)
        self.assertIn("Brightness", app.toast_text)            # - / + still brighten the picture


class RendererSizeTests(TempDirTest):
    def test_the_studio_shows_big_movies_at_screen_size(self):
        p = sm.Project.create("Movie", self.tmp)
        with p.change():
            p.add_frame(solid(RED, (3840, 2160)), 0)
            p.items.append(sm.title_item(["Hi"]))
        view = sm.Renderer(p, fit_within=(sm.VIEW_W, sm.VIEW_H))
        self.assertEqual(view.size, (sm.VIEW_W, sm.VIEW_H))
        self.assertEqual(view.image(p.items[0]).shape, (sm.VIEW_H, sm.VIEW_W, 3))
        self.assertEqual(sm.Renderer(p).image(p.items[0]).shape, (2160, 3840, 3))     # full size for export
        p.doc["size"] = [1600, 1200]
        self.assertEqual(view.size, (768, 576))

    def test_the_cache_stays_within_its_budget(self):
        p = sm.Project.create("Movie", self.tmp)
        with p.change():
            for k in range(6):
                p.add_frame(solid((k * 40, 0, 0), (640, 360)), k)
        r = sm.Renderer(p)
        with mock.patch.object(sm.Renderer, "CACHE_BYTES", 640 * 360 * 3 * 2):
            for item in p.items:
                r.image(item)
        self.assertEqual(len(r.cache), 2)
        self.assertLessEqual(r.cached_bytes, 640 * 360 * 3 * 2)


class WindowTests(unittest.TestCase):
    def test_close_button_quits_but_minimizing_doesnt(self):
        states = iter([1.0, 1.0, 0.0, 1.0, cv2.error("gone")])

        def visible(*args):
            state = next(states)
            if isinstance(state, Exception):
                raise state
            return state

        with mock.patch.object(cv2, "namedWindow"), mock.patch.object(cv2, "setMouseCallback"), \
                mock.patch.object(cv2, "imshow") as imshow, mock.patch.object(cv2, "getWindowProperty", visible):
            display = sm.Display()
            for _ in range(4):
                display.show(np.zeros((4, 4, 3), np.uint8))
            self.assertFalse(display.closed)                   # 0 = minimized, not closed
            display.show(np.zeros((4, 4, 3), np.uint8))
            self.assertTrue(display.closed)
            self.assertEqual(imshow.call_count, 4)

    def test_a_scaled_window_still_clicks_in_the_right_place(self):
        with mock.patch.object(cv2, "namedWindow"), mock.patch.object(cv2, "setMouseCallback"), \
                mock.patch.object(cv2, "imshow") as imshow, mock.patch.object(cv2, "getWindowProperty",
                                                                             return_value=1.0):
            display = sm.Display(scale=0.5)
            display.show(np.zeros((sm.CANVAS_H, sm.CANVAS_W, 3), np.uint8))
            self.assertEqual(imshow.call_args[0][1].shape[:2], (sm.CANVAS_H // 2, sm.CANVAS_W // 2))
            display._mouse(cv2.EVENT_LBUTTONDOWN, 100, 50, 0, None)
            self.assertEqual(display.take_events(), [("down", 200, 100)])

    def test_backends_that_cant_tell_never_close(self):
        with mock.patch.object(cv2, "namedWindow"), mock.patch.object(cv2, "setMouseCallback"), \
                mock.patch.object(cv2, "imshow"), mock.patch.object(cv2, "getWindowProperty", return_value=-1.0):
            display = sm.Display()
            for _ in range(5):
                display.show(np.zeros((4, 4, 3), np.uint8))
            self.assertFalse(display.closed)


class PickerTests(TempDirTest):
    def test_new_movie_then_reopen(self):
        with mock.patch("builtins.print"):
            app, _ = make_app(self.tmp, ["enter", "D", "r", "a", "g", "o", "n", "enter", "space", None, "q", "q"])
            run(app)
            app.shutdown()
            self.assertEqual(app.project.name, "Dragon")
            # Next time, the last movie is already picked: ENTER opens it.
            app2, _ = make_app(self.tmp, ["enter", None])
            run(app2)
        self.assertEqual(app2.project.path, app.project.path)
        self.assertEqual(len(app2.project.items), 1)

    def test_a_movie_open_in_another_window_isnt_opened_twice(self):
        p = sm.Project.create("Movie", self.tmp)
        other = sm.Project.open(p.path)
        self.assertTrue(other.lock())
        self.addCleanup(other.unlock)
        app, _ = make_app(self.tmp, ["esc"])
        with mock.patch("builtins.print"):
            run(app, p.path)
        self.assertIsNone(app.project)
        self.assertIn("already open", app.toast_text)

    def backup_with_one_picture(self):
        p = sm.Project.create("Movie", self.tmp)
        with p.change():
            p.add_frame(solid(RED), 0)
        p.save(backup=True)
        with p.change():
            p.add_frame(solid(BLUE), 1)
            p.add_frame(solid(GREEN), 2)
        p.save(backup=True)
        backup = next(f for f in (p.folder / "backups").glob("*.stopmo") if len(sm.Project.load_doc(f)["items"]) == 1)
        return p, backup

    def test_opening_a_backup_while_another_window_has_the_movie_changes_nothing(self):
        p, backup = self.backup_with_one_picture()
        other = sm.Project.open(p.path, lock=True)
        self.addCleanup(other.unlock)
        app, _ = make_app(self.tmp, ["esc"])
        with mock.patch("builtins.print"):
            run(app, backup)
        self.assertIsNone(app.project)
        self.assertIn("already open", app.toast_text)
        self.assertEqual(len(sm.Project.load_doc(p.path)["items"]), 3)      # untouched on disk

    def test_going_back_to_a_backup_of_the_movie_you_have_open(self):
        p, backup = self.backup_with_one_picture()
        seen = {}
        app, _ = make_app(self.tmp, [None, lambda: seen.update(ok=app.open_path(backup)), None,
                                     lambda: seen.update(went_back=len(app.project.items)), "z", None])
        with mock.patch("builtins.print"):
            run(app, p.path)
        self.assertEqual(seen, {"ok": True, "went_back": 1})
        self.assertEqual(len(app.project.items), 3)                           # Z undid going back
        self.assertEqual(len(list((p.folder / "trash").glob("*"))), 0)       # nothing thrown away

    def test_escape_at_start_quits(self):
        app, _ = make_app(self.tmp, ["esc"])
        run(app)
        self.assertFalse(app.running)
        self.assertIsNone(app.project)

    def test_old_saves_show_up(self):
        legacy = self.tmp / "work" / "stopmotion_project_2026-01-02_03-04-05"
        legacy.mkdir(parents=True)
        cv2.imwrite(str(legacy / "frame_00001.png"), solid(RED))
        cwd = os.getcwd()
        os.chdir(legacy.parent)
        self.addCleanup(os.chdir, cwd)
        projects = self.tmp / "projects"
        with mock.patch("builtins.print"):
            app, _ = make_app(projects, ["right", "enter", None])
            run(app)
        self.assertTrue(app.project.name.startswith("Old save"))
        self.assertEqual(len(app.project.items), 1)
        kinds = [e["kind"] for e in app.picker_entries()]
        self.assertNotIn("legacy", kinds)                  # not offered twice


if __name__ == "__main__":
    unittest.main()
