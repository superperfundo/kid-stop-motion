import json
import os
import shutil
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
from fakes import FakeCamera, FakeRecorder, FakeUVC, make_app, run  # noqa: E402

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
        with p.change():
            p.doc["audio"] = {"file": "audio/voice.wav"}
        app, display = make_app(self.tmp, ["p", None, "x", None])
        with mock.patch("builtins.print"), mock.patch.object(sm, "sound_module", return_value=object()):
            run(app, p.path)
        self.assertIsNotNone(display.last)
        self.assertIsNone(app.audio_seconds())

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
            p.set_audio((np.sin(2 * np.pi * 330 * t) * 9000).astype(np.int16), rate)
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
        audio = app.project.audio_file()
        self.assertIsNotNone(audio)
        samples, rate = sm.read_wav(audio)
        self.assertEqual(rate, 16000)
        self.assertEqual(app.speaker.played, [(len(samples), rate)])

    def test_silent_recording_is_not_kept(self):
        FakeRecorder.samples = np.zeros((1000, 1), np.int16)
        self.addCleanup(setattr, FakeRecorder, "samples", None)
        with mock.patch.object(sm, "sound_problem", return_value=None):
            app, _ = self.open_app(["r", None, "x", None])
        self.assertIsNone(app.project.audio_file())
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
