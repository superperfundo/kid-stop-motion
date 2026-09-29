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
from fakes import FakeCamera, FakeRecorder, make_app, run  # noqa: E402

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
        p = sm.Project.open(legacy)           # opening an old save imports it (in PROJECTS_DIR)...
        self.addCleanup(shutil.rmtree, p.folder, ignore_errors=True)
        self.assertEqual(len(p.items), 2)
        self.assertEqual(p.doc["imported_from"], str(legacy))
        p2 = sm.Project.import_legacy(legacy, self.tmp)       # ...or wherever you ask
        self.assertEqual(p2.folder.parent, self.tmp)
        self.assertTrue(p2.name.startswith("Old save 2026-05-01"))

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
        self.assertEqual([it["kind"] for it in again.items], ["frame", "title", "frame"])
        self.assertEqual(again.items[1]["lines"], ["Hi"])
        self.assertGreater(again.items[1]["hold"], 0)
        self.assertIsNone(again.items[1]["transition"])
        self.assertEqual(again.items[0]["transition"]["type"], "fade")
        self.assertNotEqual(again.items[0]["id"], again.items[2]["id"])
        self.assertGreater(again.doc["next_frame"], 1)      # never overwrite a picture
        self.assertTrue(any("missing" in note for note in again.notes))

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
        p = sm.Project.create("It's a Movie!", self.tmp)
        launcher = sm.write_launcher(p)
        self.assertTrue(os.access(launcher, os.X_OK))
        self.assertIn("It's a Movie!", launcher.name)
        subprocess.run(["sh", "-n", str(launcher)], check=True)
        text = launcher.read_text()
        self.assertIn("It'\"'\"'s a Movie!.stopmo", text)       # safely quoted for the shell


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
        colors = [cv2.imread(str(app.project.file_path(it["file"])))[0, 0] for it in app.project.items]
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
        self.assertGreater(first[0, 0, 0], 200)               # blue is first now
        self.assertEqual(app.cursor, 1)

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
