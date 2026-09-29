"""Stand-ins for the webcam, the window and the microphone, so the whole
studio can be driven by scripted key presses without any hardware."""

import numpy as np

import stopmotion as sm


class FakeCamera:
    def __init__(self, color=(0, 0, 255), size=(1280, 720)):
        self.color = color
        self.size = size
        self.cap = object()
        self.flushes = 0

    @property
    def ok(self):
        return self.cap is not None

    def frame(self):
        w, h = self.size
        return np.full((h, w, 3), self.color, np.uint8)

    def read(self):
        return None if self.cap is None else self.frame()

    def flush(self):
        self.flushes += 1

    def release(self):
        self.cap = None


class ScriptDone(Exception):
    pass


class FakeDisplay:
    """Plays back a script. Each entry is used by one wait() call:
    a key (a str like "p", a special name like "left", or an int code),
    None for "no key this time", ("click", x, y), or a callable run for
    its side effects (e.g. to change the camera)."""

    CODES = {"left": 65361, "right": 65363, "up": 65362, "down": 65364, "home": 65360, "end": 65367,
             "enter": 13, "esc": 27, "tab": 9, "space": 32, "backspace": 8, "delete": 65535}

    def __init__(self, script=()):
        self.script = list(script)
        self.events = []
        self.frames = []
        self.shown = 0

    def show(self, canvas):
        self.shown += 1
        self.last = canvas
        if len(self.frames) < 3:
            self.frames.append(canvas)
        else:
            self.frames[-1] = canvas

    def wait(self, ms):
        if not self.script:
            raise ScriptDone
        step = self.script.pop(0)
        if callable(step):
            step()
            return -1
        if step is None:
            return -1
        if isinstance(step, tuple):
            kind = step[0]
            if kind == "click":
                self.events += [("down", step[1], step[2]), ("up", step[1], step[2])]
            elif kind == "drag":
                _, x0, y0, x1, y1 = step
                self.events += [("down", x0, y0), ("drag", (x0 + x1) // 2, y0), ("drag", x1, y1), ("up", x1, y1)]
            return -1
        if isinstance(step, int):
            return step
        if step in self.CODES:
            return self.CODES[step]
        return ord(step)

    def take_events(self):
        events, self.events = self.events, []
        return events

    def close(self):
        pass


class FakeRecorder:
    samples = None

    def __init__(self):
        self.level = 0.5
        self.samplerate = 16000

    def start(self):
        pass

    def stop(self):
        if FakeRecorder.samples is not None:
            return FakeRecorder.samples, self.samplerate
        t = np.arange(self.samplerate) / self.samplerate
        return (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16).reshape(-1, 1), self.samplerate


class FakeSpeaker:
    def __init__(self):
        self.played = []

    def play(self, samples, rate):
        self.played.append((len(samples), rate))
        return True

    def stop(self):
        pass

    def click(self):
        pass


def make_app(tmp, script=(), camera=None):
    display = FakeDisplay(script)
    app = sm.App(display, camera or FakeCamera(), projects_dir=tmp)
    app.speaker = FakeSpeaker()
    app.recorder_factory = FakeRecorder
    app.countdown_seconds = 0
    return app, display


def run(app, open_path=None):
    try:
        app.run(open_path)
    except ScriptDone:
        pass
    finally:
        app.shutdown()
