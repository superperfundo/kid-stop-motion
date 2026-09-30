"""Stand-ins for the webcam, the window and the microphone, so the whole
studio can be driven by scripted key presses without any hardware."""

import math

import numpy as np

import stopmotion as sm


class FakeUVC:
    """A pretend USB webcam's controls, behaving like a Logitech C920: on auto
    it doesn't report the exposure / white balance it picked (it reports the
    last value we set), and a coarse camera only really uses a few exposures."""

    RANGES = {sm.FOCUS: (0, 250, 5), sm.EXPOSURE_TIME: (3, 2047, 1), sm.WHITE_BALANCE: (2000, 6500, 10)}

    def __init__(self, auto_exposure=600, auto_white_balance=5000, exposure_steps=None, has_focus=True):
        self.values = {sm.FOCUS: 100, sm.FOCUS_AUTO: 1, sm.AE_MODE: sm.AE_APERTURE_PRIORITY,
                       sm.EXPOSURE_TIME: 300, sm.WHITE_BALANCE: 4000, sm.WHITE_BALANCE_AUTO: 1}
        self.auto_exposure = auto_exposure
        self.auto_white_balance = auto_white_balance
        self.exposure_steps = exposure_steps
        self.has_focus = has_focus
        self.closed = False

    def get(self, control, request=sm.GET_CUR):
        if control in (sm.FOCUS, sm.FOCUS_AUTO) and not self.has_focus:
            raise OSError("LIBUSB_ERROR_PIPE")
        if control == sm.AE_MODE and request == sm.GET_RES:
            return sm.AE_MANUAL | sm.AE_AUTO | sm.AE_APERTURE_PRIORITY
        if control in self.RANGES and request != sm.GET_CUR:
            low, high, step = self.RANGES[control]
            return {sm.GET_MIN: low, sm.GET_MAX: high, sm.GET_RES: step}[request]
        return self.values[control]

    def set(self, control, value):
        self.values[control] = int(value)

    def close(self):
        self.closed = True

    def exposure(self):
        if self.values[sm.AE_MODE] != sm.AE_MANUAL:
            return self.auto_exposure
        wanted = self.values[sm.EXPOSURE_TIME]
        if self.exposure_steps:
            return min(self.exposure_steps, key=lambda step: abs(math.log(step / wanted)))
        return wanted

    def white_balance(self):
        return self.auto_white_balance if self.values[sm.WHITE_BALANCE_AUTO] else self.values[sm.WHITE_BALANCE]

    def picture(self, size):
        """What the camera sees: brighter with more exposure, redder with a higher white balance."""
        light = min(1.0, 0.25 * self.exposure() / 600)
        red, blue = light * math.sqrt(self.white_balance() / 5000), light * math.sqrt(5000 / self.white_balance())
        pixel = [255 * min(1.0, v) ** (1 / 2.2) for v in (blue, light, red)]
        w, h = size
        return np.full((h, w, 3), pixel, np.uint8)


class FakeCamera:
    def __init__(self, color=(0, 0, 255), size=(1280, 720), uvc=None, cameras=1):
        self.color = color
        self.size = size
        self.uvc = uvc
        self.cameras = cameras
        self.index = 0
        self.generation = 1
        self.cap = object()
        self.flushes = 0

    @property
    def ok(self):
        return self.cap is not None

    def frame(self):
        if self.uvc is not None:
            return self.uvc.picture(self.size)
        w, h = self.size
        img = np.full((h, w, 3), self.color, np.uint8)
        img[:h // 8, :w // 8] = 255          # a white corner, to tell which way up it is
        return img

    def switch(self):
        if self.cameras < 2:
            return False
        self.index = (self.index + 1) % self.cameras
        self.generation += 1
        return True

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
    camera = camera or FakeCamera()
    app = sm.App(display, camera, projects_dir=tmp)
    if camera.uvc is not None:
        app.cam_setup.make_controls = lambda index: sm.CameraControls(index, usb=camera.uvc, name="Fake C920")
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
