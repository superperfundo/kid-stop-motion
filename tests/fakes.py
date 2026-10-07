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

    def switch(self, on_change=None):
        if self.cameras < 2:
            return False
        if on_change is not None:
            on_change()
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
        app.cam_setup.make_controls = lambda cam: sm.CameraControls(cam.index, usb=camera.uvc, name="Fake C920")
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


class FakeVideoCapture:
    """An OpenCV VideoCapture that answers size requests the way real cameras
    do. `modes` is {(w, h): fps}. A size it hasn't got gets, by `style`:
    "nearest" (macOS, Linux), "default" (keeps what it had) or "fail" (Windows
    DirectShow: no pictures until a size it has is asked for).

    Pictures arrive on a pretend clock (`clock`) at the mode's real rate;
    `raw` gives slower rates for uncompressed pictures, and `dshow` makes
    every new size go back to uncompressed, as DirectShow does. What it
    reports as its frame rate is always 30, true or not, like many drivers."""

    def __init__(self, modes, style="nearest", start=(640, 480), backend="AVFOUNDATION", raw=None, dshow=False):
        self.modes = dict(modes)
        self.style = style
        self.size = start
        self.backend = backend
        self.raw = dict(raw or {})
        self.dshow = dshow
        self.asked = []
        self.log = []             # every property set, in order
        self.fourcc = None
        self.now = 0.0
        self._want = {}

    def clock(self):
        return self.now

    def isOpened(self):
        return True

    def getBackendName(self):
        return self.backend

    def set(self, prop, value):
        import cv2
        names = {cv2.CAP_PROP_FOURCC: "fourcc", cv2.CAP_PROP_FPS: "fps",
                 cv2.CAP_PROP_FRAME_WIDTH: "width", cv2.CAP_PROP_FRAME_HEIGHT: "height"}
        self.log.append(names.get(prop, prop))
        if prop == cv2.CAP_PROP_FOURCC:
            self.fourcc = value
            return True
        if prop == cv2.CAP_PROP_FPS:
            return True
        if prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT):
            self._want[prop] = int(value)
            if len(self._want) == 2:
                want = (self._want[cv2.CAP_PROP_FRAME_WIDTH], self._want[cv2.CAP_PROP_FRAME_HEIGHT])
                self._want = {}
                self.asked.append(want)
                if want in self.modes:
                    self.size = want
                elif self.style == "nearest":
                    self.size = min(self.modes, key=lambda m: abs(m[0] - want[0]) + abs(m[1] - want[1]))
                elif self.style == "fail":
                    self.size = None
                if self.dshow:
                    self.fourcc = None
            return True
        return False

    def get(self, prop):
        import cv2
        if prop == cv2.CAP_PROP_FPS:
            return 30 if self.size else 0
        return 0

    def rate(self):
        import cv2
        if self.fourcc != cv2.VideoWriter_fourcc(*"MJPG") and self.size in self.raw:
            return self.raw[self.size]
        return self.modes.get(self.size, 30)

    def read(self):
        if self.size is None:
            return False, None
        self.now += 1 / self.rate()
        w, h = self.size
        return True, np.zeros((h, w, 3), np.uint8)

    def release(self):
        pass


class FakeV4L2:
    """Pretend /dev/videoN, answering the kernel (V4L2) requests the app makes.
    `controls` is {control id: [min, max, step, value]}; `menus` lists the
    valid entries of menu controls."""

    def __init__(self, controls, menus=None, name="HD Pro Webcam C920", disabled=()):
        self.controls = {cid: list(c) for cid, c in controls.items()}
        self.menus = menus or {}
        self.name = name
        self.disabled = set(disabled)
        self.opened = []

    def open(self, path, flags):
        import os
        self.opened.append(path)
        return os.open(os.devnull, os.O_RDONLY)

    def ioctl(self, fd, request, buf):
        import errno
        import struct
        if request == sm.VIDIOC_QUERYCAP:
            name = self.name.encode()
            buf[16:16 + len(name)] = name
            return 0
        cid = struct.unpack_from("<I", buf)[0]
        if cid not in self.controls:
            raise OSError(errno.EINVAL, "Invalid argument")
        low, high, step, value = self.controls[cid]
        if request == sm.VIDIOC_QUERYCTRL:
            flags = sm.V4L2_CTRL_FLAG_DISABLED if cid in self.disabled else 0
            # struct v4l2_queryctrl: id, type, name[32], min, max, step, default, flags, reserved[2] (68 bytes)
            buf[:] = struct.pack("<II32siiiiI2I", cid, 1, b"name", low, high, step, value, flags, 0, 0)
        elif request == sm.VIDIOC_QUERYMENU:
            if struct.unpack_from("<II", buf)[1] not in self.menus.get(cid, ()):
                raise OSError(errno.EINVAL, "Invalid argument")
        elif request == sm.VIDIOC_G_CTRL:
            buf[:] = struct.pack("<Ii", cid, value)
        elif request == sm.VIDIOC_S_CTRL:
            new = struct.unpack("<Ii", buf)[1]
            if not low <= new <= high:
                raise OSError(errno.ERANGE, "Numerical result out of range")
            self.controls[cid][3] = new
        return 0

    def value(self, cid):
        return self.controls[cid][3]


class FakeDirectShow:
    """OpenCV's DirectShow capture as the app sees it: get() gives -1 for what
    the camera hasn't got (and for auto exposure, which DirectShow can't
    report); set() refuses values outside the camera's real ranges, or off
    its `steps`. The picture's brightness follows the exposure, like a real
    camera."""

    def __init__(self, ranges=None, missing=(), auto_exposure=-6, steps=None):
        import cv2
        self.ranges = ranges or {cv2.CAP_PROP_EXPOSURE: (-11, -2), cv2.CAP_PROP_FOCUS: (0, 250),
                                 cv2.CAP_PROP_WB_TEMPERATURE: (2800, 6500)}
        self.steps = steps or {}
        self.values = {cv2.CAP_PROP_EXPOSURE: -4, cv2.CAP_PROP_FOCUS: 0, cv2.CAP_PROP_WB_TEMPERATURE: 4000,
                       cv2.CAP_PROP_AUTOFOCUS: 1, cv2.CAP_PROP_AUTO_WB: 1, cv2.CAP_PROP_AUTO_EXPOSURE: 1}
        self.missing = set(missing)
        self.auto_exposure = auto_exposure      # what auto picks (never reported)

    def getBackendName(self):
        return "DSHOW"

    def get(self, prop):
        import cv2
        if prop in self.missing or prop == cv2.CAP_PROP_AUTO_EXPOSURE:
            return -1
        return self.values.get(prop, -1)

    def set(self, prop, value):
        import cv2
        if prop in self.missing:
            return False
        if prop in self.ranges and not self.ranges[prop][0] <= value <= self.ranges[prop][1]:
            return False
        if prop in self.steps and (value - self.ranges[prop][0]) % self.steps[prop]:
            return False
        if prop == cv2.CAP_PROP_AUTOFOCUS:
            value = 1 if round(value) == 1 else 2      # DirectShow's flags
        self.values[prop] = value
        return True

    def exposure(self):
        import cv2
        if self.values[cv2.CAP_PROP_AUTO_EXPOSURE]:
            return self.auto_exposure
        return self.values[cv2.CAP_PROP_EXPOSURE]

    def picture(self, size):
        light = min(1.0, 0.25 * 2 ** (self.exposure() + 6))
        w, h = size
        return np.full((h, w, 3), 255 * light ** (1 / 2.2), np.uint8)
