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
  C      switch camera
  R      rotate the picture 180 degrees
  F      lock / unlock focus
  X      lock / unlock exposure
  W      lock / unlock white balance
  [ / ]  nudge focus
  - / +  darker / brighter
  Q/ESC  quit

The Focus, Exposure and White balance sliders lock that setting at the
slider's value.

Assumes 12 fps.
"""

import ctypes
import ctypes.util
import cv2
import math
import numpy as np
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from datetime import datetime

# Set this to True if your camera is mounted upside down (tabletop rigs often
# end up that way). R flips the picture in the app, separately for each camera.
ROTATE_180 = False

FPS = 12

# Which camera to start with (0 = the first one). C switches in the app.
CAMERA_INDEX = 0

# Each press of - or + changes exposure by this many stops. On cameras we
# can't control, the app brightens the picture itself, up to 2 stops.
EXPOSURE_STEP = 1 / 3
MAX_EXPOSURE_STEPS = 6

WINDOW_NAME = "Kid Stop Motion - SPACE capture | P preview | E export | Z undo | O onion | T auto | N target | Q quit"


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

    def __init__(self, index):
        self.name = f"Camera {index + 1}"
        self.usb = None
        self.settings = {}  # "focus" / "exposure" / "white balance" -> LockableSetting

        if sys.platform != "darwin":
            return
        try:
            info = mac_camera_info(index)
        except (OSError, ValueError):
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
            return  # not a USB camera
        try:
            self.usb = UVCCamera(usb_id >> 16 & 0xFFFF, usb_id & 0xFFFF, usb_id >> 32)
        except OSError as err:
            print(f"No focus, exposure or white balance controls for {self.name}: {err}")
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
        """Lock a setting at a particular value."""
        try:
            self.settings[name].lock(value)
        except OSError as err:
            print(f"Couldn't set {name}: {err}")

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
        return math.log(value)

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


SLIDERS = {"focus": "Focus", "exposure": "Exposure", "white balance": "White balance"}


def open_window(controls):
    """(Re)create the window with a slider for each setting the camera has.

    Returns the sliders' positions, so we can tell when someone moves one.
    """
    cv2.destroyAllWindows()
    cv2.namedWindow(WINDOW_NAME)
    positions = {}
    for name, setting in controls.settings.items():
        positions[name] = setting.to_slider(setting.value)
        cv2.createTrackbar(SLIDERS[name], WINDOW_NAME, positions[name], 100, lambda _: None)
    return positions


def open_camera(index):
    """Open a camera and check it delivers a picture; None if it doesn't."""
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        # Try to set a friendly webcam resolution.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if cap.read()[0]:
            return cap
    cap.release()
    return None


def next_camera(index):
    """Open the camera after `index`, wrapping round to the first.

    Returns (index, capture), or None if there's no other camera.
    """
    for candidate in (index + 1, 0):
        if candidate != index:
            cap = open_camera(candidate)
            if cap is not None:
                return candidate, cap
    return None


def fit_to_size(frame, width, height):
    """Scale and centre-crop a frame to exactly width x height."""
    h, w = frame.shape[:2]
    if (w, h) == (width, height):
        return frame
    scale = max(width / w, height / h)
    frame = cv2.resize(frame, (max(width, round(w * scale)), max(height, round(h * scale))))
    top = (frame.shape[0] - height) // 2
    left = (frame.shape[1] - width) // 2
    return frame[top:top + height, left:left + width]


def exposure_lut(stops):
    """Lookup table that brightens (or darkens, if negative) by `stops`."""
    linear = (np.arange(256) / 255.0) ** 2.2
    return np.clip(np.round((linear * 2 ** stops) ** (1 / 2.2) * 255), 0, 255).astype(np.uint8)


def setting_chip(label, controls, name, matching=False):
    """Status text and colour for focus, exposure or white balance."""
    setting = controls.settings.get(name)
    if setting is None:
        return f"{label}: n/a", (150, 150, 150)
    if matching:
        return f"{label}: LOCKING...", (0, 215, 255)
    if setting.locked:
        return f"{label}: LOCKED {controls.describe(name)}", (0, 215, 255)
    return f"{label}: AUTO", (255, 255, 255)


def has_ffmpeg():
    return shutil.which("ffmpeg") is not None

def make_export_name():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"stop_motion_export_{timestamp}.mp4"

def draw_ui(frame, captured_count, is_preview=False, onion_frame=None, frame_target=0,
            camera_name="", status_chips=(), goal_entry=None):
    duration = captured_count / FPS if captured_count else 0

    overlay = frame.copy()
    h, w = frame.shape[:2]

    # Onion skin: blend the last captured frame as a ghost
    if onion_frame is not None and not is_preview:
        frame = cv2.addWeighted(frame, 0.5, onion_frame, 0.5, 0)

    cv2.rectangle(overlay, (0, 0), (w, 122), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

    status = "PREVIEWING" if is_preview else "LIVE CAMERA"
    cv2.putText(frame, status, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Focus / exposure / white balance status, right-aligned on the top line
    x = w - 16
    for text, color in reversed(status_chips):
        (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        x -= text_w
        cv2.putText(frame, text, (x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        x -= 24

    info = f"Frames: {captured_count}   Duration: {duration:.1f}s   FPS: {FPS}"
    if camera_name:
        info += f"   Camera: {camera_name}"
    cv2.putText(frame, info, (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    controls = "SPACE=capture  P=preview  E=export  Z=undo  S=save  L=load  O=onion  T=auto  N=target  Q=quit"
    cv2.putText(frame, controls, (16, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)
    camera_controls = ("C=camera  R=rotate  F/X/W=lock focus/exposure/white balance  [ ]=focus  "
                       "-/+ = darker/brighter")
    cv2.putText(frame, camera_controls, (16, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)

    # Frame target progress bar
    if frame_target > 0:
        progress = min(captured_count / frame_target, 1.0)
        bar_y = 132
        bar_h = 18
        bar_w = w - 32
        bar_x = 16
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        fill_w = int(bar_w * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 200, 0), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)
        label = f"Goal: {captured_count}/{frame_target} frames"
        cv2.putText(frame, label, (bar_x, bar_y + bar_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Frame goal entry box (opened with N)
    if goal_entry is not None:
        box_w, box_h = 560, 120
        box_x, box_y = (w - box_w) // 2, (h - box_h) // 2
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 255, 255), 2)
        cv2.putText(frame, f"Frame goal: {goal_entry}_", (box_x + 24, box_y + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(frame, "Type a number, then ENTER.  0 clears.  ESC cancels.", (box_x + 24, box_y + 94),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)

    return frame


def save_frame(frame, frames_dir, frame_number):
    filename = frames_dir / f"frame_{frame_number:05d}.png"
    cv2.imwrite(str(filename), frame)
    return filename


def export_with_opencv(frames_dir, output_path):
    """Write the MP4 with OpenCV's own encoder, for when ffmpeg isn't installed."""
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    first = cv2.imread(str(frame_files[0]))
    height, width = first.shape[:2]

    print("\nExporting MP4...")
    for codec in ("avc1", "mp4v"):  # H.264 if this OpenCV can, otherwise MPEG-4
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*codec), FPS, (width, height))
        if writer.isOpened():
            break
    else:
        print("Export failed: no video encoder available. Installing ffmpeg would fix this.")
        return False

    for file in frame_files:
        img = cv2.imread(str(file))
        if img is not None:
            writer.write(fit_to_size(img, width, height))
    writer.release()
    print(f"\nExport complete: {output_path}")
    return True


def export_mp4(frames_dir, output_path):
    if not has_ffmpeg():
        return export_with_opencv(frames_dir, output_path)

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


def main():
    camera_index = CAMERA_INDEX
    cap = open_camera(camera_index)

    if cap is None:
        print("Could not open webcam.")
        print("On macOS, you may need to allow Terminal or your Python app to access the camera:")
        print("System Settings → Privacy & Security → Camera")
        return

    controls = CameraControls(camera_index)

    with tempfile.TemporaryDirectory() as tmp:
        frames_dir = Path(tmp)
        frame_files = []
        onion_skin = False
        last_frame = None
        auto_capture = False
        next_capture_time = 0.0
        frame_target = 0
        goal_entry = None   # digits typed so far while the frame goal box is open
        rotation = {}       # camera index -> rotate 180?
        frame_size = None   # every frame is fitted to the first camera's size
        exposure_steps = 0
        exposure_table = None
        slider_positions = open_window(controls)
        next_refresh = 0.0
        matches = {}  # exposure / white balance locks still settling: name -> PictureMatch

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
        print("C     = switch camera")
        print("R     = rotate 180 degrees")
        print("F     = lock / unlock focus")
        print("X     = lock / unlock exposure")
        print("W     = lock / unlock white balance")
        print("[ / ] = nudge focus")
        print("- / + = darker / brighter")
        print("Q     = quit\n")
        print(f"Camera {camera_index + 1}: {controls.name} ({controls.summary()})")

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    print("Could not read from webcam.")
                    break

                if rotation.get(camera_index, ROTATE_180):
                    frame = cv2.flip(frame, -1)

                # Keep every frame the same size, even after switching cameras.
                if frame_size is None:
                    frame_size = (frame.shape[1], frame.shape[0])
                frame = fit_to_size(frame, *frame_size)
                camera_frame = frame  # before our own brightness change

                for name, match in list(matches.items()):
                    if match.update(camera_frame):
                        if controls.settings[name].locked:
                            shortfall = match.shortfall()
                            note = f" (as close as this camera gets: {shortfall})" if shortfall else ""
                            print(f"{name.capitalize()} LOCKED at {controls.describe(name)}{note}")
                        del matches[name]
                if matches:
                    # Keep showing the picture from auto until the lock has settled.
                    frame = next(iter(matches.values())).frame

                if exposure_steps:
                    frame = cv2.LUT(frame, exposure_table)

                exposure_text, exposure_color = setting_chip("EXPOSURE", controls, "exposure", "exposure" in matches)
                if exposure_steps:
                    exposure_text += f" {exposure_steps * EXPOSURE_STEP:+.1f}"
                chips = [setting_chip("FOCUS", controls, "focus"),
                         (exposure_text, exposure_color),
                         setting_chip("WB", controls, "white balance", "white balance" in matches)]

                display = draw_ui(frame, len(frame_files), onion_frame=last_frame if onion_skin else None,
                                  frame_target=frame_target, camera_name=controls.name,
                                  status_chips=chips, goal_entry=goal_entry)
                cv2.imshow(WINDOW_NAME, display)

                key = cv2.waitKey(1) & 0xFF

                # A moved slider locks its setting at the new value.
                for name, setting in controls.settings.items():
                    position = cv2.getTrackbarPos(SLIDERS[name], WINDOW_NAME)
                    if position != slider_positions[name]:
                        controls.set_value(name, setting.from_slider(position))
                        slider_positions[name] = position
                        matches.pop(name, None)

                # Sliders for settings on auto follow what the camera picks.
                if time.time() >= next_refresh:
                    controls.refresh()
                    next_refresh = time.time() + 0.5
                for name, setting in controls.settings.items():
                    position = setting.to_slider(setting.value)
                    if position != slider_positions[name]:
                        cv2.setTrackbarPos(SLIDERS[name], WINDOW_NAME, position)
                        slider_positions[name] = position

                # While the frame goal box is open, digits, Backspace, Enter and
                # ESC edit it. Any other key closes it and does its usual job.
                # (255 means no key.)
                if goal_entry is not None and key != 255:
                    if ord("0") <= key <= ord("9"):
                        goal_entry = (goal_entry + chr(key))[:4]
                        key = 255
                    elif key in (8, 127):
                        goal_entry = goal_entry[:-1]
                        key = 255
                    elif key in (10, 13):
                        frame_target = int(goal_entry or 0)
                        if frame_target:
                            remaining = max(frame_target - len(frame_files), 0)
                            print(f"Target set to {frame_target}.  {remaining} frame(s) to go.")
                        else:
                            print("Target cleared.")
                        goal_entry = None
                        key = 255
                    else:
                        goal_entry = None
                        if key == 27:
                            key = 255

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
                    goal_entry = ""

                elif key == ord("c"):
                    switched = next_camera(camera_index)
                    if switched is None:
                        print("No other camera found.")
                    else:
                        controls.close()
                        cap.release()
                        camera_index, cap = switched
                        controls = CameraControls(camera_index)
                        slider_positions = open_window(controls)
                        matches = {}
                        exposure_steps = 0
                        print(f"Switched to camera {camera_index + 1}: {controls.name} ({controls.summary()})")

                elif key == ord("r"):
                    rotation[camera_index] = not rotation.get(camera_index, ROTATE_180)

                elif key == ord("f"):
                    print(controls.toggle("focus"))

                elif key in (ord("x"), ord("w")):
                    name = "exposure" if key == ord("x") else "white balance"
                    setting = controls.settings.get(name)
                    if setting is not None and not setting.locked:
                        print(f"Locking {name}...")
                        matches[name] = MATCHES[name](controls, camera_frame)
                    else:
                        matches.pop(name, None)
                        print(controls.toggle(name))

                elif key in (ord("["), ord("]")):
                    focus = controls.settings.get("focus")
                    if focus is None:
                        print(f"{controls.name} can't set focus.")
                    else:
                        step = focus.step if key == ord("]") else -focus.step
                        controls.set_value("focus", focus.value + step)

                elif key in (ord("-"), ord("_"), ord("="), ord("+")):
                    step = -1 if key in (ord("-"), ord("_")) else 1
                    exposure = controls.settings.get("exposure")
                    if exposure is None:
                        # No exposure control on this camera, so brighten the picture ourselves.
                        exposure_steps = max(-MAX_EXPOSURE_STEPS, min(MAX_EXPOSURE_STEPS, exposure_steps + step))
                        exposure_table = exposure_lut(exposure_steps * EXPOSURE_STEP)
                        print(f"Brightness {exposure_steps * EXPOSURE_STEP:+.1f} stops")
                    elif "exposure" in matches:
                        matches["exposure"].target += step * EXPOSURE_STEP * math.log(2)
                    elif exposure.locked:
                        controls.set_value("exposure", exposure.value * 2 ** (step * EXPOSURE_STEP))
                        print(f"Exposure {exposure.format(exposure.value)}")
                    else:
                        # On auto: lock a step brighter or darker than auto has it.
                        print("Locking exposure...")
                        matches["exposure"] = ExposureMatch(controls, camera_frame,
                                                            step * EXPOSURE_STEP * math.log(2))

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
        finally:
            controls.close()
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
