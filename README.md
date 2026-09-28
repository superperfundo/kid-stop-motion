# Kid Stop Motion

A tiny stop motion animation studio for kids, using any webcam. Point the camera at some LEGO,
clay, or toys, tap the spacebar each time you move something, and export a movie.

It's one Python file, and everything it needs installs with pip. No accounts, no cloud, no
uploads — everything stays on your computer.

## What kids get

- **Spacebar to shoot.** One key does the main job, so a five-year-old can drive it.
- **Onion skin.** Press `O` to see a ghost of the last frame, which makes it much easier to move
  a character a small, even amount.
- **Auto-capture.** Press `T` to grab a frame every 2 seconds — good for hands-free animating,
  or for clouds and drawings.
- **Frame goal.** Press `N` to set a target like 60 frames; a progress bar fills up as you shoot.
  A 12 fps movie needs about 12 frames per second of film, so 60 frames is a 5-second short.
- **Undo.** Press `Z` when the cat walks through the set.
- **Preview before exporting.** Press `P` to watch it loop, then `E` to write an MP4.

## Setup

You need Python 3.9 or newer. The `python3` that comes with macOS is fine.

```bash
git clone https://github.com/superperfundo/kid-stop-motion.git
cd kid-stop-motion
python3 -m pip install -r requirements.txt
```

That installs OpenCV and, on macOS, a ready-built libusb for the camera controls. Nothing else is
needed.

[ffmpeg](https://ffmpeg.org/) is optional. If it's installed, the app uses it to write the MP4;
otherwise OpenCV's built-in encoder does the job. To install it on macOS with
[Homebrew](https://brew.sh/): `brew install ffmpeg`. On Debian/Ubuntu: `sudo apt install ffmpeg`.

## Run it

```bash
python3 stopmotion.py
```

A window opens showing the live camera with the controls along the top. If the picture is upside
down, press `R`. If it's the wrong camera, press `C`.

| Key | What it does |
| --- | --- |
| `SPACE` | Capture a frame |
| `P` | Preview the animation (any key stops it) |
| `E` | Export an MP4 into the current folder |
| `Z` | Delete the last frame |
| `S` | Save the frames into a `stopmotion_project_*` folder |
| `L` | Load the most recent saved project |
| `O` | Toggle onion skin |
| `T` | Toggle auto-capture every 2 seconds |
| `N` | Set a frame goal: type a number, then `ENTER` (`0` clears it) |
| `C` | Switch to the next camera |
| `R` | Rotate the picture half a turn |
| `F` | Lock or unlock focus |
| `X` | Lock or unlock exposure |
| `W` | Lock or unlock white balance |
| `[` / `]` | Nudge focus |
| `-` / `+` | Darker / brighter |
| `Q` or `ESC` | Quit |

Captured frames live in a temporary folder while you work, so **press `S` to save** if you want to
come back to a project later. Exported movies are written to the folder you ran the command from,
named with a timestamp.

## Tips for a good shoot

- Tape the camera down, or every frame will jump.
- Once the shot is set up, lock focus, exposure and white balance (`F`, `X` and `W`) so the camera
  doesn't refocus or change colour every time a hand reaches in.
- Turn off automatic room lights and close the curtains if you can. Changing daylight makes the
  movie flicker.
- Move things a *little* between shots. Small moves look smooth; big moves look jumpy.
- 12 frames per second is the default, so 24 frames is two seconds of film.

## Settings

Near the top of `stopmotion.py`:

- `FPS = 12` — frames per second of the exported movie.
- `ROTATE_180 = False` — set this to `True` if your camera is mounted upside down (tabletop rigs
  often end up that way), so the picture always starts the right way up. `R` flips it for the
  current camera while you work.
- `CAMERA_INDEX = 0` — which camera to start with. `C` switches while you work.

## Focus, exposure and white balance

Webcams keep adjusting focus, brightness and colour on their own. That's fine for video calls, but
in stop motion it makes the movie flicker and go soft whenever a hand reaches into the shot. Once
the picture looks right, lock it:

- `F`, `X` and `W` lock focus, exposure and white balance where the camera has settled. Press
  again to go back to auto. The top right of the window shows what's locked, and at what. Webcams
  don't report the exposure or white balance their auto modes pick, so `X` and `W` measure the
  picture and adjust for a moment until it matches how it looked on auto. The picture holds still
  while that happens.
- The **Focus**, **Exposure** and **White balance** sliders set a value by hand, and lock it there.
  The picture can pause while you drag a slider, so for fine focusing use `[` and `]`, which nudge
  focus a step at a time with the picture live.
- `-` and `+` make the exposure a third of a stop darker or brighter, and lock it (starting from
  what auto had, if it was on auto). On cameras without exposure control, the app darkens or
  brightens the picture itself instead.

These controls work on macOS with USB webcams. The app talks to the camera directly with standard
USB Video Class requests, which almost every USB webcam understands, because macOS's own camera
APIs can't set a focus distance. Each camera gets the controls it actually has: a webcam with a
fixed-focus lens, for instance, shows `FOCUS: n/a` but can still lock exposure and white balance.
Built-in laptop cameras aren't USB, so they show `n/a` for all three, and `-` / `+` brighten the
picture in the app instead. When the app starts, and when you switch cameras, the terminal lists
what the camera can control. The app puts the camera back on auto when you switch cameras or quit.
These controls aren't available on Windows or Linux yet; everything else works there.

### How fine the control is depends on the camera

The app can only use the settings a camera actually has, and some are coarser than they claim.
Many webcams, the Logitech C920 included, accept any exposure time but only really use a handful,
about a stop apart. On those cameras:

- The Exposure slider and `-` / `+` only change the picture when they cross one of the camera's
  steps. In between, the number changes but the picture doesn't.
- `X` can only lock to the nearest step, so the locked picture may be a bit brighter or darker
  than it was on auto. The terminal says so when that happens, e.g.
  `Exposure LOCKED at 76ms (as close as this camera gets: 0.4 stops brighter)`.

The app doesn't paper over this by turning up the camera's gain, which would add noise. If you
need something in between, adjust the light instead: move a lamp closer or further away, or put a
sheet of paper over it.

If you switch cameras partway through, new frames are cropped to match the size of the first
camera's, so the movie still exports cleanly.

## Camera permissions

The first run may need permission to use the camera. On macOS, allow it under
**System Settings → Privacy & Security → Camera** for whichever app runs Python (usually Terminal).

## A note on privacy

This app writes frames and movies to your own computer and never sends anything anywhere. The
included `.gitignore` deliberately excludes `*.mp4`, `frame_*.png`, and `stopmotion_project_*`
folders, so that if you version your own animation projects you don't accidentally publish footage
of your kids.

## License

[MIT](LICENSE) © 2026 R.F. Chapman
