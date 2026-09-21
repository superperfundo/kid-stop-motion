# Kid Stop Motion

A tiny stop motion animation studio for kids, using any webcam. Point the camera at some LEGO,
clay, or toys, tap the spacebar each time you move something, and export a movie.

It's one Python file with one dependency. No accounts, no cloud, no uploads — everything stays on
your computer.

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

You need Python 3.9+ and [ffmpeg](https://ffmpeg.org/) (used to write the MP4).

```bash
git clone https://github.com/superperfundo/kid-stop-motion.git
cd kid-stop-motion
python3 -m pip install -r requirements.txt
```

Install ffmpeg — on macOS with [Homebrew](https://brew.sh/):

```bash
brew install ffmpeg
```

On Debian/Ubuntu: `sudo apt install ffmpeg`. On Windows, see the ffmpeg download page.

## Run it

```bash
python3 stopmotion.py
```

A window opens showing the live camera with the controls along the top. If the picture is upside
down, see [Settings](#settings) — there's a one-line switch for that.

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
| `N` | Set a frame goal (typed in the terminal) |
| `Q` or `ESC` | Quit |

Captured frames live in a temporary folder while you work, so **press `S` to save** if you want to
come back to a project later. Exported movies are written to the folder you ran the command from,
named with a timestamp.

## Tips for a good shoot

- Tape the camera down, or every frame will jump.
- Turn off automatic room lights and close the curtains if you can. Changing daylight makes the
  movie flicker.
- Move things a *little* between shots. Small moves look smooth; big moves look jumpy.
- 12 frames per second is the default, so 24 frames is two seconds of film.

## Settings

Near the top of `stopmotion.py`:

- `FPS = 12` — frames per second of the exported movie.
- `ROTATE_180 = True` — the picture is rotated half a turn by default, because most tabletop rigs
  end up with the camera clamped upside down. **If your picture appears upside down, set this to
  `False`.**

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
