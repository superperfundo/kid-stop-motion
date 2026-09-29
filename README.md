# Kid Stop Motion

A tiny stop motion animation studio for kids, using any webcam. Point the camera at some LEGO,
clay, or toys, tap the spacebar each time you move something, and make a movie. Add title cards,
transitions and your own voice.

It's one Python file. No accounts, no cloud, no uploads: everything stays on your computer.

## What kids get

- **Spacebar to shoot.** One key (or the big red button) does the main job, so a five-year-old
  can drive it.
- **Ghost (onion skin).** A see-through ghost of the last picture sits on top of the camera, so
  it's easy to move a character a small, even amount. It's on by default, and it stays put
  after playing the movie, fiddling with the camera, or undoing. `O` turns it off and on.
- **A timeline of every picture.** Click a picture (or use `<` `>`) to look at it. The camera
  is a tile in the timeline too: put it anywhere and new pictures go *there*, so you can add
  frames in the middle of a movie.
- **Fix mistakes.** Delete a picture, copy it, make it stay on screen longer or shorter, move
  it (or just drag it), and undo / redo anything.
- **Title cards.** Type the words right in the window and pick from six color styles.
- **Transitions.** Put a Fade, Melt, Wipe, Circle or Slide before any picture, or at the very
  end of the movie.
- **A voice track.** Press `R` and talk while the movie plays: the microphone records a scratch
  track that plays with the movie and goes into the MP4.
- **Auto-snap, a picture goal and a speed control.** `T` takes a picture every 2 seconds, `N`
  sets a goal with a progress bar, `V` makes the movie faster or slower.
- **Nothing gets lost.** Every picture is saved the moment it's taken. There's no "save"
  step to forget.

## Setup

You need Python 3.9+ and [ffmpeg](https://ffmpeg.org/) (used to write the MP4 with sound).

```bash
git clone https://github.com/superperfundo/kid-stop-motion.git
cd kid-stop-motion
python3 -m pip install -r requirements.txt
```

Install ffmpeg on macOS with [Homebrew](https://brew.sh/):

```bash
brew install ffmpeg
```

On Debian/Ubuntu: `sudo apt install ffmpeg libportaudio2` (the second one is for the
microphone). On Windows, see the ffmpeg download page.

`requirements.txt` installs OpenCV and [sounddevice](https://python-sounddevice.readthedocs.io/)
(for the microphone). If sounddevice isn't there, everything except sound still works. Without
ffmpeg, movies are still made, just without sound.

## Run it

```bash
python3 stopmotion.py
```

First you'll see your movies: pick one to keep working on, or **New movie**. Then the studio
opens with the live camera, big buttons down the right, and the timeline along the bottom. If
the picture is upside down, see [Settings](#settings): there's a one-line switch for that.

| Key | What it does |
| --- | --- |
| `SPACE` | Take a picture. (If you're looking at an old picture, the first press moves the camera there.) |
| `P` | Play the movie with its voice track (any key stops it) |
| `O` | Ghost of the last picture on / off |
| `T` | Auto-snap: a picture every 2 seconds |
| `C` | Add a title card (or edit the one you picked) |
| `F` | Transition before the picked picture. Press again for the next kind. With the camera at the end, it's the ending. |
| `R` | Record a voice track |
| `V` | Speed: pictures per second |
| `E` | Make an MP4 movie |
| `L` | Your movies: open another one, or start a new one |
| `<` `>` or `,` `.` | Pick a picture. `HOME` / `END` jump to the start / the camera at the end. |
| `X` (or Delete) | Delete the picked picture |
| `D` | Copy the picked picture |
| `+` / `-` | Show it for longer / shorter (title cards: half a second at a time) |
| `[` / `]` | Move it left / right (or drag it with the mouse) |
| `Z` / `Y` | Undo / redo |
| `N` | Set a picture goal |
| `S` | Save now (it also saves by itself) |
| `H` | Help |
| `Q` or `ESC` | Quit (press it twice, so a stray key doesn't close the studio) |

Everything also works with the mouse: the buttons on the right, the row of buttons at the
bottom, clicking or dragging timeline tiles, and clicking a purple transition label to change it.

## Your movies are saved as you go

Each movie is a folder in **`~/Stop Motion Projects`** (your home folder):

```
Stop Motion Projects/
  Dragon Race/
    Dragon Race.stopmo          <- the project file: this is what you open
    Open Dragon Race.command    <- double-click to open it (.bat on Windows, .sh on Linux)
    Dragon Race 2026-09-29 16-20-05.mp4   <- movies you made
    frames/                     <- every picture
    audio/                      <- the voice track
    backups/                    <- older copies of the project file
    trash/                      <- pictures you deleted, just in case
```

- **Every change is written straight away**, and written so that a crash or a flat battery
  can't leave a half-written file behind.
- **Backups:** each session, and every couple of minutes while you work, the previous project
  file is copied into `backups/` (the last 30 are kept).
- **Repairs itself:** if the project file is ever damaged, the newest good backup is used and
  any pictures taken after it are put back. If it's missing altogether, it's rebuilt from the
  pictures. The damaged file is kept to one side, not deleted. A picture file that has gone
  missing shows as a grey card rather than being dropped, in case it turns up again.
- **Go back to a backup** by opening a file from `backups/`. That becomes an ordinary change,
  so `Z` undoes it.
- **One window per movie:** if a movie is already open (say it got double-clicked twice), a
  second window won't open it too, so they can't write over each other's pictures.
- **Rename a movie** by renaming its `.stopmo` file.
- **Old saves** (`stopmotion_project_*` folders from the earlier version) show up in the list
  of movies marked OLD SAVE. Opening one copies it into a new project.

### Opening a movie

- **From the studio:** press `L` (or the Projects button). Pick a movie, or press `B` / click
  **Open a file...** to use your computer's own Open window and find a `.stopmo` anywhere.
- **Double-click** the `Open <movie>.command` file in the movie's folder (`.bat` on Windows,
  `.sh` on Linux). It remembers where Python and the studio live, so if you move this repository
  folder, open the movie once from the studio and the file will be updated. Opening any file in
  a movie's folder (a picture, say) opens that movie.
- **Double-click the `.stopmo` file itself (macOS):** run this once:

  ```bash
  python3 stopmotion.py --install-mac-app
  ```

  It makes a small `Kid Stop Motion` app in `~/Applications` that opens `.stopmo` files. The
  studio still runs in Terminal, which already has permission to use the camera. If a Mac
  asks which app to use, pick Kid Stop Motion (right-click the file > Get Info > Open with >
  Change All).
- **From the command line:**

  ```bash
  python3 stopmotion.py "~/Stop Motion Projects/Dragon Race/Dragon Race.stopmo"
  python3 stopmotion.py --export "~/Stop Motion Projects/Dragon Race"   # make the MP4, no camera needed
  ```

## Title cards, transitions and the voice track

- **Title cards:** press `C` and type. The first line is big; `TAB` starts a smaller line under
  it (up to three). `LEFT` / `RIGHT` changes the colors, `ENTER` is done. A new card lasts two
  seconds; pick it in the timeline and press `+` / `-` to change that. Pick it and press `C`
  (or `ENTER`) to change the words.
- **Transitions** go *before* a picture: pick the picture and press `F` until you like it (Fade,
  Melt, Wipe, Circle, Slide, then none). A transition before the very first picture fades the
  movie in. With the camera at the end of the timeline, `F` sets how the movie ends.
- **Voice track:** press `R`. After a 3-2-1 countdown the movie plays silently while you talk,
  and recording stops when the movie ends (or when you press any key). With no pictures yet, it
  records until you press a key, which is handy for recording the lines first and animating to
  them. Press `R` again to record a new one or delete it. `Z` undoes either.

## Tips for a good shoot

- Tape the camera down, or every frame will jump.
- Turn off automatic room lights and close the curtains if you can. Changing daylight makes the
  movie flicker.
- Move things a *little* between shots. Small moves look smooth; big moves look jumpy.
- 12 pictures per second is the default, so 24 pictures is two seconds of film. Rather than
  taking the same picture over and over, pick it and press `+`.

## Settings

Near the top of `stopmotion.py`:

- `ROTATE_180 = True`: the picture is rotated half a turn by default, because most tabletop rigs
  end up with the camera clamped upside down. **If your picture appears upside down, set this to
  `False`.**
- `FPS = 12`: pictures per second for new movies. (Each movie remembers its own speed; change
  it in the studio with `V`.)
- `CAMERA_INDEX = 0`: which webcam to use. Try `1` for a second camera, or run
  `python3 stopmotion.py --camera 1`.
- `PROJECTS_DIR`: where new movies go (or run with `--projects-dir some/folder`).
- `ONION_OPACITY`, `AUTO_CAPTURE_SECONDS`, `TITLE_SECONDS`, `TRANSITION_SECONDS`,
  `SHUTTER_SOUND`.
- `WINDOW_SCALE = 1.0`: the window is 1280 x 776. Try `0.8` on a small screen.

## Camera and microphone permissions

The first run may need permission to use the camera, and the first voice recording permission
to use the microphone. On macOS, allow them under **System Settings → Privacy & Security →
Camera** and **→ Microphone** for whichever app runs Python (usually Terminal). If a recording
comes out completely silent, the studio says so. That almost always means the microphone
permission is off.

If the camera drops out (unplugged, grabbed by another app, settings changed), the studio keeps
going. You can still edit, and it reconnects by itself when the camera comes back.

## A note on privacy

This app writes pictures, sound and movies to your own computer and never sends anything
anywhere. Movies are saved in your home folder, outside this repository. The included
`.gitignore` also excludes `*.mp4`, `*.stopmo`, picture and sound folders and the like, so that
if you version your own animation projects you don't accidentally publish footage of your kids.

## For grown-ups: tests

```bash
python3 -m unittest discover -s tests
```

The tests drive the whole studio with a pretend camera and window, so they don't need a webcam
or a screen.

## License

[MIT](LICENSE) © 2026 R.F. Chapman
