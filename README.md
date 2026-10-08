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
- **Three sound tracks: DLG, SFX and MUSIC.** Add sound files (`I`), record your voice (`R`), or
  record live video with its sound (`G`). Drag the clips to move them, trim their ends, copy or
  delete them, and mute or turn down a whole track. The movie plays with all of it and the MP4
  gets the mix.
- **Auto-snap, a picture goal and a speed control.** `T` takes a picture every 2 seconds, `N`
  sets a goal with a progress bar, `V` makes the movie faster or slower.
- **A steady picture.** In the camera panel (`K`), lock focus, exposure and white balance so the
  movie doesn't flicker or go soft when a hand reaches in; switch cameras; flip the picture.
- **Nothing gets lost.** Every picture is saved the moment it's taken. There's no "save"
  step to forget.

## Setup

You need Python 3.9 or newer. The `python3` that comes with macOS is fine.

```bash
git clone https://github.com/superperfundo/kid-stop-motion.git
cd kid-stop-motion
python3 -m pip install -r requirements.txt
```

That installs OpenCV, [sounddevice](https://python-sounddevice.readthedocs.io/) for the
microphone, and on macOS a ready-built libusb for the camera controls. If sounddevice isn't there,
everything except sound still works.

[ffmpeg](https://ffmpeg.org/) is optional but recommended: it's what puts the sound into
the MP4. Without it, OpenCV's built-in encoder makes the movie, without sound. To install it on
macOS with [Homebrew](https://brew.sh/): `brew install ffmpeg`. On Debian/Ubuntu:
`sudo apt install ffmpeg libportaudio2` (the second one is for the microphone).

## Run it

```bash
python3 stopmotion.py
```

The studio asks the camera for the sharpest picture it can send at a usable speed (up to 4K;
see `CAMERA_MAX_SIZE` under [Settings](#settings)). First you'll see your movies: pick one to keep
working on, or **New movie**. Then the studio
opens with the live camera, big buttons down the right, and the timeline along the bottom. If
the picture is upside down, press `K` then `R` (it remembers, for each camera). If it's the wrong
camera, press `K` then `C`.

| Key | What it does |
| --- | --- |
| `SPACE` | Take a picture. (If you're looking at an old picture, the first press moves the camera there.) |
| `P` | Play the movie with all its sound (any key stops it) |
| `O` | Ghost of the last picture on / off |
| `T` | Auto-snap: a picture every 2 seconds |
| `C` | Add a title card (or edit the one you picked) |
| `F` | Transition before the picked picture. Press again for the next kind. With the camera at the end, it's the ending. |
| `R` | Record your voice, onto the DLG track from the start of the movie |
| `I` | Add sound files (WAV, MP3, M4A and more with ffmpeg), on the track you pick |
| `G` | Live: record the camera's video with its sound. Press `G` again to stop. |
| `V` | Speed: pictures per second |
| `E` | Make an MP4 movie |
| `L` | Your movies: open another one, or start a new one |
| `K` | Camera panel: switch camera, flip, lock focus / exposure / white balance ([below](#focus-exposure-and-white-balance)) |
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
    audio/                      <- the sounds on the tracks (as WAV files)
    live/                       <- live recordings (video)
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

## Title cards, transitions and sound

- **Title cards:** press `C` and type. The first line is big; `TAB` starts a smaller line under
  it (up to three). `LEFT` / `RIGHT` changes the colors, `ENTER` is done. A new card lasts two
  seconds; pick it in the timeline and press `+` / `-` to change that. Pick it and press `C`
  (or `ENTER`) to change the words.
- **Transitions** go *before* a picture: pick the picture and press `F` until you like it (Fade,
  Melt, Wipe, Circle, Slide, then none). A transition before the very first picture fades the
  movie in. With the camera at the end of the timeline, `F` sets how the movie ends.
- **Voice:** press `R`. After a 3-2-1 countdown the movie plays silently while you talk, and
  recording stops when the movie ends (or when you press any key). With no pictures yet, it
  records until you press a key, which is handy for recording the lines first and animating to
  them. The voice goes on the DLG track, from the start of the movie. `Z` undoes it.
- **Sound tracks:** the three lanes under the pictures are DLG (talking), SFX (bangs and
  sounds) and MUSIC (background). Each sound is a clip you can:
  - **Add** with `I`: a window opens to pick files (from Finder or Explorer), then you choose the
    track. New sounds start where the picked picture is, or where the camera is.
  - **Move** by dragging it, along the lane or onto another track. It snaps to a tenth of a second.
  - **Trim** by dragging its left or right edge: the sound starts or stops later or earlier.
  - **Pick** by clicking it (it outlines in yellow). `D` copies it (the copy goes just after it),
    `X` deletes it, and `Z` undoes either. Clicking a picture picks that instead.
  - **Mute** a whole track with its `Mute` button, or turn it down or up with the percentage
    button (100%, 50%, 150%).
  - **See** more by zooming with `+` / `-` above the lanes, and scrolling the mouse wheel over them.
  The white line in the lanes is where the movie is while it plays; the yellow line is where new
  sounds will go. Playing the movie (`P`) plays every track at once, and the white line follows
  the sound as it's heard.
- **Live video:** press `G` to record what the camera sees, with the microphone's sound, when
  stop motion won't do (a bit of talking, a dance, a surprise). It goes into the timeline as one
  picture, labelled "live", and plays for as long as it was recorded (up to five minutes). Its
  sound moves with it. Live video plays at the movie's speed (`V`), so a movie with live parts
  reads best at 24 or more pictures a second.
- **Not yet:** dragging files straight from Finder or Explorer onto the window. Use `I`, which
  opens the same kind of window. The studio is drawn in an OpenCV window, which doesn't accept
  dropped files, so dragging needs platform-specific code that isn't written yet.

## Focus, exposure and white balance

Webcams keep adjusting focus, brightness and colour on their own. That's fine for video calls, but
in stop motion it makes the movie flicker and go soft whenever a hand reaches into the shot. Once
the picture looks right, press `K` to open the camera panel and lock it. While the panel is open:

| Key | What it does |
| --- | --- |
| `F` / `X` / `W` | Lock or unlock focus / exposure / white balance |
| `[` / `]` | Nudge focus |
| `-` / `+` | Darker / brighter |
| `C` | Switch to the next camera |
| `R` | Flip the picture half a turn (remembered for each camera) |
| `K` or `ESC` | Close the panel |

(`SPACE` still takes pictures with the panel open. Outside the panel, those letters do their
usual jobs.) Above the live picture, a label shows what's locked; click it to open the panel.

- `F`, `X` and `W` lock focus, exposure and white balance where the camera has settled. Press
  again to go back to auto. Webcams don't report the exposure or white balance their auto modes
  pick, so `X` and `W` measure the picture and adjust for a moment until it matches how it looked
  on auto. The picture holds still while that happens.
- The **Focus**, **Exposure** and **White balance** sliders in the panel set a value by hand, and
  lock it there. For fine focusing, `[` and `]` nudge focus a step at a time.
- `-` and `+` make the exposure a third of a stop darker or brighter, and lock it (starting from
  what auto had, if it was on auto). On cameras without exposure control, the app darkens or
  brightens the picture itself instead.

Each camera gets the controls it actually has: a webcam with a fixed-focus lens, for instance,
shows focus as `n/a` but can still lock exposure and white balance. The terminal lists what each
camera can control. The app puts the camera back on auto when you switch cameras or quit.
Switching, flipping and `-` / `+` work with every camera.

- **macOS:** USB webcams. The app talks to the camera directly with standard USB Video Class
  requests, which almost every USB webcam understands, because macOS's own camera APIs can't set
  a focus distance. Built-in laptop cameras aren't USB, so they show `n/a` for all three, and
  `-` / `+` brighten the picture in the app instead.
- **Linux:** most USB webcams, through the system's own video interface (V4L2), which also tells
  the app each camera's real ranges. If the panel says there's no permission, add yourself to the
  `video` group.
- **Windows:** webcams, through DirectShow (the app opens cameras that way so it can reach their
  settings). Windows doesn't tell the app the ranges, so the sliders start from typical ones; if
  the camera refuses a value, the app moves on to the next one it takes, and when a slider goes
  past the end of the camera's range, it finds where the range ends and shrinks the slider to
  match. Exposure on Windows moves in whole stops, so `-` / `+` go a whole stop at a time.
- **HDMI capture boxes** (a Cam Link or a cheap HDMI-to-USB dongle, for a camera with HDMI out)
  pass the picture along but not the camera's settings. The panel says "set it on the camera":
  use the camera's own manual focus, exposure and white balance, which is the best way to shoot
  stop motion anyway. The app spots them by name, or by having no focus or exposure control at
  all.

### How fine the control is depends on the camera

The app can only use the settings a camera actually has, and some are coarser than they claim.
Many webcams, the Logitech C920 included, accept any exposure time but only really use a handful,
about a stop apart. On those cameras:

- The Exposure slider and `-` / `+` only change the picture when they cross one of the camera's
  steps. In between, the number changes but the picture doesn't.
- `X` can only lock to the nearest step, so the locked picture may be a bit brighter or darker
  than it was on auto. The studio says so when that happens, e.g.
  `Exposure LOCKED at 76ms (as close as this camera gets: 0.4 stops brighter)`.

The app doesn't paper over this by turning up the camera's gain, which would add noise. If you
need something in between, adjust the light instead: move a lamp closer or further away, or put a
sheet of paper over it.

If you switch cameras partway through a movie, new pictures are cropped to match the size of the
movie's first picture, so the movie still exports cleanly.

## Tips for a good shoot

- Tape the camera down, or every frame will jump.
- Once the shot is set up, lock focus, exposure and white balance (`K`, then `F`, `X` and `W`)
  so the camera doesn't refocus or change colour every time a hand reaches in.
- Turn off automatic room lights and close the curtains if you can. Changing daylight makes the
  movie flicker.
- Move things a *little* between shots. Small moves look smooth; big moves look jumpy.
- 12 pictures per second is the default, so 24 pictures is two seconds of film. Rather than
  taking the same picture over and over, pick it and press `+`.

## Settings

Near the top of `stopmotion.py`:

- `ROTATE_180 = False`: set this to `True` if your cameras are usually mounted upside down
  (tabletop rigs often end up that way), so the picture starts the right way up. `K` then `R`
  flips it for the camera you're using, and that's remembered.
- `FPS = 12`: pictures per second for new movies. (Each movie remembers its own speed; change
  it in the studio with `V`.)
- `CAMERA_INDEX = 0`: which camera to start with (or run `python3 stopmotion.py --camera 1`).
  `K` then `C` switches while you work.
- `CAMERA_MAX_SIZE = (3840, 2160)`: the sharpest picture to ask the camera for. Cameras that can't
  manage it give their best below it, and a size the camera only sends at under 10 pictures a
  second is skipped. Set `(1920, 1080)` for smaller files and a snappier studio on an older
  computer.
- `EXPOSURE_STEP = 1 / 3`: how many stops each `-` / `+` changes the exposure by (at least a
  whole stop on Windows, where cameras have nothing finer).
- `PROJECTS_DIR`: where new movies go (or run with `--projects-dir some/folder`).
- `ONION_OPACITY`, `AUTO_CAPTURE_SECONDS`, `TITLE_SECONDS`, `TRANSITION_SECONDS`,
  `SHUTTER_SOUND`.
- `WINDOW_SCALE = 1.0`: the window is 1280 x 900 (the sound lanes make it taller than before).
  Try `0.8` on a small screen (or run `python3 stopmotion.py --scale 0.8`).
- `MAX_LIVE_SECONDS = 300`: the longest live video `G` records, in seconds.

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

The tests drive the whole studio with a pretend camera, window and sound, so they don't need a
webcam, a screen or a speaker. Sound playback itself (the speakers, and the timing of the white
line against the sound) has only been tried with a pretend sound device.

## License

[MIT](LICENSE) © 2026 R.F. Chapman
