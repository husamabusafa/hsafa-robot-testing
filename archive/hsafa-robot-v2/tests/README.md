# hsafa-robot-v2 / tests

Per-feature standalone scripts for the modular tracker described in
`../idea.md`. Each script is runnable on its own and visualizes exactly
what one module is doing -- nothing else.

## Running

Scripts try **direct OpenCV capture first** (640×480 @ 15 fps, no daemon
needed). If the camera is already owned by the daemon, they fall back to
`reachy.media.get_frame()` automatically.

```bash
source .venv/bin/activate
python hsafa-robot-v2/tests/test_lk_points.py   # click points, watch LK
python hsafa-robot-v2/tests/test_lk_bbox.py     # drag bbox, watch LK + FB filter
```

If you prefer (or must) use the daemon:

```bash
./scripts/daemon.sh start            # from repo root
python hsafa-robot-v2/tests/test_lk_points.py   # falls back to daemon
```

## Index

| Script | Module under test | Idea.md ref |
|---|---|---|
| `test_lk_points.py` | LK on hand-clicked points + forward-backward filter | section 4.1, 3.5 |
| `test_lk_bbox.py`   | OpticalFlowModule end-to-end (bbox, GFTT, FB filter, replenish, scale) | section 4.1 |

## Conventions

* No tracker core, no fusion, no other modules. One concern per script.
* Camera access goes through `../camera_feed.py` (`make_camera_feed`), which
  wraps either `cv2.VideoCapture` or `reachy.media.get_frame()`. To run on a
  saved clip later, swap that for a `cv2.VideoCapture` pointing at a file.
* Keys: `q` / Esc to quit, `c` to clear state, anything else is documented
  in the script's docstring.
