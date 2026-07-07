# Persist Video Clips During Ingestion

## Context
The app currently chunks videos into scenes during ingestion (using `ffmpeg` via `cut_video()`), embeds the clips, but immediately discards them by placing them in a `tempfile.TemporaryDirectory()`. The search UI then loads the **full original video** and seeks to the scene start timestamp. This is sub-optimal because the user sees the full video timeline and can seek outside the matching scene.

## Approach
Persist the per-scene video clips that are already being cut during ingestion, serve them via a new `/clips` static mount, and point the search result player directly at the clip file. This is the cleanest UX: each result plays only its exact matching scene, with no timeline leakage.

## Files to modify
- `app.py` — add `/clips` static mount, ensure `data/clips` exists at startup  
- `src/omnimodal_search/es.py` — replace `tempfile.TemporaryDirectory()` with a permanent `data/clips/` directory; persist both `.mp4` and `.wav` there; **store `clip_path` (or `clip_url`) on each indexed document** so every hit explicitly references its sub-clip  
- `templates/results.html` — read the clip reference directly from `hit.clip_path` (or `hit.clip_url`), remove `data-start` attribute  
- `templates/index.html` — remove the `+0.5` seeking hack since clips natively start at 0  

## Steps
- [ ] In `app.py`, `mkdir data/clips`, add `app.mount("/clips", StaticFiles(directory="data/clips"), name="clips")`  
- [ ] In `es.py`, replace `with tempfile.TemporaryDirectory() as tmpdir:` with `clips_dir = Path("data/clips"); clips_dir.mkdir(parents=True, exist_ok=True)` and build paths with `clips_dir / ...`; add `clip_path` (e.g. `/clips/{video_id}_s{i}.mp4`) to the `scene_doc` so every ES document carries a direct reference to its sub-clip  
- [ ] In `results.html`, change `src="/videos/..."` → `src="{{ hit.clip_path }}"` and drop `data-start`  
- [ ] In `index.html`, simplify `seekVideosToStart()` to a no-op or remove entirely (kept as no-op for safety)  
- [ ] Verify `.gitignore` — `data/` is already ignored, which covers `data/clips/`  

## Verification
1. Run the ingestion script: `python ingest.py`  
2. Check `data/clips/` contains `.mp4` files like `{video_id}_s0.mp4`, `_s1.mp4`, etc.  
3. Run the app: `uvicorn app:app --reload`  
4. Perform a search; each result video should be a standalone clip that starts at 0 and ends at the scene boundary.  
