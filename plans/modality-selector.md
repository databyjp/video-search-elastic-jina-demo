# Add Modality-Filter Pulldown to Search UI

## Context
Each indexed scene produces **two vectors** in Elasticsearch: one with `modality: "fused"` (video+audio) and one with `modality: "transcript"` (speech-to-text). Currently every search queries across both. The user wants a pulldown to restrict results to transcript-only, fusion-only, or both (default).

## Approach
Add a `<select>` pulldown on the frontend labelled "Search vectors:" with options:
- **Both** → no modality filter  
- **Transcript** → `modality: transcript`  
- **Video+Audio** → `modality: fused`  

Wire the selected value into the HTMX form and the voice-search fetch body. On the backend, accept a `modality` form parameter and push an additional `term` filter into the kNN `filter` clause when it is not `"both"`.

## Files to modify
- `app.py` — `_search_hits()` gains `modality_filter` arg; three search endpoints accept `modality: str = Form("both")`  
- `templates/index.html` — add pulldown + label above the audio-mode toggle; include value in HTMX/voice requests  

## Reuse
- Uses existing `modality` field already stored in every ES document (`fused` | `transcript`).  
- No re-ingestion required.  

## Steps
- [ ] In `app.py`, update `_search_hits(query_vector, modality_filter=None)` to build a `bool` + `must` filter when `modality_filter` is `"fused"` or `"transcript"`, otherwise keep the existing `content_type: video` filter.  
- [ ] In `app.py`, add `modality: str = Form("both")` to `/search`, `/voice-search`, and `/voice-search-direct` endpoints; pass it through to `_search_hits`.  
- [ ] In `templates/index.html`, add `<label>Search vectors: <select name="modality" id="modality">` with `<option value="both" selected>Both</option>`, `<option value="transcript">Transcript</option>`, `<option value="fused">Video+Audio</option>` positioned above the audio-mode toggle.  
- [ ] In `templates/index.html`, wire the pulldown into the HTMX form (it will serialize automatically with `name="modality"`) and into the voice-search `FormData` append.  
- [ ] Verify `.py` syntax compiles.  

## Verification
1. Start the app (`uvicorn app:app --reload`).  
2. Search "cat" with "Both" — see results of both modalities (no change).  
3. Switch to "Transcript" — only `transcript` modality badges appear.  
4. Switch to "Video+Audio" — only `fused` modality badges appear.  
