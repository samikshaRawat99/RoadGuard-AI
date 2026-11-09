"""
RoadGuard AI — FastAPI Backend
===============================
Provides two API endpoints:
  POST /process   – accepts a video upload, runs the full CV pipeline, returns
                    the URL of the annotated output video.
  GET  /result/{filename} – serves a processed output video for browser playback.

Also serves the vanilla-JS frontend from the /static mount so that the browser
and the API share the same origin (no CORS issues).
"""

import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# 1. Path setup
# ---------------------------------------------------------------------------
# Resolve the project root (the directory that contains this file).
# Inserting it at position 0 in sys.path ensures "from src.xxx import yyy"
# works correctly regardless of the directory uvicorn is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import AFTER sys.path is fixed so the 'src' package is discoverable.
from src.video_processing import process_video  # noqa: E402

# ---------------------------------------------------------------------------
# 2. Directory setup
# ---------------------------------------------------------------------------
# Uploaded videos are saved here temporarily (deleted after processing).
UPLOADS_DIR = PROJECT_ROOT / "outputs" / "uploads"
# Processed (annotated) videos are kept here so the browser can play them back.
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "processed_videos"

# Create both directories if they don't already exist.
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# File extensions we accept from the browser. Must match the frontend accept attribute.
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi"}

# ---------------------------------------------------------------------------
# 3. FastAPI app instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RoadGuard AI",
    description="Intelligent road safety monitoring with YOLO + ByteTrack.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# 4. API routes  ← MUST be registered before StaticFiles is mounted.
#
# Why order matters: FastAPI evaluates routes top-to-bottom. If we mounted
# StaticFiles at "/" first, it would intercept every request including
# "/process" and "/result/*" before the API routes could handle them.
# ---------------------------------------------------------------------------

@app.post("/process")
async def process_video_endpoint(file: UploadFile = File(...)):
    """
    Accept a video file from the browser, run vehicle detection + tracking +
    speed estimation, and return the URL of the annotated output video.

    Flow:
      1. Validate the file extension.
      2. Write the uploaded bytes to a temporary file (uuid-named).
      3. Call process_video() — the existing pipeline in src/video_processing.py.
      4. Delete the temporary upload regardless of success or failure.
      5. Return JSON with the output URL and frame count.
    """

    # ── Step 1: Extension validation ──────────────────────────────────────
    # file.filename is the original filename sent by the browser.
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Please upload an MP4, MOV or AVI file."
            ),
        )

    # ── Step 2: Generate unique filenames ─────────────────────────────────
    # uuid4().hex produces a 32-character random hex string, e.g. "a3f9…".
    # This prevents filename collisions if two uploads happen simultaneously.
    request_id = uuid.uuid4().hex
    upload_path = UPLOADS_DIR / f"{request_id}{suffix}"   # temporary input
    output_filename = f"{request_id}_processed.mp4"        # permanent output
    output_path = OUTPUTS_DIR / output_filename

    try:
        # ── Step 3: Save upload to disk (chunked) ────────────────────────
        # Instead of reading the entire file into memory at once with file.read(),
        # we write it in 1 MB chunks. This keeps memory usage flat even for
        # large video files (e.g. a 500 MB highway recording).
        #
        # Why 1 MB chunks?
        #   1 MB (1_048_576 bytes) is a standard I/O buffer size. It is large
        #   enough to avoid excessive syscall overhead but small enough that we
        #   never hold the full file in RAM.
        #
        # file.read(size) on an UploadFile returns bytes; an empty bytes object
        # b"" is falsy, so the while-loop exits naturally at end-of-file.
        CHUNK_SIZE = 1_048_576  # 1 MB
        bytes_written = 0
        with upload_path.open("wb") as dest:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                dest.write(chunk)
                bytes_written += len(chunk)

        # Guard: reject an upload that contained zero bytes.
        # This catches both a genuinely empty file and a browser bug where
        # the file object exists but has no content.
        if bytes_written == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # ── Step 4: Run the pipeline ──────────────────────────────────────
        # process_video() is imported from src/video_processing.py.
        # We pass absolute string paths so the function works from any CWD.
        # return_analytics=True requests both the processed frame count and
        # the aggregated traffic metrics dictionary.
        frames_processed, analytics = process_video(
            input_path=str(upload_path),
            output_path=str(output_path),
            return_analytics=True,
        )

        # process_video() returns 0 on failure (file not found, codec error…).
        # We also verify the output file exists and is non-empty — guards against
        # silent codec failures, disk-write aborts, or zero-byte output files.
        if frames_processed <= 0 or not output_path.is_file() or output_path.stat().st_size == 0:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Processing failed: 0 frames were written. "
                    "Check the server console for details."
                ),
            )

        # ── Step 5: Return the output URL and analytics summary ───────────
        # The browser will fetch the video from GET /result/{filename}
        # and display the live analytics in the dashboard cards and panels.
        return JSONResponse({
            "output_url": f"/result/{output_filename}",
            "frames_processed": frames_processed,
            "message": f"Successfully processed {frames_processed} frame(s).",
            "analytics": analytics,
        })

    except HTTPException:
        # Re-raise our own HTTP errors unchanged so FastAPI sends the right
        # status code (400 or 500) with the correct JSON body.
        raise

    except Exception as exc:
        # Catch any unexpected error (e.g. corrupted video, codec crash)
        # and convert it to a readable 500 response instead of a raw traceback.
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected server error: {str(exc)}",
        ) from exc

    finally:
        # ── Step 6: Cleanup ───────────────────────────────────────────────
        # Close the UploadFile stream so FastAPI releases the underlying
        # temp buffer. This must come before unlink() to avoid a "file in use"
        # error on Windows and to free the file descriptor on all platforms.
        await file.close()

        # Always delete the temporary upload — even if processing crashed.
        # The output video is kept so the browser can play it back later.
        if upload_path.exists():
            upload_path.unlink()


@app.get("/result/{filename}")
async def get_result(filename: str):
    """
    Serve an annotated output video by filename.

    The browser uses the URL returned by /process (e.g. /result/abc123_processed.mp4)
    as the src of a <video> element. FileResponse streams the file efficiently.
    """
    # Sanitise the filename to block path-traversal attacks like "../../etc/passwd".
    # We only allow plain filenames with no directory separators.
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    file_path = OUTPUTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Result file '{filename}' not found.",
        )

    # FileResponse sets the correct Content-Type and supports HTTP range
    # requests, which allows the browser's <video> element to seek.
    return FileResponse(
        path=str(file_path),
        media_type="video/mp4",
        filename=filename,
    )


# ---------------------------------------------------------------------------
# 5. Frontend serving
# ---------------------------------------------------------------------------
# Serve index.html explicitly at the root URL so the browser can open "/".
# This must be a named route — not a StaticFiles mount — so that "/process"
# and "/result/*" remain reachable as API endpoints.
@app.get("/")
async def serve_index():
    """Serve the RoadGuard AI dashboard (frontend/index.html)."""
    return FileResponse(str(PROJECT_ROOT / "frontend" / "index.html"))


# Mount the entire frontend/ directory under /static.
# After this line, all files in frontend/ are reachable as:
#   /static/style.css
#   /static/script.js
# This mount is registered LAST so it cannot shadow the API routes above.
app.mount(
    "/static",
    StaticFiles(directory=str(PROJECT_ROOT / "frontend")),
    name="static",
)
