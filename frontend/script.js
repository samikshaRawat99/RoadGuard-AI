/**
 * ==============================================================================
 * RoadGuard AI — Frontend Client-Side Script
 * Vanilla JavaScript (No frameworks, libraries, or external dependencies)
 * ==============================================================================
 */

// 1. Wait for DOM content to fully load before attaching listeners
document.addEventListener("DOMContentLoaded", () => {
  // DOM Element References — Input & Controls
  const videoInput = document.getElementById("videoInput");
  const fileNameDisplay = document.getElementById("fileNameDisplay");
  const videoPreview = document.getElementById("videoPreview");
  const emptyPreviewState = document.getElementById("emptyPreviewState");
  const startBtn = document.getElementById("startBtn");
  const stopBtn = document.getElementById("stopBtn");
  const statusIndicator = document.getElementById("statusIndicator");
  const statusText = document.getElementById("statusText");

  // DOM Element References — Overview Cards
  const statTotalVehicles = document.getElementById("statTotalVehicles");
  const statTrackedVehicles = document.getElementById("statTrackedVehicles");
  const statAvgSpeed = document.getElementById("statAvgSpeed");
  const statViolations = document.getElementById("statViolations");

  // DOM Element References — Results Containers
  const processedVideoContainer = document.getElementById("processedVideoContainer");
  const analyticsContainer = document.getElementById("analyticsContainer");

  // Keep reference to generated Blob URL so we can revoke it when a new file is chosen
  let currentVideoUrl = null;

  /**
   * Helper: Update the system status indicator bar
   * @param {string} state - 'idle' | 'active' | 'working'
   * @param {string} message - User-facing status message
   */
  function setStatus(state, message) {
    statusIndicator.className = "status-indicator"; // reset classes
    if (state === "active") {
      statusIndicator.classList.add("status-active");
    } else if (state === "working") {
      statusIndicator.classList.add("status-working");
    } else {
      statusIndicator.classList.add("status-idle");
    }
    statusText.textContent = message;
  }

  /**
   * Helper: Format raw byte count into human-readable MB string
   * @param {number} bytes 
   * @returns {string} e.g. "31.8 MB"
   */
  function formatFileSize(bytes) {
    if (!bytes || bytes === 0) return "0 MB";
    const mb = bytes / (1024 * 1024);
    return `${mb.toFixed(1)} MB`;
  }

  /**
   * Helper: Format and update the statAvgSpeed DOM element safely without innerHTML
   * @param {string|number} speedVal - Numeric speed or '--' placeholder
   */
  function setAverageSpeedDisplay(speedVal) {
    statAvgSpeed.replaceChildren();

    const valueText = document.createTextNode(`${speedVal} `);
    const unitSpan = document.createElement("span");
    unitSpan.className = "unit";
    unitSpan.textContent = "px/s";

    statAvgSpeed.append(valueText, unitSpan);
  }

  /**
   * Helper: Reset overview statistics cards and result panels to placeholder states
   */
  function resetOverviewCards() {
    // 1. Reset metrics
    statTotalVehicles.textContent = "--";
    statTrackedVehicles.textContent = "--";
    setAverageSpeedDisplay("--");
    statViolations.textContent = "--";

    // 2. Reset analytics breakdown container to its initial placeholder state
    if (analyticsContainer) {
      analyticsContainer.className = "result-placeholder";
      analyticsContainer.removeAttribute("style");
      analyticsContainer.replaceChildren();

      const iconSpan = document.createElement("span");
      iconSpan.className = "placeholder-icon";
      iconSpan.textContent = "📋";

      const titleP = document.createElement("p");
      titleP.className = "placeholder-title";
      titleP.textContent = "No Active Logs";

      const subtitleP = document.createElement("p");
      subtitleP.className = "placeholder-subtitle";
      subtitleP.textContent = "Vehicle category breakdown and session metrics will populate here upon processing.";

      analyticsContainer.append(iconSpan, titleP, subtitleP);
    }

    // 3. Reset processed video container to initial placeholder state
    if (processedVideoContainer) {
      processedVideoContainer.className = "result-placeholder";
      processedVideoContainer.removeAttribute("style");
      processedVideoContainer.replaceChildren();

      const iconSpan = document.createElement("span");
      iconSpan.className = "placeholder-icon";
      iconSpan.textContent = "🎬";

      const titleP = document.createElement("p");
      titleP.className = "placeholder-title";
      titleP.textContent = "Annotated Output Video";

      const subtitleP = document.createElement("p");
      subtitleP.className = "placeholder-subtitle";
      subtitleP.textContent = "Processed video with bounding boxes, tracking IDs, and speed overlays will appear here.";

      processedVideoContainer.append(iconSpan, titleP, subtitleP);
    }
  }

  /**
   * Helper: Render live analytics into overview cards and breakdown panel
   * @param {Object} analytics - Analytics summary returned by the backend
   * @param {number} framesProcessed - Total frames successfully processed
   */
  function renderAnalytics(analytics, framesProcessed) {
    if (!analytics) return;

    // 1. Overview Cards (Distinct Metrics)
    // Card 1: Total Detections (raw bounding box count across all frames)
    statTotalVehicles.textContent = analytics.total_detections ?? "--";

    // Card 2: Vehicles Tracked (distinct non-null ByteTrack IDs observed)
    statTrackedVehicles.textContent = analytics.unique_tracked_vehicles ?? "--";

    // Card 3: Average Speed (mean px/s across tracked vehicles, or '-- px/s' if null)
    if (analytics.average_speed_px !== null && analytics.average_speed_px !== undefined) {
      setAverageSpeedDisplay(analytics.average_speed_px);
    } else {
      setAverageSpeedDisplay("--");
    }

    // Card 4: Violations count from ViolationDetector engine
    statViolations.textContent = analytics.violations_count ?? 0;

    // 2. Vehicle Breakdown Panel
    if (analyticsContainer) {
      const counts = analytics.class_counts || {};
      const carCount = counts.car || 0;
      const truckCount = counts.truck || 0;
      const busCount = counts.bus || 0;
      const motorcycleCount = counts.motorcycle || 0;
      const totalDetections = analytics.total_detections || 0;

      analyticsContainer.className = "card-body";
      analyticsContainer.style.padding = "1rem";
      analyticsContainer.replaceChildren();

      // Main wrapper
      const wrapperDiv = document.createElement("div");
      wrapperDiv.style.display = "flex";
      wrapperDiv.style.flexDirection = "column";
      wrapperDiv.style.gap = "1rem";

      // 2x2 Grid for vehicle categories
      const gridDiv = document.createElement("div");
      gridDiv.style.display = "grid";
      gridDiv.style.gridTemplateColumns = "repeat(2, 1fr)";
      gridDiv.style.gap = "0.75rem";

      const categories = [
        { label: "🚗 Cars", count: carCount },
        { label: "🚚 Trucks", count: truckCount },
        { label: "🚌 Buses", count: busCount },
        { label: "🏍️ Motorcycles", count: motorcycleCount },
      ];

      categories.forEach((cat) => {
        const itemBox = document.createElement("div");
        itemBox.style.background = "rgba(255,255,255,0.03)";
        itemBox.style.border = "1px solid var(--border-color)";
        itemBox.style.borderRadius = "var(--radius-sm)";
        itemBox.style.padding = "0.75rem";

        const labelDiv = document.createElement("div");
        labelDiv.style.fontSize = "0.75rem";
        labelDiv.style.color = "var(--text-muted)";
        labelDiv.style.textTransform = "uppercase";
        labelDiv.textContent = cat.label;

        const countDiv = document.createElement("div");
        countDiv.style.fontSize = "1.4rem";
        countDiv.style.fontWeight = "700";
        countDiv.style.color = "#fff";
        countDiv.textContent = cat.count;

        itemBox.append(labelDiv, countDiv);
        gridDiv.appendChild(itemBox);
      });

      // Footer line showing raw frame detections
      const footerDiv = document.createElement("div");
      footerDiv.style.borderTop = "1px solid rgba(255,255,255,0.08)";
      footerDiv.style.paddingTop = "0.75rem";
      footerDiv.style.fontSize = "0.82rem";
      footerDiv.style.color = "var(--text-dim)";
      footerDiv.style.display = "flex";
      footerDiv.style.justifyContent = "space-between";

      const spanLabel = document.createElement("span");
      spanLabel.textContent = "Total Raw Detections (all frames):";

      const strongCount = document.createElement("strong");
      strongCount.style.color = "var(--text-main)";
      strongCount.textContent = totalDetections;

      footerDiv.append(spanLabel, strongCount);
      wrapperDiv.append(gridDiv, footerDiv);

      // Violations Log Summary
      if (analytics.violations_count > 0 && analytics.violation_logs) {
        const violationsHeader = document.createElement("div");
        violationsHeader.style.borderTop = "1px solid rgba(255,255,255,0.08)";
        violationsHeader.style.paddingTop = "0.75rem";
        violationsHeader.style.fontSize = "0.78rem";
        violationsHeader.style.textTransform = "uppercase";
        violationsHeader.style.color = "#f87171";
        violationsHeader.style.fontWeight = "700";
        violationsHeader.textContent = `⚠️ Violations Logged (${analytics.violations_count} detected)`;

        const logList = document.createElement("div");
        logList.style.display = "flex";
        logList.style.flexDirection = "column";
        logList.style.gap = "0.4rem";
        logList.style.maxHeight = "120px";
        logList.style.overflowY = "auto";
        logList.style.fontSize = "0.8rem";

        analytics.violation_logs.slice(0, 5).forEach((v) => {
          const item = document.createElement("div");
          item.style.padding = "0.4rem 0.6rem";
          item.style.background = "rgba(239, 68, 68, 0.1)";
          item.style.border = "1px solid rgba(239, 68, 68, 0.25)";
          item.style.borderRadius = "4px";
          item.style.color = "#fca5a5";
          item.textContent = `Vehicle #${v.track_id} (${v.class_name}) · ${v.type} (${v.speed_px.toFixed(0)} px/s at frame ${v.frame})`;
          logList.appendChild(item);
        });

        wrapperDiv.append(violationsHeader, logList);
      }

      analyticsContainer.appendChild(wrapperDiv);
    }
  }

  /**
   * Helper: Safely render the processed video player using DOM elements
   * @param {string} outputUrl - URL path to the processed video
   * @param {number} framesProcessed - Number of frames written
   */
  function renderProcessedVideo(outputUrl, framesProcessed) {
    if (!processedVideoContainer) return;

    processedVideoContainer.className = "card-body";
    processedVideoContainer.replaceChildren();

    // 1. Create <video> element
    const videoElem = document.createElement("video");
    videoElem.controls = true;
    videoElem.autoplay = true;
    videoElem.style.width = "100%";
    videoElem.style.borderRadius = "8px";
    videoElem.style.backgroundColor = "#000";

    const sourceElem = document.createElement("source");
    sourceElem.src = outputUrl;
    sourceElem.type = "video/mp4";
    videoElem.appendChild(sourceElem);

    const fallbackText = document.createTextNode("Your browser does not support HTML5 video playback.");
    videoElem.appendChild(fallbackText);

    // 2. Create info & download caption paragraph
    const captionP = document.createElement("p");
    captionP.style.marginTop = "0.5rem";
    captionP.style.color = "var(--text-muted)";
    captionP.style.fontSize = "0.85rem";

    const framesText = document.createTextNode(`${framesProcessed} frames processed · `);
    captionP.appendChild(framesText);

    const downloadLink = document.createElement("a");
    downloadLink.href = outputUrl;
    downloadLink.download = "";
    downloadLink.style.color = "var(--accent-blue)";
    downloadLink.textContent = "Download video";
    captionP.appendChild(downloadLink);

    processedVideoContainer.append(videoElem, captionP);
    processedVideoContainer.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  /**
   * Event Handler: User selects a video file via file picker
   */
  videoInput.addEventListener("change", (event) => {
    const file = event.target.files && event.target.files[0];

    // Check if user cancelled selection
    if (!file) {
      return;
    }

    // Validation: Verify selected file is strictly MP4, MOV, or AVI
    const isVideo = /\.(mp4|mov|avi)$/i.test(file.name);
    if (!isVideo) {
      alert("Invalid file type. Please select a valid video file (MP4, MOV, AVI).");
      videoInput.value = ""; // clear input
      fileNameDisplay.textContent = "None";
      startBtn.disabled = true;
      return;
    }

    // 1. Display selected file details
    fileNameDisplay.textContent = `${file.name} (${formatFileSize(file.size)})`;

    // 2. Free up memory from previous video URL if one was created earlier
    if (currentVideoUrl) {
      URL.revokeObjectURL(currentVideoUrl);
    }

    // 3. Create a local temporary URL pointing to the user's selected file
    currentVideoUrl = URL.createObjectURL(file);
    videoPreview.src = currentVideoUrl;

    // 4. Update UI visibility
    emptyPreviewState.classList.add("hidden");
    videoPreview.classList.remove("hidden");

    // 5. Reset previous stats and panels to clean placeholder state
    resetOverviewCards();
    startBtn.disabled = false;
    stopBtn.disabled = true;
    setStatus("active", `Video loaded: "${file.name}". Ready for processing.`);
  });

  /**
   * Event Handler: User clicks "Start Processing"
   */
  startBtn.addEventListener("click", async () => {
    if (!videoInput.files || videoInput.files.length === 0) {
      setStatus("idle", "No video selected. Please choose a video first.");
      return;
    }

    const selectedFile = videoInput.files[0];

    // ── 1. Lock the UI while the server works ────────────────────────────
    startBtn.disabled = true;
    setStatus(
      "working",
      `Uploading and processing "${selectedFile.name}"… This may take a minute.`
    );

    // ── 2. Build the multipart payload ───────────────────────────────────
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      // ── 3. Send the request to FastAPI backend ────────────────────────────
      const response = await fetch("/process", {
        method: "POST",
        body: formData,
      });

      // ── 4. Parse JSON response body ──────────────────────────────────────
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || `Server error ${response.status}`);
      }

      // ── 5a. Success — display live analytics and render processed video ──
      setStatus(
        "active",
        `✓ Done! Processed ${data.frames_processed} frames. Live analytics populated below.`
      );

      // Render metrics to dashboard cards and vehicle breakdown panel
      renderAnalytics(data.analytics, data.frames_processed);

      // Safely render the processed video player using DOM creation
      renderProcessedVideo(data.output_url, data.frames_processed);

    } catch (err) {
      // ── 5b. Failure — display the error message ──────────────────────────
      setStatus("idle", `Error: ${err.message}`);
    } finally {
      // ── 6. Always re-enable Start ─────────────────────────────────────────
      startBtn.disabled = false;
    }
  });

  /**
   * Event Handler: User clicks "Stop Processing"
   * Stop Processing remains disabled until a real backend cancellation API is connected.
   */
  stopBtn.addEventListener("click", () => {
    setStatus("idle", "Stop Processing is unavailable until backend API cancellation is complete.");
    stopBtn.disabled = true;
  });
});
