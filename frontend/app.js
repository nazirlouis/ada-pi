const INPUT_RATE = 16000;
const OUTPUT_RATE = 24000;
const disconnectButton = document.querySelector("#disconnect");
const exitButton = document.querySelector("#exit");
const connectionStatus = document.querySelector("#connection-status");
const microphoneStatus = document.querySelector("#microphone-status");
const logElement = document.querySelector("#log");
const faceStage = document.querySelector("#face-stage");
const settingsPanel = document.querySelector("#settings-panel");
const settingsToggle = document.querySelector("#settings-toggle");
const hardwarePanel = document.querySelector("#hardware-panel");
const hardwareToggle = document.querySelector("#hardware-toggle");
const posePanel = document.querySelector("#pose-panel");
const poseToggle = document.querySelector("#pose-toggle");
const poseCanvas = document.querySelector("#pose-canvas");
const poseContext = poseCanvas.getContext("2d");
const poseStatus = document.querySelector("#pose-status");
const detectPanel = document.querySelector("#detect-panel");
const detectToggle = document.querySelector("#detect-toggle");
const detectCanvas = document.querySelector("#detect-canvas");
const detectContext = detectCanvas.getContext("2d");
const detectStatus = document.querySelector("#detect-status");
const postureCooldown = document.querySelector("#posture-cooldown");
const postureState = document.querySelector("#posture-state");
const postureScore = document.querySelector("#posture-score");
const postureToday = document.querySelector("#posture-today");
const postureHabit = document.querySelector("#posture-habit");
const postureGemini = document.querySelector("#posture-gemini");
const postureCalibration = document.querySelector("#posture-calibration");
const postureEvents = document.querySelector("#posture-events");
const habitReminder = document.querySelector("#habit-reminder");
const habitsPanel = document.querySelector("#habits-panel");
const habitsToggle = document.querySelector("#habits-toggle");
const habitSettingsPanel = document.querySelector("#habit-settings-panel");
const habitSettingsToggle = document.querySelector("#habit-settings-toggle");
const habitsList = document.querySelector("#habits-list");
const habitsEmpty = document.querySelector("#habits-empty");
const officeLightsMonitor = document.querySelector("#office-lights-monitor");
const homePanel = document.querySelector("#home-panel");
const homeToggle = document.querySelector("#home-toggle");
const homeDevices = document.querySelector("#home-devices");
const homeStatus = document.querySelector("#home-status");
const habitSignal = document.querySelector("#habit-signal");
const adaSystemPrompt = document.querySelector("#ada-system-prompt");
const adaLiveStatus = document.querySelector("#ada-live-status");
let hardwareTimer = null;
let hardwareConfigLoaded = false;
let poseSocket = null;
let poseResult = null;
let poseFrameCount = 0;
let poseFpsStarted = 0;
let detectSocket = null;
let detectionResult = null;
let detectionFrameCount = 0;
let detectionFpsStarted = 0;
let habitSocket = null;
let lastPostureEventId = null;
let reminderTimer = null;
let habitSignalTimer = null;
let activeHabitSignal = null;
const HABIT_ACK_KEY = "adaAcknowledgedHabitOccurrences";
let connectionInProgress = false;

let socket = null;
let stream = null;
let captureContext = null;
let playbackContext = null;
let captureNode = null;
let playbackNode = null;
let playbackAnalyser = null;
let playbackMeterFrame = null;
let assistantEntry = null;
let assistantPlaybackActive = false;
let localSpeechActive = false;
let speechAboveFrames = 0;
let speechBelowFrames = 0;
let microphoneNoiseFloor = .004;

function setSettingsOpen(open) {
  if (open) { closeHabits(); closeHabitSettings(); closeHome(); }
  settingsPanel.classList.toggle("open", open);
  settingsPanel.setAttribute("aria-hidden", String(!open));
  settingsToggle.setAttribute("aria-expanded", String(open));
  settingsToggle.setAttribute("aria-label", open ? "Close settings" : "Open settings");
}

function openHabitSettings() {
  closePose();
  closeDetect();
  closeHabits();
  closeHome();
  setHardwareOpen(false);
  setSettingsOpen(false);
  habitSettingsPanel.classList.add("open");
  habitSettingsPanel.setAttribute("aria-hidden", "false");
  habitSettingsToggle.setAttribute("aria-expanded", "true");
  refreshPosture();
  refreshOfficeLightSettings();
  refreshVisualMonitors();
}

const MONITOR_FIELDS = {
  sitting_too_long: [["maximum_sitting_minutes", "Maximum sitting (minutes)"], ["break_reset_minutes", "Break reset (minutes)"]],
  phone_distraction: [["confirmation_minutes", "Confirm after (minutes)"], ["reset_minutes", "Reset after (minutes)"]],
  desk_clutter: [["check_interval_seconds", "Check every (seconds)"], ["sustained_change_minutes", "Sustained change (minutes)"], ["reset_minutes", "Reset after (minutes)"]],
};

async function updateVisualMonitor(key, values) {
  const response = await fetch(`/api/habits/monitors/${key}/settings`, {method:"PATCH", headers:{"Content-Type":"application/json"}, body:JSON.stringify(values)});
  const result = await response.json(); if (!response.ok) throw new Error(result.detail || "Could not save monitor settings");
  await refreshVisualMonitors();
}

function renderVisualMonitors(monitors = {}) {
  const root = document.querySelector("#visual-monitor-settings"); root.replaceChildren();
  for (const [key, monitor] of Object.entries(monitors)) {
    const card=document.createElement("article"); card.className="visual-monitor-card";
    const controls=(MONITOR_FIELDS[key] || []).map(([field,label]) => `<label><span>${label}</span><input data-field="${field}" type="number" min="1" value="${monitor.settings[field]}"></label>`).join("");
    card.innerHTML=`<div class="settings-section-heading"><strong>${habitLabel(key)}</strong><label class="monitor-toggle"><span>Enabled</span><input data-field="enabled" type="checkbox" ${monitor.settings.enabled ? "checked" : ""}></label></div><div class="monitor-live">${monitor.state.replaceAll("_"," ")} · ${Math.round((monitor.progress||0)*100)}%${key === "desk_clutter" ? ` · ${monitor.calibrated ? "calibrated" : "needs calibration"}` : ""}</div><div class="office-timing-grid">${controls}</div>${key === "desk_clutter" ? '<button class="desk-calibrate">Calibrate clean desk</button>' : ""}`;
    card.querySelectorAll("input").forEach(input => input.addEventListener("change", () => updateVisualMonitor(key, {[input.dataset.field]: input.type === "checkbox" ? input.checked : Number(input.value)}).catch(error => document.querySelector("#visual-monitor-settings-status").textContent=error.message)));
    card.querySelector(".desk-calibrate")?.addEventListener("click", async () => { const response=await fetch("/api/habits/desk-clutter/calibration",{method:"POST"}); const result=await response.json(); document.querySelector("#visual-monitor-settings-status").textContent=response.ok ? "Calibration started · leave the clean desk and keep the scene still" : (result.detail||"Could not calibrate"); });
    root.append(card);
  }
}

async function refreshVisualMonitors() {
  try { const response=await fetch("/api/habits/monitors",{cache:"no-store"}); const result=await response.json(); if(!response.ok) throw new Error(result.detail); renderVisualMonitors(result.monitors||{}); document.querySelector("#visual-monitor-settings-status").textContent="Live · changes save automatically"; }
  catch(error) { document.querySelector("#visual-monitor-settings-status").textContent=error.message||"Visual monitors unavailable"; }
}

function closeHabitSettings() {
  habitSettingsPanel.classList.remove("open");
  habitSettingsPanel.setAttribute("aria-hidden", "true");
  habitSettingsToggle.setAttribute("aria-expanded", "false");
}

function habitLabel(key) {
  if (key === "posture") return "Posture";
  return String(key).replaceAll("_", " ").replace(/\b\w/g, letter => letter.toUpperCase());
}

function habitProgress(habit) {
  if (habit.status === "established") return 100;
  const occurrenceProgress = Math.min(1, Number(habit.rolling_occurrences || 0) / 10);
  const dayProgress = Math.min(1, Number(habit.rolling_days || 0) / 3);
  return Math.round(Math.min(occurrenceProgress, dayProgress) * 100);
}

function renderHabits(habits = []) {
  habitsList.replaceChildren();
  habitsEmpty.classList.toggle("show", !habits.length);
  for (const habit of habits) {
    const progress = habitProgress(habit);
    const card = document.createElement("article");
    card.className = "habit-card";
    card.dataset.status = habit.status;
    card.innerHTML = `
      <div class="habit-name">
        <span class="habit-status">${habit.status}</span>
        <strong>${habitLabel(habit.habit_key)}</strong>
      </div>
      <div class="habit-progress">
        <div class="habit-progress-copy"><span>Path to established</span><strong>${progress}%</strong></div>
        <div class="habit-bar" role="progressbar" aria-label="${habitLabel(habit.habit_key)} establishment progress" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${progress}"><span></span></div>
      </div>
      <div class="habit-stats">
        <div class="habit-stat"><strong>${habit.rolling_occurrences}</strong><span>of 10 times</span></div>
        <div class="habit-stat"><strong>${habit.rolling_days}</strong><span>of 3 days</span></div>
      </div>${habit.monitor ? `<div class="monitor-live">Monitor · ${habit.monitor.state} · ${Math.round((habit.monitor.progress||0)*100)}%</div>` : ""}`;
    habitsList.append(card);
    requestAnimationFrame(() => { card.querySelector(".habit-bar span").style.width = `${progress}%`; });
  }
}

async function refreshHabits() {
  try {
    const response = await fetch("/api/habits", { cache: "no-store" });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "Could not load habits");
    renderHabits((result.habits || []).map(habit => ({...habit, monitor: result.monitors?.[habit.habit_key]})));
    const pending = result.notifications?.[0];
    if (pending) showHabitSignal({kind:"occurrence", habit_key:pending.habit_key, notification_id:pending.id, ...pending.payload});
    const office = result.office_lights || {};
    const lightCount = office.lights_on?.length || 0;
    officeLightsMonitor.textContent = office.status === "monitoring"
      ? `Office lights · monitoring · ${lightCount} on · person ${office.person_state || "unknown"}`
      : `Office lights · ${office.status || "unavailable"}${office.error ? ` · ${office.error}` : ""}`;
    officeLightsMonitor.classList.toggle("offline", office.status !== "monitoring");
  } catch (error) {
    habitsList.textContent = error.message;
    officeLightsMonitor.textContent = "Office lights · unavailable";
    officeLightsMonitor.classList.add("offline");
  }
}

function openHabits() {
  closePose();
  closeDetect();
  setHardwareOpen(false);
  setSettingsOpen(false);
  closeHabitSettings();
  closeHome();
  habitsPanel.classList.add("open");
  habitsPanel.setAttribute("aria-hidden", "false");
  habitsToggle.setAttribute("aria-expanded", "true");
  refreshHabits();
}

function formatHour(hour) {
  const date = new Date(2000, 0, 1, Number(hour));
  return date.toLocaleTimeString([], { hour: "numeric" });
}

async function refreshOfficeLightSettings() {
  const status = document.querySelector("#office-light-settings-status");
  try {
    const response = await fetch("/api/habits/office-lights", { cache: "no-store" });
    const result = await response.json();
    const settings = result.settings || {};
    document.querySelector("#office-grace-minutes").value = settings.grace_minutes ?? 5;
    document.querySelector("#office-poll-seconds").value = settings.poll_seconds ?? 15;
    document.querySelector("#office-reset-hour").value = settings.reset_hour ?? 18;
    status.textContent = result.status === "monitoring" ? "Monitor online · changes save automatically" : `Monitor ${result.status || "unavailable"} · ${result.error || "check Home Assistant"}`;
  } catch (error) {
    status.textContent = error.message;
  }
}

async function updateOfficeLightSettings(values) {
  const response = await fetch("/api/habits/office-lights/settings", { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(values) });
  const result = await response.json();
  if (!response.ok) throw new Error(result.detail || "Could not save light timing");
  document.querySelector("#office-light-settings-status").textContent = "Saved · the monitor is using the new timing";
}

function closeHome() {
  homePanel.classList.remove("open");
  homePanel.setAttribute("aria-hidden", "true");
  homeToggle.setAttribute("aria-expanded", "false");
}

function openHome() {
  closePose();
  closeDetect();
  closeHabits();
  closeHabitSettings();
  setHardwareOpen(false);
  setSettingsOpen(false);
  homePanel.classList.add("open");
  homePanel.setAttribute("aria-hidden", "false");
  homeToggle.setAttribute("aria-expanded", "true");
  refreshHomeDevices();
}

function renderHomeDevices(entities) {
  homeDevices.replaceChildren();
  document.querySelector("#home-empty").classList.toggle("show", !entities.length);
  for (const entity of entities) {
    const card = document.createElement("article");
    card.className = `home-device${entity.state === "on" ? " on" : ""}`;
    const copy = document.createElement("div");
    copy.className = "home-device-copy";
    const name = document.createElement("strong");
    name.textContent = entity.name;
    const detail = document.createElement("span");
    detail.textContent = `${entity.domain} · ${entity.state}`;
    copy.append(name, detail);
    const power = document.createElement("button");
    power.className = "home-power";
    power.disabled = !entity.available;
    power.setAttribute("aria-label", `${entity.state === "on" ? "Turn off" : "Turn on"} ${entity.name}`);
    power.setAttribute("aria-pressed", String(entity.state === "on"));
    power.addEventListener("click", () => setHomeDevicePower(entity, card, detail, power));
    card.append(copy, power);
    homeDevices.append(card);
  }
}

async function refreshHomeDevices() {
  homeStatus.textContent = "Loading devices…";
  homeStatus.classList.remove("error");
  try {
    const response = await fetch("/api/home-assistant/entities", { cache: "no-store" });
    const result = await response.json();
    if (result.status !== "connected") throw new Error(result.error || "Home Assistant unavailable");
    renderHomeDevices(result.entities || []);
    homeStatus.textContent = `Connected · ${result.entities.length} controllable device${result.entities.length === 1 ? "" : "s"}`;
  } catch (error) {
    renderHomeDevices([]);
    homeStatus.textContent = error.message;
    homeStatus.classList.add("error");
  }
}

async function setHomeDevicePower(entity, card, detail, button) {
  const turnOn = entity.state !== "on";
  button.disabled = true;
  try {
    const response = await fetch(`/api/home-assistant/entities/${encodeURIComponent(entity.entity_id)}/power`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ on: turnOn }) });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "Device command failed");
    entity.state = result.state;
    card.classList.toggle("on", entity.state === "on");
    detail.textContent = `${entity.domain} · ${entity.state}`;
    button.setAttribute("aria-pressed", String(entity.state === "on"));
    button.setAttribute("aria-label", `${entity.state === "on" ? "Turn off" : "Turn on"} ${entity.name}`);
    homeStatus.textContent = `${entity.name} turned ${entity.state}`;
  } catch (error) {
    homeStatus.textContent = error.message;
    homeStatus.classList.add("error");
  } finally {
    button.disabled = !entity.available;
  }
}

function closeHabits() {
  habitsPanel.classList.remove("open");
  habitsPanel.setAttribute("aria-hidden", "true");
  habitsToggle.setAttribute("aria-expanded", "false");
}

function hideHabitSignal() {
  clearTimeout(habitSignalTimer);
  if (activeHabitSignal) {
    if (activeHabitSignal.notification_id) fetch(`/api/habits/notifications/${activeHabitSignal.notification_id}/acknowledge`, {method:"POST"}).catch(() => {});
    let acknowledged = {};
    try { acknowledged = JSON.parse(localStorage.getItem(HABIT_ACK_KEY) || "{}"); } catch (_) { acknowledged = {}; }
    acknowledged[activeHabitSignal.habit_key] = Number(activeHabitSignal.rolling_occurrences || 0);
    localStorage.setItem(HABIT_ACK_KEY, JSON.stringify(acknowledged));
  }
  activeHabitSignal = null;
  habitSignal.classList.remove("show");
  habitSignal.setAttribute("aria-hidden", "true");
}

function showHabitSignal(signal) {
  const signature = signal.notification_id ? `notification:${signal.notification_id}` : `${signal.habit_key}:${signal.rolling_occurrences}`;
  if (activeHabitSignal && (activeHabitSignal.notification_id ? `notification:${activeHabitSignal.notification_id}` : `${activeHabitSignal.habit_key}:${activeHabitSignal.rolling_occurrences}`) === signature) return;
  clearTimeout(habitSignalTimer);
  activeHabitSignal = signal;
  const established = signal.status === "established";
  const advancing = signal.kind === "status_changed" && !established;
  const officeLights = signal.habit_key === "office_lights_left_on";
  document.querySelector("#habit-signal-eyebrow").textContent = established
    ? "PATTERN ESTABLISHED"
    : (officeLights ? "LIGHTS LEFT ON WHILE AWAY" : (advancing ? "PATTERN EVOLVING" : "NEW PATTERN DETECTED"));
  document.querySelector("#habit-signal-name").textContent = habitLabel(signal.habit_key);
  document.querySelector("#habit-signal-detail").textContent = established
    ? "This pattern is now ready for focused correction"
    : (officeLights ? "The office was illuminated beautifully for absolutely nobody." : (advancing ? "Repeated often enough to move into emerging status" : "Added to the tracker. Yes, ADA noticed."));
  habitSignal.classList.add("show");
  habitSignal.setAttribute("aria-hidden", "false");
  if (!officeLights && !signal.notification_id) habitSignalTimer = setTimeout(hideHabitSignal, 6500);
}

function showPostureReminder() {
  clearTimeout(reminderTimer);
  habitReminder.classList.add("show");
  reminderTimer = setTimeout(() => habitReminder.classList.remove("show"), 8000);
}

function renderPostureEvents(events = []) {
  postureEvents.replaceChildren();
  for (const item of events.slice(0, 5)) {
    const row = document.createElement("div");
    row.className = "posture-event";
    const summary = document.createElement("span");
    const started = new Date(item.started_at).toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
    const confirmed = item.gemini_confidence == null ? "" : ` · Gemini ${Math.round(item.gemini_confidence * 100)}%`;
    summary.textContent = `${started} · ${Math.round(item.duration_seconds || 0)} sec · ${Math.round((item.worst_score || 0) * 100)}%${confirmed}`;
    const button = document.createElement("button");
    button.textContent = item.correction === "false_alarm" ? "False alarm ✓" : "False alarm";
    button.disabled = Boolean(item.correction);
    button.addEventListener("click", async () => {
      await fetch(`/api/habits/posture/events/${item.id}/correction`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ correction: "false_alarm" }) });
      await refreshPosture();
    });
    row.append(summary, button);
    postureEvents.append(row);
  }
}

function renderPosture(status, announce = true) {
  postureCooldown.value = status.cooldown_minutes;
  postureState.textContent = status.state.replaceAll("_", " ");
  postureScore.textContent = status.score == null ? (status.calibrated ? "Waiting for a clear pose" : "Not calibrated") : `Score ${Math.round(status.score * 100)}%`;
  postureToday.textContent = `Today · ${Math.round((status.today_seconds || 0) / 60)} min slouching`;
  const habit = status.habit;
  postureHabit.textContent = habit
    ? `Habit status · ${habit.status} · ${habit.rolling_occurrences}/10 this week · ${habit.rolling_days} day${habit.rolling_days === 1 ? "" : "s"}`
    : "Habit status · not observed";
  const geminiLabel = (status.gemini_status || "idle").replaceAll("_", " ");
  const geminiConfidence = status.gemini_confidence == null ? "" : ` · ${Math.round(status.gemini_confidence * 100)}%`;
  postureGemini.textContent = `Gemini confirmation · ${geminiLabel}${geminiConfidence}`;
  postureGemini.title = status.gemini_reason || "";
  if (status.state === "calibrating") {
    const instruction = status.calibration_kind === "slouch" ? "Hold your typical slouch" : "Sit tall and still";
    postureCalibration.textContent = `${instruction}… ${Math.round(status.calibration_progress * 100)}%`;
  } else if (status.calibration_error) {
    postureCalibration.textContent = status.calibration_error;
  } else {
    postureCalibration.textContent = status.slouch_calibrated ? "Good and slouch benchmarks ready." : (status.calibrated ? "Good baseline ready. Now calibrate your typical slouch." : "Sit tall for 30 seconds to establish your baseline.");
  }
  document.querySelector("#posture-calibrate-slouch").disabled = !status.calibrated || status.state === "calibrating";
  if (lastPostureEventId === null) {
    lastPostureEventId = status.last_event_id || 0;
  } else if (announce && status.last_event_id > lastPostureEventId) {
    showPostureReminder();
    lastPostureEventId = status.last_event_id;
  }
  if (status.events) renderPostureEvents(status.events);
  if (status.habits) renderHabits(status.habits.map(habit => ({...habit, monitor: status.monitors?.[habit.habit_key]})));
  if (status.monitors && habitSettingsPanel.classList.contains("open") && !document.querySelector("#visual-monitor-settings")?.contains(document.activeElement)) renderVisualMonitors(status.monitors);
  const pendingNotification = status.notifications?.[0];
  if (pendingNotification) showHabitSignal({kind:"occurrence", habit_key:pendingNotification.habit_key, notification_id:pendingNotification.id, ...pendingNotification.payload});
}

async function refreshPosture() {
  try {
    const response = await fetch("/api/habits/posture", { cache: "no-store" });
    renderPosture(await response.json(), false);
  } catch (error) {
    postureState.textContent = "Unavailable";
  }
}

async function refreshAdaSettings() {
  try {
    const response = await fetch("/api/settings/ada", { cache: "no-store" });
    const settings = await response.json();
    if (document.activeElement !== adaSystemPrompt) adaSystemPrompt.value = settings.system_prompt;
    adaLiveStatus.textContent = `Live API · ${settings.live_status}${settings.live_error ? ` · ${settings.live_error}` : ""}`;
  } catch (error) {
    adaLiveStatus.textContent = "Live API · unavailable";
  }
}

async function updatePostureSettings(values) {
  const response = await fetch("/api/habits/posture/settings", { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(values) });
  const result = await response.json();
  if (!response.ok) throw new Error(result.detail || "Could not update posture settings");
  renderPosture(result, false);
}

function connectHabitSocket() {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  habitSocket = new WebSocket(`${scheme}://${location.host}/ws/habits`);
  habitSocket.onmessage = event => {
    const message = JSON.parse(event.data);
    if (message.type === "posture") {
      renderPosture(message);
      const officeHabit = (message.habits || []).find(habit => habit.habit_key === "office_lights_left_on");
      if (officeHabit) {
        let acknowledged = {};
        try { acknowledged = JSON.parse(localStorage.getItem(HABIT_ACK_KEY) || "{}"); } catch (_) { acknowledged = {}; }
        if (Number(officeHabit.rolling_occurrences || 0) > Number(acknowledged.office_lights_left_on || 0)) {
          showHabitSignal({ kind: "occurrence", ...officeHabit });
        }
      }
      if (["first_added", "status_changed"].includes(message.habit_signal?.kind)) {
        showHabitSignal(message.habit_signal);
      }
    }
  };
  habitSocket.onclose = () => { habitSocket = null; setTimeout(connectHabitSocket, 2000); };
}

const POSE_CONNECTIONS = [
  [0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],
  [5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16],
];

function drawPoseFrame(bitmap) {
  const width = bitmap.width;
  const height = bitmap.height;
  poseCanvas.width = width;
  poseCanvas.height = height;
  poseContext.drawImage(bitmap, 0, 0);
  bitmap.close();
  if (!poseResult) return;
  const points = poseResult.keypoints;
  poseContext.lineWidth = 4;
  poseContext.lineCap = "round";
  poseContext.strokeStyle = "#17dfff";
  poseContext.shadowColor = "#17dfff";
  poseContext.shadowBlur = 8;
  for (const [start, end] of POSE_CONNECTIONS) {
    if (points[start].score < .25 || points[end].score < .25) continue;
    poseContext.beginPath();
    poseContext.moveTo(points[start].x * width, points[start].y * height);
    poseContext.lineTo(points[end].x * width, points[end].y * height);
    poseContext.stroke();
  }
  poseContext.fillStyle = "#ff3dbe";
  for (const point of points) {
    if (point.score < .25) continue;
    poseContext.beginPath();
    poseContext.arc(point.x * width, point.y * height, 6, 0, Math.PI * 2);
    poseContext.fill();
  }
  poseContext.shadowBlur = 0;
  poseFrameCount++;
  const elapsed = performance.now() - poseFpsStarted;
  if (elapsed >= 1000) {
    const fps = (poseFrameCount * 1000 / elapsed).toFixed(1);
    poseStatus.textContent = `Tracking · ${fps} FPS · ${poseResult.inference_ms} ms inference`;
    poseFrameCount = 0;
    poseFpsStarted = performance.now();
  }
}

function openPose() {
  if (poseSocket) return;
  closeDetect();
  setHardwareOpen(false);
  closeHabits();
  closeHabitSettings();
  closeHome();
  setSettingsOpen(false);
  posePanel.classList.add("open");
  posePanel.setAttribute("aria-hidden", "false");
  poseToggle.setAttribute("aria-expanded", "true");
  poseStatus.textContent = "Connecting to camera…";
  poseFpsStarted = performance.now();
  poseFrameCount = 0;
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  poseSocket = new WebSocket(`${scheme}://${location.host}/ws/pose`);
  poseSocket.binaryType = "blob";
  poseSocket.onmessage = async event => {
    if (typeof event.data === "string") {
      const message = JSON.parse(event.data);
      if (message.type === "pose") poseResult = message;
      else poseStatus.textContent = message.message;
      return;
    }
    const bitmap = await createImageBitmap(event.data);
    drawPoseFrame(bitmap);
  };
  poseSocket.onerror = () => { poseStatus.textContent = "Pose connection failed"; };
  poseSocket.onclose = () => {
    poseSocket = null;
    if (posePanel.classList.contains("open")) poseStatus.textContent = "Pose stream stopped";
  };
}

function closePose() {
  if (poseSocket && poseSocket.readyState < WebSocket.CLOSING) poseSocket.close(1000, "pose view closed");
  poseSocket = null;
  poseResult = null;
  posePanel.classList.remove("open");
  posePanel.setAttribute("aria-hidden", "true");
  poseToggle.setAttribute("aria-expanded", "false");
}

function drawDetectionFrame(bitmap) {
  const width = bitmap.width;
  const height = bitmap.height;
  detectCanvas.width = width;
  detectCanvas.height = height;
  detectContext.drawImage(bitmap, 0, 0);
  bitmap.close();
  if (!detectionResult) return;

  detectContext.lineWidth = Math.max(2, width / 240);
  detectContext.font = `700 ${Math.max(15, Math.round(width / 32))}px system-ui, sans-serif`;
  detectContext.textBaseline = "top";
  for (const detection of detectionResult.detections) {
    const x = detection.x * width;
    const y = detection.y * height;
    const boxWidth = detection.width * width;
    const boxHeight = detection.height * height;
    const label = `${detection.label} ${Math.round(detection.score * 100)}%`;
    const textWidth = detectContext.measureText(label).width;
    const textHeight = Math.max(22, Math.round(width / 24));
    const labelY = Math.max(0, y - textHeight);
    detectContext.strokeStyle = "#17dfff";
    detectContext.shadowColor = "#17dfff";
    detectContext.shadowBlur = 7;
    detectContext.strokeRect(x, y, boxWidth, boxHeight);
    detectContext.shadowBlur = 0;
    detectContext.fillStyle = "rgba(0, 18, 28, .9)";
    detectContext.fillRect(x, labelY, textWidth + 12, textHeight);
    detectContext.fillStyle = "#bffaff";
    detectContext.fillText(label, x + 6, labelY + 2);
  }

  detectionFrameCount++;
  const elapsed = performance.now() - detectionFpsStarted;
  if (elapsed >= 1000) {
    const fps = (detectionFrameCount * 1000 / elapsed).toFixed(1);
    const count = detectionResult.detections.length;
    detectStatus.textContent = `Detecting · ${fps} FPS · ${detectionResult.inference_ms} ms · ${count} object${count === 1 ? "" : "s"}`;
    detectionFrameCount = 0;
    detectionFpsStarted = performance.now();
  }
}

function openDetect() {
  if (detectSocket) return;
  closePose();
  setHardwareOpen(false);
  closeHabits();
  closeHabitSettings();
  closeHome();
  setSettingsOpen(false);
  detectPanel.classList.add("open");
  detectPanel.setAttribute("aria-hidden", "false");
  detectToggle.setAttribute("aria-expanded", "true");
  detectStatus.textContent = "Connecting to camera…";
  detectionFpsStarted = performance.now();
  detectionFrameCount = 0;
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  detectSocket = new WebSocket(`${scheme}://${location.host}/ws/detect`);
  detectSocket.binaryType = "blob";
  detectSocket.onmessage = async event => {
    if (typeof event.data === "string") {
      const message = JSON.parse(event.data);
      if (message.type === "detection") detectionResult = message;
      else detectStatus.textContent = message.message;
      return;
    }
    const bitmap = await createImageBitmap(event.data);
    drawDetectionFrame(bitmap);
  };
  detectSocket.onerror = () => { detectStatus.textContent = "Detection connection failed"; };
  detectSocket.onclose = () => {
    detectSocket = null;
    if (detectPanel.classList.contains("open")) detectStatus.textContent = "Detection stream stopped";
  };
}

function closeDetect() {
  if (detectSocket && detectSocket.readyState < WebSocket.CLOSING) detectSocket.close(1000, "detect view closed");
  detectSocket = null;
  detectionResult = null;
  detectPanel.classList.remove("open");
  detectPanel.setAttribute("aria-hidden", "true");
  detectToggle.setAttribute("aria-expanded", "false");
}

const metricGroups = [
  ["Power & battery", /battery|voltage|current|power|charging|input_plugged|power_source/i],
  ["Cooling & temperature", /temperature|fan|thermal/i],
  ["Storage", /disk|storage|filesystem|mount/i],
  ["Performance", /cpu|gpu|memory|ram|load|uptime/i],
  ["Network", /network|ip_address|mac_address|ethernet|wifi|upload|download/i],
];

function flattenMetrics(value, prefix = "", output = {}) {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    for (const [key, child] of Object.entries(value)) flattenMetrics(child, prefix ? `${prefix}.${key}` : key, output);
  } else if (Array.isArray(value)) {
    value.forEach((child, index) => flattenMetrics(child, `${prefix}.${index + 1}`, output));
  } else if (prefix && value !== null && value !== undefined) output[prefix] = value;
  return output;
}

function metricLabel(key) {
  return key.split(".").map(part => part.replaceAll("_", " ").replace(/\b\w/g, letter => letter.toUpperCase())).join(" · ");
}

function metricValue(key, value) {
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value !== "number") return String(value);
  const rounded = Math.round(value * 100) / 100;
  if (/temperature/i.test(key)) return `${rounded}°`;
  if (/percentage|percent|usage|fan_speed/i.test(key)) return `${rounded}%`;
  if (/voltage/i.test(key)) return `${rounded} V`;
  if (/current/i.test(key)) return `${rounded} A`;
  if (/power/i.test(key) && !/source/i.test(key)) return `${rounded} W`;
  return String(rounded);
}

function renderTelemetry(data) {
  const flat = flattenMetrics(data);
  const grouped = new Map(metricGroups.map(([name]) => [name, []]));
  grouped.set("System", []);
  for (const [key, value] of Object.entries(flat)) {
    const group = metricGroups.find(([, pattern]) => pattern.test(key))?.[0] || "System";
    grouped.get(group).push([key, value]);
  }
  const root = document.querySelector("#pm-telemetry");
  root.replaceChildren();
  for (const [name, metrics] of grouped) {
    if (!metrics.length) continue;
    const section = document.createElement("section");
    section.className = "metric-section";
    const heading = document.createElement("h3");
    heading.textContent = name;
    const grid = document.createElement("div");
    grid.className = "telemetry";
    for (const [key, value] of metrics) {
      const card = document.createElement("div");
      const label = document.createElement("span");
      const strong = document.createElement("strong");
      label.textContent = metricLabel(key);
      strong.textContent = metricValue(key, value);
      card.append(label, strong);
      grid.append(card);
    }
    section.append(heading, grid);
    root.append(section);
  }
  if (!root.children.length) root.textContent = "No telemetry reported by this Pironman configuration.";
}

function setHardwareOpen(open) {
  if (open) {
    closePose();
    closeDetect();
    closeHabits();
    closeHabitSettings();
    closeHome();
    setSettingsOpen(false);
  }
  hardwarePanel.classList.toggle("open", open);
  hardwarePanel.setAttribute("aria-hidden", String(!open));
  hardwareToggle.setAttribute("aria-expanded", String(open));
  document.body.classList.toggle("hardware-open", open);
  clearInterval(hardwareTimer);
  hardwareTimer = open ? setInterval(refreshHardware, 3000) : null;
  if (open) refreshHardware();
}

async function refreshHardware() {
  const state = document.querySelector("#hardware-state");
  try {
    const response = await fetch("/api/pironman", { cache: "no-store" });
    const snapshot = await response.json();
    state.textContent = snapshot.online ? "Online · live updates every 3 seconds" : (snapshot.error || "Dashboard offline");
    state.classList.toggle("offline", !snapshot.online);
    document.querySelector("#pm-dashboard-link").href = snapshot.dashboard_url;
    if (!snapshot.online) return;
    const data = snapshot.data || {};
    const config = snapshot.config || {};
    renderTelemetry(data);
    if (!hardwareConfigLoaded) {
      document.querySelector("#pm-rgb-enable").checked = config.rgb_enable ?? false;
      document.querySelector("#pm-oled-enable").checked = config.oled_enable ?? false;
      if (/^#[0-9a-f]{6}$/i.test(config.rgb_color || "")) document.querySelector("#pm-rgb-color").value = config.rgb_color;
      const brightness = Number(config.rgb_brightness ?? 50);
      document.querySelector("#pm-rgb-brightness").value = brightness;
      document.querySelector("#pm-brightness-value").value = `${brightness}%`;
      const style = document.querySelector("#pm-rgb-style");
      if (config.rgb_style && ![...style.options].some(option => option.value === config.rgb_style)) style.add(new Option(config.rgb_style, config.rgb_style));
      if (config.rgb_style) style.value = config.rgb_style;
      document.querySelector("#pm-rgb-speed").value = config.rgb_speed ?? 50;
      document.querySelector("#pm-speed-value").value = `${config.rgb_speed ?? 50}%`;
      document.querySelector("#pm-oled-rotation").value = config.oled_rotation ?? 0;
      document.querySelector("#pm-oled-sleep").value = config.oled_sleep_timeout ?? 0;
      document.querySelector("#pm-temperature-unit").value = config.temperature_unit ?? "C";
      document.querySelector("#pm-fan-mode").value = config.gpio_fan_mode ?? 3;
      document.querySelector("#pm-fan-led").value = config.gpio_fan_led ?? "follow";
      document.querySelectorAll("[data-config]").forEach(element => element.classList.toggle("unavailable", !(element.dataset.config in config)));
      hardwareConfigLoaded = true;
    }
  } catch (error) {
    state.textContent = `Pironman unavailable · ${error.message}`;
    state.classList.add("offline");
  }
}

async function updateHardware(control, value) {
  const state = document.querySelector("#hardware-state");
  state.textContent = "Applying…";
  try {
    const response = await fetch("/api/pironman/controls", { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ [control]: value }) });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "Control update failed");
    state.textContent = "Online · setting applied";
  } catch (error) {
    state.textContent = error.message;
    state.classList.add("offline");
    hardwareConfigLoaded = false;
  }
}

function logLine(role, text, className = "") {
  const line = document.createElement("p");
  line.className = `entry ${className}`;
  const label = document.createElement("span");
  label.className = "role";
  label.textContent = `${role}: `;
  const content = document.createElement("span");
  content.textContent = text;
  line.append(label, content);
  logElement.append(line);
  logElement.scrollTop = logElement.scrollHeight;
  return content;
}

function setConnected(connected) {
  disconnectButton.disabled = !connected;
  connectionStatus.textContent = connected ? "Connected" : "Disconnected";
}

function downsampleToPCM16(input, inputRate, outputRate) {
  const ratio = inputRate / outputRate;
  const length = Math.floor(input.length / ratio);
  const output = new Int16Array(length);
  for (let i = 0; i < length; i++) {
    const start = Math.floor(i * ratio);
    const end = Math.max(start + 1, Math.floor((i + 1) * ratio));
    let sum = 0;
    for (let j = start; j < end && j < input.length; j++) sum += input[j];
    const sample = Math.max(-1, Math.min(1, sum / (end - start)));
    output[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }
  return output.buffer;
}

async function createPlayback() {
  playbackContext = new AudioContext({ sampleRate: OUTPUT_RATE, latencyHint: "interactive" });
  const workletSource = `
    class PCMPlayer extends AudioWorkletProcessor {
      constructor() { super(); this.queue = []; this.offset = 0; this.bufferedSamples = 0;
        this.playing = false; this.forceStart = false; this.startThreshold = 1440;
        this.port.onmessage = e => {
        if (e.data.type === 'clear') {
          this.queue = []; this.offset = 0; this.bufferedSamples = 0;
          this.playing = false; this.forceStart = false;
        } else if (e.data.type === 'flush') {
          this.forceStart = true;
        } else {
          const chunk = new Int16Array(e.data);
          this.queue.push(chunk); this.bufferedSamples += chunk.length;
        }
      }; }
      process(inputs, outputs) {
        const out = outputs[0][0]; out.fill(0); let n = 0;
        // Hold about 60 ms before starting or recovering from an underrun.
        // This absorbs ordinary WebSocket jitter without delaying barge-in.
        if (!this.playing) {
          if (this.bufferedSamples < this.startThreshold && !(this.forceStart && this.bufferedSamples)) return true;
          this.playing = true;
        }
        while (n < out.length && this.queue.length) {
          const chunk = this.queue[0];
          while (n < out.length && this.offset < chunk.length) {
            out[n++] = chunk[this.offset++] / 32768; this.bufferedSamples--;
          }
          if (this.offset >= chunk.length) { this.queue.shift(); this.offset = 0; }
        }
        if (!this.queue.length) { this.playing = false; this.forceStart = false; }
        return true;
      }
    }
    registerProcessor('pcm-player', PCMPlayer);`;
  const url = URL.createObjectURL(new Blob([workletSource], { type: "text/javascript" }));
  await playbackContext.audioWorklet.addModule(url);
  URL.revokeObjectURL(url);
  playbackNode = new AudioWorkletNode(playbackContext, "pcm-player", { outputChannelCount: [1] });
  // Passive output metering for the mouth. The worklet remains the only PCM
  // source, and the analyser neither buffers nor modifies its audio.
  playbackAnalyser = playbackContext.createAnalyser();
  playbackAnalyser.fftSize = 256;
  playbackAnalyser.smoothingTimeConstant = .68;
  playbackNode.connect(playbackAnalyser);
  playbackAnalyser.connect(playbackContext.destination);
  await playbackContext.resume();
  startPlaybackMeter();
}

function startPlaybackMeter() {
  const samples = new Float32Array(playbackAnalyser.fftSize);
  let smoothed = 0;
  let lastUpdate = 0;

  const measure = (now) => {
    if (!playbackAnalyser) return;
    playbackAnalyser.getFloatTimeDomainData(samples);
    let power = 0;
    for (const sample of samples) power += sample * sample;
    const rms = Math.sqrt(power / samples.length);
    const level = Math.min(1, Math.max(0, (rms - .006) * 8.5));
    smoothed = level > smoothed ? smoothed * .48 + level * .52 : smoothed * .76 + level * .24;

    // Thirty visual updates per second are enough for responsive speech and
    // avoid unnecessary SVG churn on the Pi.
    if (now - lastUpdate >= 33) {
      window.idleFace?.setSpeechLevel(smoothed);
      lastUpdate = now;
    }
    playbackMeterFrame = requestAnimationFrame(measure);
  };
  playbackMeterFrame = requestAnimationFrame(measure);
}

async function startMicrophone() {
  stream = await navigator.mediaDevices.getUserMedia({
    // PipeWire's ada_aec_source has already removed speaker playback using the
    // paired ada_aec_sink as its exact render reference. A second adaptive AEC
    // here can distort near-end speech and make double-talk less reliable.
    audio: { echoCancellation: false, noiseSuppression: true, autoGainControl: false }
  });
  const track = stream.getAudioTracks()[0];
  const settings = track.getSettings();
  console.info("microphone started", settings);
  microphoneStatus.textContent = `On (PipeWire AEC; browser AEC: ${settings.echoCancellation ?? "off"})`;

  captureContext = new AudioContext({ latencyHint: "interactive" });
  const source = captureContext.createMediaStreamSource(stream);
  // ScriptProcessor is intentionally retained for broad Pi Chromium support.
  captureNode = captureContext.createScriptProcessor(2048, 1, 1);
  const silent = captureContext.createGain();
  silent.gain.value = 0;
  captureNode.onaudioprocess = (event) => {
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    const samples = event.inputBuffer.getChannelData(0);
    let power = 0;
    for (const sample of samples) power += sample * sample;
    const rms = Math.sqrt(power / samples.length);
    if (!localSpeechActive) microphoneNoiseFloor = microphoneNoiseFloor * .98 + rms * .02;
    const startThreshold = Math.max(.012, microphoneNoiseFloor * 2.8);
    const stopThreshold = Math.max(.008, microphoneNoiseFloor * 1.7);
    if (!localSpeechActive) {
      speechAboveFrames = rms > startThreshold ? speechAboveFrames + 1 : 0;
      if (speechAboveFrames >= 3) {
        localSpeechActive = true;
        speechBelowFrames = 0;
        socket.send(JSON.stringify({
          type: "local_speech_started", rms, threshold: startThreshold,
          noise_floor: microphoneNoiseFloor, playback_active: assistantPlaybackActive
        }));
      }
    } else {
      speechBelowFrames = rms < stopThreshold ? speechBelowFrames + 1 : 0;
      if (speechBelowFrames >= 12) {
        localSpeechActive = false;
        speechAboveFrames = 0;
        socket.send(JSON.stringify({
          type: "local_speech_stopped", rms, threshold: stopThreshold,
          noise_floor: microphoneNoiseFloor, playback_active: assistantPlaybackActive
        }));
      }
    }
    const pcm = downsampleToPCM16(
      samples, captureContext.sampleRate, INPUT_RATE
    );
    socket.send(pcm);
  };
  source.connect(captureNode);
  captureNode.connect(silent);
  silent.connect(captureContext.destination);
  await captureContext.resume();
}

function handleControl(event) {
  switch (event.type) {
    case "ready":
      window.idleFace?.setConnecting(false);
      setConnected(true);
      connectionStatus.textContent = "Connected — listening";
      break;
    case "speech_started":
      microphoneStatus.textContent = "Speech detected";
      break;
    case "speech_stopped":
      microphoneStatus.textContent = "On — listening";
      break;
    case "clear_audio":
      assistantPlaybackActive = false;
      playbackNode?.port.postMessage({ type: "clear" });
      window.idleFace?.setSpeechLevel(0, true);
      assistantEntry = null;
      console.info("assistant interrupted; playback queue cleared");
      break;
    case "user_transcript":
      logLine("You", event.text);
      break;
    case "assistant_transcript_delta":
      if (!assistantEntry) assistantEntry = logLine("Assistant", "");
      assistantEntry.textContent += event.text;
      logElement.scrollTop = logElement.scrollHeight;
      break;
    case "response_started":
      assistantPlaybackActive = true;
      assistantEntry = null;
      break;
    case "response_completed":
      assistantPlaybackActive = false;
      // Release a short final chunk that may be smaller than the jitter
      // buffer's normal start threshold.
      playbackNode?.port.postMessage({ type: "flush" });
      assistantEntry = null;
      break;
    case "response_interrupted":
      assistantPlaybackActive = false;
      window.idleFace?.setSpeechLevel(0, true);
      assistantEntry = null;
      break;
    case "expression":
      window.idleFace?.setExpression(event.name);
      break;
    case "error":
      logLine("Error", event.message, "system");
      break;
  }
}

async function connect() {
  if (connectionInProgress || socket) return;
  connectionInProgress = true;
  window.idleFace?.setConnecting(true);
  connectionStatus.textContent = "Requesting microphone…";
  try {
    await createPlayback();
    await startMicrophone();
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    socket = new WebSocket(`${scheme}://${location.host}/ws`);
    socket.binaryType = "arraybuffer";
    socket.onopen = () => { connectionStatus.textContent = "Connecting to AI…"; };
    socket.onmessage = (message) => {
      if (typeof message.data === "string") handleControl(JSON.parse(message.data));
      else playbackNode?.port.postMessage(message.data, [message.data]);
    };
    socket.onerror = () => logLine("System", "WebSocket error", "system");
    socket.onclose = () => disconnect(false);
  } catch (error) {
    window.idleFace?.setConnecting(false, true);
    console.error(error);
    logLine("Error", error.message, "system");
    await disconnect(false);
  } finally {
    connectionInProgress = false;
  }
}

async function disconnect(closeSocket = true) {
  window.idleFace?.setConnecting(false, true);
  if (closeSocket && socket && socket.readyState < WebSocket.CLOSING) socket.close(1000, "user disconnect");
  socket = null;
  if (captureNode) captureNode.disconnect();
  if (playbackMeterFrame) cancelAnimationFrame(playbackMeterFrame);
  playbackMeterFrame = null;
  playbackAnalyser = null;
  if (stream) stream.getTracks().forEach(track => track.stop());
  if (captureContext) await captureContext.close().catch(() => {});
  if (playbackContext) await playbackContext.close().catch(() => {});
  stream = captureContext = playbackContext = captureNode = playbackNode = null;
  localSpeechActive = false;
  speechAboveFrames = speechBelowFrames = 0;
  microphoneNoiseFloor = .004;
  assistantEntry = null;
  microphoneStatus.textContent = "Off";
  setConnected(false);
  window.idleFace?.setSpeechLevel(0, true);
  console.info("session closed");
}

disconnectButton.addEventListener("click", () => disconnect(true));
exitButton.addEventListener("click", async () => {
  exitButton.disabled = true;
  await disconnect(true);
  try {
    await fetch("/shutdown", { method: "POST", cache: "no-store" });
  } catch (error) {
    console.error("Could not stop backend", error);
    exitButton.disabled = false;
  }
});
hardwareToggle.addEventListener("click", () => setHardwareOpen(true));
poseToggle.addEventListener("click", openPose);
detectToggle.addEventListener("click", openDetect);
settingsToggle.addEventListener("click", () => {
  const opening = !settingsPanel.classList.contains("open");
  if (opening) {
    closePose();
    closeDetect();
    setHardwareOpen(false);
    closeHome();
  }
  setSettingsOpen(opening);
  if (opening) { refreshPosture(); refreshAdaSettings(); }
});
postureCooldown.addEventListener("change", () => updatePostureSettings({ cooldown_minutes: Number(postureCooldown.value) }).catch(error => { postureState.textContent = error.message; }));
const officeResetHour = document.querySelector("#office-reset-hour");
for (let hour = 0; hour < 24; hour++) {
  const option = document.createElement("option");
  option.value = String(hour);
  option.textContent = formatHour(hour);
  officeResetHour.append(option);
}
document.querySelector("#office-grace-minutes").addEventListener("change", event => updateOfficeLightSettings({ grace_minutes: Number(event.target.value) }).catch(error => { document.querySelector("#office-light-settings-status").textContent = error.message; }));
document.querySelector("#office-poll-seconds").addEventListener("change", event => updateOfficeLightSettings({ poll_seconds: Number(event.target.value) }).catch(error => { document.querySelector("#office-light-settings-status").textContent = error.message; }));
officeResetHour.addEventListener("change", event => updateOfficeLightSettings({ reset_hour: Number(event.target.value) }).catch(error => { document.querySelector("#office-light-settings-status").textContent = error.message; }));
document.querySelector("#posture-calibrate").addEventListener("click", async () => {
  const response = await fetch("/api/habits/posture/calibration/start/good", { method: "POST" });
  renderPosture(await response.json(), false);
});
document.querySelector("#posture-calibrate-slouch").addEventListener("click", async () => {
  const response = await fetch("/api/habits/posture/calibration/start/slouch", { method: "POST" });
  renderPosture(await response.json(), false);
});
document.querySelector("#habit-clear").addEventListener("click", async () => {
  if (!window.confirm("Clear all habit events and progress? Posture calibrations and settings will be kept.")) return;
  const response = await fetch("/api/habits/reset", { method: "POST" });
  const result = await response.json();
  if (!response.ok) {
    postureState.textContent = result.detail || "Could not clear habit data";
    return;
  }
  lastPostureEventId = 0;
  localStorage.removeItem(HABIT_ACK_KEY);
  renderPosture(result, false);
});
document.querySelector("#ada-prompt-save").addEventListener("click", async () => {
  const response = await fetch("/api/settings/ada", { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ system_prompt: adaSystemPrompt.value }) });
  const result = await response.json();
  adaLiveStatus.textContent = response.ok ? "Live API · reconnecting with new prompt" : (result.detail || "Could not save prompt");
  if (response.ok) setTimeout(refreshAdaSettings, 1500);
});
document.querySelector("#pm-rgb-enable").addEventListener("change", event => updateHardware("rgb_enable", event.target.checked));
document.querySelector("#pm-oled-enable").addEventListener("change", event => updateHardware("oled_enable", event.target.checked));
document.querySelector("#pm-rgb-color").addEventListener("change", event => updateHardware("rgb_color", event.target.value));
document.querySelector("#pm-rgb-style").addEventListener("change", event => updateHardware("rgb_style", event.target.value));
document.querySelector("#pm-rgb-brightness").addEventListener("input", event => { document.querySelector("#pm-brightness-value").value = `${event.target.value}%`; });
document.querySelector("#pm-rgb-brightness").addEventListener("change", event => updateHardware("rgb_brightness", Number(event.target.value)));
document.querySelector("#pm-rgb-speed").addEventListener("input", event => { document.querySelector("#pm-speed-value").value = `${event.target.value}%`; });
document.querySelector("#pm-rgb-speed").addEventListener("change", event => updateHardware("rgb_speed", Number(event.target.value)));
document.querySelector("#pm-oled-rotation").addEventListener("change", event => updateHardware("oled_rotation", Number(event.target.value)));
document.querySelector("#pm-oled-sleep").addEventListener("change", event => updateHardware("oled_sleep_timeout", Number(event.target.value)));
document.querySelector("#pm-temperature-unit").addEventListener("change", event => updateHardware("temperature_unit", event.target.value));
document.querySelector("#pm-fan-mode").addEventListener("change", event => updateHardware("gpio_fan_mode", Number(event.target.value)));
document.querySelector("#pm-fan-led").addEventListener("change", event => updateHardware("gpio_fan_led", event.target.value));
habitsToggle.addEventListener("click", openHabits);
document.querySelector("#habits-close").addEventListener("click", closeHabits);
habitSettingsToggle.addEventListener("click", openHabitSettings);
document.querySelector("#habit-settings-close").addEventListener("click", closeHabitSettings);
homeToggle.addEventListener("click", openHome);
document.querySelector("#home-close").addEventListener("click", closeHome);
document.querySelector("#home-refresh").addEventListener("click", refreshHomeDevices);
document.querySelector("#habit-signal-open").addEventListener("click", () => { hideHabitSignal(); openHabits(); });
habitSignal.addEventListener("pointerdown", event => { if (event.target === habitSignal) hideHabitSignal(); });

// Clicking the face is only an outside-click action for the Settings drawer.
faceStage.addEventListener("pointerdown", (event) => {
  if (event.target.closest("#settings-toggle, #settings-panel, #hardware-panel, #pose-panel, #detect-panel, #habits-panel, #habit-settings-panel, #home-panel, #habit-signal")) return;
  if (settingsPanel.classList.contains("open")) {
    setSettingsOpen(false);
    return;
  }
});

refreshPosture();
refreshAdaSettings();
connectHabitSocket();
// Chromium kiosk is launched with autoplay and microphone permission enabled,
// so Ada can animate, connect, and greet without requiring a screen tap.
window.idleFace?.setConnecting(true);
connect();
