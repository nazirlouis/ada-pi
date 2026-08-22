const INPUT_RATE = 16000;
const OUTPUT_RATE = 24000;
const connectButton = document.querySelector("#connect");
const disconnectButton = document.querySelector("#disconnect");
const exitButton = document.querySelector("#exit");
const connectionStatus = document.querySelector("#connection-status");
const microphoneStatus = document.querySelector("#microphone-status");
const logElement = document.querySelector("#log");
const faceStage = document.querySelector("#face-stage");
const hardwarePanel = document.querySelector("#hardware-panel");
const hardwareToggle = document.querySelector("#hardware-toggle");
const hardwareClose = document.querySelector("#hardware-close");
let hardwareTimer = null;
let hardwareConfigLoaded = false;

let socket = null;
let stream = null;
let captureContext = null;
let playbackContext = null;
let captureNode = null;
let playbackNode = null;
let playbackAnalyser = null;
let playbackMeterFrame = null;
let assistantEntry = null;
let localSpeechActive = false;
let speechAboveFrames = 0;
let speechBelowFrames = 0;
let microphoneNoiseFloor = .004;

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

async function syncManualExpression(expression) {
  try {
    const response = await fetch("/api/pironman/expression", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ expression }),
    });
    if (!response.ok) {
      const result = await response.json();
      throw new Error(result.detail || "Expression lighting failed");
    }
  } catch (error) {
    console.warn("Could not sync manual expression lighting", error);
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
  connectButton.disabled = connected;
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
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
  });
  const track = stream.getAudioTracks()[0];
  const settings = track.getSettings();
  console.info("microphone started", settings);
  microphoneStatus.textContent = `On (AEC: ${settings.echoCancellation ?? "requested"})`;

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
        socket.send(JSON.stringify({ type: "local_speech_started" }));
      }
    } else {
      speechBelowFrames = rms < stopThreshold ? speechBelowFrames + 1 : 0;
      if (speechBelowFrames >= 12) {
        localSpeechActive = false;
        speechAboveFrames = 0;
        socket.send(JSON.stringify({ type: "local_speech_stopped" }));
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
      assistantEntry = null;
      break;
    case "response_completed":
      // Release a short final chunk that may be smaller than the jitter
      // buffer's normal start threshold.
      playbackNode?.port.postMessage({ type: "flush" });
      assistantEntry = null;
      break;
    case "response_interrupted":
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
  connectButton.disabled = true;
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

connectButton.addEventListener("click", connect);
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
hardwareClose.addEventListener("click", () => setHardwareOpen(false));
document.querySelector("#expression-controls").addEventListener("click", event => {
  const expression = event.target.closest("button")?.dataset.expression;
  if (expression) syncManualExpression(expression);
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

// The full-screen idle face is the connection control. This preserves the
// user gesture Chromium requires for microphone and Web Audio access.
faceStage.addEventListener("pointerdown", (event) => {
  if (event.target.closest("#session-controls, #expression-controls, #hardware-panel")) return;
  if (!socket && !connectButton.disabled) connect();
});
