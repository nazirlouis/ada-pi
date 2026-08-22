const INPUT_RATE = 16000;
const OUTPUT_RATE = 24000;
const connectButton = document.querySelector("#connect");
const disconnectButton = document.querySelector("#disconnect");
const exitButton = document.querySelector("#exit");
const connectionStatus = document.querySelector("#connection-status");
const microphoneStatus = document.querySelector("#microphone-status");
const logElement = document.querySelector("#log");
const faceStage = document.querySelector("#face-stage");

let socket = null;
let stream = null;
let captureContext = null;
let playbackContext = null;
let captureNode = null;
let playbackNode = null;
let playbackAnalyser = null;
let playbackMeterFrame = null;
let assistantEntry = null;

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
    const pcm = downsampleToPCM16(
      event.inputBuffer.getChannelData(0), captureContext.sampleRate, INPUT_RATE
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

// The full-screen idle face is the connection control. This preserves the
// user gesture Chromium requires for microphone and Web Audio access.
faceStage.addEventListener("pointerdown", (event) => {
  if (event.target.closest("#session-controls, #expression-controls")) return;
  if (!socket && !connectButton.disabled) connect();
});
