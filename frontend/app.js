const INPUT_RATE = 16000;
const OUTPUT_RATE = 24000;
const connectButton = document.querySelector("#connect");
const disconnectButton = document.querySelector("#disconnect");
const connectionStatus = document.querySelector("#connection-status");
const microphoneStatus = document.querySelector("#microphone-status");
const logElement = document.querySelector("#log");
const adaStage = document.querySelector("#ada-stage");

let socket = null;
let stream = null;
let captureContext = null;
let playbackContext = null;
let captureNode = null;
let playbackNode = null;
let playbackAnalyser = null;
let playbackMeterFrame = null;
let assistantEntry = null;
let settleTimer = null;
let inputQuietSince = 0;

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
      constructor() { super(); this.queue = []; this.offset = 0; this.port.onmessage = e => {
        if (e.data.type === 'clear') { this.queue = []; this.offset = 0; }
        else this.queue.push(new Int16Array(e.data));
      }; }
      process(inputs, outputs) {
        const out = outputs[0][0]; out.fill(0); let n = 0;
        while (n < out.length && this.queue.length) {
          const chunk = this.queue[0];
          while (n < out.length && this.offset < chunk.length) out[n++] = chunk[this.offset++] / 32768;
          if (this.offset >= chunk.length) { this.queue.shift(); this.offset = 0; }
        }
        return true;
      }
    }
    registerProcessor('pcm-player', PCMPlayer);`;
  const url = URL.createObjectURL(new Blob([workletSource], { type: "text/javascript" }));
  await playbackContext.audioWorklet.addModule(url);
  URL.revokeObjectURL(url);
  playbackNode = new AudioWorkletNode(playbackContext, "pcm-player", { outputChannelCount: [1] });

  // Analysis is passive. The existing worklet remains the sole source of
  // assistant audio, so the rig adds no latency and no second microphone.
  playbackAnalyser = playbackContext.createAnalyser();
  playbackAnalyser.fftSize = 256;
  playbackAnalyser.smoothingTimeConstant = 0.68;
  playbackNode.connect(playbackAnalyser);
  playbackAnalyser.connect(playbackContext.destination);
  await playbackContext.resume();
  startPlaybackMeter();
}

function startPlaybackMeter() {
  const samples = new Float32Array(playbackAnalyser.fftSize);
  const spectrum = new Uint8Array(playbackAnalyser.frequencyBinCount);
  const measure = () => {
    if (!playbackAnalyser) return;
    playbackAnalyser.getFloatTimeDomainData(samples);
    playbackAnalyser.getByteFrequencyData(spectrum);

    let power = 0;
    for (const sample of samples) power += sample * sample;
    const rms = Math.sqrt(power / samples.length);
    const level = Math.min(1, Math.max(0, (rms - .007) * 8.2));

    // Vowels concentrate more energy in lower bands while consonants are
    // brighter. This is still approximate, but it produces more believable
    // viseme choices than volume-only mouth flapping.
    let lowEnergy = 0;
    let highEnergy = 0;
    for (let i = 2; i <= 11; i++) lowEnergy += spectrum[i];
    for (let i = 12; i <= 48; i++) highEnergy += spectrum[i];
    lowEnergy /= 10;
    highEnergy /= 37;
    const brightness = highEnergy / Math.max(1, lowEnergy + highEnergy);

    if (window.setSpeechFeatures) setSpeechFeatures(level, brightness);
    else setSpeechLevel(level);
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
    const input = event.inputBuffer.getChannelData(0);
    let inputPower = 0;
    for (const sample of input) inputPower += sample * sample;
    const inputRms = Math.sqrt(inputPower / input.length);
    if (adaVisual.state !== "speaking" && inputRms > .025) {
      inputQuietSince = 0;
      setAdaState("listening");
    } else if (adaVisual.state === "listening") {
      if (!inputQuietSince) inputQuietSince = performance.now();
      else if (performance.now() - inputQuietSince > 350) setAdaState("thinking");
    }
    const pcm = downsampleToPCM16(input, captureContext.sampleRate, INPUT_RATE);
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
      setConnected(true);
      connectionStatus.textContent = "Connected — listening";
      setAdaState("idle");
      break;
    case "speech_started":
      microphoneStatus.textContent = "Speech detected";
      setAdaState("listening");
      break;
    case "speech_stopped":
      microphoneStatus.textContent = "On — listening";
      break;
    case "clear_audio":
      playbackNode?.port.postMessage({ type: "clear" });
      clearTimeout(settleTimer);
      setSpeechLevel(0);
      setAdaState("listening");
      assistantEntry = null;
      console.info("assistant interrupted; playback queue cleared");
      break;
    case "user_transcript":
      logLine("You", event.text);
      setAdaState("thinking");
      break;
    case "assistant_transcript_delta":
      if (!assistantEntry) assistantEntry = logLine("Assistant", "");
      assistantEntry.textContent += event.text;
      logElement.scrollTop = logElement.scrollHeight;
      break;
    case "response_started":
      assistantEntry = null;
      clearTimeout(settleTimer);
      setAdaState("speaking");
      break;
    case "response_completed":
      settleTimer = setTimeout(() => setAdaState("idle"), 450);
      assistantEntry = null;
      break;
    case "response_interrupted":
      clearTimeout(settleTimer);
      setSpeechLevel(0);
      setAdaState("listening");
      assistantEntry = null;
      break;
    case "error":
      logLine("Error", event.message, "system");
      setAdaState("alert");
      break;
  }
}

async function connect() {
  connectButton.disabled = true;
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
      else {
        setAdaState("speaking");
        playbackNode?.port.postMessage(message.data, [message.data]);
      }
    };
    socket.onerror = () => logLine("System", "WebSocket error", "system");
    socket.onclose = () => disconnect(false);
  } catch (error) {
    console.error(error);
    logLine("Error", error.message, "system");
    await disconnect(false);
  }
}

async function disconnect(closeSocket = true) {
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
  setSpeechLevel(0);
  setAdaState("idle");
  console.info("session closed");
}

connectButton.addEventListener("click", connect);
disconnectButton.addEventListener("click", () => disconnect(true));

// Kiosk mode has no visible controls. The first tap/click is the browser
// gesture required for microphone permission and AudioContext playback.
adaStage.addEventListener("pointerdown", () => {
  if (!socket && !connectButton.disabled) connect();
});
