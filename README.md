# ADA Pi

ADA Pi is a simple continuous full-duplex voice chat for Raspberry Pi. Chromium captures the microphone and plays assistant audio, while FastAPI proxies 16 kHz microphone audio to a Gemini 3.1 Flash Live session and returns its 24 kHz native audio. Gemini's automatic voice activity detection creates turns and interrupts a response when new speech starts.

## Install

Raspberry Pi OS 64-bit and Python 3.11+ are recommended.

```bash
cd /home/naz/ada-pi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

## Configure

Keep the API key in the backend shell environment. It is never sent to the browser. You can put it in a `.env` file, which is gitignored:

```bash
GEMINI_API_KEY=your-google-ai-studio-key
```

Optional variables:

```bash
GEMINI_LIVE_MODEL=gemini-3.1-flash-live-preview
GEMINI_LIVE_VOICE=Kore
GEMINI_LIVE_INSTRUCTIONS="You are a concise, friendly voice assistant."
```

Create Gemini API keys in [Google AI Studio](https://aistudio.google.com/app/apikey). `backend/realtime_provider.py` contains the Gemini Live integration.

## Run

```bash
cd /home/naz/ada-pi
source .venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --env-file .env
```

On the Pi, open `http://localhost:8000` in Chromium, click **Connect**, and allow microphone access. The backend serves the frontend; do not open `index.html` as a `file://` URL.

Chromium only permits microphone capture in a secure context. `localhost` is treated as secure. If Chromium runs on another machine and connects to the Pi by LAN IP, use HTTPS with a trusted certificate or an appropriate secure local reverse proxy. Do not bypass this restriction for deployment.

## Verify the prototype

### Continuous microphone and echo cancellation

1. Connect and confirm the microphone status says `On` and Chromium's site indicator shows active microphone use.
2. Open DevTools Console. The `microphone started` object reports the applied track settings, including `echoCancellation` where Chromium exposes it.
3. Ask a question and let the answer play through speakers. Do not use headphones for this test.
4. Stay silent. A good AEC setup should not transcribe or respond to its own voice. Try moderate speaker volume first, keep the microphone separated from and facing away from the speaker, then increase volume gradually.
5. Backend audio receipt is summarized every five seconds rather than logged per packet.

AEC quality depends on the Pi audio device, speaker/mic placement, room reflections, Chromium build, and volume. Wired or USB audio usually gives more predictable timing than Bluetooth. The page requests echo cancellation, noise suppression, and automatic gain control, but Chromium and the audio driver decide the settings actually applied.

### Interruption / barge-in

1. Ask for an answer long enough to keep the assistant speaking.
2. While it is speaking, say “Wait, stop.” at a normal voice level.
3. Expected result: backend logs `assistant interrupted`, assistant sound stops immediately, your interruption appears in the transcript, and a new response begins naturally.
4. Repeat at several speaker volumes and distances. Check that `assistant interrupted` appears in backend logs and `assistant interrupted; playback queue cleared` appears in DevTools.

If the assistant triggers on its own speaker output, reduce speaker volume or improve physical placement. The app explicitly enables `START_OF_ACTIVITY_INTERRUPTS` and high-sensitivity automatic VAD in `backend/realtime_provider.py`.

## Logging

The backend logs browser connection, summarized incoming audio, Gemini connection, assistant interruption/completion, errors, and session close. The browser logs microphone settings, playback interruption, WebSocket errors, and session close.

## Audio path

```text
Chromium getUserMedia (AEC/NS/AGC) -> PCM16/16 kHz -> FastAPI WebSocket
-> Gemini 3.1 Flash Live API -> PCM16/24 kHz chunks -> browser AudioWorklet -> speakers
```

No recordings or WAV files are created. The microphone remains active while assistant audio plays.
