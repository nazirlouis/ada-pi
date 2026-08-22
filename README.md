# ADA Pi

ADA Pi is a full-duplex, multimodal AI assistant for Raspberry Pi 5. It combines
Gemini 3.1 Flash Live voice conversation, native Pi-camera vision, and a
lightweight animated SVG face designed for a 4.3-inch landscape touchscreen.

## Features

- Continuous microphone input while Ada speaks, with Chromium AEC/NS/AGC.
- Immediate voice interruption and playback cancellation (barge-in).
- Native 24 kHz Gemini audio playback with a small anti-jitter buffer.
- Raspberry Pi camera input through `rpicam-vid` at up to one JPEG frame/second.
- Activity-gated vision by default, with an optional continuous vision mode.
- Eleven model-controlled expressions with gaze, blinking, eyebrow movement,
  neon glow, and audio-amplitude-driven mouth shapes.
- Full-screen Chromium kiosk startup and coordinated browser/backend shutdown.
- Sliding-window context compression for extended conversations.
- Native Pironman 5 hardware drawer with live telemetry and validated RGB/OLED controls.
- Expression-synchronized case lighting, including red for Mad and orange for Alert.
- Always-on OLED guard that disables display sleep while Ada is running.

## Requirements

- Raspberry Pi 5 running 64-bit Raspberry Pi OS.
- Python 3.11 or newer.
- Chromium.
- A working microphone and speaker.
- A Raspberry Pi camera recognized by `rpicam-vid` for vision features.
- A Gemini API key with access to `gemini-3.1-flash-live-preview`.

Verify the camera before setup:

```bash
rpicam-hello --list-cameras
```

At least one camera must appear under `Available cameras`.

## Install

```bash
cd /home/naz/ada-pi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

The launch script also expects `rpicam-vid` and either `chromium` or
`chromium-browser` to be available on `PATH`.

## Configure

Create `/home/naz/ada-pi/.env`:

```bash
GEMINI_API_KEY=your-google-ai-studio-key
```

Optional `.env` settings:

```bash
GEMINI_LIVE_MODEL=gemini-3.1-flash-live-preview
GEMINI_LIVE_VOICE=Kore
GEMINI_LIVE_INSTRUCTIONS="You are a concise, friendly voice assistant."
GEMINI_VIDEO_RESOLUTION=high
PIRONMAN_URL=http://127.0.0.1:34001
```

The API key stays in the backend and is never sent to Chromium. Create a key in
[Google AI Studio](https://aistudio.google.com/app/apikey).

## Run

```bash
cd /home/naz/ada-pi
./start.sh
```

The script:

1. Validates `.venv`, `.env`, and Chromium.
2. Starts FastAPI/Uvicorn on port 8000.
3. Waits for the backend to become ready.
4. Opens ADA in Chromium kiosk mode.
5. Stops the backend when Chromium closes, and vice versa.

Tap Ada's face to connect and grant microphone permission on the first launch.

- **Disconnect** ends the Gemini session while leaving the kiosk open. Tap the
  face to reconnect.
- **Exit** closes the Gemini session, backend, and kiosk browser.
- The temporary buttons in the top-left manually test each facial expression.
  Gemini can still change the active expression through its function tool.

The host and port are shell launch settings:

```bash
ADA_HOST=127.0.0.1 ADA_PORT=8080 ./start.sh
```

Do not open `frontend/index.html` as a `file://` URL. Chromium microphone capture
requires a secure context; `localhost` qualifies.

## Pironman 5 integration

Ada uses the locally running SunFounder dashboard API instead of accessing the
GPIO hardware in parallel with Pironman's service. The default API address is
`http://127.0.0.1:34001`; override it with `PIRONMAN_URL` if necessary. Keep the
Pironman background service running even if its standalone browser autostart is
disabled.

Tap **Hardware** in Ada's lower-right controls to open the native hardware
drawer. While it is open, Ada refreshes telemetry every three seconds and
automatically organizes every value reported by the installed Pironman variant:

- CPU, GPU, memory, load, uptime, and temperatures.
- Disks, filesystems, mounted storage, and usage.
- Fan state, speed, and thermal data.
- Battery percentage, voltage, current, charging state, and power source when a
  supported UPS or PiPower device reports them.
- Input/output voltage, current, and power.
- Network, IP, and MAC information.
- Any additional scalar telemetry introduced by future dashboard versions.

Available controls are detected from the installed configuration. Depending on
the Pironman variant, the drawer can expose:

- RGB enable, color, brightness, effect, and effect speed.
- OLED enable, rotation, and sleep timeout.
- Celsius/Fahrenheit selection.
- Fan profile and fan-LED behavior.
- A link to the complete SunFounder dashboard for advanced administration.

Ada's backend validates each control and calls the dashboard's versioned
`/api/v1.0/` endpoints. It never exposes or proxies shutdown, reboot, service
restart, power-failure simulation, hardware-pin changes, or battery shutdown
thresholds. SMTP credentials and unrelated configuration also remain inside the
Pironman service and are not sent to Chromium.

### Always-on OLED

While Ada is running, a background guard checks the OLED once per minute and
enforces `oled_enable=true` with `oled_sleep_timeout=0`. It writes only when a
setting needs correction and does not change any Pi power behavior.

### Disable the standalone Pironman browser

Ada does not require Pironman's separate dashboard window. Disable only its
desktop autostart entry while retaining `pironman5.service`. A disabled entry at
`~/.config/autostart/pironman5-dashboard.desktop` should contain:

```ini
Hidden=true
X-GNOME-Autostart-enabled=false
```

## Vision modes

Activity-gated vision is the default:

```bash
./start.sh
```

Chromium detects local speech from its already echo-cancelled microphone samples.
That signal only controls video forwarding; it does not gate or modify audio sent
to Gemini. When speech begins, the backend immediately sends its most recent
camera frame, continues at up to one frame/second, and keeps a 1.5-second
post-speech grace window so the final view remains associated with the question.

For continuous scene awareness:

```bash
ADA_VIDEO_MODE=continuous ./start.sh
```

Continuous mode sends one frame/second throughout the session and consumes more
Gemini vision tokens. To lower visual processing cost further:

```bash
GEMINI_VIDEO_RESOLUTION=low ADA_VIDEO_MODE=activity ./start.sh
```

`ADA_VIDEO_MODE`, `ADA_HOST`, and `ADA_PORT` are launch-time shell variables.
Gemini settings are loaded from `.env` by Uvicorn.

## Facial expressions

Ada calls `set_facial_expression` before each spoken reply. Supported values are:

```text
neutral, sassy, amused, skeptical, annoyed, mad, concerned,
surprised, mischievous, serious, alert
```

The backend validates the function call and forwards a small expression event to
the browser. Expression geometry remains independent of gaze, blinking, glow,
and speech amplitude, so every face can continue talking naturally. The manual
test buttons use the same expression path.

Each expression changes Ada's eyes, mouth, eyebrows, aura, shimmer, and Pironman
case LEDs to the same palette:

| Expression | Color | Hex |
| --- | --- | --- |
| Neutral | Cyan | `#17dfff` |
| Sassy | Pink | `#ff3dbe` |
| Amused | Green | `#35ff9a` |
| Skeptical | Violet blue | `#7d8cff` |
| Annoyed | Orange red | `#ff6b35` |
| Mad | Red | `#ff2400` |
| Concerned | Blue | `#4a8fff` |
| Surprised | Yellow | `#ffd43b` |
| Mischievous | Purple | `#b45cff` |
| Serious | Pale cyan | `#d9f7ff` |
| Alert | Safety orange | `#ff8618` |

Case synchronization sends one latency-sensitive color request per expression.
For the quickest visible change, configure the Pironman RGB effect as **Solid**.
Lighting failures are logged without delaying Ada's animation, audio, or Gemini
session.

## Data flow

```text
Microphone
  Chromium getUserMedia (AEC/NS/AGC)
  -> PCM16 at 16 kHz
  -> FastAPI WebSocket
  -> Gemini 3.1 Flash Live

Assistant audio
  Gemini PCM16 at 24 kHz
  -> FastAPI WebSocket
  -> AudioWorklet jitter buffer
  -> Web Audio analyser (mouth amplitude only)
  -> speakers

Vision
  rpicam-vid (MJPEG 640x480 at 1 FPS)
  -> validated JPEG frame
  -> Gemini Live video input

Pironman
  Pironman service on port 34001
  -> Ada backend validation and credential filtering
  -> live hardware drawer and safe controls
  -> expression color synchronization
```

The camera is captured natively by the backend, avoiding Chromium/PipeWire camera
issues on Raspberry Pi. Chromium owns only the microphone path. No audio or video
recordings are written to disk.

## Verify

Run the test suite:

```bash
cd /home/naz/ada-pi
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip check
```

### Verify vision

Start in the default activity mode, hold an object steady, and ask Ada what she
sees. The backend should report:

```text
Pi camera streaming (mode=activity, 640x480 MJPEG at 1 FPS)
video frames forwarded (1 total, latest=... bytes)
```

If `video frames forwarded` never appears, verify the camera independently:

```bash
rpicam-hello --list-cameras
rpicam-vid --nopreview --timeout 3sec --codec mjpeg \
  --width 640 --height 480 --framerate 1 --output /dev/null
```

### Verify interruption

1. Ask Ada for a long answer.
2. While she is speaking, say “Wait, stop.” at a normal volume.
3. Her playback and mouth movement should stop immediately.
4. The backend should log `assistant interrupted`.

If Ada interrupts herself while nobody is speaking, speaker output is leaking
through Chromium's echo cancellation. Reduce speaker volume, increase physical
separation between speaker and microphone, or improve their orientation. The
Live session uses low start-of-speech sensitivity and 200 ms speech confirmation
to reduce false barge-ins without disabling full duplex.

## Troubleshooting

### `uvicorn: command not found`

Use the included script, which invokes Uvicorn through the project virtual
environment:

```bash
./start.sh
```

### No camera frames

```bash
rpicam-hello --list-cameras
```

If it reports `No cameras available`, power down the Pi and check both ends and
the orientation of the camera ribbon cable. The backend continues audio-only if
native camera capture is unavailable.

### Chromium GCM warning

`DEPRECATED_ENDPOINT` messages from Chromium's GCM registration are unrelated to
Gemini, microphone capture, and camera streaming.

### Pironman drawer is offline

Confirm the background service and API are available:

```bash
systemctl status pironman5.service
curl http://127.0.0.1:34001/api/v1.0/test
```

The expected API response contains `"status":true`. If Pironman runs elsewhere,
set `PIRONMAN_URL` in `.env` and restart Ada.

### Case color does not follow the face

Set the Pironman RGB effect to **Solid**, confirm case lighting is enabled, and
tap a manual expression button. Ada's log should show:

```text
POST /api/pironman/expression HTTP/1.1" 200 OK
```

A `503` response means Ada reached its own expression endpoint but the upstream
Pironman API rejected or could not complete the color update.

### API authentication errors

Confirm `.env` contains a valid `GEMINI_API_KEY`, then restart `./start.sh` so a
new Gemini Live session is created.
