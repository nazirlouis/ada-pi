# ADA Pi

ADA Pi turns a Raspberry Pi 5 into Ada: a voice-first desk assistant with a camera, an animated face, natural interruption, and optional habit coaching, Hailo-8 acceleration, Home Assistant controls, and Pironman integration.

Ada can listen while she speaks, answer questions about what she sees, track selected desk habits, and run as a full-screen Chromium kiosk. Audio and camera frames are processed live and are not saved as recordings.

> [!IMPORTANT]
> ADA Pi uses the Gemini Live API, so it requires an internet connection and a Gemini API key. It is not a fully local AI assistant. API usage may be billed by Google.

## What you can build

- Full-duplex voice conversations with Gemini Live
- Natural barge-in: interrupt Ada while she is speaking
- Live Pi camera context and visual questions
- Eleven animated facial expressions with audio-reactive mouth movement
- Posture, sitting-time, phone-use, clutter, hydration, late-work, junk-food, and office-light habit tracking
- Hailo-8-accelerated pose and object detection, with CPU fallback
- Optional Home Assistant device controls and office-light monitoring
- Optional Pironman 5 telemetry, lighting, OLED, and cooling controls

## Quick start

These instructions assume a clean 64-bit Raspberry Pi OS installation on a Raspberry Pi 5.

### 1. Install system packages

```bash
sudo apt update
sudo apt install -y python3-venv chromium curl git rpicam-apps
```

Reboot if the system packages or camera stack were updated:

```bash
sudo reboot
```

### 2. Check the camera

```bash
rpicam-hello --list-cameras
```

If no camera is listed, shut down the Pi and check the camera ribbon cable before continuing.

### 3. Clone and install ADA Pi

```bash
cd "$HOME"
git clone https://github.com/nazirlouis/ada-pi.git
cd ada-pi

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
cp .env.example .env
```

### 4. Add your Gemini API key

Create a key in [Google AI Studio](https://aistudio.google.com/app/apikey), then open the configuration file:

```bash
nano .env
```

At minimum, replace the Gemini key and disable Home Assistant unless you have already configured it:

```dotenv
GEMINI_API_KEY=your-google-ai-studio-key
ADA_START_HOME_ASSISTANT=false
```

Save with `Ctrl+O`, press `Enter`, then exit with `Ctrl+X`.

### 5. Start Ada

```bash
./start.sh
```

The backend starts first, then Chromium opens automatically in kiosk mode. Ada connects to Gemini and begins listening without requiring a screen tap. Use the gear icon to open settings, **Disconnect** to release the microphone, and **Exit** to close Chromium and the backend cleanly.

If `start.sh` is not executable after copying the project, run:

```bash
chmod +x start.sh scripts/*.sh
```

## Hardware

### Required

- Raspberry Pi 5 running 64-bit Raspberry Pi OS
- Python 3.11 or newer
- Pi-compatible camera
- Microphone and audio output
- A display; a touchscreen is recommended for kiosk use
- Internet connection and Gemini API key

### Optional

- SunFounder Pironman 5 or Pironman 5 Pro Max
- Hailo-8 accelerator; the bundled TFLite models provide CPU fallback
- Home Assistant for device controls and office-light monitoring

### Important audio note

The frontend is tuned for a PipeWire echo-cancel source and sink named `ada_aec_source` and `ada_aec_sink`. It intentionally leaves Chromium's own echo cancellation disabled to avoid applying AEC twice.

Check your active audio devices with:

```bash
wpctl status
```

If those PipeWire devices are not configured, the microphone will still work, but Ada may hear her own speaker output and interrupt herself. Headphones are the easiest test setup; a hands-free speaker build needs system-level echo cancellation plus sensible microphone/speaker placement.

## Using Ada

The settings drawer provides access to:

- **Ada prompt** — change her personality or instructions and reconnect the Live session
- **Pose** — inspect live pose landmarks, model, inference time, and frame rate
- **Detect** — inspect object detections and the active backend
- **Habit settings** — enable monitors, change timing, and run calibrations
- **Habit tracker** — view possible, emerging, and established patterns
- **Home Assistant** — control allow-listed devices when configured
- **Hardware** — view Pironman telemetry and reversible controls
- **Disconnect** — end the voice session and release the microphone
- **Exit** — stop the entire application cleanly

You can also ask Ada, “What habits are you tracking?” or “How are my habits doing?” She reads the current tracker rather than guessing from conversational memory.

## Habit coaching

Ada records one occurrence per prolonged episode, so repeated polling cannot inflate the tracker:

- **Possible** — first confirmed occurrence
- **Emerging** — three occurrences
- **Established** — ten occurrences across at least three days within seven days

| Monitor | How it works | Setup |
| --- | --- | --- |
| Posture | Local pose estimation detects sustained slouching; one current frame may be sent to Gemini for confirmation. | Calibrate good posture for 30 seconds, then calibrate your typical slouch. |
| Sitting too long | Tracks continuous desk presence, including standing at the desk. The default reminder is 60 minutes and resets after five minutes away. | Adjust timing under **Habit settings**. |
| Phone distraction | Requires a phone to remain near a visible wrist, hand region, or face for most of a rolling two-minute window. | Enable or disable under **Habit settings**. |
| Desk clutter | Compares the current desk with a clean numerical baseline, then requests one Gemini confirmation after a persistent change. | Calibrate a clean, still desk with nobody in frame. |
| Working too late | Combines desk presence with the configured timezone. Defaults to 10:00 PM–6:00 AM. | Set `ADA_TIMEZONE` and adjust the window in the UI. |
| Hydration | After an accumulated hour at the desk, asks you to visibly drink and observes for 15 seconds. | Timing is configurable in the UI. |
| Junk food | Uses local hand-to-mouth cues, then records only a high-confidence Gemini confirmation of a specific unhealthy item being consumed. | Enable or disable under **Habit settings**. |
| Office lights | Combines Home Assistant person/light state, local presence, a grace period, and a second check. | Requires the optional Home Assistant configuration below. |

Brief movement, unclear landmarks, failed requests, and inconclusive Gemini results do not create habit events. This project is a behavioral prototype, not a medical device or diagnostic tool.

## Configuration

The main settings live in `.env`:

```dotenv
GEMINI_API_KEY=your-google-ai-studio-key
GEMINI_LIVE_MODEL=gemini-3.1-flash-live-preview
GEMINI_LIVE_VOICE=Kore
GEMINI_LIVE_INSTRUCTIONS="You are a concise, friendly voice assistant."
GEMINI_POSTURE_MODEL=gemini-3.5-flash-lite
GEMINI_VIDEO_RESOLUTION=high

ADA_VISION_BACKEND=auto
ADA_TIMEZONE=America/New_York
ADA_LOG_LEVEL=INFO
ADA_START_HOME_ASSISTANT=false
```

Preview model names can change. If Google retires a configured model, update the corresponding value in `.env` to a currently supported Live or Flash model.

### Camera sharing modes

The default `activity` mode sends camera frames to Gemini around speech while local vision continues using the shared camera stream:

```bash
ADA_VIDEO_MODE=activity ./start.sh
```

For continuous Gemini scene awareness:

```bash
ADA_VIDEO_MODE=continuous ./start.sh
```

Continuous mode uses more API bandwidth and sends camera frames to Gemini even when you are not speaking.

### Custom host and port

```bash
ADA_HOST=127.0.0.1 ADA_PORT=8080 ./start.sh
```

Keep Chromium on the Pi at `localhost`. Browser microphone capture requires a secure context, and Chromium treats `localhost` as secure. Do not open `frontend/index.html` directly with a `file://` URL.

## Optional integrations

### Hailo-8 acceleration

Without Hailo, Ada automatically uses MoveNet and EfficientDet Lite on the CPU. To enable Hailo on Raspberry Pi OS:

```bash
sudo apt install -y hailo-all
sudo reboot
```

Verify the device:

```bash
ls -l /dev/hailo0
hailortcli fw-control identify
```

ADA Pi is matched to HailoRT 4.23, TAPPAS 5.1, and Model Zoo v2.17 HEFs. Install the checksum-verified models:

```bash
cd "$HOME/ada-pi"
./scripts/install_hailo_models.sh
ls -lh data/models/hailo8
```

`ADA_VISION_BACKEND=auto` selects Hailo when available. If initialization or inference fails, Ada uses CPU vision for the rest of that process; restart after repairing the Hailo setup.

### Pironman 5

Install SunFounder's Pironman software:

```bash
curl -sSL "https://raw.githubusercontent.com/sunfounder/pironman5/v1/install.sh" \
  -o /tmp/install-pironman5.sh
sudo bash /tmp/install-pironman5.sh
sudo reboot
```

Verify the dashboard service and local API:

```bash
systemctl status pironman5.service --no-pager
curl http://127.0.0.1:34001/api/v1.0/test
```

For a Pro Max touchscreen, set **Preferences → Control Centre → Screen → DSI-2 → Touchscreen → Mode → Multitouch**.

#### Pro Max maximum-fan service

Only run this on a Pironman 5 Pro Max with the active-low `FAN_PWM` pin:

```bash
cd "$HOME/ada-pi"
sudo ./scripts/install_max_fan_service.sh
```

Verify it with:

```bash
systemctl status ada-fan-max.service --no-pager
sudo pinctrl FAN_PWM
```

The pin should report `op dl`, meaning output driven low. Do not install this service on unrelated hardware.

### Home Assistant

If Home Assistant is already managed separately, keep `ADA_START_HOME_ASSISTANT=false` and provide its URL and token. If you want `start.sh` to start a local Docker Compose project, set it to `true` and provide the real Compose path for your Pi user.

```dotenv
ADA_START_HOME_ASSISTANT=false
ADA_HOME_ASSISTANT_COMPOSE_FILE=/home/your-user/homeassistant/compose.yaml
HOME_ASSISTANT_URL=http://127.0.0.1:8123
HOME_ASSISTANT_TOKEN=replace_with_your_long_lived_access_token
HOME_ASSISTANT_PERSON=person.your_name
HOME_ASSISTANT_OFFICE_LIGHTS=light.office_left,light.office_right
HOME_ASSISTANT_POLL_SECONDS=15
OFFICE_EMPTY_GRACE_SECONDS=300
```

Create a long-lived access token from your Home Assistant profile. The token stays in ADA Pi's backend and is never sent to the browser. Never commit `.env`.

## Privacy and data flow

- Raw microphone audio and camera images are not recorded to disk.
- Habit events, settings, and numerical calibrations are stored locally in `data/habits.db`.
- Posture and desk calibrations store numerical values, not calibration photos.
- Gemini receives live audio, activity-gated or continuous video depending on your mode, and individual confirmation frames used by enabled monitors.
- Home Assistant and Gemini credentials remain in the backend `.env` file.
- Raw audio is excluded from application logs.
- Logs are written to `logs/ada-pi.log` and rotate at 10 MB by default.

## How it works

```text
Microphone -> optional PipeWire AEC -> Chromium noise suppression -> PCM16 WebSocket -> Gemini Live
Gemini audio -> WebSocket -> AudioWorklet -> speakers + animated mouth

Pi camera -> one latest-frame MJPEG hub
          -> activity-gated or continuous Gemini vision
          -> Hailo-8 YOLO pose, or MoveNet CPU fallback
          -> Hailo-8 YOLO detection, or EfficientDet CPU fallback
          -> local desk descriptors

SQLite -> events + lifecycle + settings + calibrations + notification outbox
```

The camera hub processes only the newest 640×480 frame, so stale frames cannot build up across conversation, pose estimation, object detection, and habit monitoring.

## Verify the installation

```bash
cd "$HOME/ada-pi"
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip check
curl http://127.0.0.1:8000/api/vision/status
```

Then test a normal voice turn, interrupt Ada while she is speaking, open **Pose** and **Detect**, and calibrate posture and a clean desk. Watch the backend log with:

```bash
tail -F logs/ada-pi.log
```

## Troubleshooting

### Startup stops at Home Assistant

If you do not use Home Assistant, set this in `.env`:

```dotenv
ADA_START_HOME_ASSISTANT=false
```

If you do use it, verify `ADA_HOME_ASSISTANT_COMPOSE_FILE`, Docker access, `HOME_ASSISTANT_URL`, and the long-lived token.

### No camera frames

```bash
rpicam-hello --list-cameras
rpicam-vid --nopreview --timeout 3sec --codec mjpeg \
  --width 640 --height 480 --framerate 1 --output /dev/null
```

If no camera appears, power down the Pi and reseat the ribbon cable. Check that no other process is using the camera.

### Ada interrupts herself

```bash
wpctl status
```

Confirm the expected `ada_aec_sink` and `ada_aec_source` are present and selected. If they are not configured, test with headphones. With speakers, lower the volume, separate the microphone from the speakers, and reduce room reflections.

### Hailo is using CPU fallback

```bash
ls -l /dev/hailo0
hailortcli fw-control identify
ls -lh data/models/hailo8
curl http://127.0.0.1:8000/api/vision/status
```

Hailo mode requires the device, HEF files, and TAPPAS post-processors. Restart Ada after fixing the setup.

### Pironman hardware is offline

```bash
systemctl status pironman5.service --no-pager
curl http://127.0.0.1:34001/api/v1.0/test
```

Set the RGB effect to **Solid** for the fastest expression-color changes.

### Gemini connection errors

Confirm that `.env` contains a valid `GEMINI_API_KEY`, that the configured models are available to your account, and that the Pi has internet access. Restart `./start.sh`, then inspect the settings panel and `logs/ada-pi.log`.

### Chromium does not open

```bash
command -v chromium || command -v chromium-browser
```

Install Chromium if neither command returns a path. If you are working over SSH without a graphical desktop session, start ADA Pi from the Pi's desktop terminal or run only the backend manually for debugging.

## Project layout

```text
backend/       FastAPI, Gemini Live, camera, vision, habits, and integrations
frontend/      Kiosk interface, animated face, browser audio, and controls
scripts/       Optional Hailo, Pironman, and system setup helpers
tests/         Automated unit and contract tests
data/          Local runtime database and downloaded models (gitignored)
logs/          Rotating runtime logs (gitignored)
start.sh       Backend and Chromium kiosk launcher
```

When reporting a problem, include your Raspberry Pi OS version, Python version, relevant hardware, the output of `/api/vision/status`, and the smallest relevant section of `logs/ada-pi.log`. Remove API keys, Home Assistant tokens, and personal entity names before sharing logs or configuration.

## License

ADA Pi is open source under the [MIT License](LICENSE).
