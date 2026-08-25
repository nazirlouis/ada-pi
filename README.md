# ADA Pi

ADA is a voice-first desk assistant for Raspberry Pi 5. She holds natural,
full-duplex conversations, sees through a Pi camera, displays an animated face,
and helps build healthier desk habits without recording your day.

## 1. Features and what Ada can do

### Natural voice conversation

- Talk and listen at the same time through Gemini Live.
- Interrupt Ada while she is speaking; queued audio stops immediately.
- Continue long conversations using session resumption and context compression.
- Use Chromium echo cancellation and noise suppression.
- Configure Ada's model, voice, instructions, and video quality.

### Camera and visual awareness

- Ask questions about visible objects or the current scene.
- Share one 640×480, 5 FPS camera stream across conversation, pose estimation,
  object detection, and habit monitoring.
- Process only the newest frame so stale video cannot accumulate.
- Use Hailo-8 acceleration when available, with automatic TFLite CPU fallback.
- Inspect live pose landmarks, object boxes, model names, and backend status.

The default `activity` mode sends video to Gemini only around speech. Continuous
scene awareness is optional:

```bash
ADA_VIDEO_MODE=continuous ./start.sh
```

### Habit coaching

Ada tracks posture, long desk sessions, late-night work, hydration, distracting
phone use, junk food, desk clutter, and office lights left on. She records one
occurrence per prolonged episode, so repeated polling cannot inflate progress.

- **Possible** — first confirmed occurrence.
- **Emerging** — three occurrences.
- **Established** — ten occurrences across at least three days in seven days.

Confirmed habits create durable on-screen alerts and concise spoken suggestions.
Settings and numerical calibrations survive restarts; clearing history preserves
those settings and calibrations.

#### Posture

Local pose estimation looks for sustained slouching. Under **Habit settings**,
calibrate good posture for 30 seconds, then calibrate the slouch Ada should
recognize. A suspected slouch sends one current frame to Gemini for confirmation.
Brief movement and unclear landmarks do not create events; images are never saved.

#### Sitting too long

Continuous desk presence counts as a session, including standing at the desk.
The default alert is after 60 minutes, with short visibility gaps tolerated and
a reset after five continuous minutes away. Timing is configurable.

#### Phone distraction

A phone must remain close to a visible wrist, hand region, or face for most of a
rolling two-minute window. A phone resting on the desk and brief checks are ignored.

#### Desk clutter

Calibrate a clean desk while the frame is still and nobody is visible. Ada stores
only normalized numerical descriptors. A persistent change is sent to Gemini
once for confirmation; calibration photos are not retained.

#### Working too late

Ada combines desk presence with the configured timezone. The default late window
begins at 10:00 PM and resets at 6:00 AM. Both times are configurable.

#### Hydration

After each accumulated hour at the desk, Ada asks you to visibly drink water and
observes for 15 seconds. A confirmed drink resets the timer. Camera, Gemini, and
inconclusive-verdict failures never count against you.

#### Junk food

Local hand-to-mouth cues can trigger a Gemini review. Ada records an event only
when a specific unhealthy item and visible consumption are both confirmed with
high confidence. Face touching, chewing alone, or visible food does not count.

#### Office lights left on

With Home Assistant configured, Ada combines person state, local presence, a
grace period, and a second check before reporting lights left on. Unknown or
stale presence fails safely.

### Animated kiosk and controls

- Eleven expressive animated faces with synchronized mouth movement.
- **Settings** for Ada's prompt and Gemini connection.
- **Habit settings** for calibration and monitor configuration.
- **Habit tracker** for possible, emerging, and established patterns.
- **Home Assistant** for allow-listed lights, fans, switches, and helpers.
- **Pose** and **Detect** diagnostic previews.
- **Hardware** for Pironman telemetry and safe reversible controls.
- **Disconnect** to release the microphone while alerts remain active.
- **Exit** to stop Chromium, the backend, camera, and Gemini cleanly.

You can ask “What habits are you tracking?” or “How are my habits doing?” Ada
reads the current tracker instead of relying on conversational memory.

### Pironman integration

With the Pironman 5 dashboard installed, Ada shows temperatures, load, storage,
fan configuration, OLED settings, and RGB controls. Expression colors can
synchronize with case lighting. Ada uses the dashboard API instead of competing
for GPIO access.

On Pironman 5 Pro Max, the included system service drives the active-low
`FAN_PWM` pin low immediately, keeps the fans at maximum, and starts at boot.
While Ada runs, a guard also selects always-on fan mode and keeps the OLED awake.

### Privacy and local data

- Camera images and raw audio are never recorded to disk.
- Posture and desk calibrations contain numerical values only.
- Gemini receives conversational and individual confirmation frames as needed.
- Gemini and Home Assistant credentials remain in the backend.
- Habit data is stored locally in `data/habits.db`.
- Raw audio is excluded from logs.

### How it works

```text
Microphone -> Chromium AEC/NS -> PCM16 WebSocket -> Gemini Live
Gemini audio -> WebSocket -> AudioWorklet -> speakers + animated mouth

Pi camera -> one latest-frame MJPEG hub
          -> activity-gated Gemini vision
          -> Hailo-8 YOLO pose, or MoveNet CPU fallback
          -> Hailo-8 YOLO detection, or EfficientDet CPU fallback
          -> local desk descriptors

SQLite -> events + lifecycle + settings + calibrations + notification outbox
```

## 2. Setup and installation

### Requirements

Required:

- Raspberry Pi 5 with 64-bit Raspberry Pi OS.
- Python 3.11 or newer and Chromium.
- Pi camera, microphone, and speakers.
- Internet access and a Gemini API key.

Optional:

- SunFounder Pironman 5 or Pironman 5 Pro Max.
- Hailo-8 accelerator; bundled CPU models work without one.
- Home Assistant for office-light monitoring and device controls.

Install the base packages:

```bash
sudo apt update
sudo apt full-upgrade -y
sudo apt install -y python3-venv chromium curl git
sudo reboot
```

After reboot, confirm the camera is detected:

```bash
rpicam-hello --list-cameras
```

### Install Pironman 5 (optional)

Skip this section without a Pironman case. Install SunFounder's software:

```bash
curl -sSL "https://raw.githubusercontent.com/sunfounder/pironman5/v1/install.sh" \
  -o /tmp/install-pironman5.sh
sudo bash /tmp/install-pironman5.sh
sudo reboot
```

Verify the dashboard service and API after reboot:

```bash
systemctl status pironman5.service --no-pager
curl http://127.0.0.1:34001/api/v1.0/test
```

For a Pro Max touchscreen, set **Preferences → Control Centre → Screen → DSI-2
→ Touchscreen → Mode → Multitouch** for normal touch gestures.

### Install Hailo-8 support (optional)

Skip this section without a Hailo device. Ada will use MoveNet and EfficientDet
Lite on the CPU. Install Raspberry Pi's Hailo runtime, firmware, GStreamer
integration, and TAPPAS post-processors:

```bash
sudo apt install -y hailo-all
sudo reboot
```

Verify the accelerator:

```bash
ls -l /dev/hailo0
hailortcli fw-control identify
```

ADA's implementation is matched to HailoRT 4.23 and TAPPAS 5.1. Install its
checksum-verified Model Zoo v2.17 HEFs:

```bash
cd /home/naz/ada-pi
./scripts/install_hailo_models.sh
ls -lh data/models/hailo8
```

`ADA_VISION_BACKEND=auto` selects Hailo and falls back to CPU for the remainder
of the process if Hailo initialization or inference fails.

### Install ADA Pi

Clone or copy the repository to `/home/naz/ada-pi`, then run:

```bash
cd /home/naz/ada-pi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
cp .env.example .env
```

Create a Gemini API key in [Google AI Studio](https://aistudio.google.com/app/apikey),
then replace the placeholder in `.env`:

```dotenv
GEMINI_API_KEY=your-google-ai-studio-key
```

Common optional settings:

```dotenv
GEMINI_LIVE_MODEL=gemini-3.1-flash-live-preview
GEMINI_LIVE_VOICE=Kore
GEMINI_POSTURE_MODEL=gemini-3.5-flash-lite
GEMINI_VIDEO_RESOLUTION=high
PIRONMAN_URL=http://127.0.0.1:34001
ADA_VISION_BACKEND=auto
ADA_LOG_LEVEL=INFO
ADA_TIMEZONE=America/New_York
```

### Install Pro Max maximum-fan startup (optional)

This one-time step sets Pironman 5 Pro Max fans to full speed immediately and
enables that policy at every boot:

```bash
cd /home/naz/ada-pi
sudo ./scripts/install_max_fan_service.sh
```

Verify the service and active-low PWM output:

```bash
systemctl status ada-fan-max.service --no-pager
sudo pinctrl FAN_PWM
```

The pin should include `op dl`, meaning output driven low. Skip this service on
systems without the Pironman Pro Max `FAN_PWM` pin.

### Configure Home Assistant (optional)

Create a long-lived access token and add these values to `.env`:

```dotenv
ADA_START_HOME_ASSISTANT=true
ADA_HOME_ASSISTANT_COMPOSE_FILE=/home/naz/homeassistant/compose.yaml
HOME_ASSISTANT_URL=http://127.0.0.1:8123
HOME_ASSISTANT_TOKEN=replace_with_your_token
HOME_ASSISTANT_PERSON=person.naz
HOME_ASSISTANT_OFFICE_LIGHTS=light.left_office_light,light.right_office_light
HOME_ASSISTANT_POLL_SECONDS=15
OFFICE_EMPTY_GRACE_SECONDS=300
```

`start.sh` starts the configured Compose project and waits for Home Assistant.
It leaves Home Assistant running when Ada exits. Set
`ADA_START_HOME_ASSISTANT=false` if Home Assistant is managed separately.

### Start Ada

```bash
cd /home/naz/ada-pi
./start.sh
```

Ada opens Chromium in kiosk mode after the backend is ready. Do not open
`frontend/index.html` as a `file://` URL; microphone capture requires the secure
`localhost` context served by the backend.

Optional launch overrides:

```bash
ADA_HOST=127.0.0.1 ADA_PORT=8080 ADA_VIDEO_MODE=activity ./start.sh
```

### Verify the installation

```bash
cd /home/naz/ada-pi
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip check
```

Then confirm speech and interruption, inspect **Pose** and **Detect**, check
**Hardware** if Pironman is installed, and calibrate posture and a clean desk.

```bash
curl http://127.0.0.1:8000/api/vision/status
tail -F logs/ada-pi.log
```

### Troubleshooting

#### No camera frames

```bash
rpicam-hello --list-cameras
rpicam-vid --nopreview --timeout 3sec --codec mjpeg \
  --width 640 --height 480 --framerate 1 --output /dev/null
```

If no camera appears, power down the Pi and check the ribbon cable. Audio can
continue without vision.

#### Hailo is using CPU fallback

```bash
ls -l /dev/hailo0
hailortcli fw-control identify
ls -lh data/models/hailo8
curl http://127.0.0.1:8000/api/vision/status
```

Hailo mode requires the device, HEF files, and TAPPAS post-processors. Restart
Ada after repair; fallback does not switch back during the same process.

#### Pironman hardware is offline

```bash
systemctl status pironman5.service --no-pager
curl http://127.0.0.1:34001/api/v1.0/test
```

Set the RGB effect to **Solid** for the fastest expression-color changes.

#### Fans do not start at maximum

```bash
systemctl status ada-fan-max.service --no-pager
sudo pinctrl FAN_PWM
```

The service should be active and the pin should report `op dl`. Reinstall it
with `sudo ./scripts/install_max_fan_service.sh` if missing.

#### Ada interrupts herself

Run `wpctl status` and confirm `ada_aec_sink` and `ada_aec_source` are selected.
Lower speaker volume or improve microphone/speaker separation if echo remains.

#### Gemini connection errors

Confirm `.env` contains a valid `GEMINI_API_KEY`, restart `./start.sh`, and
inspect **Settings** or `logs/ada-pi.log`.
