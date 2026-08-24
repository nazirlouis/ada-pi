# ADA Pi

ADA is a voice-first desk assistant for Raspberry Pi 5. She holds natural,
full-duplex conversations, sees through a fixed Pi camera, displays an animated
face, and helps you notice recurring desk habits without recording your day.

## What Ada can do

- Talk and listen at the same time, with immediate voice interruption.
- Use the camera to answer questions about objects and the current scene.
- Show eleven expressive faces and synchronize compatible Pironman case lighting.
- Track posture, long desk sessions, late-night work, hydration, distracting phone use, junk food, and desk clutter.
- Notice office lights left on using Home Assistant and local presence detection.
- Show live MoveNet pose and EfficientDet object-detection previews.
- Display Pironman telemetry and safe hardware controls.
- Maintain long conversations with Gemini Live resumption and context compression.

Ada shares one camera stream across conversation, pose, object detection, and
habit monitoring. Local inference always processes the newest frame, preventing
stale video from accumulating behind audio or UI work.

## Using Ada

```bash
cd /home/naz/ada-pi
./start.sh
```

Speak naturally once Ada indicates that she is connected. You can interrupt her
while she is speaking; queued playback is discarded immediately.

The kiosk controls provide:

- **Settings** — edit Ada's prompt and inspect the Gemini connection.
- **Habit settings** — calibrate and configure all habit monitors.
- **Habit tracker** — view possible, emerging, and established patterns.
- **Home Assistant** — view and locally control allow-listed devices.
- **Pose** — inspect shared MoveNet landmarks and confidence.
- **Detect** — inspect shared EfficientDet boxes and confidence.
- **Hardware** — view Pironman telemetry and safe controls.
- **Disconnect** — disconnect the microphone while background alerts remain active.
- **Exit** — stop the browser, backend, camera, and Gemini session.

You can also ask Ada “What habits are you tracking?” or “How are my habits doing?”
She uses a read-only Live tool to retrieve the current complete tracker instead
of relying on conversational memory.

## Habit coaching

Ada records one occurrence per prolonged episode. Repeated polling cannot inflate
progress while a condition remains active. A habit is:

- **Possible** after its first confirmed occurrence.
- **Emerging** after three occurrences.
- **Established** after ten occurrences across at least three days in seven days.

Each confirmed visual habit creates a durable on-screen alert and one concise
spoken response with a practical correction. Alerts remain until acknowledged.
Settings, monitor state, and numerical calibrations survive restarts. Clearing
history removes events and progress while preserving settings and calibrations.

### Posture

Open **Habit settings**, choose **Calibrate good posture**, and sit tall and still
for 30 seconds. Keep both shoulders and at least one ear visible. Then choose
**Calibrate slouch posture** and hold the slouch Ada should recognize.

MoveNet evaluates posture locally. A sustained high score sends one current frame
to Gemini for structured confirmation. Brief movement, unclear landmarks, and
rejected confirmations do not create events. Images are never saved.

### Sitting too long

Ada treats continuous desk presence as sitting, including standing at the desk.
By default she alerts after 60 minutes, tolerates visibility gaps up to 15
seconds, and resets after five continuous minutes away. Timing and enablement are
available under **Habit settings → Sitting Too Long**.

### Phone distraction

A phone must remain near a visible wrist, hand region, or face for most of a
rolling two-minute window. A phone resting on the desk and brief checks are
ignored. Two continuous minutes without active-use evidence resets the episode.

### Desk clutter

Arrange a clean desk, choose **Calibrate clean desk**, leave the camera frame,
and keep the scene still. Ada stores normalized, downscaled numerical descriptors,
not calibration photos.

When nobody has been visible for 30 seconds, Ada compares the scene with the
baseline. A change must persist before Gemini receives one current frame to
confirm clutter with at least 65% confidence. Speech waits until you return; the
visual alert remains available immediately. Cleanup or recalibration resets it.

### Working too late

Ada uses desk presence and the configured local timezone to recognize work after
the nightly cutoff, which defaults to 10:00 PM. She records at most one occurrence
between that cutoff and the 6:00 AM reset, including when the session crosses
midnight. Both times are configurable under **Habit settings**. Select **Run
check now** to return to Ada's face and evaluate the current cutoff and presence
without creating a false occurrence outside the late window.

### Not drinking enough water

After each accumulated hour of desk presence, Ada asks you to visibly drink
water. Once she finishes the prompt, the newest camera frames are sent to Gemini
Live for a 15-second observation. A confirmed drink resets the timer without an
event; a high-confidence confirmed miss or ignored request records one
occurrence. Camera, Gemini, or inconclusive-verdict failures never count against
you and retry later. During the check, Ada's main screen shows the spoken-prompt
state, a server-synchronized 15-to-0 countdown, and the Gemini review state.
Select **Run check now** under Habit Settings to start the same flow immediately.

### Junk food

Repeated local hand-to-mouth cues only trigger a 15-second Gemini Live review;
the gesture itself is never treated as junk-food evidence. Ada records only when
Gemini identifies a specific unhealthy item and confirms high-confidence visible
consumption of it,
including sugary drinks, candy, chips, pastries, desserts, fast food, and heavily
processed snacks or meals. Face touching, scratching, empty-hand gestures,
ambiguous chewing, mere food presence, and ambiguous meals do not count.
The episode resets after 30 minutes without eating evidence.

### Office lights left on

With Home Assistant configured, Ada checks that a tracked light is on while the
configured person is away, waits through a grace period, and checks again before
recording. Local presence prevents false alerts while you remain in the office.
Unknown or stale presence fails safely. The episode resets when all lights turn
off or at the configured next-day reset time.

## Privacy

- Camera images and audio are never recorded to disk.
- Posture and desk calibrations contain numerical values only.
- Gemini receives conversational frames and individual confirmation frames as needed.
- Gemini and Home Assistant credentials stay in the backend.
- Habit data is stored locally in `data/habits.db`.
- Raw audio is not included in logs.

The default vision mode forwards frames to Gemini only around speech. Continuous
scene awareness is optional:

```bash
ADA_VIDEO_MODE=continuous ./start.sh
```

## Requirements

- Raspberry Pi 5 with 64-bit Raspberry Pi OS.
- Python 3.11 or newer.
- Chromium or Chromium Browser.
- A microphone, speaker, and Pi camera.
- A Gemini API key with access to the configured Live model.

Check the camera first:

```bash
rpicam-hello --list-cameras
```

## Installation

```bash
cd /home/naz/ada-pi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
cp .env.example .env
```

Add your [Google AI Studio](https://aistudio.google.com/app/apikey) key to `.env`:

```bash
GEMINI_API_KEY=your-google-ai-studio-key
```

Useful optional settings:

```bash
GEMINI_LIVE_MODEL=gemini-3.1-flash-live-preview
GEMINI_LIVE_VOICE=Kore
GEMINI_POSTURE_MODEL=gemini-3.5-flash-lite
GEMINI_VIDEO_RESOLUTION=high
PIRONMAN_URL=http://127.0.0.1:34001
ADA_LOG_LEVEL=INFO
ADA_TIMEZONE=America/New_York
```

The bundled MoveNet and EfficientDet Lite INT8 models use LiteRT's ARM64 XNNPACK
CPU delegate; no Hailo hardware is required.

Launch options can be supplied to the script:

```bash
ADA_HOST=127.0.0.1 ADA_PORT=8080 ADA_VIDEO_MODE=activity ./start.sh
```

Do not open `frontend/index.html` as a `file://` URL. Microphone capture requires
the secure `localhost` context provided by the backend.

## Home Assistant setup

Create a long-lived token and configure `.env`:

```bash
ADA_START_HOME_ASSISTANT=true
ADA_HOME_ASSISTANT_COMPOSE_FILE=/home/naz/homeassistant/compose.yaml
HOME_ASSISTANT_URL=http://127.0.0.1:8123
HOME_ASSISTANT_TOKEN=replace_with_your_token
HOME_ASSISTANT_PERSON=person.naz
HOME_ASSISTANT_OFFICE_LIGHTS=light.left_office_light,light.right_office_light
HOME_ASSISTANT_POLL_SECONDS=15
OFFICE_EMPTY_GRACE_SECONDS=300
```

When enabled, `start.sh` starts the configured Compose project and waits for Home
Assistant before launching Ada. It leaves Home Assistant running on exit. Set
`ADA_START_HOME_ASSISTANT=false` when it is managed separately.

## Pironman integration

Ada uses the running SunFounder dashboard API instead of accessing GPIO in
parallel. Keep `pironman5.service` active. The default API is
`http://127.0.0.1:34001`; override it with `PIRONMAN_URL`.

The Hardware drawer exposes available telemetry and reversible controls. Ada
does not proxy shutdown, reboot, service restart, hardware-pin changes, battery
thresholds, credentials, or unrelated configuration. An OLED guard keeps the
display enabled with sleep disabled while Ada runs.

## How it works

```text
Microphone -> Chromium AEC/NS -> PCM16 WebSocket -> Gemini Live
Gemini audio -> WebSocket -> AudioWorklet -> speakers + animated mouth

Pi camera -> one latest-frame MJPEG hub
          -> activity-gated Gemini vision
          -> MoveNet pose and presence
          -> EfficientDet phone/object detection
          -> local desk descriptors

SQLite -> events + lifecycle + settings + calibrations + notification outbox
```

The camera runs at 640×480 and 5 FPS. Local inference runs in worker threads.
The Detect preview consumes existing monitor results instead of starting a
competing loop. Gemini Live reconnects automatically when sessions expire.

## Verification

```bash
cd /home/naz/ada-pi
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip check
```

For a device smoke test:

1. Confirm speech, interruption, and expression changes.
2. Open **Pose** and verify camera framing and stable landmarks.
3. Open **Detect** and verify current boxes without delayed frames.
4. Calibrate posture and a clean desk.
5. Confirm audio remains smooth while both local models run.
6. Watch CPU load and alert timing during a representative session.

Logs are written to `logs/ada-pi.log`:

```bash
tail -F logs/ada-pi.log
```

## Troubleshooting

### No camera frames

```bash
rpicam-hello --list-cameras
rpicam-vid --nopreview --timeout 3sec --codec mjpeg \
  --width 640 --height 480 --framerate 1 --output /dev/null
```

If no camera appears, power down the Pi and check the ribbon cable. Audio can
continue without vision.

### Ada interrupts herself

Run `wpctl status` and confirm `ada_aec_sink` and `ada_aec_source` are selected.
Lower speaker volume or improve microphone/speaker separation if echo remains.

### Pironman drawer is offline

```bash
systemctl status pironman5.service
curl http://127.0.0.1:34001/api/v1.0/test
```

Set the RGB effect to **Solid** for the fastest expression-color changes.

### Gemini authentication or connection errors

Confirm `.env` has a valid `GEMINI_API_KEY`, restart `./start.sh`, and inspect
the connection under **Settings** or in `logs/ada-pi.log`.
