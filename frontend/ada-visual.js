(() => {
  "use strict";

  const STATES = new Set(["idle", "listening", "thinking", "speaking", "alert"]);
  const clamp = (value, low = 0, high = 1) => Math.max(low, Math.min(high, Number(value) || 0));

  class AdaVisualController {
    constructor(stage, rig) {
      this.stage = stage;
      this.rig = rig;
      this.state = "idle";
      this.speechLevel = 0;
      this.brightness = .5;
      this.mouth = "rest";
      this.lastMouthChange = 0;
      this.gaze = { x: 0, y: 0 };
      this.gazeTarget = { x: 0, y: 0 };
      this.nextGaze = performance.now() + 900;
      this.pointerUntil = 0;
      this.blinkStarted = 0;
      this.nextBlink = performance.now() + this.randomBlinkDelay();
      this.lastFrame = performance.now();
      this.pupils = [...stage.querySelectorAll(".pupil")];
      this.eyeWindows = [...stage.querySelectorAll(".eye-window")];
      this.blinkLayer = stage.querySelector(".blink-layer");
      this.quality = new URLSearchParams(location.search).get("quality") === "low" ||
        localStorage.getItem("ada-quality") === "low" ? "low" : "high";
      this.stage.dataset.quality = this.quality;
      this.frame = this.frame.bind(this);
      requestAnimationFrame(this.frame);
    }

    setAdaState(state) {
      if (!STATES.has(state)) return false;
      this.state = state;
      this.stage.dataset.state = state;
      if (state !== "speaking") this.setSpeechFeatures(0, this.brightness, true);
      if (state === "listening") this.setGaze(0, 0);
      if (state === "thinking") this.setGaze(-.38, -.2);
      if (state === "alert") this.setGaze(0, -.1);
      return true;
    }

    setGaze(x, y) {
      this.gazeTarget.x = clamp(x, -1, 1);
      this.gazeTarget.y = clamp(y, -1, 1);
      return { ...this.gazeTarget };
    }

    setExpression(name, intensity = 1) {
      const expression = name || "neutral";
      this.stage.dataset.expression = expression;
      this.stage.style.setProperty("--expression", clamp(intensity).toFixed(3));
    }

    setSpeechLevel(value, immediate = false) {
      this.setSpeechFeatures(value, this.brightness, immediate);
    }

    setSpeechFeatures(level, brightness = .5, immediate = false) {
      const target = clamp(level);
      this.speechLevel = immediate ? target : this.speechLevel * .62 + target * .38;
      this.brightness = this.brightness * .72 + clamp(brightness) * .28;
      this.stage.style.setProperty("--speech", this.speechLevel.toFixed(3));
      this.updateMouth(performance.now(), immediate);
    }

    setQuality(mode) {
      this.quality = mode === "low" ? "low" : "high";
      this.stage.dataset.quality = this.quality;
      localStorage.setItem("ada-quality", this.quality);
      return this.quality;
    }

    updateMouth(now, immediate = false) {
      let next = "rest";
      if (this.state === "speaking" && this.speechLevel > .032) {
        if (this.speechLevel > .68) next = "wide";
        else if (this.brightness < .31 && this.speechLevel > .14) next = "round";
        else if (this.speechLevel > .3) next = "medium";
        else next = "small";
      }
      if (next === this.mouth) return;
      if (!immediate && now - this.lastMouthChange < 72) return;
      this.mouth = next;
      this.lastMouthChange = now;
      this.stage.dataset.mouth = next;
    }

    randomBlinkDelay() {
      const stateFactor = this.state === "listening" ? .78 : this.state === "thinking" ? 1.2 : 1;
      return (2600 + Math.random() * 4100) * stateFactor;
    }

    updateBlink(now) {
      if (!this.blinkStarted && now >= this.nextBlink) this.blinkStarted = now;
      if (!this.blinkStarted) return;

      const elapsed = now - this.blinkStarted;
      let amount;
      if (elapsed < 82) amount = elapsed / 82;
      else if (elapsed < 126) amount = 1;
      else amount = 1 - (elapsed - 126) / 118;
      amount = clamp(amount);
      const eased = amount * amount * (3 - 2 * amount);
      this.blinkLayer.style.opacity = eased.toFixed(3);
      this.eyeWindows.forEach((eye) => { eye.style.opacity = (1 - eased).toFixed(3); });

      if (elapsed >= 244) {
        this.blinkLayer.style.opacity = "0";
        this.eyeWindows.forEach((eye) => { eye.style.opacity = "1"; });
        this.blinkStarted = 0;
        this.nextBlink = now + this.randomBlinkDelay();
      }
    }

    updateAutonomousGaze(now) {
      if (now < this.pointerUntil || now < this.nextGaze) return;
      if (this.state === "idle") {
        this.setGaze((Math.random() - .5) * 1.15, (Math.random() - .5) * .48);
        this.nextGaze = now + 1500 + Math.random() * 3100;
      } else if (this.state === "speaking") {
        this.setGaze((Math.random() - .5) * .34, (Math.random() - .5) * .16);
        this.nextGaze = now + 1900 + Math.random() * 2500;
      } else if (this.state === "thinking") {
        this.setGaze(-.48 + Math.random() * .22, -.25 + Math.random() * .16);
        this.nextGaze = now + 1200 + Math.random() * 1700;
      } else {
        this.setGaze(0, 0);
        this.nextGaze = now + 2400;
      }
    }

    updatePupils(now, elapsed) {
      const response = 1 - Math.pow(.0008, elapsed / 1000);
      this.gaze.x += (this.gazeTarget.x - this.gaze.x) * response;
      this.gaze.y += (this.gazeTarget.y - this.gaze.y) * response;
      const scale = this.rig.clientWidth / 800;
      const microX = Math.sin(now * .0061) * .38 + Math.sin(now * .0137) * .17;
      const microY = Math.cos(now * .0053) * .2;
      const tx = (this.gaze.x * 9 + microX) * scale;
      const ty = (this.gaze.y * 5 + microY) * scale;
      this.pupils.forEach((pupil, index) => {
        const asymmetry = index ? .12 : -.12;
        pupil.style.transform = `translate(calc(-50% + ${tx + asymmetry * scale}px), calc(-50% + ${ty}px))`;
      });
    }

    frame(now) {
      const elapsed = Math.min(50, Math.max(0, now - this.lastFrame));
      this.lastFrame = now;
      this.updateBlink(now);
      this.updateAutonomousGaze(now);
      this.updatePupils(now, elapsed);
      this.updateMouth(now);
      requestAnimationFrame(this.frame);
    }
  }

  const stage = document.querySelector("#ada-stage");
  const controller = new AdaVisualController(stage, document.querySelector("#ada-rig"));
  window.adaVisual = controller;
  window.setAdaState = (state) => controller.setAdaState(state);
  window.setGaze = (x, y) => controller.setGaze(x, y);
  window.setExpression = (name, intensity) => controller.setExpression(name, intensity);
  window.setSpeechLevel = (value) => controller.setSpeechLevel(value);
  window.setSpeechFeatures = (level, brightness) => controller.setSpeechFeatures(level, brightness);

  stage.addEventListener("pointermove", (event) => {
    if (controller.state !== "idle") return;
    const bounds = controller.rig.getBoundingClientRect();
    controller.pointerUntil = performance.now() + 1300;
    controller.setGaze(
      ((event.clientX - bounds.left) / bounds.width) * 2 - 1,
      ((event.clientY - bounds.top) / bounds.height) * 2 - 1
    );
  }, { passive: true });

  const panel = document.querySelector("#dev-panel");
  panel.addEventListener("click", (event) => {
    const state = event.target.dataset?.state;
    if (state) controller.setAdaState(state);
    if (event.target.id === "quality-toggle") {
      controller.setQuality(controller.quality === "low" ? "high" : "low");
    }
  });

  addEventListener("keydown", (event) => {
    if (event.key.toLowerCase() === "d") panel.hidden = !panel.hidden;
    const state = ["idle", "listening", "thinking", "speaking", "alert"][Number(event.key) - 1];
    if (state) controller.setAdaState(state);
    if (event.key.toLowerCase() === "q") {
      controller.setQuality(controller.quality === "low" ? "high" : "low");
    }
  });
})();
