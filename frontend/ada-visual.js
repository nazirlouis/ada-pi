(() => {
  "use strict";

  const STATES = new Set(["idle", "listening", "thinking", "speaking", "alert"]);
  const MOUTHS = {
    rest: ["M-65 0 C-34-8-18-18 0-10 C18-18 34-8 65 0 C32 5 14 4 0 3 C-14 4-32 5-65 0Z", "M-60 2 C-27 7 27 7 60 2 C31 16-31 16-60 2Z", "M-60 3 C-29 20 29 20 60 3 C31 34-31 34-60 3Z"],
    narrow: ["M-66 0 C-36-9-16-15 0-8 C16-15 36-9 66 0 C31 3-31 3-66 0Z", "M-58 2 C-25 6 25 6 58 2 C29 13-29 13-58 2Z", "M-58 3 C-26 15 26 15 58 3 C29 25-29 25-58 3Z"],
    open: ["M-61-2 C-31-13-16-17 0-8 C16-17 31-13 61-2 C30 4-30 4-61-2Z", "M-54 2 C-25 7 25 7 54 2 C38 42-38 42-54 2Z", "M-54 3 C-28 39 28 39 54 3 C31 55-31 55-54 3Z"],
    round: ["M-44-2 C-24-17-11-19 0-9 C11-19 24-17 44-2 C24 3-24 3-44-2Z", "M-38 1 C-22 5 22 5 38 1 C30 49-30 49-38 1Z", "M-38 3 C-25 44 25 44 38 3 C24 56-24 56-38 3Z"],
  };

  class AdaVisualController {
    constructor(stage, canvas) {
      this.stage = stage;
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d", { alpha: true });
      this.state = "idle";
      this.speechLevel = 0;
      this.gaze = { x: 0, y: 0 };
      this.quality = new URLSearchParams(location.search).get("quality") === "low" || localStorage.getItem("ada-quality") === "low" ? "low" : "high";
      this.particles = [];
      this.lastFrame = 0;
      this.lastMouth = "rest";
      this.blinkTimer = 0;
      this.nextBlink = performance.now() + this.randomBlinkDelay();
      this.eyes = [...stage.querySelectorAll(".eye")];
      this.gazes = [...stage.querySelectorAll(".gaze")];
      this.lips = [stage.querySelector(".lip.upper"), stage.querySelector(".mouth-opening"), stage.querySelector(".lip.lower")];
      this.resize = this.resize.bind(this);
      this.frame = this.frame.bind(this);
      addEventListener("resize", this.resize, { passive: true });
      this.resize();
      requestAnimationFrame(this.frame);
    }

    setAdaState(state) {
      if (!STATES.has(state)) return false;
      this.state = state;
      this.stage.dataset.state = state;
      if (state !== "speaking") this.setSpeechLevel(0, true);
      if (state === "listening") this.setGaze(0, 0);
      return true;
    }

    setGaze(x, y) {
      this.gaze.x = Math.max(-1, Math.min(1, Number(x) || 0));
      this.gaze.y = Math.max(-1, Math.min(1, Number(y) || 0));
      const tx = this.gaze.x * 8;
      const ty = this.gaze.y * 5;
      this.gazes.forEach((node) => { node.style.transform = `translate(${tx}px,${ty}px)`; });
    }

    setExpression(name, intensity = 1) {
      const amount = Math.max(0, Math.min(1, Number(intensity) || 0));
      this.stage.style.setProperty("--expression", amount);
      this.stage.dataset.expression = name || "neutral";
      if (name === "stern") this.eyes.forEach((eye) => { eye.style.filter = `brightness(${1 + amount * .35})`; });
      else this.eyes.forEach((eye) => { eye.style.filter = ""; });
    }

    setSpeechLevel(value, immediate = false) {
      const target = Math.max(0, Math.min(1, Number(value) || 0));
      this.speechLevel = immediate ? target : this.speechLevel * .68 + target * .32;
      this.stage.style.setProperty("--speech", this.speechLevel.toFixed(3));
      this.updateMouth();
    }

    setQuality(mode) {
      this.quality = mode === "low" ? "low" : "high";
      localStorage.setItem("ada-quality", this.quality);
      this.resize();
      return this.quality;
    }

    updateMouth() {
      let shape = "rest";
      if (this.state === "speaking" && this.speechLevel > .035) {
        if (this.speechLevel > .62) shape = "open";
        else if (this.speechLevel > .31) shape = Math.floor(performance.now() / 150) % 2 ? "round" : "open";
        else shape = "narrow";
      }
      if (shape === this.lastMouth) return;
      this.lastMouth = shape;
      MOUTHS[shape].forEach((path, index) => this.lips[index].setAttribute("d", path));
    }

    resize() {
      const dpr = this.quality === "low" ? 1 : Math.min(devicePixelRatio || 1, 1.5);
      this.canvas.width = Math.floor(innerWidth * dpr);
      this.canvas.height = Math.floor(innerHeight * dpr);
      this.canvas.style.width = `${innerWidth}px`;
      this.canvas.style.height = `${innerHeight}px`;
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const count = this.quality === "low" ? 36 : Math.min(130, Math.floor(innerWidth * innerHeight / 7000));
      this.particles = Array.from({ length: count }, (_, i) => this.makeParticle(i));
    }

    makeParticle(index) {
      const side = index % 2 ? -1 : 1;
      return { x: .5 + side * (.09 + Math.random() * .27), y: .19 + Math.random() * .68, vx: (Math.random() - .5) * .000025, vy: (Math.random() - .5) * .000035, size: .4 + Math.random() * 1.8, alpha: .14 + Math.random() * .58, phase: Math.random() * Math.PI * 2 };
    }

    blink(now) {
      if (now >= this.nextBlink && !this.blinkTimer) this.blinkTimer = now;
      if (!this.blinkTimer) return;
      const elapsed = now - this.blinkTimer;
      const scale = elapsed < 90 ? 1 - elapsed / 100 : elapsed < 180 ? (elapsed - 80) / 100 : 1;
      this.eyes.forEach((eye) => { eye.style.transform = `scaleY(${Math.max(.08, Math.min(1, scale))})`; });
      if (elapsed > 190) {
        this.eyes.forEach((eye) => { eye.style.transform = ""; });
        this.blinkTimer = 0;
        this.nextBlink = now + this.randomBlinkDelay();
      }
    }

    randomBlinkDelay() { return 2600 + Math.random() * 4200; }

    frame(now) {
      const elapsed = Math.min(40, now - (this.lastFrame || now));
      this.lastFrame = now;
      this.blink(now);
      const ctx = this.ctx;
      ctx.clearRect(0, 0, innerWidth, innerHeight);
      const alert = this.state === "alert";
      const gather = this.state === "listening" ? .045 : 0;
      for (const p of this.particles) {
        p.x += p.vx * elapsed;
        p.y += p.vy * elapsed;
        if (p.x < .15 || p.x > .85) p.vx *= -1;
        if (p.y < .1 || p.y > .93) p.vy *= -1;
        const eyeY = .414;
        const eyeX = p.x < .5 ? .394 : .606;
        const x = (p.x + (eyeX - p.x) * gather) * innerWidth;
        const y = (p.y + (eyeY - p.y) * gather) * innerHeight;
        const pulse = .65 + Math.sin(now * .0012 + p.phase) * .35;
        ctx.fillStyle = alert ? `rgba(255,126,20,${p.alpha * pulse})` : `rgba(73,232,255,${p.alpha * pulse})`;
        ctx.beginPath(); ctx.arc(x, y, p.size, 0, Math.PI * 2); ctx.fill();
      }
      if (this.state === "idle" && now > this.nextBlink - 700 && Math.random() < .012) this.setGaze((Math.random() - .5) * .5, (Math.random() - .5) * .25);
      this.updateMouth();
      requestAnimationFrame(this.frame);
    }
  }

  const stage = document.querySelector("#ada-stage");
  const controller = new AdaVisualController(stage, document.querySelector("#particle-field"));
  window.adaVisual = controller;
  window.setAdaState = (state) => controller.setAdaState(state);
  window.setGaze = (x, y) => controller.setGaze(x, y);
  window.setExpression = (name, intensity) => controller.setExpression(name, intensity);
  window.setSpeechLevel = (value) => controller.setSpeechLevel(value);

  stage.addEventListener("pointermove", (event) => {
    if (controller.state === "idle") controller.setGaze(event.clientX / innerWidth * 2 - 1, event.clientY / innerHeight * 2 - 1);
  }, { passive: true });

  const panel = document.querySelector("#dev-panel");
  panel.addEventListener("click", (event) => {
    const state = event.target.dataset?.state;
    if (state) controller.setAdaState(state);
    if (event.target.id === "quality-toggle") controller.setQuality(controller.quality === "low" ? "high" : "low");
  });
  addEventListener("keydown", (event) => {
    if (event.key.toLowerCase() === "d") panel.hidden = !panel.hidden;
    const state = ["idle", "listening", "thinking", "speaking", "alert"][Number(event.key) - 1];
    if (state) controller.setAdaState(state);
    if (event.key.toLowerCase() === "q") controller.setQuality(controller.quality === "low" ? "high" : "low");
  });
})();
