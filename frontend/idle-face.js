(() => {
  "use strict";

  const face = document.querySelector("#idle-face");
  if (!face) return;
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const eyes = [...face.querySelectorAll(".eye")];
  const mouthPaths = [...face.querySelectorAll(".mouth, .mouth-aura, .mouth-shimmer")];
  const expressionBrows = [face.querySelector(".left-brow"), face.querySelector(".right-brow")];

  // Every shape uses the same SVG command structure so Chromium can smoothly
  // interpolate between them. Geometry changes create speech; the mouth group
  // itself remains reserved for following ADA's gaze.
  const mouthShapes = {
    rest: "M690 742 C758 776 842 776 910 742 C858 805 742 805 690 742Z",
    narrow: "M704 750 C754 735 846 735 896 750 C858 788 742 788 704 750Z",
    open: "M710 754 C754 722 846 722 890 754 C876 835 724 835 710 754Z",
    wide: "M680 752 C746 720 854 720 920 752 C895 828 705 828 680 752Z",
    round: "M742 748 C770 706 830 706 858 748 C884 842 716 842 742 748Z"
  };
  const expressions = [
    "neutral", "sassy", "amused", "skeptical", "annoyed", "mad",
    "concerned", "surprised", "mischievous", "serious", "alert"
  ];
  const expressionMouths = {
    neutral: mouthShapes.rest,
    sassy: "M686 750 C754 779 846 769 914 730 C866 800 746 812 686 750Z",
    amused: "M674 738 C752 783 848 783 926 738 C872 817 728 817 674 738Z",
    skeptical: "M704 754 C760 746 840 746 896 754 C848 777 752 777 704 754Z",
    annoyed: "M690 758 C754 746 846 746 910 758 C852 771 748 771 690 758Z",
    mad: "M688 774 C754 742 846 742 912 774 C854 752 746 752 688 774Z",
    concerned: "M696 779 C758 742 842 742 904 779 C852 760 748 760 696 779Z",
    surprised: "M752 748 C778 718 822 718 848 748 C866 823 734 823 752 748Z",
    mischievous: "M682 758 C750 782 846 766 918 728 C866 805 746 810 682 758Z",
    serious: "M700 756 C760 750 840 750 900 756 C848 770 752 770 700 756Z",
    alert: "M692 760 C756 744 844 744 908 760 C850 779 750 779 692 760Z"
  };
  const expressionBrowPaths = {
    neutral: ["M270 326 C372 305 500 305 600 326", "M1330 326 C1228 305 1100 305 1000 326"],
    sassy: ["M255 337 C350 300 500 296 603 325", ""],
    amused: ["M266 325 C370 286 508 290 603 326", "M1334 325 C1230 286 1092 290 997 326"],
    skeptical: ["M264 342 C370 322 508 320 603 339", "M1338 309 C1230 266 1090 276 990 318"],
    annoyed: ["M260 331 C372 317 505 318 610 342", "M1340 331 C1228 317 1095 318 990 342"],
    mad: ["M258 315 C378 308 516 327 619 365", "M1342 315 C1222 308 1084 327 981 365"],
    concerned: ["M264 337 C378 323 510 293 610 277", "M1336 337 C1222 323 1090 293 990 277"],
    surprised: ["M270 305 C380 252 502 258 595 305", "M1330 305 C1220 252 1098 258 1005 305"],
    mischievous: ["M252 326 C374 286 518 305 622 352", "M1348 326 C1226 286 1082 305 978 352"],
    serious: ["M268 323 C370 306 502 306 604 323", "M1332 323 C1230 306 1098 306 996 323"],
    alert: ["M258 319 C374 305 510 323 614 354", "M1342 319 C1226 305 1090 323 986 354"]
  };

  let gazeTimer = null;
  let saccadeTimer = null;
  let blinkTimer = null;
  let sequenceTimers = [];
  let baseGaze = { x: 0, y: 0 };
  let speechLevel = 0;
  let mouthShape = "rest";
  let lastMouthChange = 0;
  let connectingTimer = null;
  let expression = "neutral";

  const clamp = (value, limit) => Math.max(-limit, Math.min(limit, value));
  const randomBetween = (min, max) => min + Math.random() * (max - min);
  const later = (fn, delay) => {
    const timer = setTimeout(() => {
      sequenceTimers = sequenceTimers.filter((item) => item !== timer);
      fn();
    }, delay);
    sequenceTimers.push(timer);
    return timer;
  };

  function applyGaze(x, y) {
    face.style.setProperty("--gaze-x", `${clamp(x, 18).toFixed(1)}px`);
    face.style.setProperty("--gaze-y", `${clamp(y, 10).toFixed(1)}px`);
  }

  function setGaze(x, y) {
    baseGaze = { x: clamp(x, 18), y: clamp(y, 10) };
    applyGaze(baseGaze.x, baseGaze.y);
  }

  function applyMouthShape(name, force = false) {
    if (name === mouthShape && !force) return;
    mouthShape = name;
    const pathData = name === "rest" ? expressionMouths[expression] : mouthShapes[name];
    mouthPaths.forEach((path) => path.setAttribute("d", pathData));
  }

  function setExpression(name) {
    if (!expressions.includes(name)) return false;
    face.classList.remove(...expressions.map((item) => `expression-${item}`));
    expression = name;
    face.classList.add(`expression-${name}`);
    face.dataset.expression = name;
    expressionBrows.forEach((brow, index) => {
      const pathData = expressionBrowPaths[name][index];
      brow.setAttribute("d", pathData);
      brow.style.opacity = pathData ? "1" : "0";
    });
    if (speechLevel <= .025) applyMouthShape("rest", true);
    document.querySelectorAll("#expression-controls button").forEach((button) => {
      button.classList.toggle("active", button.dataset.expression === name);
    });
    return true;
  }

  function setSpeechLevel(value, immediate = false) {
    const next = Math.max(0, Math.min(1, Number(value) || 0));
    speechLevel = immediate ? next : speechLevel * .3 + next * .7;
    face.style.setProperty("--speech-level", speechLevel.toFixed(3));
    face.classList.toggle("is-speaking", speechLevel > .025);

    if (immediate || speechLevel <= .025) {
      applyMouthShape("rest");
      lastMouthChange = performance.now();
      return;
    }

    const now = performance.now();
    if (now - lastMouthChange < 58) return;
    let shape;
    if (expression === "surprised") {
      shape = speechLevel < .16 ? "narrow" : "round";
    } else if (["serious", "annoyed", "mad", "concerned"].includes(expression)) {
      shape = speechLevel < .2 || mouthShape === "open" ? "narrow" : "open";
    } else if (speechLevel < .13) shape = "narrow";
    else if (speechLevel < .34) shape = mouthShape === "narrow" ? "open" : "narrow";
    else if (speechLevel < .62) shape = mouthShape === "round" ? "open" : "round";
    else shape = mouthShape === "wide" ? "open" : "wide";
    applyMouthShape(shape);
    lastMouthChange = now;
  }

  function setConnecting(connecting, immediate = false) {
    clearTimeout(connectingTimer);
    connectingTimer = null;
    if (connecting) {
      face.classList.add("is-connecting");
    } else if (immediate) {
      face.classList.remove("is-connecting");
    } else {
      // Let the visible synchronization finish one last complete cycle after
      // Gemini becomes ready instead of snapping straight back to idle.
      connectingTimer = setTimeout(() => {
        face.classList.remove("is-connecting");
        connectingTimer = null;
      }, 1450);
    }
  }

  function scheduleGaze() {
    clearTimeout(gazeTimer);
    // Longer focus holds are mixed with ordinary glances so the movement
    // feels intentional instead of metronomic.
    const delay = Math.random() < .34 ? randomBetween(4500, 7000) : randomBetween(1800, 3800);
    gazeTimer = setTimeout(() => {
      const magnitude = Math.random() < .62 ? .68 : 1;
      setGaze(randomBetween(-18, 18) * magnitude, randomBetween(-10, 10) * magnitude);
      scheduleGaze();
    }, delay);
  }

  function scheduleSaccade() {
    clearTimeout(saccadeTimer);
    saccadeTimer = setTimeout(() => {
      face.classList.add("micro-saccade");
      applyGaze(baseGaze.x + randomBetween(-2.6, 2.6), baseGaze.y + randomBetween(-1.5, 1.5));
      later(() => face.classList.remove("micro-saccade"), 125);
      scheduleSaccade();
    }, randomBetween(520, 1250));
  }

  function blinkOnce(onFinished, strength = 1) {
    const baseLift = ["surprised", "concerned"].includes(expression)
      ? 4
      : (["serious", "annoyed", "mad"].includes(expression) ? 2 : 3);
    const lift = `${(-baseLift * strength).toFixed(1)}px`;
    eyes.forEach((eye) => eye.classList.add("is-blinking"));
    face.style.setProperty("--blink-left", ".055");
    face.style.setProperty("--brow-left-lift", lift);
    later(() => {
      face.style.setProperty("--blink-right", ".055");
      face.style.setProperty("--brow-right-lift", lift);
    }, 18);
    later(() => {
      face.style.setProperty("--blink-left", "1");
      face.style.setProperty("--brow-left-lift", "0px");
    }, 108);
    later(() => {
      face.style.setProperty("--blink-right", "1");
      face.style.setProperty("--brow-right-lift", "0px");
    }, 126);
    later(() => {
      eyes.forEach((eye) => eye.classList.remove("is-blinking"));
      onFinished();
    }, 255);
  }

  function scheduleBlink() {
    clearTimeout(blinkTimer);
    blinkTimer = setTimeout(() => {
      blinkOnce(() => {
        if (Math.random() < .18) later(() => blinkOnce(scheduleBlink, 1.3), 115);
        else scheduleBlink();
      });
    }, randomBetween(2800, 6500));
  }

  function stop() {
    clearTimeout(gazeTimer);
    clearTimeout(saccadeTimer);
    clearTimeout(blinkTimer);
    sequenceTimers.forEach(clearTimeout);
    sequenceTimers = [];
  }

  function start() {
    stop();
    face.style.setProperty("--blink-left", "1");
    face.style.setProperty("--blink-right", "1");
    face.style.setProperty("--brow-left-lift", "0px");
    face.style.setProperty("--brow-right-lift", "0px");
    if (reducedMotion) return;
    scheduleGaze();
    scheduleSaccade();
    scheduleBlink();
  }

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stop();
    else start();
  });

  window.idleFace = { setGaze, setSpeechLevel, setConnecting, setExpression, expressions };
  const expressionControls = document.querySelector("#expression-controls");
  if (expressionControls) {
    expressions.forEach((name) => {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.expression = name;
      button.textContent = name;
      button.addEventListener("click", () => setExpression(name));
      expressionControls.append(button);
    });
  }
  setExpression("neutral");
  start();
})();
