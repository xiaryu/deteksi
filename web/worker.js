// Lensa Deteksi - Web Worker untuk AI
// Semua proses AI berjalan di sini (thread terpisah), jadi tampilan video
// di halaman utama tetap mulus walaupun HP butuh waktu lama untuk satu deteksi.

const VER = "0.10.14";
const CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VER}`;
const MP = "https://storage.googleapis.com/mediapipe-models/";
const MODEL_HAND = MP + "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const MODELS = {
  akurat: [MP + "object_detector/efficientdet_lite2/float16/latest/efficientdet_lite2.tflite",
           MP + "object_detector/efficientdet_lite2/float32/latest/efficientdet_lite2.tflite",
           MP + "object_detector/efficientdet_lite2/int8/latest/efficientdet_lite2.tflite"],
  cepat:  [MP + "object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite"]
};

// Muat build CommonJS MediaPipe di worker klasik
self.module = { exports: {} };
self.exports = self.module.exports;
importScripts(`${CDN}/vision_bundle.cjs`);
const { FilesetResolver, ObjectDetector, HandLandmarker } = self.module.exports;

let vision = null, detector = null, hands = null, delegate = "GPU", mode = "cepat";

function makeHands(d) {
  return HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_HAND, delegate: d },
    runningMode: "VIDEO", numHands: 2
  });
}

async function makeDetector(m) {
  const urls = [...MODELS[m], ...(m === "akurat" ? MODELS.cepat : [])];
  let lastErr;
  for (const url of urls) {
    try {
      const d = await ObjectDetector.createFromOptions(vision, {
        baseOptions: { modelAssetPath: url, delegate },
        runningMode: "VIDEO", scoreThreshold: 0.5, maxResults: 10
      });
      return { d, mode: url.includes("lite0") ? "cepat" : "akurat" };
    } catch (e) { lastErr = e; }
  }
  throw lastErr;
}

async function init(m) {
  vision = await FilesetResolver.forVisionTasks(`${CDN}/wasm`);
  try { hands = await makeHands("GPU"); }
  catch { delegate = "CPU"; hands = await makeHands("CPU"); }
  const r = await makeDetector(m);
  detector = r.d; mode = r.mode;
}

// hasil MediaPipe disalin ke objek biasa supaya aman dikirim ke halaman
const slimObjs = list => list.map(d => ({
  name: d.categories[0].categoryName,
  score: d.categories[0].score,
  box: { x: d.boundingBox.originX, y: d.boundingBox.originY, w: d.boundingBox.width, h: d.boundingBox.height }
}));
const slimHands = list => list.map(lm => lm.map(p => ({ x: p.x, y: p.y })));

self.onmessage = async (e) => {
  const msg = e.data;

  if (msg.type === "init") {
    try {
      await init(msg.mode);
      self.postMessage({ type: "ready", delegate, mode });
    } catch (err) {
      self.postMessage({ type: "error", message: String(err && err.message || err) });
    }
    return;
  }

  if (msg.type === "model") {
    try {
      const r = await makeDetector(msg.mode);
      const old = detector; detector = r.d; mode = r.mode;
      try { old && old.close(); } catch {}
      self.postMessage({ type: "model", mode, wanted: msg.mode });
    } catch (err) {
      self.postMessage({ type: "model", mode, wanted: msg.mode, failed: true });
    }
    return;
  }

  if (msg.type === "frame") {
    const { bitmap, ts, doObj } = msg;
    const out = { type: "result" };
    let failed = false;
    if (switching) { bitmap.close(); self.postMessage(out); return; }
    try {
      if (doObj && detector) out.objs = slimObjs(detector.detectForVideo(bitmap, ts).detections);
      if (hands) out.hands = slimHands(hands.detectForVideo(bitmap, ts).landmarks);
    } catch (err) {
      failed = true;
      out.warn = String(err && err.message || err);
    } finally {
      bitmap.close();
    }
    self.postMessage(out);
    if (failed && delegate === "GPU") fallbackToCPU();
  }
};

// Ada HP yang bisa membuat model GPU tapi gagal saat deteksi: pindah ke CPU sekali.
let switching = false;
async function fallbackToCPU() {
  switching = true;
  delegate = "CPU";
  try {
    const h = await makeHands("CPU");
    const r = await makeDetector(mode);
    try { hands && hands.close(); detector && detector.close(); } catch {}
    hands = h; detector = r.d; mode = r.mode;
  } catch (e) {
    self.postMessage({ type: "result", warn: "CPU fallback gagal: " + e });
  }
  switching = false;
}
