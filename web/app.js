// Lensa Deteksi - kamera, tampilan, dan efek
// AI berjalan di worker.js (thread terpisah) supaya video tetap mulus ~30 FPS.
// Kalau worker tidak didukung, AI otomatis jalan di halaman ini (mode cadangan).

/* ================= Konfigurasi ================= */
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

const ID = { person:"manusia", bicycle:"sepeda", car:"mobil", motorcycle:"motor", airplane:"pesawat", bus:"bus", train:"kereta",
  truck:"truk", boat:"perahu", "traffic light":"lampu lalu lintas", "fire hydrant":"hidran", "stop sign":"rambu stop", bench:"bangku",
  bird:"burung", cat:"kucing", dog:"anjing", horse:"kuda", sheep:"domba", cow:"sapi", elephant:"gajah", bear:"beruang", zebra:"zebra",
  giraffe:"jerapah", backpack:"tas ransel", umbrella:"payung", handbag:"tas tangan", tie:"dasi", suitcase:"koper", frisbee:"frisbee",
  skis:"ski", snowboard:"snowboard", "sports ball":"bola", kite:"layangan", "baseball bat":"tongkat bisbol", "baseball glove":"sarung tangan",
  skateboard:"skateboard", surfboard:"papan selancar", "tennis racket":"raket", bottle:"botol", "wine glass":"gelas", cup:"cangkir",
  fork:"garpu", knife:"pisau", spoon:"sendok", bowl:"mangkuk", banana:"pisang", apple:"apel", sandwich:"roti lapis", orange:"jeruk",
  broccoli:"brokoli", carrot:"wortel", "hot dog":"hot dog", pizza:"pizza", donut:"donat", cake:"kue", chair:"kursi", couch:"sofa",
  "potted plant":"tanaman pot", bed:"kasur", "dining table":"meja", toilet:"toilet", tv:"TV", laptop:"laptop", mouse:"mouse",
  remote:"remote", keyboard:"keyboard", "cell phone":"HP", microwave:"microwave", oven:"oven", toaster:"pemanggang roti", sink:"wastafel",
  refrigerator:"kulkas", book:"buku", clock:"jam", vase:"vas", scissors:"gunting", "teddy bear":"boneka", "hair drier":"pengering rambut",
  toothbrush:"sikat gigi" };
const ANIMALS = new Set(["bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe"]);
const colorFor = c => c === "person" ? "#3cf2ff" : ANIMALS.has(c) ? "#ffd23c" : "#ff3ca8";
const nameFor = c => ID[c] || c;

const TIPS = [4, 8, 12, 16, 20];                        // jempol, telunjuk, tengah, manis, kelingking
const TIP_COLORS = ["#ff3ca8", "#3cf2ff", "#b86bff", "#4dff9a", "#ffd23c"];
const BONES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
               [9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];

// HP: pengaturan ringan
const IS_MOBILE = /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent) || matchMedia("(pointer: coarse)").matches;
const GLOW = !IS_MOBILE;                       // shadowBlur sangat lambat di canvas HP
const OBJ_INTERVAL = IS_MOBILE ? 300 : 120;    // jeda deteksi objek (ms)
const SMOOTH = 0.45;                           // 0..1, makin besar makin cepat mengejar posisi tangan

/* ================= State ================= */
const $ = id => document.getElementById(id);
const cv = $("cv"), ctx = cv.getContext("2d"), vid = $("vid");
let stream = null, facing = "user";
let showBox = true, showHand = true;
let mode = IS_MOBILE ? "cepat" : "akurat";
try { mode = localStorage.getItem("lensa-mode") || mode; } catch {}

let engine = null;            // "worker" | "main" | null
let aiReady = false, busy = false;
let worker = null;
let mainVision = null, mainDetector = null, mainHands = null, mainDelegate = "GPU";

let objs = [];                // [{name, score, box:{x,y,w,h}}]  (piksel video)
let handTarget = [];          // [[{x,y}...21]]  (0..1) hasil AI terakhir
let handShow = [];            // posisi yang digambar (dihaluskan)
let lastObjT = 0, lastTs = 0;
let fpsT = performance.now(), fpsN = 0, aiN = 0;

/* ================= Kamera ================= */
async function startCam() {
  if (stream) stream.getTracks().forEach(t => t.stop());
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: {
      facingMode: facing,
      width: { ideal: IS_MOBILE ? 640 : 1280 }, height: { ideal: IS_MOBILE ? 480 : 720 },
      frameRate: { ideal: 30 }
    }, audio: false });
  } catch (e) {
    if (e.name === "NotAllowedError") throw e;
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  }
  vid.srcObject = stream;
  await vid.play().catch(() => {});
  sizeCanvas();
}
function sizeCanvas() {
  const s = stream?.getVideoTracks()[0]?.getSettings() || {};
  cv.width = vid.videoWidth || s.width || 1280;
  cv.height = vid.videoHeight || s.height || 720;
}
vid.addEventListener("loadedmetadata", sizeCanvas);
vid.addEventListener("resize", sizeCanvas);

/* ================= AI: worker (utama) ================= */
function startWorker() {
  return new Promise((resolve, reject) => {
    let w;
    try { w = new Worker("worker.js"); } catch (e) { return reject(e); }
    const timer = setTimeout(() => { w.terminate(); reject(new Error("worker timeout")); }, 90000);
    w.onerror = (e) => { clearTimeout(timer); w.terminate(); reject(new Error(e.message || "worker error")); };
    w.onmessage = (e) => {
      const m = e.data;
      if (m.type === "ready") {
        clearTimeout(timer);
        w.onmessage = onWorkerMessage; w.onerror = null;
        worker = w; mode = m.mode; updateModeBtn();
        resolve(m.delegate);
      } else if (m.type === "error") {
        clearTimeout(timer); w.terminate(); reject(new Error(m.message));
      }
    };
    w.postMessage({ type: "init", mode });
  });
}

function onWorkerMessage(e) {
  const m = e.data;
  if (m.type === "result") {
    busy = false; aiN++;
    if (m.objs) objs = m.objs;
    if (m.hands) handTarget = m.hands;
    if (m.warn) console.warn("AI:", m.warn);
  } else if (m.type === "model") {
    modeSwitched(m.mode, m.wanted, m.failed);
  }
}

/* ================= AI: halaman utama (cadangan) ================= */
let MPlib = null;
async function startMain() {
  MPlib = await import(CDN);
  mainVision = await MPlib.FilesetResolver.forVisionTasks(`${CDN}/wasm`);
  const mk = d => MPlib.HandLandmarker.createFromOptions(mainVision, {
    baseOptions: { modelAssetPath: MODEL_HAND, delegate: d }, runningMode: "VIDEO", numHands: 2 });
  try { mainHands = await mk("GPU"); } catch { mainDelegate = "CPU"; mainHands = await mk("CPU"); }
  mainDetector = await mainMakeDetector(mode);
  return mainDelegate;
}
async function mainMakeDetector(m) {
  const urls = [...MODELS[m], ...(m === "akurat" ? MODELS.cepat : [])];
  let lastErr;
  for (const url of urls) {
    try {
      const d = await MPlib.ObjectDetector.createFromOptions(mainVision, {
        baseOptions: { modelAssetPath: url, delegate: mainDelegate },
        runningMode: "VIDEO", scoreThreshold: 0.5, maxResults: 10 });
      mode = url.includes("lite0") ? "cepat" : "akurat";
      return d;
    } catch (e) { lastErr = e; }
  }
  throw lastErr;
}
function mainDetect(now, doObj) {
  try {
    if (doObj) objs = mainDetector.detectForVideo(vid, now).detections.map(d => ({
      name: d.categories[0].categoryName, score: d.categories[0].score,
      box: { x: d.boundingBox.originX, y: d.boundingBox.originY, w: d.boundingBox.width, h: d.boundingBox.height } }));
    else handTarget = mainHands.detectForVideo(vid, now).landmarks.map(lm => lm.map(p => ({ x: p.x, y: p.y })));
    aiN++;
  } catch (e) { console.warn(e); }
}

/* ================= Status ================= */
function setAI(text, cls) { const el = $("aiStatus"); el.textContent = text; el.className = "pill " + cls; }
function updateModeBtn() {
  $("tMode").textContent = mode === "akurat" ? "Mode: Akurat" : "Mode: Cepat";
  $("tMode").title = mode === "akurat"
    ? "Lebih teliti, lebih berat. Klik untuk mode cepat."
    : "Lebih ringan, kurang teliti untuk benda kecil. Klik untuk mode akurat.";
}

/* ================= Mulai ================= */
$("start").addEventListener("click", async () => {
  const btn = $("start");
  btn.disabled = true; btn.textContent = "Membuka kamera…"; $("msg").textContent = "";
  try {
    await startCam();
  } catch (e) {
    console.error(e);
    btn.disabled = false; btn.textContent = "Coba lagi";
    $("msg").textContent =
      e.name === "NotAllowedError" ? "Izin kamera ditolak. Klik ikon kamera di address bar, pilih Izinkan, lalu coba lagi." :
      e.name === "NotReadableError" ? "Kamera sedang dipakai aplikasi lain (Zoom, Camera, tab lain). Tutup dulu lalu coba lagi." :
      e.name === "NotFoundError" ? "Kamera tidak ditemukan di perangkat ini." :
      "Kamera gagal dibuka: " + e.message;
    return;
  }
  $("intro").hidden = true;
  $("app").hidden = false;
  requestAnimationFrame(loop);
  loadGallery();

  setAI("Memuat AI…", "loading");
  try {
    const d = await startWorker();
    engine = "worker";
    console.info("AI berjalan di worker, delegate:", d);
  } catch (e) {
    console.warn("Worker gagal, pakai mode cadangan:", e);
    try { await startMain(); engine = "main"; }
    catch (e2) { console.error(e2); setAI("AI gagal, cek internet", "err"); return; }
  }
  aiReady = true;
  updateModeBtn();
  setAI(engine === "worker" ? "AI aktif" : "AI aktif (cadangan)", "ok");
});

/* ================= Loop utama ================= */
function loop(now) {
  requestAnimationFrame(loop);
  const W = cv.width, H = cv.height, mirror = facing === "user";

  // kirim frame ke AI tanpa menahan tampilan
  if (aiReady && vid.readyState >= 2) {
    const doObj = now - lastObjT >= OBJ_INTERVAL;
    const ts = Math.max(now, lastTs + 1); lastTs = ts;     // timestamp harus selalu naik
    if (engine === "worker" && !busy) {
      busy = true;
      if (doObj) lastObjT = now;
      createImageBitmap(vid).then(bitmap => {
        worker.postMessage({ type: "frame", bitmap, ts, doObj }, [bitmap]);
      }).catch(() => { busy = false; });
    } else if (engine === "main") {
      if (doObj) lastObjT = now;
      mainDetect(ts, doObj);
    }
  }

  smoothHands();

  ctx.save();
  if (mirror) { ctx.translate(W, 0); ctx.scale(-1, 1); }
  ctx.drawImage(vid, 0, 0, W, H);
  ctx.restore();

  if (showBox) drawBoxes(W, mirror);
  if (showHand) drawHands(W, H, mirror);

  $("no").textContent = objs.length;
  $("nh").textContent = handShow.length;
  fpsN++;
  if (now - fpsT > 500) {
    $("fps").textContent = Math.round(fpsN * 1000 / (now - fpsT));
    $("aifps").textContent = Math.round(aiN * 1000 / (now - fpsT));
    fpsT = now; fpsN = 0; aiN = 0;
    updateChips();
  }
}

// garis tangan meluncur halus ke posisi terbaru, walau AI lebih lambat dari video
function smoothHands() {
  if (handTarget.length !== handShow.length) { handShow = handTarget.map(h => h.map(p => ({ ...p }))); return; }
  for (let i = 0; i < handTarget.length; i++)
    for (let j = 0; j < 21; j++) {
      const s = handShow[i][j], t = handTarget[i][j];
      s.x += (t.x - s.x) * SMOOTH; s.y += (t.y - s.y) * SMOOTH;
    }
}

/* ================= Gambar kotak objek ================= */
function drawBoxes(W, mirror) {
  const s = W / 640;
  for (const o of objs) {
    const col = colorFor(o.name), b = o.box;
    const x = mirror ? W - b.x - b.w : b.x, y = b.y, w = b.w, h = b.h;

    ctx.fillStyle = col + "14"; ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = col; ctx.lineWidth = 1.5 * s;
    if (GLOW) { ctx.shadowColor = col; ctx.shadowBlur = 10 * s; }
    ctx.strokeRect(x, y, w, h);

    const L = Math.min(w, h) * .18; ctx.lineWidth = 5 * s; ctx.beginPath();
    for (const [px, py, dx, dy] of [[x,y,1,1],[x+w,y,-1,1],[x,y+h,1,-1],[x+w,y+h,-1,-1]]) {
      ctx.moveTo(px + dx * L, py); ctx.lineTo(px, py); ctx.lineTo(px, py + dy * L);
    }
    ctx.stroke(); ctx.shadowBlur = 0;

    const label = `${nameFor(o.name).toUpperCase()}  ${Math.round(o.score * 100)}%`;
    ctx.font = `600 ${13 * s}px "JetBrains Mono", monospace`;
    const tw = ctx.measureText(label).width + 14 * s, th = 22 * s, ly = y > th ? y - th : y;
    ctx.fillStyle = col; ctx.fillRect(x, ly, tw, th);
    ctx.fillStyle = "#05070d"; ctx.fillText(label, x + 7 * s, ly + 15.5 * s);
  }
}

/* ================= Efek garis tangan (tren TikTok) ================= */
function drawHands(W, H, mirror) {
  const s = W / 640, t = performance.now() / 1000;
  const P = handShow.map(lm => lm.map(p => ({ x: (mirror ? 1 - p.x : p.x) * W, y: p.y * H })));
  ctx.lineCap = "round";

  for (const pts of P) {
    ctx.strokeStyle = "rgba(232,241,255,.5)"; ctx.lineWidth = 2 * s; ctx.beginPath();
    for (const [a, b] of BONES) { ctx.moveTo(pts[a].x, pts[a].y); ctx.lineTo(pts[b].x, pts[b].y); }
    ctx.stroke();
  }

  if (P.length === 2) {
    TIPS.forEach((tip, i) => neonLine(P[0][tip], P[1][tip], TIP_COLORS[i], s, t + i));
    ctx.globalAlpha = .35;
    neonLine(P[0][4], P[1][8], "#ffffff", s * .5, t);
    neonLine(P[0][8], P[1][4], "#ffffff", s * .5, t + 1);
    ctx.globalAlpha = 1;
  } else if (P.length === 1) {
    TIPS.slice(1).forEach((tip, i) => neonLine(P[0][4], P[0][tip], TIP_COLORS[i + 1], s * .8, t + i));
  }

  for (const pts of P) TIPS.forEach((tip, i) => {
    const p = pts[tip];
    ctx.fillStyle = TIP_COLORS[i];
    if (GLOW) { ctx.shadowColor = TIP_COLORS[i]; ctx.shadowBlur = 22 * s; }
    else { ctx.globalAlpha = .3; ctx.beginPath(); ctx.arc(p.x, p.y, 13 * s, 0, 7); ctx.fill(); ctx.globalAlpha = 1; }
    ctx.beginPath(); ctx.arc(p.x, p.y, 6 * s, 0, 7); ctx.fill();
  });
  ctx.shadowBlur = 0;
}

function neonLine(a, b, col, s, t) {
  const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
  const dx = b.x - a.x, dy = b.y - a.y, len = Math.hypot(dx, dy) || 1;
  const wob = Math.sin(t * 9) * Math.min(14, len * .03) * s;   // getar seperti senar
  const cx = mx - dy / len * wob, cy = my + dx / len * wob;
  const path = () => { ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.quadraticCurveTo(cx, cy, b.x, b.y); ctx.stroke(); };
  if (GLOW) {
    ctx.shadowColor = col; ctx.shadowBlur = 25 * s; ctx.strokeStyle = col; ctx.lineWidth = 5 * s; path();
    ctx.shadowBlur = 0;
  } else {
    const a0 = ctx.globalAlpha;
    ctx.strokeStyle = col; ctx.globalAlpha = a0 * .25; ctx.lineWidth = 14 * s; path();
    ctx.globalAlpha = a0; ctx.lineWidth = 5 * s; path();
  }
  ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1.5 * s; path();
}

/* ================= UI ================= */
function updateChips() {
  const count = {};
  for (const o of objs) count[o.name] = (count[o.name] || 0) + 1;
  $("chips").innerHTML = Object.entries(count)
    .map(([n, k]) => `<span class="chip" style="--c:${colorFor(n)}">${nameFor(n)} ×${k}</span>`).join("");
}

function toggle(btn, fn) {
  btn.addEventListener("click", () => {
    const v = btn.getAttribute("aria-pressed") !== "true";
    btn.setAttribute("aria-pressed", v); fn(v);
  });
}
toggle($("tBox"), v => showBox = v);
toggle($("tHand"), v => showHand = v);

updateModeBtn();
function modeSwitched(newMode, wanted, failed) {
  mode = newMode;
  try { localStorage.setItem("lensa-mode", mode); } catch {}
  objs = []; updateModeBtn(); setAI("AI aktif", "ok"); $("tMode").disabled = false;
  toast(failed ? "Gagal ganti model, cek internet"
      : mode !== wanted ? "Mode akurat tidak tersedia, tetap mode cepat"
      : mode === "akurat" ? "Mode akurat aktif" : "Mode cepat aktif");
}
$("tMode").addEventListener("click", async () => {
  if (!aiReady) return;
  const next = mode === "akurat" ? "cepat" : "akurat";
  $("tMode").disabled = true; setAI("Ganti model…", "loading");
  if (engine === "worker") { worker.postMessage({ type: "model", mode: next }); return; }
  const prev = mode;
  try {
    const d = await mainMakeDetector(next);
    const old = mainDetector; mainDetector = d; try { old?.close(); } catch {}
    modeSwitched(mode, next, false);
  } catch { mode = prev; modeSwitched(prev, next, true); }
});

$("tFlip").addEventListener("click", async () => {
  facing = facing === "user" ? "environment" : "user";
  try { await startCam(); } catch { toast("Kamera lain tidak tersedia"); facing = "user"; await startCam().catch(() => {}); }
});

function toast(text) {
  const el = $("toast"); el.textContent = text; el.hidden = false;
  clearTimeout(toast.t); toast.t = setTimeout(() => el.hidden = true, 2200);
}

/* ================= Foto ================= */
// Dengan server Python: foto disimpan ke folder hasil_foto.
// Tanpa server (mis. di Vercel): foto langsung diunduh ke perangkat.
let hasServer = false;

$("shot").addEventListener("click", async () => {
  const f = $("flash"); f.classList.remove("on"); void f.offsetWidth; f.classList.add("on");
  const gambar = cv.toDataURL("image/jpeg", 0.9);
  const nama = "deteksi_" + new Date().toISOString().slice(0, 19).replace(/[-:T]/g, "") + ".jpg";

  if (hasServer) {
    try {
      const r = await fetch("/api/foto", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ gambar, objek: objs.map(o => nameFor(o.name)) })
      });
      const j = await r.json();
      if (!j.ok) throw new Error(j.pesan);
      toast("Foto tersimpan: " + j.nama);
      loadGallery();
      return;
    } catch (e) { console.warn("Simpan ke server gagal, unduh langsung", e); }
  }
  const a = document.createElement("a");
  a.href = gambar; a.download = nama;
  document.body.appendChild(a); a.click(); a.remove();
  toast("Foto diunduh: " + nama);
});

async function loadGallery() {
  const box = $("thumbs");
  try {
    const r = await fetch("/api/foto");
    const files = await r.json();
    if (!r.ok || !Array.isArray(files)) throw 0;
    hasServer = true;
    $("gallery").hidden = false; $("app").classList.remove("no-gallery");
    box.innerHTML = files.length
      ? files.map(f => `<a href="/hasil_foto/${f}" target="_blank" title="${f}"><img src="/hasil_foto/${f}" alt="${f}" loading="lazy"></a>`).join("")
      : `<p class="empty">Belum ada foto. Tekan "Ambil foto" untuk menyimpan hasil deteksi.</p>`;
  } catch {
    hasServer = false;
    $("gallery").hidden = true; $("app").classList.add("no-gallery");
  }
}
