// Lensa Deteksi - logika kamera + AI (MediaPipe Tasks Vision)
import { FilesetResolver, ObjectDetector, HandLandmarker }
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

/* ================= Konfigurasi ================= */
const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MP = "https://storage.googleapis.com/mediapipe-models/object_detector/";
// Mode "akurat" = EfficientDet Lite2 (lebih teliti, lebih berat). Mode "cepat" = Lite0.
// Tiap mode punya beberapa alamat cadangan; yang pertama berhasil dipakai.
const MODELS = {
  akurat: [MP + "efficientdet_lite2/float16/latest/efficientdet_lite2.tflite",
           MP + "efficientdet_lite2/float32/latest/efficientdet_lite2.tflite",
           MP + "efficientdet_lite2/int8/latest/efficientdet_lite2.tflite"],
  cepat:  [MP + "efficientdet_lite0/float16/1/efficientdet_lite0.tflite"]
};
const MODEL_HAND = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

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

/* ================= State ================= */
const $ = id => document.getElementById(id);
const cv = $("cv"), ctx = cv.getContext("2d"), vid = $("vid");
let detector = null, hands = null, stream = null, facing = "user";
let showBox = true, showHand = true;
let vision = null, delegateUsed = "GPU", mode = "akurat";
// HP: pakai pengaturan ringan (model cepat, resolusi kecil, tanpa efek blur yang berat)
const IS_MOBILE = /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent) || matchMedia("(pointer: coarse)").matches;
const GLOW = !IS_MOBILE;                       // shadowBlur sangat lambat di canvas HP
const OBJ_INTERVAL = IS_MOBILE ? 300 : 120;    // jeda deteksi objek (ms); tangan diproses di sela-selanya
mode = IS_MOBILE ? "cepat" : "akurat";
try { mode = localStorage.getItem("lensa-mode") || mode; } catch {}
let objs = [], handPts = [], lastVidTime = -1, lastObjT = 0;
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

/* ================= Model AI ================= */
async function loadModels() {
  vision = await FilesetResolver.forVisionTasks(WASM);
  try {
    hands = await makeHands("GPU");
  } catch (e) {
    console.warn("GPU tidak didukung, pindah ke CPU", e);
    delegateUsed = "CPU";
    hands = await makeHands("CPU");
  }
  detector = await makeDetector(mode);
}
function makeHands(delegate) {
  return HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_HAND, delegate },
    runningMode: "VIDEO", numHands: 2
  });
}
async function makeDetector(m) {
  // coba model pilihan; kalau semua alamatnya gagal, turun ke mode cepat
  const urls = [...MODELS[m], ...(m === "akurat" ? MODELS.cepat : [])];
  let lastErr;
  for (const url of urls) {
    try {
      const d = await ObjectDetector.createFromOptions(vision, {
        baseOptions: { modelAssetPath: url, delegate: delegateUsed },
        runningMode: "VIDEO", scoreThreshold: 0.5, maxResults: 10
      });
      if (url.includes("lite0") && m === "akurat") { mode = "cepat"; updateModeBtn(); }
      return d;
    } catch (e) { lastErr = e; console.warn("Model gagal:", url, e); }
  }
  throw lastErr;
}

function updateModeBtn() {
  $("tMode").textContent = mode === "akurat" ? "Mode: Akurat" : "Mode: Cepat";
  $("tMode").title = mode === "akurat"
    ? "Lebih teliti, sedikit lebih berat. Klik untuk mode cepat."
    : "Lebih ringan, kurang teliti untuk benda kecil. Klik untuk mode akurat.";
}

function setAI(text, cls) { const el = $("aiStatus"); el.textContent = text; el.className = "pill " + cls; }

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
  try { await loadModels(); setAI("AI aktif", "ok"); }
  catch (e) { console.error(e); setAI("AI gagal, cek internet", "err"); }
});

/* ================= Loop utama ================= */
function loop(now) {
  requestAnimationFrame(loop);
  const W = cv.width, H = cv.height, mirror = facing === "user";

  // AI hanya diproses saat kamera mengirim frame BARU, dan cuma SATU model per frame:
  // objek tiap OBJ_INTERVAL ms, sisanya untuk tangan. Video tetap digambar setiap frame layar.
  if (detector && hands && vid.readyState >= 2 && vid.currentTime !== lastVidTime) {
    lastVidTime = vid.currentTime;
    try {
      if (now - lastObjT >= OBJ_INTERVAL) { objs = detector.detectForVideo(vid, now).detections; lastObjT = now; }
      else handPts = hands.detectForVideo(vid, now).landmarks;
      aiN++;
    } catch (e) { console.warn(e); }
  }

  ctx.save();
  if (mirror) { ctx.translate(W, 0); ctx.scale(-1, 1); }
  ctx.drawImage(vid, 0, 0, W, H);
  ctx.restore();

  if (showBox) drawBoxes(W, mirror);
  if (showHand) drawHands(W, H, mirror);

  $("no").textContent = objs.length;
  $("nh").textContent = handPts.length;
  fpsN++;
  if (now - fpsT > 500) {
    $("fps").textContent = Math.round(fpsN * 1000 / (now - fpsT));
    $("aifps").textContent = Math.round(aiN * 1000 / (now - fpsT));
    fpsT = now; fpsN = 0; aiN = 0;
    updateChips();
  }
}

/* ================= Gambar kotak objek ================= */
function drawBoxes(W, mirror) {
  const s = W / 640;
  for (const d of objs) {
    const c = d.categories[0], col = colorFor(c.categoryName), b = d.boundingBox;
    const x = mirror ? W - b.originX - b.width : b.originX, y = b.originY, w = b.width, h = b.height;

    ctx.fillStyle = col + "14"; ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = col; ctx.lineWidth = 1.5 * s; ctx.shadowColor = col; ctx.shadowBlur = GLOW ? 10 * s : 0;
    ctx.strokeRect(x, y, w, h);

    // sudut tebal ala viewfinder
    const L = Math.min(w, h) * .18; ctx.lineWidth = 5 * s; ctx.beginPath();
    for (const [px, py, dx, dy] of [[x,y,1,1],[x+w,y,-1,1],[x,y+h,1,-1],[x+w,y+h,-1,-1]]) {
      ctx.moveTo(px + dx * L, py); ctx.lineTo(px, py); ctx.lineTo(px, py + dy * L);
    }
    ctx.stroke(); ctx.shadowBlur = 0;

    const label = `${nameFor(c.categoryName).toUpperCase()}  ${Math.round(c.score * 100)}%`;
    ctx.font = `600 ${13 * s}px "JetBrains Mono", monospace`;
    const tw = ctx.measureText(label).width + 14 * s, th = 22 * s, ly = y > th ? y - th : y;
    ctx.fillStyle = col; ctx.fillRect(x, ly, tw, th);
    ctx.fillStyle = "#05070d"; ctx.fillText(label, x + 7 * s, ly + 15.5 * s);
  }
}

/* ================= Efek garis tangan (tren TikTok) ================= */
function drawHands(W, H, mirror) {
  const s = W / 640, t = performance.now() / 1000;
  const P = handPts.map(lm => lm.map(p => ({ x: (mirror ? 1 - p.x : p.x) * W, y: p.y * H })));
  ctx.lineCap = "round";

  for (const pts of P) {                                  // kerangka tangan
    ctx.strokeStyle = "rgba(232,241,255,.5)"; ctx.lineWidth = 2 * s; ctx.beginPath();
    for (const [a, b] of BONES) { ctx.moveTo(pts[a].x, pts[a].y); ctx.lineTo(pts[b].x, pts[b].y); }
    ctx.stroke();
    ctx.fillStyle = "rgba(232,241,255,.8)";
    for (const p of pts) { ctx.beginPath(); ctx.arc(p.x, p.y, 2.2 * s, 0, 7); ctx.fill(); }
  }

  if (P.length === 2) {
    // jari ke jari antar dua tangan
    TIPS.forEach((tip, i) => neonLine(P[0][tip], P[1][tip], TIP_COLORS[i], s, t + i));
    ctx.globalAlpha = .35;
    neonLine(P[0][4], P[1][8], "#ffffff", s * .5, t);
    neonLine(P[0][8], P[1][4], "#ffffff", s * .5, t + 1);
    ctx.globalAlpha = 1;
  } else if (P.length === 1) {
    // satu tangan: jempol ke tiap ujung jari
    TIPS.slice(1).forEach((tip, i) => neonLine(P[0][4], P[0][tip], TIP_COLORS[i + 1], s * .8, t + i));
  }

  for (const pts of P) TIPS.forEach((tip, i) => {          // ujung jari bercahaya
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
    // glow murah: garis lebar transparan + garis warna, tanpa blur
    const a = ctx.globalAlpha;
    ctx.strokeStyle = col; ctx.globalAlpha = a * .25; ctx.lineWidth = 14 * s; path();
    ctx.globalAlpha = a; ctx.lineWidth = 5 * s; path();
  }
  ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1.5 * s; path();
}

/* ================= UI ================= */
function updateChips() {
  const count = {};
  for (const d of objs) { const n = d.categories[0].categoryName; count[n] = (count[n] || 0) + 1; }
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
$("tMode").addEventListener("click", async () => {
  if (!vision) return;
  const btn = $("tMode"), next = mode === "akurat" ? "cepat" : "akurat";
  btn.disabled = true; setAI("Ganti model…", "loading");
  const prev = mode;
  mode = next;                                   // makeDetector bisa menurunkannya ke "cepat"
  try {
    const d = await makeDetector(next);
    const old = detector; detector = d; old?.close();
    try { localStorage.setItem("lensa-mode", mode); } catch {}
    objs = []; updateModeBtn(); setAI("AI aktif", "ok");
    toast(mode !== next ? "Mode akurat tidak tersedia, tetap mode cepat"
        : mode === "akurat" ? "Mode akurat aktif" : "Mode cepat aktif");
  } catch (e) {
    console.error(e); mode = prev; updateModeBtn(); setAI("AI aktif", "ok"); toast("Gagal ganti model, cek internet");
  }
  btn.disabled = false;
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
        body: JSON.stringify({ gambar, objek: objs.map(d => nameFor(d.categories[0].categoryName)) })
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
