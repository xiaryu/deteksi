# Lensa Deteksi

Deteksi manusia, hewan, dan benda secara real-time dengan kotak berlabel, plus efek garis neon antar ujung jari (tren TikTok).

## Struktur

```
lensa-deteksi/
├── app.py              server Flask + API simpan foto (untuk laptop)
├── requirements.txt
├── web/                tampilan website (dipakai laptop DAN Vercel)
│   ├── index.html
│   ├── style.css
│   └── app.js          kamera, deteksi AI, efek garis tangan
└── hasil_foto/         foto dari tombol "Ambil foto" (dibuat otomatis)
```

## Jalan di laptop

```
python -m pip install -r requirements.txt
python app.py          # buka http://localhost:5000
python app.py --hp     # mode HP di WiFi yang sama (HTTPS)
```

Foto tersimpan di folder `hasil_foto` dan muncul di galeri samping.

## Online lewat Vercel

Yang di-deploy hanya folder `web`. Di Vercel tidak ada server Python, jadi tombol "Ambil foto" langsung mengunduh gambar ke perangkat dan galeri samping disembunyikan.

Cara 1, Vercel CLI (perlu Node.js):
```
npm install -g vercel
cd web
vercel --prod
```

Cara 2, lewat GitHub: upload proyek ke GitHub, buka vercel.com → Add New → Project → pilih repo,
lalu isi **Root Directory = `web`**, Framework Preset = **Other**, klik Deploy.
