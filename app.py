"""
Lensa Deteksi - server Python (Flask)

Jalankan:
    pip install -r requirements.txt
    python app.py

Lalu buka http://localhost:5000 (otomatis terbuka di browser).
Deteksi AI berjalan di browser (MediaPipe), Python menyajikan halaman
dan menyimpan foto hasil tangkapan ke folder "hasil_foto".
"""
import base64
import os
import socket
import sys
import threading
import webbrowser
from datetime import datetime

from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")          # folder yang sama dipakai untuk Vercel
FOTO_DIR = os.path.join(BASE_DIR, "hasil_foto")
os.makedirs(FOTO_DIR, exist_ok=True)

app = Flask(__name__, static_folder=WEB_DIR, static_url_path="")
PORT = 5000


@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.post("/api/foto")
def simpan_foto():
    """Terima gambar (base64 PNG) + daftar objek dari browser, simpan ke disk."""
    data = request.get_json(silent=True) or {}
    gambar = data.get("gambar", "")
    if not gambar.startswith("data:image/"):
        return jsonify(ok=False, pesan="Data gambar tidak valid"), 400

    _, isi = gambar.split(",", 1)
    nama = datetime.now().strftime("deteksi_%Y%m%d_%H%M%S.jpg")
    with open(os.path.join(FOTO_DIR, nama), "wb") as f:
        f.write(base64.b64decode(isi))

    objek = data.get("objek", [])
    with open(os.path.join(FOTO_DIR, "log.txt"), "a", encoding="utf-8") as log:
        log.write(f"{nama}: {', '.join(objek) if objek else '-'}\n")

    print(f"[foto] {nama}  objek: {objek}")
    return jsonify(ok=True, nama=nama)


@app.get("/api/foto")
def daftar_foto():
    files = sorted((f for f in os.listdir(FOTO_DIR) if f.endswith(".jpg")), reverse=True)
    return jsonify(files[:12])


@app.get("/hasil_foto/<path:nama>")
def lihat_foto(nama):
    return send_from_directory(FOTO_DIR, nama)


def ip_lokal():
    """Cari IP laptop di jaringan WiFi (mis. 192.168.1.5)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


if __name__ == "__main__":
    mode_hp = "--hp" in sys.argv

    if mode_hp:
        # HTTPS wajib supaya kamera HP diizinkan browser.
        # Sertifikat dibuat otomatis (self-signed), jadi HP akan memberi peringatan sekali.
        ip = ip_lokal()
        print("\n  Lensa Deteksi - MODE HP")
        print(f"  Laptop : https://localhost:{PORT}")
        print(f"  HP     : https://{ip}:{PORT}   <- ketik ini di Chrome HP")
        print("  HP & laptop harus di WiFi yang sama.")
        print("  Kalau muncul 'Koneksi tidak aman': Lanjutan > Lanjutkan ke situs.")
        print("  Tekan Ctrl+C untuk berhenti.\n")
        app.run(host="0.0.0.0", port=PORT, debug=False, ssl_context="adhoc")
    else:
        url = f"http://localhost:{PORT}"
        print(f"\n  Lensa Deteksi berjalan di {url}")
        print("  Mau dibuka di HP? Jalankan: python app.py --hp")
        print("  Tekan Ctrl+C untuk berhenti.\n")
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
        app.run(host="127.0.0.1", port=PORT, debug=False)
