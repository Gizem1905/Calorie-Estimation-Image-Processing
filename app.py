import cv2
import numpy as np
import os
from flask import Flask, render_template, Response

app = Flask(__name__)

# --- 1. OPTİMİZE EDİLMİŞ KALORİ VE RENK VERİTABANI ---
# Ten rengini ve arka planı elemek için Doygunluk (Saturation) alt sınırları artırıldı.
KALORI_VERITABANI = {
    "Elma (Kirmizi)": {"kalori": 95, "hsv_alt": np.array([0, 160, 100]), "hsv_ust": np.array([10, 255, 255])},
    "Elma (Yesil)": {"kalori": 95, "hsv_alt": np.array([35, 80, 70]), "hsv_ust": np.array([85, 255, 255])}, 
    "Muz": {"kalori": 105, "hsv_alt": np.array([22, 130, 130]), "hsv_ust": np.array([33, 255, 255])},
    "Portakal": {"kalori": 62, "hsv_alt": np.array([10, 180, 150]), "hsv_ust": np.array([20, 255, 255])},
    "Patates": {"kalori": 77, "hsv_alt": np.array([10, 30, 50]), "hsv_ust": np.array([25, 120, 160])},
    "Domates": {"kalori": 20, "hsv_alt": np.array([0, 160, 60]), "hsv_ust": np.array([8, 255, 255])},
    "Salatalik": {"kalori": 15, "hsv_alt": np.array([40, 60, 40]), "hsv_ust": np.array([80, 255, 255])},
    "Havuc": {"kalori": 25, "hsv_alt": np.array([5, 180, 100]), "hsv_ust": np.array([15, 255, 255])},
    "Cilek": {"kalori": 5, "hsv_alt": np.array([0, 160, 100]), "hsv_ust": np.array([5, 255, 255])},
    "Kiraz": {"kalori": 5, "hsv_alt": np.array([170, 160, 50]), "hsv_ust": np.array([180, 255, 255])},
    "Ananas": {"kalori": 50, "hsv_alt": np.array([20, 120, 100]), "hsv_ust": np.array([30, 255, 255])},
    "Pizza": {"kalori": 285, "hsv_alt": np.array([10, 80, 80]), "hsv_ust": np.array([25, 255, 255])},
    "Hamburger": {"kalori": 250, "hsv_alt": np.array([10, 50, 20]), "hsv_ust": np.array([20, 150, 200])}
}

# --- 2. ŞABLONLARI YÜKLEME ---
TUM_SABLONLAR = []
dosya_adlari = [
    ("Elma (Kirmizi)", 'kirmizi.png'), ("Elma (Kirmizi)", 'yarim.png'), ("Elma (Kirmizi)", 'ustten.png'),
    ("Elma (Yesil)", 'yesil.png'), ("Muz", 'muz_tek.png'), ("Muz", 'muz_cok.png'),
    ("Portakal", 'portakal_yan.png'), ("Domates", 'domates.png'), ("Salatalik", 'salatalik.png'),
    ("Havuc", 'havuc.png'), ("Patates", 'patates.png'), ("Pizza", 'pizza.png'),
    ("Hamburger", 'hamburger.png'), ("Ananas", 'ananas.png'), ("Cilek", 'cilek.png'), ("Kiraz", 'kiraz.png')
]

for nesne_adi, dosya_adi in dosya_adlari:
    tam_yol = os.path.join(os.getcwd(), dosya_adi)
    sablon = cv2.imread(tam_yol, cv2.IMREAD_GRAYSCALE)
    if sablon is not None:
        TUM_SABLONLAR.append((nesne_adi, sablon))

def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): return

    # Filtreleme için kernel
    kernel = np.ones((5, 5), np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        gri_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        tespit_edilen_nesne = "Bilinmiyor"
        yontem = "Renk"
        max_val_global = 0

        # 1. Şablon Eşleştirme (Önce Şekle Bak)
        if TUM_SABLONLAR:
            for nesne_adi, sablon in TUM_SABLONLAR:
                w, h = sablon.shape[::-1]
                if gri_frame.shape[0] < h or gri_frame.shape[1] < w: continue
                res = cv2.matchTemplate(gri_frame, sablon, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val > 0.70 and max_val > max_val_global: # Eşik 0.70'e çıkarıldı
                    max_val_global = max_val
                    tespit_edilen_nesne = nesne_adi
                    yontem = "Sablon"

        # 2. Renk Tespiti (Şablon yoksa veya başarısızsa)
        if tespit_edilen_nesne == "Bilinmiyor":
            for nesne, veri in KALORI_VERITABANI.items():
                mask = cv2.inRange(hsv, veri["hsv_alt"], veri["hsv_ust"])
                
                # Arka plan gürültü temizliği (Morfolojik Açınım)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    # Alan eşiği 4000'e çıkarıldı (Yüzünü ve küçük lekeleri elemek için)
                    if cv2.contourArea(c) > 4000:
                        tespit_edilen_nesne = nesne
                        yontem = "Renk"
                        break

        # 3. Sonuçları Çiz ve Yaz
        nesne_sayisi = 0
        if tespit_edilen_nesne != "Bilinmiyor":
            v = KALORI_VERITABANI[tespit_edilen_nesne]
            mask_final = cv2.inRange(hsv, v["hsv_alt"], v["hsv_ust"])
            mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, kernel)
            cnts, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in cnts:
                if cv2.contourArea(cnt) > 4000:
                    nesne_sayisi += 1
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            toplam_kalori = nesne_sayisi * v["kalori"]
            sonuc_metni = f"{nesne_sayisi} ADET {tespit_edilen_nesne} | KALORI: {toplam_kalori} kcal ({yontem})"
            cv2.putText(frame, sonuc_metni, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "NESNE BEKLENIYOR...", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, threaded=False)