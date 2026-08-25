import cv2
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Trafik İşareti Tanıma", layout="centered")


# Modeli önbelleğe alma
@st.cache_resource
def load_traffic_model():
  return load_model("traffic_sign_model.keras")


model = load_traffic_model()

# GTSRB Resmi Standart Sınıf Listesi (0 - 42)
classes = {
    0: "Hız Sınırı (20km/s)",
    1: "Hız Sınırı (30km/s)",
    2: "Hız Sınırı (50km/s)",
    3: "Hız Sınırı (60km/s)",
    4: "Hız Sınırı (70km/s)",
    5: "Hız Sınırı (80km/s)",
    6: "Hız Sınırı Sonu (80km/s)",
    7: "Hız Sınırı (100km/s)",
    8: "Hız Sınırı (120km/s)",
    9: "Geçme Yasağı",
    10: "3.5 Ton Üstü Araçlar İçin Geçme Yasağı",
    11: "Kavşakta Geçiş Hakkı",
    12: "Ana Yol (Öncelikli Yol)",
    13: "Yol Ver",
    14: "Dur",
    15: "Taşıt Giremez",
    16: "3.5 Ton Üstü Taşıt Giremez",
    17: "Girişi Olmayan Yol (Ters Yön)",
    18: "Genel Tehlike Uyarı",
    19: "Sola Tehlikeli Viraj",
    20: "Sağa Tehlikeli Viraj",
    21: "Tehlikeli Devamlı Virajlar",
    22: "Kasisli / Engebeli Yol",
    23: "Kaygan Yol",
    24: "Sağdan Daralan Yol",
    25: "Yol Çalışması",
    26: "Trafik Işıkları",
    27: "Yaya Geçidi",
    28: "Okul Geçidi / Çocuklar",
    29: "Bisiklet Geçebilir",
    30: "Gizli Buzlanma / Don",
    31: "Vahşi Hayvan Geçebilir",
    32: "Tüm Hız ve Geçiş Yasaklarının Sonu",
    33: "Sadece Sağa Dönüş",
    34: "Sadece Sola Dönüş",
    35: "Sadece İleri Mecburi Yön",
    36: "İleri veya Sağa Mecburi Yön",
    37: "İleri veya Sola Mecburi Yön",
    38: "Sağdan Gidiniz",
    39: "Soldan Gidiniz",
    40: "Dönel Kavşak",
    41: "Geçme Yasağı Sonu",
    42: "3.5 Ton Üstü Araçlar İçin Geçme Yasağı Sonu",
}

st.title(" Trafik İşareti Sınıflandırma")
st.write(
    "Bir trafik işareti görseli yükleyin (en iyi sonuç için görselin sadece"
    " tabelaya odaklandığından emin olun)."
)

uploaded_file = st.file_uploader(
    "Bir görsel seçin", type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
  image = Image.open(uploaded_file)
  st.image(image, caption="Yüklenen Görsel", use_container_width=True)

  if st.button("İşareti Analiz Et"):
    with st.spinner("Model tahmin yapıyor..."):
      try:
        # PIL Görselini RGB olarak alma
        img_rgb = np.array(image.convert("RGB"))

        # Model eğitiminde cv2.imread kullanıldığı için BGR formatına çeviriyoruz
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # 32x32 boyutlandırma ve normalizasyon
        img_resized = cv2.resize(img_bgr, (32, 32))
        img_normalized = img_resized.astype("float32") / 255.0

        # Batch boyutu ekleme: (1, 32, 32, 3)
        img_batch = np.expand_dims(img_normalized, axis=0)

        # Tahmin alma
        predictions = model.predict(img_batch)
        class_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions)) * 100

        st.success("Analiz tamamlandı!")
        st.subheader(" Tahmin Sonucu")
        st.info(f"**Tabela Sınıfı (ID: {class_index}):** {classes[class_index]}")
        st.write(f"**Güven Skoru (Eminlik):** %{confidence:.2f}")

      except Exception as e:
        st.error(f"Hata oluştu: {e}")