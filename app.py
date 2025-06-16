import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb

# --- Load Model & Fitur ---
model = joblib.load('model_xgboost.pkl')
fitur_model = joblib.load('fitur_model.pkl')

# Daftar lokasi lengkap
lokasi_list = [
    'Andir, Bandung', 'Antapani, Bandung', 'Arcamanik, Bandung', 'Arjasari, Bandung',
    'Asia Afrika, Bandung', 'Astanaanyar, Bandung', 'Awiligar, Bandung', 'Babakanciparay, Bandung',
    'Baleendah, Bandung', 'Bandung Barat, Bandung', 'Bandung Kidul, Bandung', 'Bandung Kota, Bandung',
    'Bandung Kulon, Bandung', 'Bandung Selatan, Bandung', 'Bandung Timur, Bandung', 'Bandung Utara, Bandung',
    'Bandung Wetan, Bandung', 'Banjaran, Bandung', 'Batujajar, Bandung', 'Batununggal, Bandung',
    'Bkr, Bandung', 'Bojongloa Kidul, Bandung', 'Bojongloa, Bandung', 'Bojongsoang, Bandung',
    'Braga, Bandung', 'Buah Batu, Bandung', 'Burangrang, Bandung', 'Cangkuang, Bandung',
    'Caringin, Bandung', 'Ciateul, Bandung', 'Cibaduyut, Bandung', 'Cibeunying Kidul, Bandung',
    'Cibeunying, Bandung', 'Cibeureum, Bandung', 'Cibiru, Bandung', 'Cibogo, Bandung', 'Cicadas, Bandung',
    'Cicaheum, Bandung', 'Cicalengka, Bandung', 'Cicendo, Bandung', 'Cidadap, Bandung', 'Cigadung, Bandung',
    'Cigondewah, Bandung', 'Cihampelas, Bandung', 'Cihanjuang, Bandung', 'Cijagra, Bandung', 'Cijerah, Bandung',
    'Cikalong Wetan, Bandung', 'Cikancung, Bandung', 'Cikutra, Bandung', 'Cilengkrang, Bandung',
    'Cileunyi, Bandung', 'Cililin, Bandung', 'Cimahi, Bandung', 'Cimaung, Bandung', 'Cimenyan, Bandung',
    'Cimindi, Bandung', 'Cinambo, Bandung', 'Cipaganti, Bandung', 'Cipaku, Bandung', 'Ciparay, Bandung',
    'Cipedes, Bandung', 'Cipeundeuy, Bandung', 'Cisarua, Bandung', 'Ciumbuleuit, Bandung', 'Ciwastra, Bandung',
    'Ciwidey, Bandung', 'Coblong, Bandung', 'Dadali, Bandung', 'Dago Pakar, Bandung', 'Dago, Bandung',
    'Dayeuhkolot, Bandung', 'Derwati, Bandung', 'Diponegoro, Bandung', 'Gardu Jati, Bandung', 'Garuda, Bandung',
    'Gatot Subroto, Bandung', 'Gede Bage, Bandung', 'Geger Kalong, Bandung', 'Gunung Batu, Bandung',
    'Hegarmanah, Bandung', 'Holis Cigondewah, Bandung', 'Jatinangor, Bandung', 'Katapang, Bandung',
    'Kebon Kawung, Bandung', 'Kebonjati, Bandung', 'Kertasari, Bandung', 'Kiaracondong, Bandung',
    'Kopo Permai, Bandung', 'Kopo, Bandung', 'Kota Baru Parahyangan, Bandung', 'Kutawaringin, Bandung',
    'Laswi, Bandung', 'Lembang, Bandung', 'Lengkong, Bandung', 'Leuwi Panjang, Bandung', 'Lombok, Bandung',
    'Majalaya, Bandung', 'Mandalajati, Bandung', 'Margaasih, Bandung', 'Margacinta, Bandung',
    'Margahayu, Bandung', 'Mekar Wangi, Bandung', 'Moch Toha, Bandung', 'Muara, Bandung', 'Nagreg, Bandung',
    'Ngamprah, Bandung', 'Otista, Bandung', 'Padalarang, Bandung', 'Padasuka, Bandung', 'Pajajaran, Bandung',
    'Pameungpeuk, Bandung', 'Pangalengan, Bandung', 'Panyileukan, Bandung', 'Parongpong, Bandung',
    'Paseh, Bandung', 'Pasir Kaliki, Bandung', 'Pasir Koja, Bandung', 'Pasir Luyu, Bandung',
    'Pasirjambu, Bandung', 'Pasteur, Bandung', 'Pelajar Pejuang, Bandung', 'Peta, Bandung', 'Pinus, Bandung',
    'Podomoro Park Bandung, Bandung', 'Pondok Hijau, Bandung', 'Pungkur, Bandung', 'Rancabali, Bandung',
    'Rancaekek, Bandung', 'Rancamanyar, Bandung', 'Rancasari, Bandung', 'Regol, Bandung',
    'Resor Dago Pakar, Bandung', 'Riau, Bandung', 'Sarijadi, Bandung', 'Sariwangi, Bandung',
    'Sayap Dago, Bandung', 'Setiabudi, Bandung', 'Setra Duta, Bandung', 'Setra Indah, Bandung',
    'Setra Murni, Bandung', 'Setra Sari, Bandung', 'Singgasana, Bandung', 'Soekarno Hatta, Bandung',
    'Solokan Jeruk, Bandung', 'Soreang, Bandung', 'Suci, Bandung', 'Sudirman, Bandung', 'Sukahaji, Bandung',
    'Sukajadi, Bandung', 'Sukaluyu, Bandung', 'Sukasari, Bandung', 'Summarecon Bandung, Bandung',
    'Sumurbandung, Bandung', 'Supratman, Bandung', 'Surapati, Bandung', 'Surya Sumantri, Bandung',
    'Talaga Bodas, Bandung', 'Taman Kopo Indah, Bandung', 'Tegalega, Bandung', 'Terusan Buah Batu, Bandung',
    'Tubagus Ismail, Bandung', 'Turangga, Bandung', 'Ujungberung, Bandung', 'Veteran, Bandung',
    'Wastukencana, Bandung'
]

# --- Judul ---
st.title("Prediksi Harga Rumah 🏠")

# --- Input User ---
st.header("Input Fitur Rumah")

kamar_tidur = st.number_input("Jumlah Kamar Tidur", min_value=0, step=1)
kamar_mandi = st.number_input("Jumlah Kamar Mandi", min_value=0, step=1)
garasi = st.number_input("Jumlah Garasi", min_value=0, step=1)
luas_tanah = st.number_input("Luas Tanah (m²)", min_value=0, step=1)
luas_bangunan = st.number_input("Luas Bangunan (m²)", min_value=0, step=1)

lokasi = st.selectbox("Lokasi", sorted(lokasi_list))
waktu_penjualan = st.text_input("Waktu Penjualan (format: YYYY-MM)")

# --- Fungsi Preprocessing ---
def preprocess_input_streamlit():
    data = {
        'kamar_tidur': kamar_tidur,
        'kamar_mandi': kamar_mandi,
        'garasi': garasi,
        'luas_tanah': luas_tanah,
        'luas_bangunan': luas_bangunan,
        'lokasi': lokasi,
        'waktu_penjualan': waktu_penjualan
    }

    df = pd.DataFrame([data])
    df['lokasi'] = df['lokasi'].astype(str)
    df['waktu_penjualan'] = df['waktu_penjualan'].astype(str)

    df_encoded = pd.get_dummies(df, columns=['lokasi', 'waktu_penjualan'], drop_first=True)

    # Tambahkan kolom yang hilang
    missing_cols = set(fitur_model) - set(df_encoded.columns)
    for col in missing_cols:
        df_encoded[col] = 0

    # Urutkan kolom sesuai model
    df_encoded = df_encoded[fitur_model]
    return df_encoded

# --- Prediksi ---
if st.button("Prediksi Harga"):
    try:
        input_data = preprocess_input_streamlit()
        prediksi = model.predict(input_data)[0]
        st.success(f"💰 Prediksi Harga Rumah: Rp {int(prediksi):,}")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")