# Import library yang dibutuhkan
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Aplikasi Prediksi Gaji (Rp)",
    page_icon="",
    layout="centered"
)

# --- Fungsi untuk Memuat & Melatih Model ---
@st.cache_data
def load_and_train_model():
    # Memuat dataset dari file CSV baru
    df = pd.read_csv('Gaji_Indonesia.csv') # <-- PERUBAHAN NAMA FILE
    
    # Memisahkan fitur (X) dan target (y)
    X = df[['YearsExperience']]
    y = df['Salary']
    
    # Membuat dan melatih model Regresi Linier
    model = LinearRegression()
    model.fit(X, y)
    
    return model, df, X, y

# --- Judul dan Deskripsi Aplikasi ---
st.title("🇮🇩 Aplikasi Prediksi Gaji di Indonesia")
st.markdown("""
Aplikasi ini menggunakan **Regresi Linier** untuk memprediksi gaji di Indonesia (dalam Rupiah) berdasarkan pengalaman kerja.
""")

# --- Memuat Data dan Melatih Model ---
with st.spinner('Memuat data dan melatih model...'):
    model, df, X, y = load_and_train_model()

# --- Tampilkan Dataset (Opsional) ---
with st.expander("Klik untuk melihat dataset yang digunakan"):
    # Memformat kolom gaji agar mudah dibaca
    formatted_df = df.copy()
    formatted_df['Salary'] = formatted_df['Salary'].apply(lambda x: f"Rp {x:,.0f}")
    st.dataframe(formatted_df, use_container_width=True)

# --- Visualisasi Data ---
st.header("1. Visualisasi Data")
st.markdown("Grafik di bawah ini menunjukkan hubungan antara Pengalaman Kerja dan Gaji.")

fig_scatter = px.scatter(
    df, x='YearsExperience', y='Salary',
    title="Hubungan Antara Pengalaman Kerja dan Gaji",
    labels={'YearsExperience': 'Pengalaman Kerja (Tahun)', 'Salary': 'Gaji (Rp)'}, # <-- PERUBAHAN LABEL
    template='plotly_white'
)
# Mengupdate format sumbu Y agar menampilkan format Rupiah
fig_scatter.update_layout(yaxis_tickprefix='Rp ', yaxis_tickformat=',.0f')
st.plotly_chart(fig_scatter, use_container_width=True)

# --- Informasi Model dan Garis Regresi ---
st.header("2. Model Regresi Linier")
st.markdown("Model machine learning telah menemukan 'garis lurus terbaik' yang paling sesuai dengan sebaran data untuk melakukan prediksi.")

intercept = model.intercept_
coefficient = model.coef_[0]

st.markdown("Formula garis regresi yang dihasilkan:")
# Menampilkan formula dengan format angka yang lebih sederhana
st.latex(f"Gaji = {coefficient:.0f} \\times (Pengalaman Kerja) + {intercept:.0f}")

st.markdown(f"- **Koefisien (Slope):** Setiap 1 tahun pengalaman, gaji diprediksi meningkat sebesar **Rp {coefficient:,.0f}**.") # <-- PERUBAHAN FORMAT
st.markdown(f"- **Intercept:** Gaji awal teoretis (0 tahun pengalaman) adalah **Rp {intercept:,.0f}**.") # <-- PERUBAHAN FORMAT

# Visualisasi Garis Regresi
fig_line = go.Figure()
fig_line.add_trace(go.Scatter(x=X['YearsExperience'], y=y, mode='markers', name='Data Aktual'))
fig_line.add_trace(go.Scatter(x=X['YearsExperience'], y=model.predict(X), mode='lines', name='Garis Regresi', line=dict(color='red', width=3)))
fig_line.update_layout(
    title="Garis Regresi Linier pada Data Latih",
    xaxis_title="Pengalaman Kerja (Tahun)",
    yaxis_title="Gaji (Rp)", # <-- PERUBAHAN LABEL
    yaxis_tickprefix='Rp ', yaxis_tickformat=',.0f', # <-- PERUBAHAN FORMAT AXIS
    template='plotly_white'
)
st.plotly_chart(fig_line, use_container_width=True)

# --- Antarmuka Prediksi Interaktif ---
st.header("3. Coba Prediksi Gaji Anda!")
st.markdown("Gunakan slider di bawah untuk memilih pengalaman kerja dan lihat estimasi gajinya.")

pengalaman = st.slider(
    "Pilih Pengalaman Kerja (Tahun):", 
    min_value=0.0, 
    max_value=20.0, 
    value=5.0,
    step=0.5
)

input_data = [[pengalaman]]
prediksi_gaji = model.predict(input_data)[0]

st.subheader("Hasil Estimasi:")
# <-- PERUBAHAN FORMAT HASIL PREDIKSI
st.success(f"Estimasi gaji untuk **{pengalaman} tahun** pengalaman adalah **Rp {prediksi_gaji:,.0f}**")

st.markdown("---")
st.write("Dibuat dengan semangat lokal 🇮🇩")
