# Import library yang dibutuhkan
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Aplikasi Prediksi Gaji",
    page_icon="💼",
    layout="centered"
)

# --- Fungsi untuk Memuat & Melatih Model ---
# Menggunakan cache agar model tidak perlu dilatih ulang setiap kali ada interaksi pengguna
@st.cache_data
def load_and_train_model():
    # Memuat dataset dari file CSV
    df = pd.read_csv('Salary_Data.csv')
    
    # Memisahkan fitur (X) dan target (y)
    # Fitur: YearsExperience, Target: Salary
    X = df[['YearsExperience']]
    y = df['Salary']
    
    # Membuat instance model Regresi Linier
    model = LinearRegression()
    
    # Melatih model dengan data yang ada
    model.fit(X, y)
    
    # Mengembalikan model yang sudah dilatih dan dataframe
    return model, df, X, y

# --- Judul dan Deskripsi Aplikasi ---
st.title("Aplikasi Prediksi Gaji Menggunakan Regresi Linier")
st.markdown("""
Selamat datang di Aplikasi Prediksi Gaji! 
Aplikasi ini menggunakan algoritma **Regresi Linier** untuk memprediksi besaran gaji berdasarkan lama pengalaman kerja seseorang.
""")

# --- Memuat Data dan Melatih Model ---
# Menampilkan spinner saat proses berjalan
with st.spinner('Memuat data dan melatih model, mohon tunggu...'):
    model, df, X, y = load_and_train_model()

# --- Tampilkan Dataset (Opsional) ---
# Menggunakan expander agar tidak memakan tempat
with st.expander("Klik untuk melihat dataset yang digunakan"):
    st.dataframe(df, use_container_width=True)

# --- Visualisasi Data ---
st.header("1. Visualisasi Data")
st.markdown("Grafik di bawah ini menunjukkan hubungan antara Pengalaman Kerja (sumbu-X) dan Gaji (sumbu-Y). Terlihat ada tren positif, di mana semakin lama pengalaman kerja, semakin tinggi gajinya.")

# Membuat scatter plot menggunakan Plotly Express
fig_scatter = px.scatter(
    df, x='YearsExperience', y='Salary',
    title="Hubungan Antara Pengalaman Kerja dan Gaji",
    labels={'YearsExperience': 'Pengalaman Kerja (Tahun)', 'Salary': 'Gaji ($)'},
    template='plotly_white'
)
st.plotly_chart(fig_scatter, use_container_width=True)

# --- Informasi Model dan Garis Regresi ---
st.header("2. Model Regresi Linier")
st.markdown("Model machine learning telah dilatih untuk menemukan 'garis lurus terbaik' yang paling sesuai dengan sebaran data di atas. Garis ini kemudian digunakan untuk melakukan prediksi.")

# Menampilkan formula regresi yang ditemukan oleh model
intercept = model.intercept_
coefficient = model.coef_[0]

st.markdown("Garis regresi yang dihasilkan oleh model memiliki formula:")
st.latex(f"Gaji = {coefficient:.2f} \\times (Pengalaman Kerja) + {intercept:.2f}")

# Visualisasi Garis Regresi
st.markdown("**Visualisasi Garis Regresi pada Data**")
fig_line = go.Figure()
# Menambahkan titik-titik data (scatter)
fig_line.add_trace(go.Scatter(x=X['YearsExperience'], y=y, mode='markers', name='Data Aktual'))
# Menambahkan garis regresi (line)
fig_line.add_trace(go.Scatter(x=X['YearsExperience'], y=model.predict(X), mode='lines', name='Garis Regresi (Prediksi)', line=dict(color='red', width=3)))
fig_line.update_layout(
    title="Garis Regresi Linier pada Data Latih",
    xaxis_title="Pengalaman Kerja (Tahun)",
    yaxis_title="Gaji ($)",
    template='plotly_white'
)
st.plotly_chart(fig_line, use_container_width=True)

# --- Antarmuka Prediksi Interaktif ---
st.header("3. Coba Prediksi Sendiri!")
st.markdown("Gunakan slider di bawah ini untuk memilih tahun pengalaman kerja dan lihat prediksi gaji yang dihasilkan oleh model.")

# Slider untuk input dari pengguna
pengalaman = st.slider(
    "Pilih Pengalaman Kerja (Tahun):", 
    min_value=0.0, 
    max_value=20.0, 
    value=5.0,  # Nilai default saat pertama kali dimuat
    step=0.5
)

# Melakukan prediksi berdasarkan input slider
input_data = [[pengalaman]]
prediksi_gaji = model.predict(input_data)[0]

# Menampilkan hasil prediksi dengan format yang menarik
st.subheader("Hasil Prediksi:")
st.success(f"Prediksi gaji untuk **{pengalaman} tahun** pengalaman adalah **$ {prediksi_gaji:,.2f}**")

st.markdown("---")
st.write("Dibuat dengan Python, Streamlit, dan Scikit-learn.")