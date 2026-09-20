import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.joblib")
selected_features = joblib.load("selected_features.joblib")

st.set_page_config(page_title="Prediksi Dropout Siswa - Jaya Jaya Institut", page_icon="🎓", layout="centered")

st.title("🎓 Prediksi Risiko Dropout Siswa - Jaya Jaya Institut")
st.write(
    "Masukkan data akademik dan administratif siswa untuk memprediksi kemungkinan "
    "siswa tersebut **Dropout** atau **Graduate** (lulus). Cocok dipakai untuk siswa yang "
    "masih aktif berkuliah (belum lulus/keluar), sebagai deteksi dini risiko dropout."
)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Performa Akademik**")
        curricular_2nd_approved = st.number_input("Jumlah mata kuliah lulus - semester 2", min_value=0, max_value=20, value=5)
        curricular_2nd_grade = st.number_input("Rata-rata nilai semester 2 (skala 0-20)", min_value=0.0, max_value=20.0, value=12.0, step=0.1)
        curricular_2nd_eval = st.number_input("Jumlah evaluasi diikuti - semester 2", min_value=0, max_value=33, value=6)
        curricular_1st_approved = st.number_input("Jumlah mata kuliah lulus - semester 1", min_value=0, max_value=26, value=5)
        curricular_1st_grade = st.number_input("Rata-rata nilai semester 1 (skala 0-20)", min_value=0.0, max_value=20.0, value=12.0, step=0.1)
        curricular_1st_eval = st.number_input("Jumlah evaluasi diikuti - semester 1", min_value=0, max_value=45, value=6)

    with col2:
        st.markdown("**Data Penerimaan & Administrasi**")
        admission_grade = st.number_input("Nilai penerimaan / admission grade (skala 0-200)", min_value=0.0, max_value=200.0, value=130.0, step=0.1)
        previous_qualification_grade = st.number_input("Nilai kualifikasi sebelumnya (skala 0-200)", min_value=0.0, max_value=200.0, value=130.0, step=0.1)
        age = st.number_input("Usia saat mendaftar", min_value=17, max_value=70, value=20)
        tuition = st.selectbox("Status pembayaran uang kuliah", ["Sudah lunas", "Belum lunas"])
        scholarship = st.selectbox("Penerima beasiswa?", ["Tidak", "Ya"])
        debtor = st.selectbox("Memiliki tunggakan (debtor)?", ["Tidak", "Ya"])

    submitted = st.form_submit_button("Prediksi Risiko Dropout")

if submitted:
    input_data = pd.DataFrame([{
        "Curricular_units_2nd_sem_approved": curricular_2nd_approved,
        "Curricular_units_2nd_sem_grade": curricular_2nd_grade,
        "Curricular_units_1st_sem_approved": curricular_1st_approved,
        "Curricular_units_1st_sem_grade": curricular_1st_grade,
        "Admission_grade": admission_grade,
        "Tuition_fees_up_to_date": 1 if tuition == "Sudah lunas" else 0,
        "Age_at_enrollment": age,
        "Curricular_units_2nd_sem_evaluations": curricular_2nd_eval,
        "Previous_qualification_grade": previous_qualification_grade,
        "Curricular_units_1st_sem_evaluations": curricular_1st_eval,
        "Scholarship_holder": 1 if scholarship == "Ya" else 0,
        "Debtor": 1 if debtor == "Ya" else 0,
    }])[selected_features]

    prediction = model.predict(input_data)[0]
    prob_dropout = model.predict_proba(input_data)[0][1]

    st.subheader("Hasil Prediksi")

    if prediction == 1:
        st.error(f"⚠️ Prediksi: **Berisiko Dropout** (probabilitas {prob_dropout:.1%}). Disarankan diberi bimbingan khusus.")
    else:
        st.success(f"🎓 Prediksi: **Kemungkinan Graduate** (probabilitas dropout hanya {prob_dropout:.1%}).")

    st.write("**Probabilitas Dropout:**")
    proba_df = pd.DataFrame({
        "Status": ["Graduate", "Dropout"],
        "Probabilitas": [1 - prob_dropout, prob_dropout],
    })
    st.bar_chart(proba_df.set_index("Status"))
    st.dataframe(
        proba_df.assign(Probabilitas=lambda d: (d["Probabilitas"] * 100).round(1).astype(str) + "%"),
        hide_index=True,
        use_container_width=True,
    )

st.divider()
st.caption(
    "Model: Random Forest Classifier (klasifikasi biner Dropout vs Graduate), dilatih pada dataset "
    "\"Predict Students' Dropout and Academic Success\" (UCI Machine Learning Repository). "
    "Siswa berstatus Enrolled tidak digunakan untuk melatih model karena belum memiliki hasil akhir."
)
