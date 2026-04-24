"""
app.py — Streamlit Deployment (Monolithic)
==========================================
Muat model .pkl dari exp/placement/ dan tampilkan UI prediksi.

Jalankan:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

EXP_PATH = '/exp/placement/'

st.set_page_config(
    page_title='Student Placement Predictor',
    page_icon='🎓',
    layout='wide',
)

# ─────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────

@st.cache_resource
def load_artifacts():
    artifacts = {}
    files = {
        'clf_model'     : 'best_clf_pipeline.pkl',
        'reg_model'     : 'best_reg_pipeline.pkl',
        'bin_enc_clf'   : 'bin_enc_dict.pkl',
        'bin_enc_reg'   : 'bin_enc_dict_reg.pkl',
        'clf_feat'      : 'clf_feature_cols.pkl',
        'reg_feat'      : 'reg_feature_cols.pkl',
    }
    for key, fname in files.items():
        path = os.path.join(EXP_PATH, fname)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                artifacts[key] = pickle.load(f)
        else:
            artifacts[key] = None
    return artifacts


arts = load_artifacts()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.title('🎓 Student Placement Predictor')
st.markdown(
    'Masukkan data mahasiswa untuk memprediksi **status penempatan kerja** dan '
    '**estimasi gaji (LPA)**.'
)

if arts['clf_model'] is None:
    st.error(
        '⚠️ Model belum tersedia. Jalankan `python pipeline.py` terlebih dahulu '
        'untuk melatih dan menyimpan model.'
    )
    st.stop()

st.success('✅ Model berhasil dimuat dari `exp/placement/`')
st.divider()

# ─────────────────────────────────────────────
# SIDEBAR — INPUT FORM
# ─────────────────────────────────────────────

with st.sidebar:
    st.header('📋 Data Mahasiswa')

    gender          = st.selectbox('Gender',          ['Male', 'Female'])
    branch          = st.selectbox('Branch',          ['CSE', 'ECE', 'IT', 'ME', 'CE'])
    cgpa            = st.slider('CGPA',               min_value=5.0, max_value=10.0, value=7.5, step=0.01)
    tenth_pct       = st.slider('10th Percentage',    min_value=40.0, max_value=100.0, value=75.0, step=0.1)
    twelfth_pct     = st.slider('12th Percentage',    min_value=40.0, max_value=100.0, value=75.0, step=0.1)
    backlogs        = st.number_input('Backlogs',      min_value=0, max_value=10, value=0)
    study_hours     = st.slider('Study Hours/Day',    min_value=0.0, max_value=12.0, value=4.0, step=0.5)
    attendance      = st.slider('Attendance (%)',     min_value=50.0, max_value=100.0, value=80.0, step=0.5)
    projects        = st.number_input('Projects Completed',   min_value=0, max_value=20, value=3)
    internships     = st.number_input('Internships Completed', min_value=0, max_value=5,  value=1)
    coding_skill    = st.slider('Coding Skill Rating',     min_value=1, max_value=10, value=6)
    comm_skill      = st.slider('Communication Skill',     min_value=1, max_value=10, value=6)
    apt_skill       = st.slider('Aptitude Skill Rating',   min_value=1, max_value=10, value=6)
    hackathons      = st.number_input('Hackathons Participated', min_value=0, max_value=20, value=2)
    certifications  = st.number_input('Certifications Count',   min_value=0, max_value=20, value=2)
    sleep_hours     = st.slider('Sleep Hours/Day',    min_value=3.0, max_value=10.0, value=7.0, step=0.5)
    stress_level    = st.selectbox('Stress Level',   [1, 2, 3], format_func=lambda x: {1:'Low',2:'Medium',3:'High'}[x])
    part_time_job   = st.selectbox('Part-Time Job',  ['Yes', 'No'])
    family_income   = st.selectbox('Family Income Level', ['Low', 'Medium', 'High'])
    city_tier       = st.selectbox('City Tier',      ['Tier 1', 'Tier 2', 'Tier 3'])
    internet_access = st.selectbox('Internet Access', ['Yes', 'No'])
    extracurricular = st.selectbox('Extracurricular Involvement', ['None', 'Low', 'Medium', 'High'])

    predict_btn = st.button('🔮 Prediksi Sekarang', use_container_width=True, type='primary')


# ─────────────────────────────────────────────
# HELPER — ENCODE INPUT & PREDICT
# ─────────────────────────────────────────────

def build_input_df(bin_enc_dict: dict) -> pd.DataFrame:
    """Susun raw input ke DataFrame, encode binary cols."""
    raw = {
        'cgpa'                      : cgpa,
        'tenth_percentage'          : tenth_pct,
        'twelfth_percentage'        : twelfth_pct,
        'study_hours_per_day'       : study_hours,
        'attendance_percentage'     : attendance,
        'projects_completed'        : int(projects),
        'internships_completed'     : int(internships),
        'coding_skill_rating'       : coding_skill,
        'communication_skill_rating': comm_skill,
        'aptitude_skill_rating'     : apt_skill,
        'hackathons_participated'   : int(hackathons),
        'certifications_count'      : int(certifications),
        'sleep_hours'               : sleep_hours,
        'backlogs'                  : int(backlogs),
        'stress_level'              : stress_level,
        'gender'                    : gender,
        'part_time_job'             : part_time_job,
        'internet_access'           : internet_access,
        'family_income_level'       : family_income,
        'city_tier'                 : city_tier,
        'extracurricular_involvement': extracurricular,
        'branch'                    : branch,
    }
    df_in = pd.DataFrame([raw])

    # Encode binary cols dengan encoder yang sudah di-fit
    for col in ['gender', 'part_time_job', 'internet_access']:
        if col in bin_enc_dict and bin_enc_dict[col] is not None:
            df_in[col] = bin_enc_dict[col].transform(df_in[col].astype(str))

    return df_in


# ─────────────────────────────────────────────
# MAIN — PREDICTION & DISPLAY
# ─────────────────────────────────────────────

col1, col2 = st.columns(2)

# ── Ringkasan Input ──
with col1:
    st.subheader('📊 Ringkasan Input')
    input_display = {
        'CGPA'              : cgpa,
        'Attendance (%)'    : attendance,
        'Coding Skill'      : coding_skill,
        'Aptitude Skill'    : apt_skill,
        'Internships'       : internships,
        'Projects'          : projects,
        'Certifications'    : certifications,
        'Hackathons'        : hackathons,
        'Backlogs'          : backlogs,
        'Study Hours/Day'   : study_hours,
        'Branch'            : branch,
        'City Tier'         : city_tier,
        'Family Income'     : family_income,
    }
    st.dataframe(
        pd.DataFrame(input_display.items(), columns=['Fitur', 'Nilai']),
        hide_index=True, use_container_width=True
    )

# ── Hasil Prediksi ──
with col2:
    st.subheader('🔮 Hasil Prediksi')

    if predict_btn:
        try:
            # ── Klasifikasi ──
            clf_input = build_input_df(arts['bin_enc_clf'])
            if arts['clf_feat'] is not None:
                clf_input = clf_input[arts['clf_feat']]

            placement_pred  = arts['clf_model'].predict(clf_input)[0]
            placement_proba = arts['clf_model'].predict_proba(clf_input)[0]

            label = 'Placed ✅' if placement_pred == 1 else 'Not Placed ❌'
            conf  = placement_proba[placement_pred] * 100

            if placement_pred == 1:
                st.success(f'**Status: {label}**')
            else:
                st.error(f'**Status: {label}**')

            st.metric('Confidence', f'{conf:.1f}%')

            # Gauge bar probabilitas
            st.write('**Probabilitas:**')
            prob_df = pd.DataFrame({
                'Kelas'      : ['Not Placed', 'Placed'],
                'Probabilitas': [placement_proba[0], placement_proba[1]]
            })
            st.bar_chart(prob_df.set_index('Kelas'))

            # ── Regresi (hanya jika diprediksi Placed) ──
            if placement_pred == 1:
                st.divider()
                reg_input = build_input_df(arts['bin_enc_reg'])
                if arts['reg_feat'] is not None:
                    reg_input = reg_input[arts['reg_feat']]

                salary_pred = arts['reg_model'].predict(reg_input)[0]

                st.metric(
                    label='💰 Estimasi Gaji',
                    value=f'{salary_pred:.2f} LPA',
                    delta='Prediksi berdasarkan profil mahasiswa'
                )
                st.info(
                    f'Gaji diperkirakan sekitar **{salary_pred:.2f} Lakh Per Annum** '
                    f'berdasarkan profil akademik dan skill mahasiswa.'
                )
            else:
                st.divider()
                st.warning('Estimasi gaji tidak tersedia — mahasiswa diprediksi belum ditempatkan.')

        except Exception as e:
            st.error(f'Terjadi error saat prediksi: {e}')
            st.exception(e)

    else:
        st.info('👈 Isi data di sidebar dan klik **Prediksi Sekarang**')

# ─────────────────────────────────────────────
# VISUALISASI — PROFIL MAHASISWA
# ─────────────────────────────────────────────

st.divider()
st.subheader('📈 Profil Skill Mahasiswa')

skill_labels = ['Coding', 'Communication', 'Aptitude', 'Study Hours', 'Attendance (÷10)']
skill_values = [
    coding_skill,
    comm_skill,
    apt_skill,
    min(study_hours, 10),
    attendance / 10
]

fig, ax = plt.subplots(figsize=(8, 3))
bars = ax.barh(skill_labels, skill_values, color='steelblue', alpha=0.8)
ax.set_xlim(0, 10)
ax.set_xlabel('Skor (skala 10)')
ax.set_title('Profil Skill Mahasiswa')
for bar, val in zip(bars, skill_values):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}', va='center', fontsize=9)
plt.tight_layout()
st.pyplot(fig)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.divider()
st.markdown(
    '<div style="text-align:center; color:gray; font-size:13px;">'
    'Student Placement Predictor | Model: Random Forest (HyperOpt Tuned) | '
    'Data: 5000 Students'
    '</div>',
    unsafe_allow_html=True
)
