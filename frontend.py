"""
frontend.py — Streamlit Client (Decoupled Architecture)
========================================================
Streamlit bertindak sebagai client yang mengirim request ke FastAPI.

Pastikan FastAPI sudah berjalan:
    uvicorn main:app --reload

Lalu jalankan:
    streamlit run frontend.py
"""

import streamlit as st
import requests
import pandas as pd

API_BASE = 'http://localhost:8000'

st.set_page_config(
    page_title='Placement Predictor (Decoupled)',
    page_icon='🚀',
    layout='wide',
)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.title('🚀 Student Placement Predictor')
st.caption('Frontend Streamlit → Backend FastAPI (Decoupled Architecture)')

# Cek koneksi API
try:
    health = requests.get(f'{API_BASE}/health', timeout=3).json()
    st.success(f"✅ API terhubung | clf: {health['clf_model_loaded']} | reg: {health['reg_model_loaded']}")
except Exception:
    st.error(f'❌ FastAPI tidak dapat diakses di `{API_BASE}`. Jalankan: `uvicorn main:app --reload`')
    st.stop()

st.divider()

# ─────────────────────────────────────────────
# TAB LAYOUT
# ─────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(['🔵 Prediksi Placement', '🟠 Prediksi Salary', '🔮 Full Prediction'])

# ─────────────────────────────────────────────
# SHARED INPUT FORM
# ─────────────────────────────────────────────

def input_form(key_prefix: str) -> dict:
    """Form input data mahasiswa, dikembalikan sebagai dict."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('**👤 Data Pribadi & Akademik**')
        gender     = st.selectbox('Gender', ['Male', 'Female'], key=f'{key_prefix}_gender')
        branch     = st.selectbox('Branch', ['CSE', 'ECE', 'IT', 'ME', 'CE'], key=f'{key_prefix}_branch')
        cgpa       = st.slider('CGPA', 5.0, 10.0, 7.5, 0.01, key=f'{key_prefix}_cgpa')
        tenth_pct  = st.slider('10th Percentage', 40.0, 100.0, 75.0, 0.1, key=f'{key_prefix}_tenth')
        twelfth_pct= st.slider('12th Percentage', 40.0, 100.0, 75.0, 0.1, key=f'{key_prefix}_twelfth')
        backlogs   = st.number_input('Backlogs', 0, 10, 0, key=f'{key_prefix}_backlogs')

    with col2:
        st.markdown('**📚 Aktivitas Akademik**')
        study_hours  = st.slider('Study Hours/Day', 0.0, 12.0, 4.0, 0.5, key=f'{key_prefix}_study')
        attendance   = st.slider('Attendance (%)',  50.0, 100.0, 80.0, 0.5, key=f'{key_prefix}_attend')
        projects     = st.number_input('Projects Completed', 0, 20, 3, key=f'{key_prefix}_proj')
        internships  = st.number_input('Internships', 0, 5, 1, key=f'{key_prefix}_intern')
        hackathons   = st.number_input('Hackathons', 0, 20, 2, key=f'{key_prefix}_hack')
        certifications = st.number_input('Certifications', 0, 20, 2, key=f'{key_prefix}_cert')

    with col3:
        st.markdown('**🧠 Skill & Kondisi**')
        coding_skill = st.slider('Coding Skill',        1, 10, 6, key=f'{key_prefix}_code')
        comm_skill   = st.slider('Communication Skill', 1, 10, 6, key=f'{key_prefix}_comm')
        apt_skill    = st.slider('Aptitude Skill',      1, 10, 6, key=f'{key_prefix}_apt')
        sleep_hours  = st.slider('Sleep Hours/Day', 3.0, 10.0, 7.0, 0.5, key=f'{key_prefix}_sleep')
        stress_level = st.selectbox('Stress Level', [1, 2, 3],
                                    format_func=lambda x: {1:'Low',2:'Medium',3:'High'}[x],
                                    key=f'{key_prefix}_stress')
        part_time    = st.selectbox('Part-Time Job', ['Yes', 'No'], key=f'{key_prefix}_pt')
        family_inc   = st.selectbox('Family Income', ['Low', 'Medium', 'High'], key=f'{key_prefix}_fam')
        city_tier    = st.selectbox('City Tier', ['Tier 1', 'Tier 2', 'Tier 3'], key=f'{key_prefix}_city')
        internet     = st.selectbox('Internet Access', ['Yes', 'No'], key=f'{key_prefix}_net')
        extra        = st.selectbox('Extracurricular', ['None', 'Low', 'Medium', 'High'], key=f'{key_prefix}_extra')

    return {
        'gender'                     : gender,
        'branch'                     : branch,
        'cgpa'                       : cgpa,
        'tenth_percentage'           : tenth_pct,
        'twelfth_percentage'         : twelfth_pct,
        'backlogs'                   : backlogs,
        'study_hours_per_day'        : study_hours,
        'attendance_percentage'      : attendance,
        'projects_completed'         : int(projects),
        'internships_completed'      : int(internships),
        'coding_skill_rating'        : coding_skill,
        'communication_skill_rating' : comm_skill,
        'aptitude_skill_rating'      : apt_skill,
        'hackathons_participated'    : int(hackathons),
        'certifications_count'       : int(certifications),
        'sleep_hours'                : sleep_hours,
        'stress_level'               : stress_level,
        'part_time_job'              : part_time,
        'family_income_level'        : family_inc,
        'city_tier'                  : city_tier,
        'internet_access'            : internet,
        'extracurricular_involvement': extra,
    }


# ─────────────────────────────────────────────
# TAB 1 — Prediksi Placement
# ─────────────────────────────────────────────

with tab1:
    st.subheader('🔵 Prediksi Status Placement')
    st.info('Mengirim request `POST /predict/placement` ke FastAPI.')

    payload_clf = input_form('clf')
    btn_clf     = st.button('🔮 Prediksi Placement', key='btn_clf', type='primary')

    if btn_clf:
        with st.spinner('Mengirim request ke API ...'):
            try:
                resp = requests.post(f'{API_BASE}/predict/placement', json=payload_clf, timeout=10)
                resp.raise_for_status()
                result = resp.json()

                st.divider()
                st.markdown('**📡 Response dari FastAPI:**')
                st.json(result)

                status = result['placement_status']
                conf   = result['confidence_placed'] * 100

                if status == 'Placed':
                    st.success(f'✅ **{status}** — Confidence: {conf:.1f}%')
                else:
                    st.error(f'❌ **{status}** — Confidence Placed: {conf:.1f}%')

                # Bar chart probabilitas
                prob_data = pd.DataFrame({
                    'Status'      : ['Not Placed', 'Placed'],
                    'Probabilitas': [result['confidence_not_placed'], result['confidence_placed']]
                })
                st.bar_chart(prob_data.set_index('Status'))

            except requests.exceptions.RequestException as e:
                st.error(f'Request error: {e}')


# ─────────────────────────────────────────────
# TAB 2 — Prediksi Salary
# ─────────────────────────────────────────────

with tab2:
    st.subheader('🟠 Prediksi Estimasi Salary')
    st.info('Mengirim request `POST /predict/salary` ke FastAPI.')

    payload_reg = input_form('reg')
    btn_reg     = st.button('💰 Prediksi Salary', key='btn_reg', type='primary')

    if btn_reg:
        with st.spinner('Mengirim request ke API ...'):
            try:
                resp = requests.post(f'{API_BASE}/predict/salary', json=payload_reg, timeout=10)
                resp.raise_for_status()
                result = resp.json()

                st.divider()
                st.markdown('**📡 Response dari FastAPI:**')
                st.json(result)

                st.metric(
                    label='💰 Estimasi Gaji',
                    value=f"{result['salary_lpa']:.2f} LPA",
                )
                st.info(result['note'])

            except requests.exceptions.RequestException as e:
                st.error(f'Request error: {e}')


# ─────────────────────────────────────────────
# TAB 3 — Full Prediction
# ─────────────────────────────────────────────

with tab3:
    st.subheader('🔮 Full Prediction (Placement + Salary)')
    st.info('Mengirim request `POST /predict/full` ke FastAPI — satu endpoint untuk keduanya.')

    payload_full = input_form('full')
    btn_full     = st.button('🚀 Full Predict', key='btn_full', type='primary')

    if btn_full:
        with st.spinner('Mengirim request ke API ...'):
            try:
                resp = requests.post(f'{API_BASE}/predict/full', json=payload_full, timeout=10)
                resp.raise_for_status()
                result = resp.json()

                st.divider()
                st.markdown('**📡 Response dari FastAPI:**')
                st.json(result)

                placement = result['placement']
                salary    = result.get('salary')

                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown('**🔵 Placement**')
                    status = placement['placement_status']
                    conf   = placement['confidence_placed'] * 100
                    if status == 'Placed':
                        st.success(f'✅ **{status}** | Confidence: {conf:.1f}%')
                    else:
                        st.error(f'❌ **{status}** | Confidence: {conf:.1f}%')

                with col_b:
                    st.markdown('**🟠 Salary Estimation**')
                    if salary:
                        st.metric('Estimasi Gaji', f"{salary['salary_lpa']:.2f} LPA")
                    else:
                        st.warning('Tidak tersedia (Not Placed)')

            except requests.exceptions.RequestException as e:
                st.error(f'Request error: {e}')

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.divider()
st.markdown(
    '<div style="text-align:center; color:gray; font-size:12px;">'
    'Decoupled Architecture | Streamlit Frontend → FastAPI Backend | '
    'Swagger UI: <a href="http://localhost:8000/docs">localhost:8000/docs</a>'
    '</div>',
    unsafe_allow_html=True
)
