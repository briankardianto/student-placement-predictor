"""
main.py — FastAPI Backend (Decoupled Architecture)
===================================================
API Server untuk prediksi placement & salary.
Test via Swagger UI: http://localhost:8000/docs

Jalankan:
    pip install fastapi uvicorn pydantic
    uvicorn main:app --reload
"""

import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, Optional

# ─────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────

EXP_PATH = './exp/placement/'


def load_pkl(filename: str):
    path = os.path.join(EXP_PATH, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Artifact '{filename}' tidak ditemukan di {EXP_PATH}. "
            "Jalankan `python pipeline.py` terlebih dahulu."
        )
    with open(path, 'rb') as f:
        return pickle.load(f)


clf_model     = load_pkl('best_clf_pipeline.pkl')
reg_model     = load_pkl('best_reg_pipeline.pkl')
bin_enc_clf   = load_pkl('bin_enc_dict.pkl')
bin_enc_reg   = load_pkl('bin_enc_dict_reg.pkl')
clf_feat      = load_pkl('clf_feature_cols.pkl')
reg_feat      = load_pkl('reg_feature_cols.pkl')

# ─────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────

app = FastAPI(
    title       ='🎓 Student Placement API',
    description =(
        'API untuk prediksi **placement status** (Klasifikasi) dan '
        '**estimasi salary** (Regresi) mahasiswa.\n\n'
        '- `POST /predict/placement` — prediksi Placed / Not Placed\n'
        '- `POST /predict/salary`    — estimasi gaji LPA\n'
        '- `POST /predict/full`      — keduanya sekaligus\n'
    ),
    version='1.0.0',
)


# ─────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────

class StudentFeatures(BaseModel):
    gender                      : Literal['Male', 'Female']           = Field(..., example='Male')
    branch                      : Literal['CSE', 'ECE', 'IT', 'ME', 'CE'] = Field(..., example='CSE')
    cgpa                        : float = Field(..., ge=5.0, le=10.0,  example=8.5)
    tenth_percentage            : float = Field(..., ge=40.0, le=100.0, example=78.5)
    twelfth_percentage          : float = Field(..., ge=40.0, le=100.0, example=80.0)
    backlogs                    : int   = Field(..., ge=0, le=10,       example=0)
    study_hours_per_day         : float = Field(..., ge=0.0, le=12.0,  example=5.0)
    attendance_percentage       : float = Field(..., ge=50.0, le=100.0, example=85.0)
    projects_completed          : int   = Field(..., ge=0, le=20,       example=3)
    internships_completed       : int   = Field(..., ge=0, le=5,        example=1)
    coding_skill_rating         : int   = Field(..., ge=1, le=10,       example=7)
    communication_skill_rating  : int   = Field(..., ge=1, le=10,       example=7)
    aptitude_skill_rating       : int   = Field(..., ge=1, le=10,       example=7)
    hackathons_participated     : int   = Field(..., ge=0, le=20,       example=2)
    certifications_count        : int   = Field(..., ge=0, le=20,       example=3)
    sleep_hours                 : float = Field(..., ge=3.0, le=10.0,  example=7.0)
    stress_level                : Literal[1, 2, 3]                    = Field(..., example=2)
    part_time_job               : Literal['Yes', 'No']                = Field(..., example='No')
    family_income_level         : Literal['Low', 'Medium', 'High']    = Field(..., example='Medium')
    city_tier                   : Literal['Tier 1', 'Tier 2', 'Tier 3'] = Field(..., example='Tier 1')
    internet_access             : Literal['Yes', 'No']                = Field(..., example='Yes')
    extracurricular_involvement : Literal['None', 'Low', 'Medium', 'High'] = Field(..., example='Medium')


class PlacementResponse(BaseModel):
    placement_status     : Literal['Placed', 'Not Placed']
    confidence_placed    : float = Field(..., description='Probabilitas Placed (0–1)')
    confidence_not_placed: float = Field(..., description='Probabilitas Not Placed (0–1)')


class SalaryResponse(BaseModel):
    salary_lpa     : float = Field(..., description='Estimasi gaji dalam Lakh Per Annum')
    note           : str


class FullPredictionResponse(BaseModel):
    placement      : PlacementResponse
    salary         : Optional[SalaryResponse]


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────

def prepare_df(student: StudentFeatures, bin_enc: dict) -> pd.DataFrame:
    """Ubah input Pydantic → DataFrame dengan binary encoding."""
    data = student.model_dump()
    df   = pd.DataFrame([data])

    for col in ['gender', 'part_time_job', 'internet_access']:
        if col in bin_enc and bin_enc[col] is not None:
            df[col] = bin_enc[col].transform(df[col].astype(str))

    return df


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get('/', tags=['Health'])
def root():
    return {
        'status' : 'ok',
        'message': 'Student Placement API is running',
        'docs'   : '/docs',
    }


@app.get('/health', tags=['Health'])
def health_check():
    """Cek apakah semua model berhasil dimuat."""
    return {
        'clf_model_loaded': clf_model is not None,
        'reg_model_loaded': reg_model is not None,
        'status'          : 'healthy',
    }


@app.post('/predict/placement', response_model=PlacementResponse, tags=['Prediction'])
def predict_placement(student: StudentFeatures):
    """
    **Klasifikasi** — Prediksi apakah mahasiswa akan Placed atau Not Placed.

    Returns probabilitas untuk setiap kelas.
    """
    try:
        df    = prepare_df(student, bin_enc_clf)
        df    = df[clf_feat]
        pred  = clf_model.predict(df)[0]
        proba = clf_model.predict_proba(df)[0]

        return PlacementResponse(
            placement_status      = 'Placed' if pred == 1 else 'Not Placed',
            confidence_placed     = round(float(proba[1]), 4),
            confidence_not_placed = round(float(proba[0]), 4),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/predict/salary', response_model=SalaryResponse, tags=['Prediction'])
def predict_salary(student: StudentFeatures):
    """
    **Regresi** — Estimasi gaji dalam Lakh Per Annum (LPA).

    Catatan: prediksi ini bermakna paling baik untuk mahasiswa yang diprediksi Placed.
    """
    try:
        df   = prepare_df(student, bin_enc_reg)
        df   = df[reg_feat]
        pred = reg_model.predict(df)[0]

        return SalaryResponse(
            salary_lpa=round(float(pred), 2),
            note='Estimasi berdasarkan profil akademik dan skill mahasiswa.'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/predict/full', response_model=FullPredictionResponse, tags=['Prediction'])
def predict_full(student: StudentFeatures):
    """
    **Full Prediction** — Jalankan klasifikasi dan regresi sekaligus.

    - Selalu mengembalikan `placement`.
    - `salary` hanya dikembalikan jika mahasiswa diprediksi **Placed**.
    """
    try:
        # Klasifikasi
        df_clf = prepare_df(student, bin_enc_clf)[clf_feat]
        pred   = clf_model.predict(df_clf)[0]
        proba  = clf_model.predict_proba(df_clf)[0]

        placement = PlacementResponse(
            placement_status      = 'Placed' if pred == 1 else 'Not Placed',
            confidence_placed     = round(float(proba[1]), 4),
            confidence_not_placed = round(float(proba[0]), 4),
        )

        # Regresi hanya jika Placed
        salary = None
        if pred == 1:
            df_reg     = prepare_df(student, bin_enc_reg)[reg_feat]
            salary_val = reg_model.predict(df_reg)[0]
            salary = SalaryResponse(
                salary_lpa=round(float(salary_val), 2),
                note='Estimasi berdasarkan profil akademik dan skill mahasiswa.'
            )

        return FullPredictionResponse(placement=placement, salary=salary)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
