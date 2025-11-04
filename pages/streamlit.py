import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st


def load_training_schema() -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Load the original dataset and reproduce the preprocessing feature set."""

    data_path = Path('../data/dataset.csv')
    if not data_path.exists():
        raise FileNotFoundError('dataset.csv 파일이 없습니다. 학습 데이터 경로를 확인하세요.')

    df = pd.read_csv(data_path)
    if 'Target' not in df.columns:
        raise KeyError('dataset.csv에 Target 컬럼이 없습니다.')

    # Drop Enrolled rows and map target
    df = df.copy()
    df = df[df['Target'] != 'Enrolled']

    target_map = {'Dropout': 0, 'Graduate': 1}
    df['Target'] = df['Target'].map(target_map)
    if df['Target'].isna().any():
        raise ValueError('정의되지 않은 타깃 라벨이 존재합니다.')

    df = df.drop_duplicates()

    feature_cols = [col for col in df.columns if col != 'Target']
    X = df[feature_cols]
    y = df['Target']

    columns_to_drop = set()

    # ID-like and constant columns
    id_like = [col for col in feature_cols if 'id' in col.lower()]
    constant_cols = [col for col in feature_cols if X[col].nunique(dropna=False) <= 1]
    columns_to_drop.update(id_like)
    columns_to_drop.update(constant_cols)

    # Numerics with near-zero correlation
    numeric_candidates = X.select_dtypes(include=['number']).columns.tolist()
    low_corr_numeric = []
    for col in numeric_candidates:
        corr = X[col].corr(y)
        if pd.isna(corr) or abs(corr) < 0.02:
            low_corr_numeric.append(col)
    columns_to_drop.update(low_corr_numeric)

    # High-cardinality categoricals
    categorical_candidates = X.select_dtypes(include=['object', 'category']).columns.tolist()
    high_cardinality = [
        col
        for col in categorical_candidates
        if (X[col].nunique(dropna=False) / len(X)) > 0.6
    ]
    columns_to_drop.update(high_cardinality)

    if columns_to_drop:
        X = X.drop(columns=columns_to_drop)

    feature_cols = X.columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()

    return X, numeric_cols, categorical_cols


@st.cache_resource(show_spinner=False)
def load_pipeline():
    model_path = Path('../model/model_trained.pkl')
    if not model_path.exists():
        raise FileNotFoundError('학습된 모델 파일(model_trained.pkl)을 찾을 수 없습니다. 먼저 학습을 수행하세요.')
    return joblib.load(model_path)


@st.cache_data(show_spinner=False)
def load_metadata():
    X, numeric_cols, categorical_cols = load_training_schema()
    numeric_stats: Dict[str, Dict[str, float]] = {}
    for col in numeric_cols:
        numeric_stats[col] = {
            'min': float(X[col].min()),
            'max': float(X[col].max()),
            'mean': float(X[col].mean()),
            'median': float(X[col].median()),
        }
    categorical_options: Dict[str, List[str]] = {}
    for col in categorical_cols:
        categorical_options[col] = sorted(X[col].dropna().unique().tolist())
    return X.columns.tolist(), numeric_cols, categorical_cols, numeric_stats, categorical_options


st.set_page_config(page_title='학생 이탈 예측', layout='wide')
st.title('학생 이탈(졸업 여부) 예측')
st.write('저장된 파이프라인을 사용하여 학생 정보를 입력하면 Dropout 확률을 예측합니다.')

try:
    feature_cols, numeric_cols, categorical_cols, numeric_stats, categorical_options = load_metadata()
    pipeline = load_pipeline()
except Exception as exc:
    st.error(f'모델 또는 메타데이터 로드 중 오류가 발생했습니다: {exc}')
    st.stop()

with st.sidebar:
    st.header('입력 방법 안내')
    st.markdown(
        """- 숫자형 값은 과거 데이터의 범위를 참고하여 자동으로 채워집니다.\n"
        "- 범주형 값은 드롭다운에서 선택하세요.\n"
        "- 값을 변경한 후 **예측 실행** 버튼을 클릭하면 결과가 표시됩니다."""
    )

with st.form('prediction_form'):
    st.subheader('학생 특성 입력')
    input_data: Dict[str, object] = {}

    numeric_columns = st.columns(min(len(numeric_cols), 3) or 1)
    for idx, col in enumerate(numeric_cols):
        stats = numeric_stats[col]
        column = numeric_columns[idx % len(numeric_columns)]
        default = stats['median'] if not np.isnan(stats['median']) else stats['mean']
        min_value = stats['min'] if not np.isnan(stats['min']) else None
        max_value = stats['max'] if not np.isnan(stats['max']) else None
        with column:
            value = st.number_input(
                label=col,
                value=default,
                min_value=min_value,
                max_value=max_value,
                step=1.0 if isinstance(default, float) else 1,
            )
        input_data[col] = value

    if categorical_cols:
        st.markdown('---')
        cat_columns = st.columns(min(len(categorical_cols), 2) or 1)
        for idx, col in enumerate(categorical_cols):
            options = categorical_options[col]
            with cat_columns[idx % len(cat_columns)]:
                if options:
                    selection = st.selectbox(col, options, index=0)
                else:
                    selection = st.text_input(col, value='')
            input_data[col] = selection

    submitted = st.form_submit_button('예측 실행')

if submitted:
    try:
        input_df = pd.DataFrame([input_data], columns=feature_cols)
        prediction = pipeline.predict(input_df)[0]
        proba = pipeline.predict_proba(input_df)[0]
        dropout_prob = float(proba[0])
        graduate_prob = float(proba[1])

        st.success('예측이 완료되었습니다.')
        col_pred, col_prob = st.columns(2)
        with col_pred:
            st.metric('예측 결과', 'Dropout' if prediction == 0 else 'Graduate')
        with col_prob:
            st.metric('Dropout 확률', f'{dropout_prob * 100:.2f}%')
            st.metric('Graduate 확률', f'{graduate_prob * 100:.2f}%')

        st.markdown('### 입력 값 확인')
        st.json(json.dumps(input_data, ensure_ascii=False, indent=2))
    except Exception as exc:
        st.error(f'예측 중 오류가 발생했습니다: {exc}')
