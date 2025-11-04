import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / 'model' / 'model_trained.pkl'
DATASET_PATH = ROOT_DIR / 'data' / 'dataset.csv'


def unwrap_estimator(estimator: Any) -> Any:
    """Follow best_estimator_ references until the fitted pipeline is reached."""

    current = estimator
    while hasattr(current, 'best_estimator_'):
        current = current.best_estimator_
    return current


def resolve_feature_names(estimator: Any) -> List[str]:
    names = getattr(estimator, 'feature_names_in_', None)
    if names is None:
        raise AttributeError('학습된 파이프라인에서 feature_names_in_ 정보를 찾을 수 없습니다.')
    return [str(name) for name in list(names)]


def normalize_columns(selection: Any, feature_names: Sequence[str]) -> List[str]:
    if selection is None:
        return []
    if isinstance(selection, slice):
        return list(np.array(feature_names)[selection].tolist())
    if isinstance(selection, (list, tuple, set)):
        return [str(col) for col in selection]
    if isinstance(selection, np.ndarray):
        if selection.dtype == bool:
            return [name for name, flag in zip(feature_names, selection) if flag]
        return [feature_names[int(idx)] for idx in selection]
    if isinstance(selection, pd.Index):
        return selection.astype(str).tolist()
    return [str(selection)]


def extract_step(transformer: Any, target_cls: type) -> Optional[Any]:
    if isinstance(transformer, target_cls):
        return transformer
    if isinstance(transformer, SkPipeline):
        for step in transformer.named_steps.values():
            found = extract_step(step, target_cls)
            if found is not None:
                return found
    return None


def find_column_transformer(estimator: Any) -> Optional[ColumnTransformer]:
    if isinstance(estimator, ColumnTransformer):
        return estimator
    if isinstance(estimator, SkPipeline):
        for _, step in estimator.steps:
            found = find_column_transformer(step)
            if found is not None:
                return found
    return None


def clean_scalar(value: Any) -> Optional[Any]:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def compute_feature_modes(
    dataset: pd.DataFrame,
    feature_names: Sequence[str],
) -> Dict[str, Any]:
    modes: Dict[str, Any] = {}
    for column in feature_names:
        if column not in dataset.columns:
            continue
        series = dataset[column].dropna()
        if series.empty:
            continue
        mode_values = series.mode(dropna=True)
        if not mode_values.empty:
            modes[column] = clean_scalar(mode_values.iloc[0])
    return modes


def compute_numeric_bounds(
    dataset: pd.DataFrame,
    numeric_columns: Sequence[str],
) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for column in numeric_columns:
        if column not in dataset.columns:
            continue
        series = pd.to_numeric(dataset[column], errors='coerce').dropna()
        if series.empty:
            continue
        lower = clean_scalar(series.min())
        upper = clean_scalar(series.max())
        bounds[column] = (lower, upper)
    return bounds


def sanitize_categories(values: Sequence[Any]) -> List[str]:
    cleaned: List[str] = []
    for value in values:
        scalar = clean_scalar(value)
        if scalar is None:
            continue
        text = str(scalar).strip()
        if not text:
            continue
        cleaned.append(text)
    return cleaned


def extract_schema_from_preprocessor(
    preprocessor: ColumnTransformer,
    feature_names: Sequence[str],
) -> Tuple[List[str], List[str], Dict[str, Any], Dict[str, Any], Dict[str, List[str]]]:
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    numeric_defaults: Dict[str, Any] = {}
    categorical_defaults: Dict[str, Any] = {}
    categorical_options: Dict[str, List[str]] = {}

    for _, transformer, cols in getattr(preprocessor, 'transformers_', []):
        if transformer in ('drop', None):
            continue
        column_list = normalize_columns(cols, feature_names)
        if not column_list:
            continue

        if transformer == 'passthrough':
            numeric_cols.extend(column_list)
            continue

        imputer = extract_step(transformer, SimpleImputer)
        encoder = extract_step(transformer, OneHotEncoder)

        if encoder is not None:
            categorical_cols.extend(column_list)
            categories = getattr(encoder, 'categories_', [])
            for idx, column in enumerate(column_list):
                options = categories[idx] if idx < len(categories) else []
                categorical_options[column] = sanitize_categories(options)
        else:
            numeric_cols.extend(column_list)

        if imputer is not None and hasattr(imputer, 'statistics_'):
            stats = getattr(imputer, 'statistics_', [])
            for idx, column in enumerate(column_list):
                value = stats[idx] if idx < len(stats) else None
                if encoder is not None:
                    categorical_defaults[column] = clean_scalar(value)
                else:
                    numeric_defaults[column] = clean_scalar(value)

    ordered_numeric = [col for col in feature_names if col in set(numeric_cols)]
    ordered_categorical = [col for col in feature_names if col in set(categorical_cols)]

    return ordered_numeric, ordered_categorical, numeric_defaults, categorical_defaults, categorical_options


def sanitize_numeric_default(value: Any) -> Tuple[float | int, float | int]:
    value = clean_scalar(value)
    if isinstance(value, bool):
        return int(value), 1
    if isinstance(value, (int, np.integer)):
        return int(value), 1
    if isinstance(value, (float, np.floating)):
        float_value = float(value)
        if math.isnan(float_value):
            return 0.0, 0.1
        if float_value.is_integer():
            return int(float_value), 1
        return float_value, 0.1
    try:
        converted = float(value) if value is not None else 0.0
        if converted.is_integer():
            return int(converted), 1
        return converted, 0.1
    except (TypeError, ValueError):
        return 0.0, 0.1


def sanitize_categorical_default(candidate: Any, options: List[str]) -> str:
    candidate_value = clean_scalar(candidate)
    if candidate_value is None:
        return options[0] if options else ''
    candidate_text = str(candidate_value)
    if options and candidate_text not in options:
        return options[0]
    return candidate_text


def get_field_label(column: str) -> str:
    return FIELD_LABELS.get(column, column)


def format_codebook_option(option: Dict[str, object]) -> str:
    if isinstance(option, dict):
        return str(option.get('label', option.get('value')))
    return str(option)


def coerce_codebook_value(raw_value: str, sample: Optional[Any]) -> Any:
    if sample is not None:
        sample_type = type(sample)
        is_numeric_like = False
        if sample_type in {int, float}:
            is_numeric_like = True
        else:
            try:
                is_numeric_like = np.issubdtype(sample_type, np.number)
            except TypeError:
                is_numeric_like = False
        if is_numeric_like:
            try:
                return sample_type(raw_value)
            except Exception:
                pass
    try:
        return int(raw_value)
    except ValueError:
        try:
            return float(raw_value)
        except ValueError:
            return raw_value


def build_codebook_options(
    column: str,
    categorical_choices: Dict[str, List[Any]],
) -> List[Dict[str, object]]:
    label_map = CODEBOOK_LABELS.get(column, {})
    observed_values = categorical_choices.get(column, [])
    sample_value = observed_values[0] if observed_values else None

    options_values: List[Any] = []
    seen_keys: set[str] = set()

    for value in observed_values:
        key = str(value)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        options_values.append(value)

    for key_str in label_map.keys():
        if key_str in seen_keys:
            continue
        value = coerce_codebook_value(key_str, sample_value)
        key = str(value)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        options_values.append(value)

    result: List[Dict[str, object]] = []
    for value in options_values:
        key = str(value)
        label_text = label_map.get(key)
        display_label = label_text if label_text else str(value)
        result.append({'value': value, 'label': display_label})

    return result


HIDDEN_FEATURES = {
    'Application order',
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (without evaluations)',
    'Daytime/evening attendance',
    'Displaced',
    "Father's occupation",
    'GDP',
    'Inflation rate',
    'Marital status',
    "Mother's occupation",
    "Mother's qualification",
    'Previous qualification',
}


CODEBOOK_LABELS: Dict[str, Dict[str, str]] = {
    'Application mode': {
        '1': '일반 전형 / 국가 경쟁 입학시험',
        '2': '특수 쿼터',
        '6': '외국인 학생 전형',
        '8': '편입',
        '12': '재입학',
    },
    'Previous qualification': {
        '1': '고등학교 졸업',
        '2': '학위 취득 이전 (학사)',
        '10': '학위 취득 이후 (석사)',
    },
    'Course': {
        '33': '회계',
        '171': '관리',
        '8014': '정보 시스템',
        '9070': '사회 서비스',
    },
    'Daytime/evening attendance': {
        '1': '주간',
        '0': '야간',
    },
    'Marital status': {
        '1': '미혼',
        '2': '기혼',
        '3': '별거/이혼',
        '6': '사별',
    },
    'Gender': {
        '1': '남성',
        '0': '여성',
    },
    'Debtor': {
        '1': '채무 있음',
        '0': '채무 없음',
    },
    'Tuition fees up to date': {
        '1': '납부 완료',
        '0': '미납',
    },
    'Scholarship holder': {
        '1': '장학금 수혜',
        '0': '장학금 없음',
    },
}


FIELD_LABELS: Dict[str, str] = {
    'Application mode': '지원 유형',
    'Gender': '성별',
    'Debtor': '채무 여부',
    'Tuition fees up to date': '등록금 납부 여부',
    'Scholarship holder': '장학금 수혜 여부',
    'Age at enrollment': '입학 시 나이',
    'Curricular units 1st sem (approved)': '1학기 이수 과목 수',
    'Curricular units 1st sem (grade)': '1학기 이수 학점',
    'Curricular units 2nd sem (approved)': '2학기 이수 과목 수',
    'Curricular units 2nd sem (grade)': '2학기 이수 학점',
    'Curricular units 1st sem (enrolled)': '1학기 수강 학점',
    'Curricular units 2nd sem (enrolled)': '2학기 수강 학점',
    'Curricular units 1st sem (evaluations)': '1학기 평가 횟수',
    'Curricular units 2nd sem (evaluations)': '2학기 평가 횟수',
    'Curricular units 1st sem (without evaluations)': '1학기 평가 제외 학점',
    'Curricular units 2nd sem (without evaluations)': '2학기 평가 제외 학점',
    'Curricular units 1st sem (credited)': '1학기 학점 인정 수',
    'Curricular units 2nd sem (credited)': '2학기 학점 인정 수',
    'Application order': '지원 순위',
    'Daytime/evening attendance': '주간/야간 구분',
    'Displaced': '거주 이전 여부',
    "Father's occupation": '부 직업',
    'GDP': '국내총생산(GDP)',
    'Inflation rate': '물가상승률',
    'Marital status': '결혼 상태',
    "Mother's occupation": '모 직업',
    "Mother's qualification": '모 학력',
    'Previous qualification': '이전 학력',
}


def render_metric_card(column, label: str, value: str) -> None:
    column.markdown(
        f"""
        <div class="metric-wrapper">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_pipeline():
    if not MODEL_PATH.exists():
        raise FileNotFoundError('학습된 모델 파일(model_trained.pkl)을 찾을 수 없습니다. 먼저 학습을 수행하세요.')
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_metadata():
    pipeline = load_pipeline()
    fitted_estimator = unwrap_estimator(pipeline)
    feature_names = resolve_feature_names(fitted_estimator)

    preprocessor = find_column_transformer(fitted_estimator)
    if preprocessor is not None:
        (
            numeric_cols,
            categorical_cols,
            numeric_defaults_raw,
            categorical_defaults_raw,
            categorical_options_raw,
        ) = extract_schema_from_preprocessor(preprocessor, feature_names)
    else:
        numeric_cols = feature_names
        categorical_cols = []
        numeric_defaults_raw = {}
        categorical_defaults_raw = {}
        categorical_options_raw = {}

    remaining = [
        col for col in feature_names if col not in set(numeric_cols) | set(categorical_cols)
    ]
    if remaining:
        numeric_cols = list(numeric_cols) + remaining

    dataset_modes: Dict[str, Any] = {}
    numeric_bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    dataset_summary: Dict[str, Any] = {}
    if DATASET_PATH.exists():
        try:
            dataset_df = pd.read_csv(DATASET_PATH)
            dataset_modes = compute_feature_modes(dataset_df, feature_names)
            numeric_bounds = compute_numeric_bounds(dataset_df, numeric_cols)
            dataset_summary = {
                'row_count': int(len(dataset_df)),
                'feature_count': int(dataset_df.shape[1]),
            }
            if 'Target' in dataset_df.columns:
                target_counts_series = dataset_df['Target'].value_counts(dropna=False)
                target_counts: Dict[str, int] = {
                    str(index): int(count) for index, count in target_counts_series.items()
                }
                dataset_summary['target_counts'] = target_counts
                total_count = sum(target_counts.values())
                if total_count > 0:
                    dataset_summary['dropout_ratio'] = target_counts.get('Dropout', 0) / total_count
                    dataset_summary['graduate_ratio'] = target_counts.get('Graduate', 0) / total_count
        except Exception:
            dataset_modes = {}
            numeric_bounds = {}
            dataset_summary = {}

    auto_fill_defaults: Dict[str, Any] = {
        column: dataset_modes.get(column) for column in feature_names
    }

    numeric_defaults: Dict[str, Dict[str, float | int]] = {}
    for col in numeric_cols:
        default_candidate = auto_fill_defaults.get(col)
        if default_candidate is None:
            default_candidate = numeric_defaults_raw.get(col)
        value, step = sanitize_numeric_default(default_candidate)
        numeric_defaults[col] = {'value': value, 'step': step}
        auto_fill_defaults[col] = value

    categorical_defaults: Dict[str, str] = {}
    categorical_options: Dict[str, List[str]] = {}
    for col in categorical_cols:
        options = categorical_options_raw.get(col, [])
        categorical_options[col] = options
        default_candidate = auto_fill_defaults.get(col)
        if default_candidate is None:
            default_candidate = categorical_defaults_raw.get(col)
        categorical_defaults[col] = sanitize_categorical_default(default_candidate, options)
        auto_fill_defaults[col] = categorical_defaults[col]

    for col in feature_names:
        if auto_fill_defaults.get(col) is None:
            fallback = dataset_modes.get(col)
            auto_fill_defaults[col] = fallback if fallback is not None else ''

    return (
        feature_names,
        numeric_cols,
        categorical_cols,
        numeric_defaults,
        categorical_defaults,
        categorical_options,
        auto_fill_defaults,
        numeric_bounds,
        dataset_summary,
    )


st.set_page_config(page_title='학생 이탈 예측', layout='wide')

st.markdown(
    """
    <style>
    :root {
        --primary-color: #3b82f6;
        --accent-color: #0ea5e9;
    }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #ffffff 0%, #f4f7fb 100%);
    }
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #111827 0%, #1f2937 100%);
        color: #f9fafb;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #f9fafb;
    }
    .hero-section {
        padding: 2.5rem 3rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(59,130,246,0.95), rgba(14,165,233,0.9));
        color: #ffffff;
        box-shadow: 0 18px 35px rgba(15, 23, 42, 0.18);
        margin-bottom: 1.5rem;
    }
    .hero-section h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    .hero-section p {
        margin-top: 0.75rem;
        font-size: 1.05rem;
        opacity: 0.9;
    }
    .metric-wrapper {
        padding: 1.1rem 1.4rem;
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.85);
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.12);
        border: 1px solid rgba(148, 163, 184, 0.25);
    }
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.08em;
        margin-bottom: 0.35rem;
    }
    .metric-value {
        font-size: 1.45rem;
        font-weight: 600;
        color: #0f172a;
    }
    .result-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.6rem;
        box-shadow: 0 18px 32px rgba(15, 23, 42, 0.16);
        border: 1px solid rgba(148, 163, 184, 0.25);
    }
    .result-card h3 {
        margin-top: 0;
        margin-bottom: 0.9rem;
        font-weight: 600;
    }
    .result-badge {
        display: inline-block;
        padding: 0.6rem 1.1rem;
        border-radius: 999px;
        background: rgba(59,130,246,0.12);
        color: #1d4ed8;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
    .prob-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    .prob-box {
        padding: 0.9rem 1.1rem;
        border-radius: 12px;
        background: rgba(241,245,249,0.7);
    }
    .prob-label {
        font-size: 0.85rem;
        color: #475569;
        font-weight: 500;
        margin-bottom: 0.25rem;
    }
    .prob-value {
        font-size: 1.35rem;
        font-weight: 600;
        color: #0f172a;
    }
    .sidebar-tips {
        padding: 1rem 1.1rem;
        border-radius: 14px;
        background: rgba(15,23,42,0.35);
        border: 1px solid rgba(148,163,184,0.2);
    }
    .stTabs [role="tab"] {
        padding: 0.75rem 1.4rem;
        border-radius: 12px 12px 0 0;
        margin-right: 0.5rem;
        background-color: rgba(255,255,255,0.55);
        font-weight: 600;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background: #ffffff;
        box-shadow: 0 -6px 18px rgba(15, 23, 42, 0.12);
        border-bottom: 2px solid transparent;
    }
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
        border: none;
        color: #ffffff;
        padding: 0.7rem 1.8rem;
        border-radius: 999px;
        font-weight: 600;
        box-shadow: 0 12px 24px rgba(59, 130, 246, 0.25);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        filter: brightness(1.05);
        transform: translateY(-2px);
        box-shadow: 0 16px 32px rgba(59, 130, 246, 0.35);
    }
    .stExpander {
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }
    .stSelectbox, .stNumberInput, .stTextInput {
        border-radius: 8px;
    }
    /* 입력 폼 스타일 */
    [data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.5);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.15);
    }
    /* 성공 메시지 스타일 */
    .stSuccess {
        border-radius: 12px;
        padding: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-section">
        <h1>🎓 학생 이탈 예측 시스템</h1>
        <p>머신러닝 모델로 학생의 중도 이탈 위험을 미리 예측하고, 맞춤형 지원 방안을 마련하세요.</p>
        <div style="margin-top: 1rem; font-size: 0.95rem; opacity: 0.85;">
            ✨ 간편한 입력 → 🤖 모델 분석 → 📊 시각화된 결과
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    (
        feature_cols,
        numeric_cols,
        categorical_cols,
        numeric_defaults,
        categorical_defaults,
        categorical_options,
        auto_fill_defaults,
        numeric_bounds,
        dataset_summary,
    ) = load_metadata()
    pipeline = load_pipeline()
except Exception as exc:
    st.error(f'모델 또는 메타데이터 로드 중 오류가 발생했습니다: {exc}')
    st.stop()

display_numeric_cols = [col for col in numeric_cols if col not in HIDDEN_FEATURES]
display_categorical_cols = [col for col in categorical_cols if col not in HIDDEN_FEATURES]

auto_fill_values: Dict[str, Any] = {}
for col in numeric_cols:
    default_config = numeric_defaults.get(col, {'value': 0.0})
    auto_fill_values[col] = default_config.get('value')
for col in categorical_cols:
    auto_fill_values[col] = categorical_defaults.get(col)
for col in feature_cols:
    if col not in auto_fill_values:
        auto_fill_values[col] = auto_fill_defaults.get(col)

feature_overview_rows: List[Dict[str, Any]] = []
for column in feature_cols:
    if column in numeric_cols:
        feature_type = '숫자형'
    elif column in categorical_cols:
        feature_type = '범주형'
    else:
        feature_type = '기타'
    preview_value = auto_fill_values.get(column)
    feature_overview_rows.append(
        {
            '피처': column,
            '한글 라벨': get_field_label(column),
            '유형': feature_type,
            '기본값 미리보기': '' if preview_value is None else preview_value,
        }
    )
feature_overview_df = pd.DataFrame(feature_overview_rows)

codebook_options_map: Dict[str, List[Dict[str, object]]] = {}
for column in CODEBOOK_LABELS:
    if column not in feature_cols or column in HIDDEN_FEATURES:
        continue
    options = build_codebook_options(column, categorical_options)
    if options:
        codebook_options_map[column] = options

codebook_display_cols = list(codebook_options_map.keys())
display_numeric_cols = [col for col in display_numeric_cols if col not in codebook_display_cols]
display_categorical_cols = [col for col in display_categorical_cols if col not in codebook_display_cols]

with st.sidebar:
    st.markdown('## 📘 사용 가이드')
    st.markdown(
        """
        <div class="sidebar-tips">
            <h4 style="color: #f9fafb; margin-bottom: 1rem; font-size: 1.1rem;">
                💡 이렇게 사용하세요!
            </h4>
            <ul style="list-style-type:none; padding-left:0; margin:0; line-height: 2;">
                <li style="margin-bottom: 0.8rem;">
                    <strong style="color: #60a5fa;">1단계</strong><br/>
                    아래 입력 폼에서 학생 정보를 입력하세요
                </li>
                <li style="margin-bottom: 0.8rem;">
                    <strong style="color: #60a5fa;">2단계</strong><br/>
                    기본값이 자동으로 채워져 있어요
                </li>
                <li style="margin-bottom: 0.8rem;">
                    <strong style="color: #60a5fa;">3단계</strong><br/>
                    필요한 항목만 수정하세요
                </li>
                <li>
                    <strong style="color: #60a5fa;">4단계</strong><br/>
                    예측 실행 버튼을 클릭! 🚀
                </li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.info("💡 **팁**: 모든 항목을 입력하지 않아도 예측이 가능합니다!")

if dataset_summary:
    st.markdown('### 📊 학습 데이터 통계')
    st.caption('모델이 학습한 데이터의 주요 통계입니다')
    metric_cols = st.columns(4)
    total_records = int(dataset_summary.get('row_count', 0))
    feature_count = int(dataset_summary.get('feature_count', 0))
    target_counts = dataset_summary.get('target_counts', {}) or {}
    total_target_count = sum(target_counts.values()) if target_counts else 0
    dropout_ratio = dataset_summary.get('dropout_ratio')
    graduate_ratio = dataset_summary.get('graduate_ratio')

    render_metric_card(metric_cols[0], '📚 학습 데이터', f"{total_records:,}명")
    render_metric_card(metric_cols[1], '🔍 분석 항목', f"{feature_count}개")
    dropout_display = f"{dropout_ratio * 100:.1f}%" if dropout_ratio is not None else '--'
    render_metric_card(metric_cols[2], '⚠️ 중퇴율', dropout_display)
    if graduate_ratio is not None:
        render_metric_card(metric_cols[3], '🎓 졸업률', f"{graduate_ratio * 100:.1f}%")
    elif total_target_count > 0 and target_counts:
        top_label = max(target_counts, key=target_counts.get)
        top_share = target_counts[top_label] / total_target_count
        render_metric_card(metric_cols[3], f'📈 최다 ({top_label})', f"{top_share * 100:.1f}%")
    else:
        render_metric_card(metric_cols[3], '🎓 졸업률', '--')
    st.markdown('---')
else:
    st.warning('⚠️ dataset.csv 파일을 찾을 수 없습니다.')

tab_predict, tab_feature, tab_insight = st.tabs(['🎯 예측하기', '📖 입력 항목 안내', '📊 데이터 분석'])

with tab_predict:
    st.markdown('### 🎯 학생 정보 입력')
    
    # 안내 메시지
    st.info(
        """
        💡 **입력 방법**
        - 기본값은 자동으로 설정되어 있습니다
        - 변경하고 싶은 항목만 수정하세요
        - 모든 항목을 입력하지 않아도 예측이 가능합니다
        """
    )

    with st.form('prediction_form'):
        input_data: Dict[str, Any] = {}

        if codebook_display_cols:
            st.markdown('#### 📝 기본 정보')
            st.caption('학생의 기본 정보를 선택해주세요')
            codebook_layout = st.columns(max(1, min(len(codebook_display_cols), 3)))
            for idx, column in enumerate(codebook_display_cols):
                options = codebook_options_map.get(column, [])
                if not options:
                    continue
                default_value = auto_fill_values.get(column)
                default_index = 0
                if default_value is not None:
                    for opt_idx, option in enumerate(options):
                        if str(option['value']) == str(default_value):
                            default_index = opt_idx
                            break
                with codebook_layout[idx % len(codebook_layout)]:
                    selection = st.selectbox(
                        get_field_label(column),
                        options,
                        index=default_index,
                        format_func=format_codebook_option,
                    )
                input_data[column] = selection['value']

        if display_numeric_cols:
            st.markdown('---')
            st.markdown('#### 📊 학업 성적 정보')
            st.caption('학생의 성적 및 학업 관련 정보를 입력하세요 (각 항목의 유효 범위 내에서 입력)')
            numeric_layout = st.columns(max(1, min(len(display_numeric_cols), 3)))
            for idx, column in enumerate(display_numeric_cols):
                default_config = numeric_defaults.get(column, {'value': 0.0, 'step': 0.1})
                raw_default = default_config['value']
                raw_step = default_config['step']
                
                # 최소/최대값 설정
                min_value = None
                max_value = None
                help_text = None
                min_max = numeric_bounds.get(column)
                
                # 학업 성적 정보 영역은 전부 정수형으로 처리
                is_integer_type = True
                
                if min_max is not None:
                    lower, upper = min_max
                    if lower is not None and upper is not None:
                        try:
                            min_value = int(float(lower))
                            max_value = int(float(upper))
                            help_text = f"⚠️ 유효 범위: {min_value} ~ {max_value}"
                        except (TypeError, ValueError):
                            pass
                
                with numeric_layout[idx % len(numeric_layout)]:
                    if is_integer_type:
                        # 정수형 입력
                        number_kwargs: Dict[str, Any] = {
                            'label': get_field_label(column),
                            'value': int(raw_default),
                            'step': 1,
                        }
                        if min_value is not None:
                            number_kwargs['min_value'] = min_value
                        if max_value is not None:
                            number_kwargs['max_value'] = max_value
                    else:
                        # 실수형 입력
                        number_kwargs: Dict[str, Any] = {
                            'label': get_field_label(column),
                            'value': float(raw_default),
                            'step': float(raw_step),
                        }
                        if min_value is not None:
                            number_kwargs['min_value'] = float(min_value)
                        if max_value is not None:
                            number_kwargs['max_value'] = float(max_value)
                    
                    if help_text is not None:
                        number_kwargs['help'] = help_text
                    
                    value = st.number_input(**number_kwargs)
                input_data[column] = value

        if display_categorical_cols:
            st.markdown('---')
            st.markdown('#### 📂 추가 정보')
            st.caption('기타 카테고리 정보를 선택하세요')
            categorical_layout = st.columns(max(1, min(len(display_categorical_cols), 2)))
            for idx, column in enumerate(display_categorical_cols):
                options = categorical_options.get(column, [])
                default_option = categorical_defaults.get(column, options[0] if options else '')
                with categorical_layout[idx % len(categorical_layout)]:
                    if options:
                        try:
                            default_index = options.index(default_option)
                        except ValueError:
                            default_index = 0
                        selection = st.selectbox(
                            get_field_label(column),
                            options,
                            index=default_index,
                        )
                    else:
                        selection = st.text_input(get_field_label(column), value=default_option)
                input_data[column] = selection

        other_columns = [
            col for col in feature_cols if col not in set(numeric_cols) | set(categorical_cols)
        ]
        if other_columns:
            st.markdown('---')
            st.markdown('##### 기타 피처')
            for column in other_columns:
                default_text = auto_fill_values.get(column)
                if default_text is None:
                    default_text = ''
                else:
                    default_text = str(default_text)
                input_data[column] = st.text_input(get_field_label(column), value=default_text)

        for hidden_feature in HIDDEN_FEATURES:
            if hidden_feature in feature_cols and hidden_feature not in input_data:
                input_data[hidden_feature] = auto_fill_values.get(hidden_feature)

        # 입력값 검증
        st.markdown('---')
        validation_errors = []
        
        for column in display_numeric_cols:
            value = input_data.get(column)
            if value is None:
                continue
                
            min_max = numeric_bounds.get(column)
            if min_max is not None:
                lower, upper = min_max
                if lower is not None and upper is not None:
                    try:
                        min_val = float(lower)
                        max_val = float(upper)
                        
                        if value < min_val or value > max_val:
                            validation_errors.append({
                                'column': column,
                                'label': get_field_label(column),
                                'value': value,
                                'min': min_val,
                                'max': max_val
                            })
                    except (TypeError, ValueError):
                        pass
        
        # 검증 결과 표시
        if validation_errors:
            st.error('❌ **입력값 오류가 발견되었습니다!**')
            st.markdown('**다음 항목들을 수정해주세요:**')
            
            for error in validation_errors:
                st.markdown(
                    f"""
                    <div style="background: #fee2e2; padding: 1rem; border-radius: 8px; 
                                margin: 0.5rem 0; border-left: 4px solid #ef4444;">
                        <strong style="color: #991b1b;">📍 {error['label']}</strong><br/>
                        <span style="color: #7f1d1d;">
                            입력값: <strong>{error['value']:.2f}</strong><br/>
                            유효 범위: <strong>{error['min']:.1f} ~ {error['max']:.1f}</strong>
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            st.warning('⚠️ 위 항목들을 유효 범위 내로 수정한 후 다시 시도해주세요.')
        else:
            if display_numeric_cols:
                st.success('✅ 모든 입력값이 유효합니다!')
        
        # Submit 버튼은 항상 생성 (조건부로 비활성화)
        submitted = st.form_submit_button(
            '🚀 예측 시작하기', 
            use_container_width=True, 
            type='primary',
            disabled=len(validation_errors) > 0
        )

    if submitted:
        try:
            for column in feature_cols:
                input_data.setdefault(column, None)
            input_df = pd.DataFrame([input_data], columns=feature_cols)

            if not hasattr(pipeline, 'predict'):
                raise AttributeError('로딩된 객체는 예측 기능을 제공하지 않습니다.')

            prediction = pipeline.predict(input_df)[0]
            dropout_prob = graduate_prob = None
            if hasattr(pipeline, 'predict_proba'):
                probabilities = pipeline.predict_proba(input_df)[0]
                dropout_prob = float(probabilities[0])
                graduate_prob = float(probabilities[1])

            st.success('✨ 예측이 완료되었습니다!')
            
            # 예측 결과 결정
            badge_text = 'Dropout' if prediction == 0 else 'Graduate'
            badge_color = '#ef4444' if prediction == 0 else '#10b981'
            badge_icon = '⚠️' if prediction == 0 else '🎓'
            description_text = (
                '학생의 중도 이탈 가능성이 더 높게 예측되었습니다.'
                if prediction == 0
                else '학생이 졸업할 가능성이 더 높게 예측되었습니다.'
            )
            
            # 예측 결과 헤더
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, {badge_color}15, {badge_color}25); 
                            padding: 2rem; border-radius: 20px; text-align: center; 
                            border: 2px solid {badge_color}50; margin-bottom: 2rem;
                            box-shadow: 0 8px 24px rgba(0,0,0,0.12);">
                    <div style="font-size: 4rem; margin-bottom: 1rem;">{badge_icon}</div>
                    <h2 style="margin: 0; color: #1f2937; font-size: 2rem;">🎯 예측 결과</h2>
                    <div style="margin: 1.5rem 0;">
                        <span style="background: {badge_color}; color: white; 
                                     padding: 0.8rem 2.5rem; border-radius: 50px; 
                                     font-size: 1.8rem; font-weight: bold; 
                                     box-shadow: 0 4px 12px {badge_color}40;">
                            {badge_text}
                        </span>
                    </div>
                    <p style="margin: 1rem 0 0 0; color: #475569; font-size: 1.15rem; font-weight: 500;">
                        {description_text}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            # 확률 시각화
            if dropout_prob is not None and graduate_prob is not None:
                st.markdown("### 📊 예측 확률 분석")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # 도넛 차트
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=['Dropout', 'Graduate'],
                        values=[dropout_prob * 100, graduate_prob * 100],
                        hole=.6,
                        marker=dict(
                            colors=['#ef4444', '#10b981'],
                            line=dict(color='#ffffff', width=4)
                        ),
                        textinfo='label+percent',
                        textfont=dict(size=16, color='#ffffff', family='Arial Black'),
                        hovertemplate='<b>%{label}</b><br>확률: %{value:.2f}%<extra></extra>',
                        pull=[0.05 if prediction == 0 else 0, 0.05 if prediction == 1 else 0]
                    )])
                    
                    # 중앙 텍스트
                    max_prob = max(dropout_prob, graduate_prob)
                    center_color = "#ef4444" if dropout_prob > graduate_prob else "#10b981"
                    
                    fig_pie.add_annotation(
                        text=f'<b>{max_prob * 100:.1f}%</b>',
                        x=0.5, y=0.5,
                        font=dict(size=36, color=center_color, family='Arial Black'),
                        showarrow=False
                    )
                    
                    fig_pie.update_layout(
                        title=dict(
                            text='<b>🍩 확률 분포</b>',
                            font=dict(size=20, color='#1f2937'),
                            x=0.5,
                            xanchor='center'
                        ),
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.2,
                            xanchor="center",
                            x=0.5,
                            font=dict(size=14, color='#1f2937', family='Arial')
                        ),
                        paper_bgcolor="rgba(0,0,0,0)",
                        height=400,
                        margin=dict(l=20, r=20, t=80, b=20)
                    )
                    
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # 게이지 차트
                    prediction_label = "Dropout" if dropout_prob > graduate_prob else "Graduate"
                    main_prob = max(dropout_prob, graduate_prob)
                    gauge_color = "#ef4444" if dropout_prob > graduate_prob else "#10b981"
                    
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=main_prob * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': f"<b>{prediction_label} 확률</b>", 
                               'font': {'size': 20, 'color': '#1f2937', 'family': 'Arial Black'}},
                        number={'suffix': '%', 'font': {'size': 48, 'color': gauge_color, 'family': 'Arial Black'}},
                        gauge={
                            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': gauge_color},
                            'bar': {'color': gauge_color, 'thickness': 0.8},
                            'bgcolor': "white",
                            'borderwidth': 3,
                            'bordercolor': "#e2e8f0",
                            'steps': [
                                {'range': [0, 33], 'color': '#fee2e2'},
                                {'range': [33, 66], 'color': '#fef3c7'},
                                {'range': [66, 100], 'color': '#d1fae5'}
                            ],
                            'threshold': {
                                'line': {'color': gauge_color, 'width': 6},
                                'thickness': 0.85,
                                'value': main_prob * 100
                            }
                        }
                    ))
                    
                    fig_gauge.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        font={'color': "#1f2937", 'family': "Arial"},
                        height=400,
                        margin=dict(l=20, r=20, t=80, b=20)
                    )
                    
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                # 상세 확률 카드
                st.markdown("---")
                detail_col1, detail_col2 = st.columns(2)
                
                with detail_col1:
                    st.markdown(
                        f"""
                        <div style="background: linear-gradient(135deg, #fee2e2, #fecaca); 
                                    padding: 2rem; border-radius: 16px; 
                                    border-left: 6px solid #ef4444;
                                    box-shadow: 0 4px 16px rgba(239, 68, 68, 0.2);
                                    transition: transform 0.2s;">
                            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                                <span style="font-size: 2.5rem; margin-right: 1rem;">⚠️</span>
                                <h3 style="margin: 0; color: #991b1b; font-size: 1.5rem;">Dropout</h3>
                            </div>
                            <p style="font-size: 3rem; font-weight: bold; margin: 1rem 0; 
                                      color: #7f1d1d; text-align: center;">
                                {dropout_prob * 100:.2f}%
                            </p>
                            <p style="margin: 0; color: #991b1b; font-size: 1rem; text-align: center;">
                                중도 이탈 가능성
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                with detail_col2:
                    st.markdown(
                        f"""
                        <div style="background: linear-gradient(135deg, #d1fae5, #a7f3d0); 
                                    padding: 2rem; border-radius: 16px; 
                                    border-left: 6px solid #10b981;
                                    box-shadow: 0 4px 16px rgba(16, 185, 129, 0.2);
                                    transition: transform 0.2s;">
                            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                                <span style="font-size: 2.5rem; margin-right: 1rem;">🎓</span>
                                <h3 style="margin: 0; color: #065f46; font-size: 1.5rem;">Graduate</h3>
                            </div>
                            <p style="font-size: 3rem; font-weight: bold; margin: 1rem 0; 
                                      color: #064e3b; text-align: center;">
                                {graduate_prob * 100:.2f}%
                            </p>
                            <p style="margin: 0; color: #065f46; font-size: 1rem; text-align: center;">
                                졸업 가능성
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # 인사이트 및 조언
            st.markdown("---")
            st.markdown("### 💬 모델의 조언")
            
            if dropout_prob > 0.7:
                st.error(
                    """
                    **⚠️ 높은 중도 이탈 위험**
                    - 학생 상담 및 멘토링 프로그램 참여를 권장합니다
                    - 학업 지원 프로그램을 적극 활용하세요
                    - 정기적인 학습 진도 체크가 필요합니다
                    """
                )
            elif dropout_prob > 0.4:
                st.warning(
                    """
                    **⚡ 주의가 필요한 상태**
                    - 학습 패턴을 점검해보세요
                    - 교수님 또는 학업 상담사와 면담을 고려하세요
                    - 동료 학습 그룹 참여를 추천합니다
                    """
                )
            else:
                st.success(
                    """
                    **✅ 안정적인 학업 상태**
                    - 현재의 좋은 패턴을 유지하세요
                    - 지속적인 자기 관리가 중요합니다
                    - 학업 목표를 향해 꾸준히 나아가세요
                    """
                )

            with st.expander('📋 입력한 데이터 확인하기', expanded=False):
                st.json(json.dumps(input_data, ensure_ascii=False, indent=2))
        except Exception as exc:
            st.error(f'❌ 예측 중 오류가 발생했습니다: {exc}')
    else:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #dbeafe, #bfdbfe); 
                        padding: 2rem; border-radius: 16px; text-align: center;
                        border: 2px solid #3b82f6;">
                <h3 style="color: #1e40af; margin-bottom: 1rem;">� 시작해볼까요?</h3>
                <p style="color: #1e40af; font-size: 1.1rem; margin: 0;">
                    위에서 학생 정보를 입력하고<br/>
                    <strong>"🚀 예측 시작하기"</strong> 버튼을 눌러주세요!
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

with tab_feature:
    st.markdown('### 📖 입력 항목 가이드')
    st.caption('각 입력 항목에 대한 설명입니다')
    if not feature_overview_df.empty:
        st.dataframe(
            feature_overview_df.sort_values(by='피처'),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info('피처 정보를 불러오지 못했습니다.')

    st.markdown("---")
    
    if codebook_display_cols:
        st.markdown("#### 🏷️ 선택 항목별 코드 설명")
        with st.expander('📚 코드북 상세 보기', expanded=False):
            for column in codebook_display_cols:
                options = codebook_options_map.get(column, [])
                if not options:
                    continue
                st.markdown(f"##### {get_field_label(column)}")
                st.dataframe(pd.DataFrame(options), use_container_width=True, hide_index=True)
                st.markdown("")

    visible_categorical = [
        col for col in categorical_cols if col not in HIDDEN_FEATURES and col not in codebook_display_cols
    ]
    if visible_categorical:
        st.markdown("#### 📂 카테고리 항목 선택지")
        with st.expander('🔍 가능한 값 확인하기', expanded=False):
            for column in visible_categorical:
                options = categorical_options.get(column, [])
                if not options:
                    continue
                st.markdown(f"##### {get_field_label(column)}")
                st.info(', '.join(str(opt) for opt in options))

    hidden_columns = sorted(set(feature_cols).intersection(HIDDEN_FEATURES))
    if hidden_columns:
        st.markdown("---")
        with st.expander('🔒 자동 처리되는 숨김 항목', expanded=False):
            st.caption('다음 항목들은 자동으로 처리되어 입력하지 않아도 됩니다:')
            st.write(', '.join(hidden_columns))

with tab_insight:
    st.markdown('### 📊 데이터 분석 인사이트')
    st.caption('모델이 학습한 데이터의 패턴과 통계를 확인하세요')
    
    if dataset_summary:
        target_counts = dataset_summary.get('target_counts', {}) or {}
        if target_counts:
            st.markdown('##### Target 분포')
            target_df = pd.DataFrame(
                {
                    'Target': list(target_counts.keys()),
                    'Count': [int(value) for value in target_counts.values()],
                }
            ).set_index('Target')
            st.bar_chart(target_df)

        numeric_range_rows: List[Dict[str, Any]] = []
        for column, bounds in numeric_bounds.items():
            if column in HIDDEN_FEATURES:
                continue
            lower, upper = bounds
            lower_display: Any = ''
            upper_display: Any = ''
            if lower is not None:
                try:
                    lower_float = float(lower)
                    lower_display = int(lower_float) if lower_float.is_integer() else round(lower_float, 2)
                except (TypeError, ValueError):
                    lower_display = lower
            if upper is not None:
                try:
                    upper_float = float(upper)
                    upper_display = int(upper_float) if upper_float.is_integer() else round(upper_float, 2)
                except (TypeError, ValueError):
                    upper_display = upper
            numeric_range_rows.append(
                {
                    '피처': column,
                    '최소값': lower_display,
                    '최대값': upper_display,
                }
            )

        if numeric_range_rows:
            st.markdown("---")
            st.markdown("#### 📏 숫자형 데이터 범위")
            with st.expander('📊 상세 범위 확인하기', expanded=False):
                range_df = pd.DataFrame(numeric_range_rows).sort_values(by='피처')
                st.dataframe(
                    range_df, 
                    use_container_width=True, 
                    hide_index=True,
                    column_config={
                        '피처': st.column_config.TextColumn('항목', width='large'),
                        '최소값': st.column_config.NumberColumn('최소값', format='%.2f'),
                        '최대값': st.column_config.NumberColumn('최대값', format='%.2f'),
                    }
                )

        st.markdown("---")
        st.info('📌 **참고**: 모든 통계와 범위는 실제 학습에 사용된 dataset.csv 데이터를 기준으로 합니다.')
    else:
        st.warning('⚠️ dataset.csv 파일을 찾을 수 없습니다. 데이터 분석을 위해서는 데이터 파일이 필요합니다.')
