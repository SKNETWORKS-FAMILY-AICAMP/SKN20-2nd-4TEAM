import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder


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
    'Curricular units 1st sem (approved)': '1학기 이수 학점(승인)',
    'Curricular units 1st sem (grade)': '1학기 이수 학점(성적)',
    'Curricular units 2nd sem (approved)': '2학기 이수 학점(승인)',
    'Curricular units 2nd sem (grade)': '2학기 이수 학점(성적)',
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


@st.cache_resource(show_spinner=False)
def load_pipeline():
    model_path = Path('../model/model_trained.pkl')
    if not model_path.exists():
        raise FileNotFoundError('학습된 모델 파일(model_trained.pkl)을 찾을 수 없습니다. 먼저 학습을 수행하세요.')
    return joblib.load(model_path)


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

    numeric_defaults: Dict[str, Dict[str, float | int]] = {}
    for col in numeric_cols:
        value, step = sanitize_numeric_default(numeric_defaults_raw.get(col))
        numeric_defaults[col] = {'value': value, 'step': step}

    categorical_defaults: Dict[str, str] = {}
    categorical_options: Dict[str, List[str]] = {}
    for col in categorical_cols:
        options = categorical_options_raw.get(col, [])
        categorical_options[col] = options
        categorical_defaults[col] = sanitize_categorical_default(
            categorical_defaults_raw.get(col), options
        )

    return (
        feature_names,
        numeric_cols,
        categorical_cols,
        numeric_defaults,
        categorical_defaults,
        categorical_options,
    )


st.set_page_config(page_title='학생 이탈 예측', layout='wide')
st.title('학생 이탈(졸업 여부) 예측')
st.write('저장된 파이프라인을 사용하여 학생 정보를 입력하면 Dropout 확률을 예측합니다.')

try:
    (
        feature_cols,
        numeric_cols,
        categorical_cols,
        numeric_defaults,
        categorical_defaults,
        categorical_options,
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
    auto_fill_values.setdefault(col, None)

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
    st.header('입력 방법 안내')
    st.markdown(
        """
        - 학습된 파이프라인이 요구하는 피처만 입력하면 됩니다.
        - 숫자형 값은 기본 제안값을 사용하거나 자유롭게 수정하세요.
        - 범주형 값은 제공된 선택지를 사용하거나 직접 입력할 수도 있습니다.
        - 값을 변경한 후 **예측 실행** 버튼을 눌러 결과를 확인하세요.
        """
    )

with st.form('prediction_form'):
    st.subheader('학생 특성 입력')
    input_data: Dict[str, Any] = {}

    if codebook_display_cols:
        st.markdown('---')
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
        numeric_layout = st.columns(max(1, min(len(display_numeric_cols), 3)))
        for idx, column in enumerate(display_numeric_cols):
            default_config = numeric_defaults.get(column, {'value': 0.0, 'step': 0.1})
            default_value = default_config['value']
            step_value = default_config['step']
            with numeric_layout[idx % len(numeric_layout)]:
                value = st.number_input(
                    label=get_field_label(column),
                    value=default_value,
                    step=step_value,
                )
            input_data[column] = value

    if display_categorical_cols:
        st.markdown('---')
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
        for column in other_columns:
            input_data[column] = st.text_input(get_field_label(column), value='')

    for hidden_feature in HIDDEN_FEATURES:
        if hidden_feature in feature_cols and hidden_feature not in input_data:
            input_data[hidden_feature] = auto_fill_values.get(hidden_feature)

    submitted = st.form_submit_button('예측 실행')

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

        st.success('예측이 완료되었습니다.')
        col_pred, col_prob = st.columns(2)
        with col_pred:
            st.metric('예측 결과', 'Dropout' if prediction == 0 else 'Graduate')
        if dropout_prob is not None and graduate_prob is not None:
            with col_prob:
                st.metric('Dropout 확률', f'{dropout_prob * 100:.2f}%')
                st.metric('Graduate 확률', f'{graduate_prob * 100:.2f}%')

        st.markdown('### 입력 값 확인')
        st.json(json.dumps(input_data, ensure_ascii=False, indent=2))
    except Exception as exc:
        st.error(f'예측 중 오류가 발생했습니다: {exc}')
