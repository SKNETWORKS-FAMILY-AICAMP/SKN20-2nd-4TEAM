![팀명 소개](https://capsule-render.vercel.app/api?type=waving&color=gradient&height=200&section=header&text=Drop%20Signal%20Detector&fontSize=60)
# SKN20-2nd-4TEAM  
# 🎓 학생 학업 중도 이탈률 예측 프로젝트  
---
## 👥 팀원 소개  
### 팀명 : Drop Signal Detector (자퇴 시그널 감지조)

<table>
  <tr>
    <td valign="top" align="center" width="200">
      <img src="./img/final_squid.png" width="110"><br>
      <b>김황현</b><br>
      데이터 엔지니어<br>
      <sub>전체 프로젝트 기획<br>파이프라인 설계 및 모델링 전략 수립</sub>
    </td>
    <td valign="top" align="center" width="200">
      <img src="./img/final_ddoong.png" width="110"><br>
      <b>이도경</b><br>
      머신러닝 담당<br>
      <sub>EDA, 전처리 및 변수 중요도 분석<br>피처 엔지니어링</sub>
    </td>
    <td valign="top" align="center" width="200">
      <img src="./img/final_snail.png" width="110"><br>
      <b>이지은</b><br>
      데이터 분석 담당<br>
      <sub>모델 학습 및 튜닝<br>성능 평가, 시각화 보고서 제작</sub>
    </td>
    <td valign="top" align="center" width="200">
      <img src="./img/haepari.png" width="110"><br>
      <b>정소영</b><br>
      머신러닝 담당<br>
      <sub>데이터 전처리<br>모델링 전략 수립</sub>
    </td>
    <td valign="top" align="center" width="200">
      <img src="./img/final_squirral.png" width="110"><br>
      <b>최유정</b><br>
      데이터 엔지니어 / UI<br>
      <sub>EDA, 데이터 전처리<br>Streamlit 화면 설계, GitHub 관리</sub>
    </td>
  </tr>
</table>



🔗 **GitHub Repository**: [SKNETWORKS-FAMILY-AICAMP / SKN20-2nd-4TEAM](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN20-2nd-4TEAM)

---
## 📖 WBS (Work Breakdown Structure)

| 작업 명 | 담당자 | 산출물 |
|----------|---------|---------|
| 프로젝트 주제 및 데이터 선정 | 김황현, 이도경, 이지은, 정소영, 최유정 | 없음 |
| 데이터 수집 및 전처리 | 정소영, 이도경, 최유정 | 전처리 코드, 이상치/결측치 처리 보고서 |
| 탐색적 데이터 분석 (EDA) | 이도경, 최유정 | Heatmap, 변수 상관분석 결과, EDA 마크다운 보고서 |
| 피처 엔지니어링 및 변수 선택 | 이도경, 이지은 | Feature Importance, 변수 선정 코드 |
| 머신러닝 모델링 (Logistic, RandomForest, LightGBM) | 김황현, 정소영 | 모델 학습 코드, 파이프라인 설계, 성능 비교 결과 |
| 모델 성능 평가 및 튜닝 | 이지은, 김황현 | F1, AUC, Confusion Matrix, 최적화 결과 |
| 딥러닝 모델링 (MLP) | 이도경 | PyTorch MLP 학습 코드 및 실험 결과 |
| Streamlit 웹 구현 | 이도경, 최유정 | Streamlit UI 코드, `app.py`, 페이지별 구성 |
| Streamlit 통합 및 테스트 | 김황현, 최유정 | 실시간 예측 테스트 결과, UI 동작 점검 |
| README.md 작성 및 프로젝트 구조 정리 | 최유정 | GitHub README.md, 프로젝트 폴더 구조 |
| 발표자료 제작 (PPT / PDF) | 최유정 | 프로젝트 발표자료, Github 시각자료 |
| 최종 점검 및 회고 작성 | 전원 | 회고 섹션, 기술 블로그 초안 |

---
## 🧩 프로젝트 개요  

학생 개개인의 학적, 학업 성취, 재정 상태, 가정 요인 등을 종합 분석하여  
**학업 중도 이탈(Dropout) 가능성을 예측**하는 머신러닝 기반 프로젝트입니다.  

- **목표**: 학생이 졸업(Graduate) 혹은 자퇴(Dropout) 중 어떤 경로를 밟을지 사전에 예측  
- **데이터 출처**: [Predict Students’ Dropout and Academic Success (Kaggle)](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention)  
- **데이터 규모**: 4,424명 × 35개 변수  
- **주요 변수**: 학적, 학업 성취, 장학금, 등록금, 부모 학력·직업, 거시경제 변수 등  

---

## 📊 데이터 분석 및 전처리  

| 단계 | 주요 내용 |
|------|------------|
| 1️⃣ 데이터 확인 | 결측치 없음 / 수치형·범주형 혼합 데이터 |
| 2️⃣ 타깃 인코딩 | Dropout=1, Enrolled/Graduate=0 |
| 3️⃣ 필요 없는 컬럼 제거 | `Unemployment rate`, `Nacionality`, `Course` 등 6개 제거 |
| 4️⃣ 범주형 처리 | One-Hot Encoding, Label Encoding 병행 |
| 5️⃣ 수치형 처리 | StandardScaler로 정규화 |
| 6️⃣ 불균형 데이터 대응 | `class_weight='balanced'` + `SMOTE` 적용 |

---
## 📊 탐색적 데이터 분석 (EDA)

### 1. 전체 상관 행렬 (Full Correlation Heatmap)
<p align="center">
  <img src="./figures/correlation_heatmap_full.png" width="800">
</p>

> Spearman 상관계수를 기반으로 변수 간 관계를 분석했습니다.  
> `Curricular units`(학업 성취 관련 변수) 간 강한 양의 상관을 보였으며,  
> `Debtor`, `Tuition fees up to date`, `Scholarship holder` 등 재정 요인은 Dropout과 음의 상관을 보였습니다.

---

### 2. 주요 변수 중심 상관 행렬 (Top Features Heatmap)
<p align="center">
  <img src="./figures/correlation_heatmap_top.png" width="800">
</p>

> 학업 성취도 및 재정 요인 중심으로 주요 상관 변수를 시각화했습니다.  
> Dropout 예측에 특히 기여하는 핵심 변수군을 도출했습니다.

---

### 3. 타깃별 수치형 변수 분포 (Numeric Distribution by Target)
<p align="center">
  <img src="./figures/numeric_distribution_by_target.png" width="800">
</p>

> 학업 성취 관련 변수들이 Dropout/Graduate 그룹 간 어떻게 차이를 보이는지 비교했습니다.  
> Dropout 학생군은 성취도 지표(`Curricular units grade`, `enrolled`, `approved`)가 전반적으로 낮게 나타났습니다.

---

### 4. 수치형 변수 히스토그램 (Numeric Histogram by Target)
<p align="center">
  <img src="./figures/numeric_histogram_by_target.png" width="800">
</p>

> `Age at enrollment`, `Curricular units grade`, `Evaluations` 등 주요 변수의 분포를 시각화하여  
> 이탈 학생의 연령대 및 학업 패턴의 특징을 파악했습니다.

---

### 5. 타깃 상관 변수 바그래프 (Target Correlation Bar)
<p align="center">
  <img src="./figures/target_correlation_bar.png" width="800">
</p>

> Dropout 여부(Target)와 각 주요 변수 간의 상관계수를 막대그래프로 표현했습니다.  
> 재정 요인(`Debtor`, `Tuition fees up to date`)과 학업 성취 변수(`Curricular units grade`)가  
> 모델 예측에 높은 기여도를 보였습니다.

---

### 6. 타깃 분포 (Target Distribution)
<p align="center">
  <img src="./figures/target_distribution.png" width="600">
</p>

> 전체 데이터에서 Dropout 학생의 비율은 약 40%로 확인되었으며,  
> 데이터 불균형 문제 해결을 위해 `SMOTE` 및 `class_weight='balanced'` 기법을 병행 적용했습니다.

### 사용 컬럼 요약  

| 구분 | 컬럼 수 | 컬럼명 |
|----------|----------|--------|
| **사용 컬럼 (28)** | 28 | `Age at enrollment`, `Application mode`, `Application order`, `Curricular units 1st/2nd sem (approved, grade, enrolled, evaluations...)`, `Daytime/evening attendance`, `Debtor`, `Displaced`, `Gender`, `Scholarship holder`, `Tuition fees up to date`, `Mother's qualification`, `Father's occupation`, `GDP`, `Inflation rate`, `Previous qualification`, `Marital status` 등 |
| **제거 컬럼 (6)** | 6 | `Course`, `Educational special needs`, `Father's qualification`, `International`, `Nacionality`, `Unemployment rate` |

> **제거 사유:**  
> - 개인 특성과 상관이 낮은 외부 변수(`Unemployment rate`, `Nacionality`)  
> - 불균형 또는 다중공선성(`Father's qualification`)  

---

## ⚙️ 사용 기술 스택  

| 구분 | 기술 |
|------|------|
| 언어 | ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) |
| 분석 | ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white) |
| 모델링 | ![RandomForest](https://img.shields.io/badge/RandomForest-228B22?style=for-the-badge&logo=scikitlearn&logoColor=white) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) |
| 시각화 | ![Matplotlib](https://img.shields.io/badge/Matplotlib-003049?style=for-the-badge&logo=plotly&logoColor=white) ![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white) |
| 웹 | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white) |
| 협업 | ![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white) ![Discord](https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white) |



---

## 🤖 머신러닝 및 딥러닝 모델링  

| 모델 | Accuracy | Precision | Recall | F1 |
|------|-----------|------------|--------|----|
| **Random Forest (최종)** | **0.9146** | **0.8975** | **0.9706** | **0.9326** |
| XGBoost | 0.9118 | 0.9004 | 0.9615 | 0.9299 |
| AdaBoost | 0.9049 | 0.8926 | 0.9593 | 0.9248 |
| LightGBM | 0.9036 | 0.8924 | 0.9570 | 0.9236 |
| Decision Tree | 0.9008 | 0.8854 | 0.9615 | 0.9219 |
| Logistic Regression | 0.9008 | 0.9093 | 0.9299 | 0.9195 |

> **최종 모델: Random Forest**  
> - **Accuracy:** 0.91  
> - **Precision:** 0.92  
> - **Recall:** 0.91  
> - **F1-score:** 0.91  
>
> **클래스별 세부 성능:**  
> - Graduate → Precision **0.95**, Recall **0.83**, F1 **0.88**  
> - Dropout → Precision **0.90**, Recall **0.97**, F1 **0.93**  
>
> 모델은 **Dropout(이탈 학생)** 탐지에서 높은 재현율(Recall)을 기록했으며,  
> 실제 이탈 학생의 약 **97%**를 정확히 분류했습니다.  
> 따라서 **조기 위험 감지(Early Warning)** 목적에 가장 적합한 모델로 **RandomForest**를 선정했습니다.

---

## 💻 Streamlit 구현  

> 머신러닝 모델을 기반으로 학생 개별 데이터를 입력받아 **이탈 확률을 예측**하고,  
> 예측 결과를 시각적으로 표시하는 웹 애플리케이션입니다.  

---

### 🎯 주요 기능  
- 학생 정보 입력 시 **Dropout 확률 실시간 예측**  
- **Low / Medium / High Risk** 단계별 위험도 시각화  
- 주요 **Feature Importance 및 입력 데이터 분석 결과** 표시  
- 사용 가이드를 포함한 **대시보드형 UI 구성**  
- Streamlit 기반 **실시간 상호작용형 분석 시스템**

---

### 🖥️ 서비스 화면

#### 1️⃣ 메인 대시보드  
<p align="center">
  <img src="./img/streamlit1.png" width="900"><br>
  <sub>학생 이탈 예측 시스템 메인 페이지 — 학습 데이터 통계 및 입력 가이드</sub>
</p>

---

#### 2️⃣ 데이터 인사이트 페이지  
<p align="center">
  <img src="./img/streamlit2.png" width="900"><br>
  <sub>데이터 분포, 주요 변수 범위 및 모델 특징 요약을 제공</sub>
</p>

---

#### 3️⃣ 예측 결과 페이지  
<p align="center">
  <img src="./img/streamlit3.png" width="900"><br>
  <sub>입력값 기반 예측 결과 — Dropout vs Graduate 확률 시각화</sub>
</p>

---

## 🔍 예측 결과 및 성능 평가  

| 지표 | 값 |
|------|------|
| **Precision** | 0.89 |
| **Recall** | 0.87 |
| **F1-score** | 0.93 |
| **ROC-AUC** | 0.94 |

> 모델은 이탈 학생을 높은 정밀도와 재현율로 분류했으며,  
> Confusion Matrix 분석 결과 실제 Dropout 학생의 87%를 정확히 탐지했습니다.

---

## 🔍 모델 활용 방안  

- 학업 이탈 가능성이 높은 학생을 조기 탐지하여 상담·장학 지원 연계  
- 예측 결과를 학교 EWS(Early Warning System)에 적용  
- 장기적으로 학업 유지율(RETENTION RATE) 향상 기여  

---

## 💭 팀 회고  

| 이름 | 한 줄 회고 |
|------|-------------|
| 김황현 | 여러 시행착오 끝에 주제가 최종적으로 정해졌을 때 조원들의 어두웠던 얼굴이 환하게 밝아지는 모습을 보며 주제 선택의 중요성에 대해 느낄 수 있었습니다. 수업시간에 배운 내용을 활용해볼 수 있는 좋은 시간이었고, 좋은 조원분들 만나 즐겁게 프로젝트를 할 수 있어 좋았습니다. 다들 고생하셨습니다. |
| 이도경 | 데이터가 생각처럼 움직이지 않아 답답했지만, 포기하지 않고 꾸준히 문제를 해결하는 과정에서 머신러닝의 진정한 재미를 느꼈습니다. 이번 경험으로 기술을 익히는 것도 중요하지만 '끈기'가 정말 중요하다는 것을 깨달았습니다. |
| 이지은 | 활용할 데이터를 찾기가 쉽지가 않아서 어떻게 해야하나 걱정이 많았는데, 다행히 좋은 데이터를 찾고 팀원분들과 함께 노력하여 잘 마무리할 수 있어서 뿌듯했습니다. 딥러닝 시간에 배웠던 다양한 모델들을 직접 적용해보고, 관련 데이터를 수집하며 학습했던 내용을 실제로 적용해볼 수 있어 좋았습니다. 여전히 많이 부족하지만 이번 팀 프로젝트로 한층 더 성장할 수 있게 된 거 같아 기쁩니다. 함께 해주신 조원분들 정말 고생 많으셨고 감사합니다! |
| 정소영 | 데이터 전처리부터 머신러닝·딥러닝 모델 실험까지 진행하며, 직접 F1 스코어가 나왔을 때의 뿌듯함이 아직도 기억납니다. 주제를 조금 더 빨리 정했다면 더 깊게 탐구할 수 있지 않았을까 하는 아쉬움도 있지만, 멋진 팀원분들 덕분에 훌륭한 결과물을 낼 수 있었다고 생각합니다. 다들 정말 고생 많으셨습니다! |
| 최유정 | 너무 고생하셨습니다 ♡!! 이번 프로젝트를 통해 실제 데이터를 바탕으로 학생 이탈 예측 모델을 성공적으로 구현했으며, 데이터 분석부터 웹 서비스 구현까지 머신러닝 파이프라인 전반을 경험할 수 있었습니다. 특히, 팀원들의 적극적인 협업 덕분에 데이터셋 찾는 난관을 극복하고 기대 이상의 완성도 높은 결과물을 만들어낼 수 있었던 것 같습니다! |

---

## 🗂️ 프로젝트 구조  

<pre>
SKN20-2nd-4TEAM
│
├── 01_preprocessing_report/
│   └── Students'_EDA
│
├── 02_training_report/
│   ├── project.ipynb
│   └── data/
│       └── dataset.csv
│
├── 03_trained_model/
│   ├── model_trained.pkl
│   └── feature_importance.png
│
├── images/
│   ├── heatmap_all_image.png
│   ├── heatmap_later.png
│   ├── streamlit_ui.png
│   └── delete_column.png
│
├── app.py
│
├── pages/
│   ├── input_form.py
│   └── result.py
│
├── only_test_streamlit_guidance.txt
├── 2차 프로젝트 발표자료.pdf
└── README.md
</pre>
