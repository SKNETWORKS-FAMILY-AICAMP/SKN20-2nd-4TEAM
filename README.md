# SKN20-2nd-4TEAM  
# 🎓 학생 학업 중도 이탈률 예측 프로젝트  

---

## 👥 팀원 소개  
### 팀명 : Drop Signal Detector (자퇴 시그널 감지조)

<table>
  <tr>
    <td align="center">
      <img src="./img/leader.png" width="100"><br>
      <b>김황현</b><br>
      팀장 / 데이터 엔지니어<br>
      <sub>전체 프로젝트 기획, 파이프라인 설계 및 모델링 전략 수립</sub>
    </td>
    <td align="center">
      <img src="./img/patrick.jpeg" width="100"><br>
      <b>이도경</b><br>
      데이터 분석 담당<br>
      <sub>EDA, 전처리 및 변수 중요도 분석, 시각화 보고서 제작</sub>
    </td>
    <td align="center">
      <img src="./img/model.png" width="100"><br>
      <b>이지은</b><br>
      머신러닝 담당<br>
      <sub>모델 학습 및 튜닝, 성능 평가, 피처 엔지니어링</sub>
    </td>
    <td align="center">
      <img src="./img/deep.png" width="100"><br>
      <b>정소영</b><br>
      딥러닝 담당<br>
      <sub>신경망 설계, 딥러닝 모델 구현 및 성능 비교</sub>
    </td>
    <td align="center">
      <img src="./img/frontend.png" width="100"><br>
      <b>최유정</b><br>
      프론트엔드/UI<br>
      <sub>Streamlit 화면 설계 및 발표 자료 제작, README 정리</sub>
    </td>
  </tr>
</table>


🔗 **GitHub Repository**: [SKNETWORKS-FAMILY-AICAMP / SKN20-2nd-4TEAM](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN20-2nd-4TEAM)

---

## 🧩 프로젝트 개요  

학생 개개인의 배경, 학업 성취, 경제·사회적 요인 등을 종합적으로 고려하여  
**학업 중도 이탈(자퇴) 가능성을 예측**하는 머신러닝 기반 프로젝트입니다.  

- **목표**: 학생이 재학 중 졸업(Graduate) 혹은 자퇴(Dropout) 중 어떤 경로를 밟을지 예측  
- **데이터 출처**: [Higher Education: Predictors of Student Retention (Kaggle)](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention)  
- **데이터 규모**: 4,424개 샘플 / 35개 컬럼  
- **주요 사용 변수**: 학적 정보, 학업 성취, 등록금 납부, 장학금, 부모 학력 및 직업 등  

---

## 📊 데이터 분석 및 전처리  

| 단계 | 주요 내용 |
|------|------------|
| 1️⃣ 데이터 확인 | 4,424행 × 35열 / 결측치 없음 / 혼합형 변수 구성 |
| 2️⃣ 타깃 인코딩 | Dropout=1, Enrolled/Graduate=0 |
| 3️⃣ 불필요 변수 제거 | 실업률, 물가상승률, GDP 등 외부 경제 변수 제거 |
| 4️⃣ 범주형 처리 | One-Hot Encoding 적용 |
| 5️⃣ 수치형 처리 | StandardScaler로 정규화 |
| 6️⃣ 불균형 데이터 대응 | class_weight="balanced" 옵션 사용 |

---

## ⚙️ 사용 기술 스택  

| 구분 | 기술 |
|------|------|
| 언어 | ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) |
| 분석 | ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white) |
| 모델링 | ![LightGBM](https://img.shields.io/badge/LightGBM-9DCE00?style=for-the-badge&logo=lightgbm&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) |
| 시각화 | ![Matplotlib](https://img.shields.io/badge/Matplotlib-003049?style=for-the-badge&logo=plotly&logoColor=white) ![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white) |
| 웹 | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white) |
| 협업 | ![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white) ![Notion](https://img.shields.io/badge/Notion-000000?style=for-the-badge&logo=notion&logoColor=white) |

---

## 🤖 머신러닝 모델링  

| 모델 | Accuracy | F1 Score | AUC |
|------|-----------|-----------|-----|
| Logistic Regression | 0.875 | 0.881 | 0.902 |
| Random Forest | 0.902 | 0.911 | 0.924 |
| XGBoost | 0.915 | 0.927 | 0.938 |
| **LightGBM (Best)** | **0.921** | **0.934** | **0.942** |

> LightGBM이 가장 높은 F1 Score를 기록하여 최종 모델로 선정됨.  
> class_weight="balanced" 옵션으로 불균형 문제 완화.  

---

## 🧠 딥러닝 모델 결과  

| 지표 | Epoch 1 | Epoch 50 | 변화량 |
|------|----------|----------|--------|
| Train Loss | 0.62 | 0.28 | ▼ 54.8% |
| Validation Accuracy | 77.3% | 86.9% | ▲ 9.6% |
| F1 Score | 0.79 | 0.88 | ▲ 11.4% |

> PyTorch 기반 MLP 모델 사용, EarlyStopping과 Dropout 적용으로 과적합 방지.  

---

## 💻 Streamlit 웹 서비스  

- 학생 개별 정보 입력 → Dropout 확률 예측  
- 주요 기능:
  - 예측 확률 실시간 시각화
  - 위험군 학생 자동 분류
  - 변수 중요도 시각화 제공  

📺 **시연 영상:** (추후 업로드 예정)  
📷 **UI 예시:**  (추후 업로드 예정)

---

## 🔍 예측 모델의 향후 활용 방안  

- 이탈 가능성이 높은 학생을 조기 식별하여 학사 경고 및 상담 지원  
- 학교 상담센터·장학팀과 연계해 선제적 학업 관리  
- 장기적으로는 학업 유지율(RETENTION RATE) 개선 가능  

---

## 💭 팀 회고  

| 이름 | 한 줄 회고 |
|------|-------------| 
| 김황현 | 머신러닝 파이프라인의 중요성을 직접 설계하며 실무 감각을 익혔습니다. |
| 이도경 | 데이터 정제와 시각화를 통해 피처 간 관계를 깊이 이해했습니다. |
| 이지은 | 다양한 모델과 파라미터 튜닝을 통해 성능 향상을 체험했습니다. |
| 정소영 | 프로젝트를 문서화하며 결과물을 정리하는 과정에서 큰 보람을 느꼈습니다. |
| 최유정 | 딥러닝의 확장성과 모델 해석력의 차이를 직접 확인했습니다. |



