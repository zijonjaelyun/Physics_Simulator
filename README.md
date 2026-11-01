# Nernst Lab v2 · 네른스트 전지 실험실

**네른스트 식과 값 대입을 첫 화면에 놓고, 농도·온도·부하를 바꾸며 다니엘 전지와 이온 수송을 탐구하는 Streamlit 앱입니다.**

## 이번 버전에서 바뀐 핵심

- 첫 화면: 일반식, 현재 값 대입, Q·log₁₀Q·전압 보정·평형 전압을 항상 표시합니다.
- Cu²⁺·Zn²⁺ 농도와 온도를 바꾸면 초기 전지 그림과 계산값이 즉시 갱신됩니다. 별도 적용 버튼이 없습니다.
- 시간 재생: 현재 시각의 **표면 농도**를 식에 대입하여 전압과 전류, 농도 분포를 같이 표시합니다.
- 심화: 평균/표면/단자 전압 분리, 균일 혼합 vs 유한 저장액 확산, 휴지 중 회복, 같은 Q/다른 총량, 통제 변수 비교, 무작위 보행.
- 실제 측정 자료는 포함되어 있지 않습니다. 모든 결과는 명시된 이상 모델의 계산입니다.

## 1. GitHub → Streamlit 배포: 이 순서대로

### GitHub에 올릴 파일

ZIP 자체를 올리지 말고 **압축을 푼 내용물**을 저장소 최상위에 올리세요.

```text
Physics_Simulator/             ← GitHub 저장소의 최상위
├── app.py                    ← Main file path = app.py
├── physics.py
├── visuals.py
├── requirements.txt
├── .streamlit/
│   └── config.toml           ← 권장: 화면 테마
├── README.md
├── MODEL.md
├── VALIDATION.md
├── requirements-dev.txt
├── tests/
└── sample_replay.html        ← 선택: 한 실험의 독립 재생본
```

**필수 파일은 `app.py`, `physics.py`, `visuals.py`, `requirements.txt` 네 개입니다.**
이전 버전의 `app.py`만 덮어쓰지 말고 네 파일을 모두 이번 버전으로 맞추세요.
새 앱은 예전 `rendering.py`, `brownian_view.py`, `assets/player.html`을 사용하지 않습니다.

GitHub 웹사이트에서:

1. 기존 저장소를 엽니다.
2. **Add file → Upload files**로 위 내용물을 업로드합니다.
3. 같은 이름의 파일은 이번 버전으로 교체하고 **Commit changes**합니다.
4. 저장소 파일 목록에 `app.py`가 바로 보이는지 확인합니다.
5. 같은 폴더의 `requirements.txt`가 이번 ZIP의 내용인지 확인합니다.

폴더 전체를 한 단계 아래에 올렸다면 파일 경로도 달라집니다. 예를 들어
`nernst_lab/app.py`로 올라갔다면 Main file path도 `nernst_lab/app.py`여야 합니다.
가장 단순한 구조는 위 예시처럼 저장소 루트에 네 파일을 놓는 것입니다.

### Streamlit Community Cloud 설정

| 입력란 | 설정 |
|---|---|
| Repository | 방금 파일을 올린 GitHub 저장소 |
| Branch | 그 파일들이 있는 브랜치(보통 `main`) |
| Main file path | **`app.py`** |
| Advanced settings → Python | **3.12 권장** |
| Secrets | 필요 없음 |

이후 Deploy를 누릅니다. 이미 같은 경로로 배포한 앱이 있다면 커밋 변경 사항이 반영됩니다.
화면이 그대로면 저장소 경로·브랜치·커밋을 확인하고 앱 관리 메뉴에서 재시작하세요.

**공개 URL의 실제 배포는 이 프로젝트 전달에 포함되지 않았습니다.** 내 계정이나 API 키는 필요하지 않습니다.

공식 안내:
- [파일 배치](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization)
- [배포 입력란](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [파일 변경 반영](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/edit-your-app)

## 2. 로컬 실행

Python 3.12를 권장합니다.

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Windows에서 독립 환경을 쓰는 경우:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

브라우저에서 터미널에 표시된 Local URL을 여세요.
JavaScript 빌드, Node.js, API 키, 외부 이미지·폰트 CDN은 필요 없습니다.

`streamlit==1.55.0`을 지정했습니다. 최신 버전이라고 주장하는 것이 아니라,
이 소스가 사용하는 HTML 구성 요소 API를 고정하기 위한 선택입니다.
다른 버전으로 변경하면 포함된 AppTest와 브라우저 검사를 다시 실행하세요.

## 3. 사용 순서

### 01 · 네른스트 식 조절

왼쪽에서 초기 Cu²⁺ 농도, Zn²⁺ 농도, 온도를 바꿉니다.

화면에 동시에 표시되는 값:

1. 네른스트 일반식.
2. 현재 수치를 넣은 로그 계산.
3. 반응 지수 Q와 log₁₀Q.
4. E°에 더해지는 농도 보정 ΔE.
5. 전지 그림, 평형 전압, 단자 전압, 부하 전류.

이 화면은 **초기 조건의 즉시 비교**입니다. 그림 속 대표점의 움직임만으로
농도가 시간에 따라 줄지는 않습니다. 시간 변화는 다음 화면에서 계산합니다.
점은 같은 시각 두 용액의 농도비를 근사하는 표식이고, 색은 실제 용액 색이 아니라 종 구분용입니다.

E°와 n은 자유 슬라이더가 아닙니다. 이 반응의 전자 수는 n=2이며 E°는 교육용 1.10 V로 고정합니다.
온도를 바꾸면 RT/(2F)만 바뀝니다. 실제 E°(T), D(T), r(T)는 계산하지 않습니다.

### 02 · 방전과 이온 이동

선택한 초기 조건으로 두 모델을 계산합니다. 재생·일시정지·초기화·시간 이동·배속·모델 선택을 지원합니다.

- **평균 농도 네른스트 값 E_avg**
- **표면 농도 네른스트 값 E_s**
- **단자 전압 V_term = E_s − Ir**

세 값을 구분합니다. 재생 상단의 식은 **현재 표시 시각의 농도**로 갱신됩니다.
그래프의 시각과 전지 그림도 동일한 기록을 사용합니다.
계산값은 최대 약 800개 프레임으로 추려 재생하므로 시간 슬라이더의 요청 시각 바로 이전 기록을 표시할 수 있습니다.
표시되는 시각·식·그림은 서로 동일합니다. CSV에는 모든 계산 기록이 들어갑니다.

회로 연결 방식을 ‘중간 휴지 후 재방전’으로 바꾸면 전체 관찰 시간의 35–65%에서 I=0이 됩니다.
휴지 중에도 확산은 계속되므로 표면 농도와 전압의 회복을 관찰할 수 있습니다.
너무 강한 부하로 휴지 전에 모델 하한에 도달하면 그 이전에 종료될 수 있습니다.

### 03 · 심화 탐구

A. **로그 법칙:** 현재 계산값을 가상 관측 기록으로 저장하고 온도별 E–logQ 기울기를 비교합니다. 실제 측정 데이터가 아닙니다.  
B. **같은 Q, 다른 총량:** 같은 초기 전압과 화학량론적 전하량 상한(mAh)의 차이를 확인합니다.  
C. **한 변수 비교:** 외부 저항 또는 확산계수만 바꿔 다섯 후보를 동일한 유효 시간에서 비교합니다.  
D. **입자 운동:** 독립적인 1차원 무작위 보행과 평균제곱변위 2Dt를 비교합니다. 전지의 반응 입자를 추적하는 모델은 아닙니다.

### 04 · 검증과 모델

모델 식·경계 조건·생략한 효과·출처를 읽고, 물질·전하·회로식의 보존 오차와 수렴을 검사합니다.

## 4. 처음 해 볼 실험

**실험 1: 농도비 10배**  
25 °C에서 Zn²⁺ 농도만 0.1 M → 1.0 M로 바꿉니다. 초기 전압은 약 29.58 mV 낮아집니다.
Cu²⁺만 10배로 바꾸면 반대로 높아집니다.

**실험 2: 같은 전압, 다른 양**  
두 농도를 같은 배수로 바꿉니다. Q가 같으므로 초기 평형 전압은 같지만 구리 이온의 총몰수는 달라집니다.

**실험 3: 네른스트 식의 입력 농도는 어느 농도인가?**  
평균 농도와 구리 전극 표면 농도를 비교합니다. 강한 부하에서 표면 고갈이 먼저 나타날 수 있습니다.

**실험 4: 휴지 중 무엇이 회복되는가?**  
전류가 0인 구간에서 누적 전하가 증가하지 않는지, 표면 농도는 어떻게 변하는지 확인합니다.

## 5. 오류가 나타날 때

| 증상 | 먼저 확인할 것 |
|---|---|
| Main file path를 찾지 못함 | GitHub에서 `app.py`가 실제 어느 폴더에 있는지 |
| `No module named physics` 또는 `visuals` | 두 모듈을 `app.py`와 같은 폴더에 업로드했는지 |
| `No module named scipy/plotly` | 이번 `requirements.txt`가 함께 올라갔는지 |
| 예전 화면이 나옴 | 다른 브랜치·폴더의 `app.py`를 배포하고 있지 않은지 |
| 계산이 바로 종료됨 | 강한 부하에서 표면 농도 하한에 도달한 것은 아닌지 |
| 확산층 부피 오류 | 전극 면적×확산층 두께보다 구리 총부피가 충분히 큰지 |
| 입력 후 로딩이 길어짐 | 관찰 시간·격자 수·시간 간격을 기본값으로 되돌리기 |
| 브라우저 입력 요소가 비정상 | 앱을 재시작하고 브라우저를 새로고침한 뒤 Cloud 로그 확인 |

`requirements-dev.txt`는 개발 테스트용이며 배포 의존 목록은 `requirements.txt`입니다.

## 6. 테스트

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

선택적인 브라우저 검사(Playwright와 Chromium 설치 필요):

```bash
python -m pip install playwright
python -m playwright install chromium
python tests/browser_smoke.py
```

브라우저 검사 스크립트는 시스템 Chromium이 있으면 이를 사용하고, 없으면 Playwright Chromium을 사용합니다.

제작 환경에서 실제로 통과한 검증과 실행하지 못한 검증은 **VALIDATION.md**를 확인하세요.
브라우저 구성 요소의 실행을 Streamlit 전체 서버 실행과 혼동하지 않습니다.

## 7. 중요한 모델 한계

활동도를 농도로 근사하고, 충분한 지지전해질과 빠른 전극 반응을 가정합니다.
농도 구배가 있는 상태의 `E_s`는 국소 가역 계면 가정의 값이지 전지 전체가 평형이라는 뜻이 아닙니다.
실제 염다리 전위·부반응·활성화 과전압·전극 구조 변화는 생략합니다.

표면 농도가 10⁻⁸ M에 접근하면 계산을 종료합니다.
이는 0 V 도달, 구리 이온 전체 소진, 실제 전지 수명이 아닙니다.
전압을 보기 좋게 만들려고 0 V로 강제하지 않습니다.

수식과 출처의 상세 연결은 **MODEL.md**에 있습니다.
