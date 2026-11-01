# 배포만 빠르게 하기

1. ZIP을 다운로드하고 압축을 풉니다.
2. GitHub 저장소의 최상위에서 **Add file → Upload files**를 엽니다.
3. **app.py / physics.py / visuals.py / requirements.txt** 네 파일을 함께 올립니다.
   같은 이름의 예전 파일은 이번 것으로 교체합니다. `app (1).py`처럼 이름이 바뀌지 않게 주의하세요.
4. **Commit changes**를 누릅니다.
5. Streamlit의 설정을 다음처럼 지정합니다.

```text
Repository: 파일을 올린 저장소
Branch: main  (실제 올린 브랜치)
Main file path: app.py
Advanced settings → Python: 3.12 권장
Secrets: 비워 둠
```

**ZIP 파일 자체는 앱의 진입 파일이 아닙니다.**
Main file path에 ZIP 이름이나 로컬 컴퓨터 경로를 넣지 마세요.

저장소 화면을 열었을 때 바로 `app.py`가 보이는 구조가 가장 간단합니다.
만약 `nernst_lab/app.py`처럼 하위 폴더에 넣었다면 Main file path도 똑같이 써야 합니다.

이번 버전은 첫 화면에 큰 네른스트 식과 실제 숫자 대입이 보입니다.
그 화면이 아니라면 예전 파일 또는 다른 브랜치의 앱을 보고 있을 수 있습니다.

상세 사용법과 오류 해결은 README.md, 검증 범위는 VALIDATION.md를 보세요.
