# PTM-CoScientist Web UI

Streamlit 기반의 테스트용 Web UI입니다. 추후 PTM-platform 프론트엔드에 통합될 예정입니다.

## 기능

- **파이프라인 실행:** Order Code와 연구 목표를 입력하여 Co-Scientist 파이프라인 시작
- **가설 & 토너먼트 결과:** Elo 레이팅 분포 차트, 가설 상세 내용 (IF/THEN/BECAUSE), 지지/반박 문헌 근거, 토너먼트 기록
- **실험 설계:** 상위 가설에 대한 구체적인 실험 프로토콜 (접근법, 시약, 대조군, 예상 결과)
- **Scientist Feedback:** 방향 제시, 제약 조건, 초기 아이디어를 통한 AI 가설 방향 조정

## 실행 방법

### 로컬 실행 (개발용)

```bash
# 1. API 서버 먼저 실행
coscientist serve

# 2. 별도 터미널에서 Web UI 실행
cd PTM-CoScientist
streamlit run webui/app.py
```

브라우저에서 `http://localhost:8501` 접속

### Docker로 실행

```bash
docker-compose up -d
```

- Web UI: `http://localhost:8501`
- API: `http://localhost:8080`
