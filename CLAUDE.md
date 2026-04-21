# Project: Dota 2 Match Analyzer

## Development Workflow
- Use 4 parallel agents for large implementation tasks, splitting by file ownership to avoid edit conflicts

## Run
```bash
./run.sh
# or: backend/venv/bin/uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Tech Stack
- Backend: Python 3 + FastAPI (uvicorn)
- Frontend: Vue 3 + ECharts + Tailwind CSS (CDN)
- Parser: Clarity (Java) with Python wrapper
- Language: Chinese，English

## Key Conventions
- All user-facing text supports `lang` parameter ("zh" or "en")
- Log messages should be in English (developer-facing)
- Parser is abstraction layer (mock/clarity/future parsers)
- Map coords: game space (-8000,8000) mapped to (0,100) for ECharts
