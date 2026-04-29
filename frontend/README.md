# ELARA Frontend

## Dev mode
```bash
npm install
./setup.sh   # installs shadcn UI components
npm run dev  # runs at http://localhost:3000
```

## Production
```bash
docker compose up --build frontend
```

## Notes
- Logo images (`memora-dark.png`, `memora-light.png`, `memora-favicon.png`) are not included.
  Copy them from `LLM-Agent-For--elders-main/frontend/public/` into `frontend/public/` before building.
- All API calls proxy through Next.js routes to avoid CORS issues:
  - `/api/auth/login` → `localhost:8001/auth/login`
  - `/api/auth/signup` → `localhost:8001/auth/signup`
  - `/api/chat` → `localhost:8001/chat`
  - `/api/profile/[userId]` → `localhost:8001/get_profile/{userId}`
  - `/api/memories/[userId]` → `localhost:8001/get_memories/{userId}`
