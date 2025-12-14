start "Backend Server" cmd /k "cd backend && py -m uvicorn main:app --reload --port 8000"
start "Frontend Client" cmd /k "cd frontend && npm run dev"
timeout /t 3 >nul
start http://localhost:5173