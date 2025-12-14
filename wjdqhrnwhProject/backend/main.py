from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from rag_core import NasdaqRagBot

app = FastAPI()

# 1. CORS 설정 (기본)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. ★ [특단의 조치] OPTIONS 요청 강제 처리 (에러 방지용) ★
@app.options("/{rest_of_path:path}")
async def preflight_handler(rest_of_path: str):
    return JSONResponse(
        content="OK",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        },
    )

class QueryRequest(BaseModel):
    query: str

# 봇 초기화
try:
    bot = NasdaqRagBot()
    print("✅ Bot Loaded")
except Exception as e:
    print(f"❌ Bot Fail: {e}")
    bot = None

@app.post("/chat")
def chat(request: QueryRequest):
    if not bot:
        raise HTTPException(status_code=500, detail="Bot not initialized")
    return {"answer": bot.get_answer(request.query)}

@app.get("/")
def read_root():
    return {"status": "alive"}
