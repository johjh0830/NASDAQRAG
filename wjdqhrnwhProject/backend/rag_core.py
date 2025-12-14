import os
import shutil
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# ★★★ [여기가 수정되었습니다] ★★★
from langchain_community.vectorstores import Chroma 
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage

# 0.1.20 버전 호환 임포트
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain import hub
from langchain.tools.retriever import create_retriever_tool

load_dotenv()

class NasdaqRagBot:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        
        self.stock_cache = {}
        self.cache_expire_minutes = 10
        
        self.llm = ChatOpenAI(
            openai_api_key=self.api_key, 
            model_name="gpt-4o-mini", 
            temperature=0
        )
        self.embedding_model = OpenAIEmbeddings(
            openai_api_key=self.api_key, 
            model="text-embedding-3-small"
        )
        
        self.vectorstore = None
        self.retriever = None
        self.setup_rag_system()
        
        self.agent_executor = self.setup_agent()

    def setup_rag_system(self):
        """나스닥 데이터 관리 및 RAG 초기화"""
        print("[System] 나스닥 데이터 캐싱 확인 중...")
        os.makedirs("data", exist_ok=True)
        csv_path = "data/nasdaq_history.csv"
        db_path = "./chroma_db"
        
        today_str = datetime.now().strftime("%Y-%m-%d")
        is_cache_valid = False
        
        if os.path.exists(csv_path) and os.path.exists(db_path):
            file_timestamp = os.path.getmtime(csv_path)
            file_date = datetime.fromtimestamp(file_timestamp).strftime("%Y-%m-%d")
            if file_date == today_str:
                is_cache_valid = True

        if is_cache_valid:
            print(f"✅ [Smart Skip] 오늘의 나스닥 데이터가 이미 존재합니다.")
            self.vectorstore = Chroma(
                persist_directory=db_path,
                embedding_function=self.embedding_model,
                collection_name="nasdaq_history_v2"
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            return

        print(f"🔄 [Update] 나스닥 최신 데이터를 다운로드합니다...")
        if os.path.exists(db_path): shutil.rmtree(db_path)

        try:
            df = yf.download("^IXIC", start="2010-01-01", end=today_str, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(1)
            df = df.reset_index()
            df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"[Warning] 다운로드 실패: {e}")
            return

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, parse_dates=["Date"])
            df.set_index("Date", inplace=True)
            if len(df.columns) >= 4:
                df.columns = ["Open", "High", "Low", "Close", "Volume"][:len(df.columns)]
            
            try:
                monthly = df["Close"].resample("ME").agg(["first", "last"])
            except:
                monthly = df["Close"].resample("M").agg(["first", "last"])
            monthly["return"] = (monthly["last"] / monthly["first"] - 1) * 100
            
            docs_text = []
            docs_meta = []
            for date, row in monthly.iterrows():
                if pd.isna(row['first']): continue
                text = (f"{date.year}년 {date.month}월 나스닥 시장: "
                        f"{'상승' if row['return'] > 0 else '하락'} 마감 ({row['return']:.2f}%).")
                docs_text.append(text)
                docs_meta.append({"year": date.year, "month": date.month})
            
            self.vectorstore = Chroma.from_texts(
                texts=docs_text,
                metadatas=docs_meta,
                embedding=self.embedding_model,
                collection_name="nasdaq_history_v2",
                persist_directory=db_path
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

    def setup_agent(self):
        # 1. RAG 도구
        retriever_tool = create_retriever_tool(
            self.retriever,
            "nasdaq_history_search",
            "나스닥의 과거 흐름이나 역사적 데이터를 검색할 때 사용."
        )
        
        # 2. 단순 조회 도구
        @tool
        def get_stock_price(ticker: str):
            """현재 주가 단순 조회"""
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                return str({"price": info.get("currentPrice"), "pe": info.get("trailingPE")})
            except:
                return "조회 실패"

        # 3. 기술적 분석 도구
        @tool
        def analyze_technical_indicators(ticker: str):
            """매수/매도 판단을 위한 기술적 지표 분석"""
            ticker = ticker.upper().strip()
            now = datetime.now()

            if ticker in self.stock_cache:
                cached = self.stock_cache[ticker]
                time_diff = now - cached["timestamp"]
                if time_diff < timedelta(minutes=self.cache_expire_minutes):
                    print(f"🚀 [Fast Load] '{ticker}' 캐시 사용.")
                    return str(cached["data"]) + " (Note: 캐시된 데이터)"

            print(f"🌍 [Download] '{ticker}' API 호출...")
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1y")
                
                if len(hist) < 100: return "데이터 부족"

                hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
                hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
                
                delta = hist['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs)).iloc[-1]

                current = hist['Close'].iloc[-1]
                sma_50 = hist['SMA_50'].iloc[-1]
                sma_200 = hist['SMA_200'].iloc[-1]
                
                result_data = {
                    "ticker": ticker,
                    "current_price": round(current, 2),
                    "rsi": round(rsi, 2),
                    "sma_50": round(sma_50, 2),
                    "sma_200": round(sma_200, 2),
                    "golden_cross": bool(sma_50 > sma_200),
                    "price_above_sma200": bool(current > sma_200)
                }
                
                self.stock_cache[ticker] = {"data": result_data, "timestamp": now}
                return str(result_data)

            except Exception as e:
                return f"에러: {e}"

        tools = [retriever_tool, get_stock_price, analyze_technical_indicators]
        
        system_msg = """
        당신은 냉철한 '주식 투자 보조 에이전트'입니다.
        사용자가 특정 종목의 매수/매도 여부를 물으면 반드시 아래 포맷을 엄격하게 지켜서 답변하세요.

        [답변 포맷]
        # 1. 결론: [매수 추천 / 매도 추천 / 관망] 중 택 1
        
        # 2. 투자 근거 (5가지)
        1. (기술적 지표 분석 - RSI, 이평선 등)
        2. (시장 상황 - RAG 도구 활용)
        3. (현재 주가 위치)
        4. (추세 설명)
        5. (종합 평가)

        # 3. 반대 의견 및 리스크 (3가지)
        - (반대 논리 1)
        - (반대 논리 2)
        - (반대 논리 3)

        (마지막 줄에 '투자의 책임은 본인에게 있습니다' 명시)
        """
        
        prompt = hub.pull("hwchase17/openai-functions-agent")
        prompt.messages[0] = SystemMessage(content=system_msg)
        
        agent = create_openai_functions_agent(self.llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=True)

    def get_answer(self, query: str):
        try:
            return self.agent_executor.invoke({"input": query})["output"]
        except Exception as e:
            return f"오류 발생: {str(e)}"
