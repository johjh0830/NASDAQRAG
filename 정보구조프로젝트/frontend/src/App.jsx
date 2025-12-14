import { useState, useRef, useEffect } from 'react';
import './App.css';

function App() {
  const [messages, setMessages] = useState([
    { role: 'bot', text: '주식 투자 보조 에이전트입니다. 무엇을 도와드릴까요?' }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = { role: 'user', text: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      // Backend API 호출
      const response = await fetch('https://nasdaqrag.onrender.com/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: userMessage.text }),
      });

      if (!response.ok) {
        throw new Error('Server Error');
      }

      const data = await response.json();
      const botMessage = { role: 'bot', text: data.answer };
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error("Error:", error);
      setMessages(prev => [...prev, { role: 'bot', text: '오류가 발생했습니다. 서버 상태를 확인해주세요.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="container">
      <h1 className="header">📈 Nasdaq AI Analyst</h1>
      
      <div className="chat-window">
        {messages.map((msg, index) => (
          <div 
            key={index} 
            className={`message ${msg.role}`}
            // ▼▼▼ [핵심 수정] 줄바꿈을 인식하도록 스타일 추가 ▼▼▼
            style={{ whiteSpace: "pre-wrap", textAlign: "left" }} 
          >
            {msg.text}
          </div>
        ))}
        {isLoading && <div className="loading">분석 중입니다...</div>}
        <div ref={messagesEndRef} />
      </div>

      <div className="input-area">
        <input 
          type="text" 
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyPress}
          placeholder="예: 엔비디아 숏칠까?"
          disabled={isLoading}
        />
        <button onClick={handleSend} disabled={isLoading}>
          {isLoading ? '...' : '전송'}
        </button>
      </div>
    </div>
  );
}

export default App;