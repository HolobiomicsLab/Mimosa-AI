import React, { useState, useEffect, useRef, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import axios from 'axios';
import './App.css';
import { colors } from './colors';

// Constants
const API_BASE_URL = process.env.BACKEND_PORT || 'http://0.0.0.0:8000';
const POLLING_INTERVAL = 3000;
const VIEW_TYPES = {
  BLOCKS: 'blocks',
  SCREENSHOT: 'screenshot',
  THINKING: 'thinking'
};

const MESSAGE_TYPES = {
  USER: 'user',
  AGENT: 'agent',
  ERROR: 'error'
};

// Custom hooks
const usePolling = (callback, interval, dependencies = []) => {
  useEffect(() => {
    const intervalId = setInterval(callback, interval);
    return () => clearInterval(intervalId);
  }, dependencies);
};

const useScrollToBottom = () => {
  const messagesEndRef = useRef(null);
  
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  return { messagesEndRef, scrollToBottom };
};

// Utility functions
const normalizeAnswer = (answer) => {
  if (!answer) return '';
  return answer
    .trim()
    .toLowerCase()
    .replace(/\s+/g, ' ')
    .replace(/[.,!?]/g, '');
};

const isDuplicateMessage = (messages, newAnswer) => {
  if (!newAnswer?.trim()) return false;
  
  const normalizedNewAnswer = normalizeAnswer(newAnswer);
  return messages.some(msg => 
    normalizeAnswer(msg.content) === normalizedNewAnswer
  );
};

// API service
const apiService = {
  async checkHealth() {
    try {
      await axios.get(`${API_BASE_URL}/health`);
      return true;
    } catch {
      return false;
    }
  },

  async fetchScreenshot() {
    const timestamp = new Date().getTime();
    const response = await axios.get(
      `${API_BASE_URL}/screenshots/updated_screen.png?timestamp=${timestamp}`,
      { responseType: 'blob' }
    );
    return URL.createObjectURL(response.data);
  },

  async fetchLatestAnswer() {
    const response = await axios.get(`${API_BASE_URL}/latest_answer`);
    return response.data;
  },

  async submitQuery(query) {
    const response = await axios.post(`${API_BASE_URL}/query`, {
      query,
      tts_enabled: false
    });
    return response.data;
  },

  async stopAgent() {
    const response = await axios.get(`${API_BASE_URL}/stop`);
    return response.data;
  }
};

// Components
const Header = () => (
  <header className="header">
    <img src="/icons/mimosa.svg" alt="Mimosa Icon" className="header-icon" />
    <h1>Mimosa-AI</h1>
  </header>
);

const Message = ({ message, index }) => {
  const getMessageClassName = (type) => {
    switch (type) {
      case MESSAGE_TYPES.USER:
        return 'user-message';
      case MESSAGE_TYPES.AGENT:
        return 'agent-message';
      default:
        return 'error-message';
    }
  };

  return (
    <div key={index} className={`message ${getMessageClassName(message.type)}`}>
      {message.type === MESSAGE_TYPES.AGENT && (
        <span className="agent-name">{message.agentName}</span>
      )}
      <ReactMarkdown>{message.content}</ReactMarkdown>
    </div>
  );
};

const MessagesContainer = ({ messages, messagesEndRef, isThinkingView = false }) => (
  <div className="messages">
    {messages.length === 0 ? (
      <p className="placeholder">
        {isThinkingView ? 'No thinking yet.' : 'No messages yet. Type below to start!'}
      </p>
    ) : (
      messages.map((msg, index) => (
        <div key={index} className={`message ${
          msg.type === MESSAGE_TYPES.USER ? 'user-message' :
          msg.type === MESSAGE_TYPES.AGENT ? 'agent-message' : 'error-message'
        }`}>
          {msg.type === MESSAGE_TYPES.AGENT && (
            <span className="agent-name">{msg.agentName}</span>
          )}
          <ReactMarkdown>
            {isThinkingView ? msg.reasoning : msg.content}
          </ReactMarkdown>
        </div>
      ))
    )}
    <div ref={messagesEndRef} />
  </div>
);

const StatusIndicator = ({ isOnline, isLoading, status }) => {
  if (!isLoading && !isOnline) {
    return <p className="loading-animation">System offline. Deploy backend first.</p>;
  }
  
  if (isOnline) {
    return <div className="loading-animation">{status}</div>;
  }
  
  return null;
};

const InputForm = ({ query, setQuery, onSubmit, onStop, isLoading }) => (
  <form onSubmit={onSubmit} className="input-form">
    <input
      type="text"
      value={query}
      onChange={(e) => setQuery(e.target.value)}
      placeholder="Type your query..."
      disabled={isLoading}
    />
    <button type="submit" disabled={isLoading}>
      Send
    </button>
    <button type="button" onClick={onStop}>
      Stop
    </button>
  </form>
);

const ViewSelector = ({ currentView, setCurrentView, hasScreenshot, onScreenshotClick }) => (
  <div className="view-selector">
    <button
      className={currentView === VIEW_TYPES.BLOCKS ? 'active' : ''}
      onClick={() => setCurrentView(VIEW_TYPES.BLOCKS)}
    >
      Editor View
    </button>
    <button
      className={currentView === VIEW_TYPES.SCREENSHOT ? 'active' : ''}
      onClick={hasScreenshot ? () => setCurrentView(VIEW_TYPES.SCREENSHOT) : onScreenshotClick}
    >
      Browser View
    </button>
    <button
      className={currentView === VIEW_TYPES.THINKING ? 'active' : ''}
      onClick={() => setCurrentView(VIEW_TYPES.THINKING)}
    >
      Reasoning view
    </button>
  </div>
);

const BlocksView = ({ responseData }) => (
  <div className="blocks">
    {responseData?.blocks && Object.values(responseData.blocks).length > 0 ? (
      Object.values(responseData.blocks).map((block, index) => (
        <div key={index} className="block">
          <p className="block-tool">Tool: {block.tool_type}</p>
          <pre>{block.block}</pre>
          <p className="block-feedback">Feedback: {block.feedback}</p>
          {block.success ? (
            <p className="block-success">Success</p>
          ) : (
            <p className="block-failure">Failure</p>
          )}
        </div>
      ))
    ) : (
      <div className="block">
        <p className="block-tool">Tool: No tool in use</p>
        <pre>No file opened</pre>
      </div>
    )}
  </div>
);

const ScreenshotView = ({ responseData }) => (
  <div className="screenshot">
    <img
      src={responseData?.screenshot || 'placeholder.png'}
      alt="Screenshot"
      onError={(e) => {
        e.target.src = 'placeholder.png';
        console.error('Failed to load screenshot');
      }}
      key={responseData?.screenshotTimestamp || 'default'}
    />
  </div>
);

const ThinkingView = ({ messages, messagesEndRef }) => (
  <div className="thinking">
    <MessagesContainer 
      messages={messages} 
      messagesEndRef={messagesEndRef} 
      isThinkingView={true} 
    />
  </div>
);

const ChatSection = ({ 
  messages, 
  query, 
  setQuery, 
  onSubmit, 
  onStop, 
  isLoading, 
  isOnline, 
  status, 
  messagesEndRef 
}) => (
  <div className="chat-section">
    <h2>Chat Interface</h2>
    <MessagesContainer messages={messages} messagesEndRef={messagesEndRef} />
    <StatusIndicator isOnline={isOnline} isLoading={isLoading} status={status} />
    <InputForm 
      query={query}
      setQuery={setQuery}
      onSubmit={onSubmit}
      onStop={onStop}
      isLoading={isLoading}
    />
  </div>
);

const ComputerSection = ({ 
  currentView, 
  setCurrentView, 
  responseData, 
  error, 
  messages, 
  messagesEndRef, 
  onScreenshotClick 
}) => {
  const renderContent = () => {
    if (error) {
      return <p className="error">{error}</p>;
    }

    switch (currentView) {
      case VIEW_TYPES.BLOCKS:
        return <BlocksView responseData={responseData} />;
      case VIEW_TYPES.THINKING:
        return <ThinkingView messages={messages} messagesEndRef={messagesEndRef} />;
      case VIEW_TYPES.SCREENSHOT:
        return <ScreenshotView responseData={responseData} />;
      default:
        return <BlocksView responseData={responseData} />;
    }
  };

  return (
    <div className="computer-section">
      <h2>Computer View</h2>
      <ViewSelector
        currentView={currentView}
        setCurrentView={setCurrentView}
        hasScreenshot={!!responseData?.screenshot}
        onScreenshotClick={onScreenshotClick}
      />
      <div className="content">
        {renderContent()}
      </div>
    </div>
  );
};

// Main App Component
function App() {
  // State management
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentView, setCurrentView] = useState(VIEW_TYPES.BLOCKS);
  const [responseData, setResponseData] = useState(null);
  const [isOnline, setIsOnline] = useState(false);
  const [status, setStatus] = useState('Agents ready');
  
  const { messagesEndRef, scrollToBottom } = useScrollToBottom();

  // API functions with error handling
  const checkHealth = useCallback(async () => {
    try {
      const online = await apiService.checkHealth();
      setIsOnline(online);
      console.log(`System is ${online ? 'online' : 'offline'}`);
    } catch (error) {
      console.error('Health check failed:', error);
      setIsOnline(false);
    }
  }, []);

  const fetchScreenshot = useCallback(async () => {
    try {
      const imageUrl = await apiService.fetchScreenshot();
      console.log('Screenshot fetched successfully');
      
      setResponseData((prev) => {
        // Clean up previous blob URL
        if (prev?.screenshot && prev.screenshot !== 'placeholder.png') {
          URL.revokeObjectURL(prev.screenshot);
        }
        return {
          ...prev,
          screenshot: imageUrl,
          screenshotTimestamp: new Date().getTime()
        };
      });
    } catch (error) {
      console.error('Error fetching screenshot:', error);
      setResponseData((prev) => ({
        ...prev,
        screenshot: 'placeholder.png',
        screenshotTimestamp: new Date().getTime()
      }));
    }
  }, []);

  const fetchLatestAnswer = useCallback(async () => {
    try {
      const data = await apiService.fetchLatestAnswer();
      updateResponseData(data);
      
      if (!data.answer?.trim()) return;
      
      if (!isDuplicateMessage(messages, data.answer)) {
        const newMessage = {
          type: MESSAGE_TYPES.AGENT,
          content: data.answer,
          reasoning: data.reasoning,
          agentName: data.agent_name,
          status: data.status,
          uid: data.uid,
        };
        
        setMessages(prev => [...prev, newMessage]);
        setStatus(data.status);
        scrollToBottom();
      } else {
        console.log('Duplicate answer detected, skipping:', data.answer);
      }
    } catch (error) {
      console.error('Error fetching latest answer:', error);
    }
  }, [messages, scrollToBottom]);

  const updateResponseData = useCallback((data) => {
    setResponseData(prev => ({
      ...prev,
      blocks: data.blocks || prev?.blocks || null,
      done: data.done,
      answer: data.answer,
      agent_name: data.agent_name,
      status: data.status,
      uid: data.uid,
    }));
  }, []);

  // Event handlers
  const handleSubmit = async (e) => {
    e.preventDefault();
    await checkHealth();
    
    if (!query.trim()) {
      console.log('Empty query');
      return;
    }

    // Add user message
    setMessages(prev => [...prev, { type: MESSAGE_TYPES.USER, content: query }]);
    setIsLoading(true);
    setError(null);

    try {
      console.log('Sending query:', query);
      setQuery('waiting for response...');
      
      const data = await apiService.submitQuery(query);
      
      setQuery('Enter your query...');
      console.log('Response:', data);
      updateResponseData(data);
    } catch (error) {
      console.error('Error:', error);
      setError('Failed to process query.');
      setMessages(prev => [
        ...prev,
        { type: MESSAGE_TYPES.ERROR, content: 'Error: Unable to get a response.' }
      ]);
    } finally {
      console.log('Query completed');
      setIsLoading(false);
      setQuery('');
    }
  };

  const handleStop = async (e) => {
    e.preventDefault();
    await checkHealth();
    setIsLoading(false);
    setError(null);
    
    try {
      await apiService.stopAgent();
      setStatus("Requesting stop...");
    } catch (error) {
      console.error('Error stopping the agent:', error);
    }
  };

  const handleGetScreenshot = async () => {
    try {
      setCurrentView(VIEW_TYPES.SCREENSHOT);
    } catch (error) {
      setError('Browser not in use');
    }
  };

  // Polling effect
  const pollingCallback = useCallback(() => {
    checkHealth();
    fetchLatestAnswer();
    fetchScreenshot();
  }, [checkHealth, fetchLatestAnswer, fetchScreenshot]);

  usePolling(pollingCallback, POLLING_INTERVAL, [messages]);

  // Cleanup effect for blob URLs
  useEffect(() => {
    return () => {
      if (responseData?.screenshot && responseData.screenshot !== 'placeholder.png') {
        URL.revokeObjectURL(responseData.screenshot);
      }
    };
  }, [responseData?.screenshot]);

  return (
    <div className="app">
      <Header />
      <main className="main">
        <div className="app-sections">
          <ChatSection
            messages={messages}
            query={query}
            setQuery={setQuery}
            onSubmit={handleSubmit}
            onStop={handleStop}
            isLoading={isLoading}
            isOnline={isOnline}
            status={status}
            messagesEndRef={messagesEndRef}
          />
          <ComputerSection
            currentView={currentView}
            setCurrentView={setCurrentView}
            responseData={responseData}
            error={error}
            messages={messages}
            messagesEndRef={messagesEndRef}
            onScreenshotClick={handleGetScreenshot}
          />
        </div>
      </main>
    </div>
  );
}

export default App;