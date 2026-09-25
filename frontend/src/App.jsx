import { useEffect, useRef, useState } from 'react';
import {
  Bot,
  CheckCircle2,
  ChevronRight,
  FileText,
  LoaderCircle,
  MessageCircle,
  Paperclip,
  Plus,
  RefreshCw,
  Send,
  Sparkles,
  UploadCloud,
  UserRound,
  X,
} from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  // chats: { [chat_id]: { chat_id, display_name, senders, chunk_count, message_count, ingested_at, source_filename } }
  const [chats, setChats] = useState({});
  const [isIngesting, setIsIngesting] = useState(false);
  const [reingestingId, setReingestingId] = useState(null);
  const [toast, setToast] = useState('');
  const [chatOpen, setChatOpen] = useState(false);
  const [activeChatId, setActiveChatId] = useState(null);
  const [activeChatName, setActiveChatName] = useState('');
  const [draft, setDraft] = useState('');
  const [messages, setMessages] = useState([]);
  const [isSending, setIsSending] = useState(false);
  const fileInputRef = useRef(null);
  const reingestInputRef = useRef(null);
  const chatInputRef = useRef(null);

  useEffect(() => {
    fetch(`${API_URL}/api/chats`)
      .then((r) => r.json())
      .then((data) => {
        if (data.chats && typeof data.chats === 'object') setChats(data.chats);
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!toast) return undefined;
    const timeout = window.setTimeout(() => setToast(''), 3200);
    return () => window.clearTimeout(timeout);
  }, [toast]);

  const openFilePicker = () => fileInputRef.current?.click();

  const ingestFile = async (event) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.txt')) {
      setToast('Choose a WhatsApp .txt export.');
      return;
    }

    setIsIngesting(true);
    setToast(`Indexing ${file.name}...`);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const response = await fetch(`${API_URL}/api/ingest`, { method: 'POST', body: formData });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Ingestion failed.');
      setChats(data.chats || {});
      setToast(`${file.name} indexed — ${data.chat?.chunk_count || 0} chunks stored.`);
    } catch (error) {
      setToast(error.message);
    } finally {
      setIsIngesting(false);
    }
  };

  const reingestChat = async (chatId, event) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.txt')) {
      setToast('Choose a WhatsApp .txt export.');
      return;
    }

    setReingestingId(chatId);
    setToast(`Re-indexing ${file.name}...`);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const response = await fetch(`${API_URL}/api/ingest`, { method: 'POST', body: formData });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Re-ingestion failed.');
      setChats(data.chats || {});
      setToast(`${file.name} re-indexed — old chunks replaced.`);
    } catch (error) {
      setToast(error.message);
    } finally {
      setReingestingId(null);
    }
  };

  const openChat = (chatId, displayName) => {
    setActiveChatId(chatId);
    setActiveChatName(displayName);
    setMessages([]);
    setChatOpen(true);
    window.setTimeout(() => chatInputRef.current?.focus(), 50);
  };

  const sendMessage = async (event) => {
    event?.preventDefault();
    const message = draft.trim();
    if (!message || isSending || !activeChatId) return;
    setDraft('');
    setMessages((current) => [...current, { role: 'user', content: message }]);
    setIsSending(true);
    try {
      const response = await fetch(`${API_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, chat_id: activeChatId }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Chat request failed.');
      setMessages((current) => [...current, { role: 'assistant', content: data.answer }]);
    } catch (error) {
      setMessages((current) => [...current, { role: 'assistant', error: true, content: error.message }]);
    } finally {
      setIsSending(false);
    }
  };

  const chatList = Object.values(chats);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark"><MessageCircle size={19} strokeWidth={2.4} /></div>
          <div>
            <div className="brand-name">WhatsRAG AI</div>
            <div className="brand-status"><span className="status-dot" /> local intelligence layer</div>
          </div>
        </div>
      </header>

      <main className="content">
        <section className="page-intro">
          <div>
            <h1>Conversation, made searchable.</h1>
          </div>
        </section>

        <section className="ingest-card panel">
          <div className="section-heading">
            <div className="icon-tile"><UploadCloud size={21} /></div>
            <div><h2>Ingest export</h2><p>Turn a raw WhatsApp archive into searchable memory.</p></div>
          </div>
          <button className="drop-zone" type="button" onClick={openFilePicker} disabled={isIngesting}>
            {isIngesting ? <LoaderCircle className="spin" size={27} /> : <Plus size={27} />}
            <strong>{isIngesting ? 'Building your index...' : <>Add new <span>.txt</span> chat export</>}</strong>
            <small>{isIngesting ? 'Parsing messages, creating chunks, and storing embeddings' : 'Tap to browse or drag a WhatsApp text export here'}</small>
            <div className="format-tags"><span><FileText size={12} /> .txt</span><span>iOS + Android</span><span>media tags okay</span></div>
          </button>
          <input ref={fileInputRef} className="visually-hidden" type="file" accept=".txt" onChange={ingestFile} />
        </section>

        {chatList.map((chat) => (
          <section className="featured panel" key={chat.chat_id} style={{ marginBottom: '16px' }}>
            <div className="feature-copy" style={{ marginBottom: '20px' }}>
              <div className="avatar avatar-large"><Bot size={25} /></div>
              <div>
                <h2>{chat.display_name}</h2>
              </div>
            </div>
            <div className="feature-actions">
              <button
                className="button secondary"
                type="button"
                disabled={reingestingId === chat.chat_id}
                onClick={() => {
                  const input = document.createElement('input');
                  input.type = 'file';
                  input.accept = '.txt';
                  input.onchange = (e) => reingestChat(chat.chat_id, e);
                  input.click();
                }}
              >
                {reingestingId === chat.chat_id
                  ? <><LoaderCircle className="spin" size={17} /> Re-ingesting...</>
                  : <><RefreshCw size={17} /> Re-ingest</>}
              </button>
              <button className="button primary" type="button" onClick={() => openChat(chat.chat_id, chat.display_name)}>
                <Sparkles size={17} /> Chat AI <ChevronRight size={16} />
              </button>
            </div>
          </section>
        ))}
      </main>

      <nav className="bottom-nav"><div className="nav-active"><MessageCircle size={18} /><span>Ingest & chats</span></div><div className="nav-caption">private by design</div></nav>

      {chatOpen && <div className="chat-backdrop" onClick={() => setChatOpen(false)}><aside className="chat-drawer" onClick={(event) => event.stopPropagation()}>
        <div className="chat-header"><div><p className="eyebrow">RAG ASSISTANT</p><h2>{activeChatName}</h2></div><button className="icon-button" title="Close chat" type="button" onClick={() => setChatOpen(false)}><X size={19} /></button></div>
        <div className="chat-messages">
          {!messages.length && <div className="chat-empty"><div className="empty-mark"><Sparkles size={23} /></div><h3>Ask your archive.</h3><p>Search messages by memory, sender, date, or topic. Every answer is grounded in your indexed export.</p></div>}
          {messages.map((item, index) => <div className={`message-bubble ${item.role} ${item.error ? 'error' : ''}`} key={`${item.role}-${index}`}><span className="bubble-label">{item.role === 'user' ? 'YOU' : 'WHATS RAG'}</span>{item.content}</div>)}
          {isSending && <div className="thinking"><LoaderCircle className="spin" size={16} /> Searching your indexed memory...</div>}
        </div>
        <form className="chat-composer" onSubmit={sendMessage}><button className="icon-button" title="Attach a file" type="button" onClick={openFilePicker}><Paperclip size={18} /></button><input ref={chatInputRef} value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="Ask about your chats..." disabled={isSending} /><button className="send-button" title="Send message" type="submit" disabled={!draft.trim() || isSending}><Send size={17} /></button></form>
      </aside></div>}

      {toast && <div className="toast"><CheckCircle2 size={17} /> {toast}</div>}
    </div>
  );
}

export default App;
