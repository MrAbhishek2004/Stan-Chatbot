# 🤖 STAN Chatbot Agent

A human-like conversational chatbot built for the **STAN Internship Challenge**.  
Stan goes beyond a basic Q&A bot — it demonstrates **empathy, contextual awareness, memory persistence, and tone adaptation**.  

The chatbot is implemented using:  
- **Streamlit** → lightweight, interactive frontend UI  
- **LangChain + LangGraph** → conversation logic & memory management  
- **Google Gemini API** → LLM for natural, adaptive responses  
- **MySQL** → persistent long-term memory per user  

---

## ✨ Features

- 🧠 **Human-like interaction**: Engages in natural, diverse conversations without robotic replies.  
- 🎭 **Context-aware tone adaptation**: Matches user’s style (formal, casual, empathetic).  
- 📝 **Persistent memory**: Stores and recalls per-user conversation history via MySQL.  
- 🧩 **Personality consistency**: Always stays in character as “Stan” — a friendly, empathetic friend.  
- 💡 **Hallucination resistance**: Avoids false claims, gives safe or playful answers to uncertain questions.  
- ⚡ **Token trimming**: Keeps conversation efficient by compressing older history.  
- 🛠 **Modular**: Easily swap Gemini with OpenAI, Mistral, or Ollama if needed.  

---

## 🗂️ Architecture

```mermaid
flowchart TD
    A[Streamlit UI] --> B[LangChain + LangGraph]
    B --> C[Google Gemini API]
    B --> D[MySQL Database]
