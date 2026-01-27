#StudyBuddy
![study buddy ui](assets/StudyBuddyUi.png)
StudyBuddy is an AI-powered study assistant that lets you upload documents and interactively ask questions about them. Think of it as a smart study companion that understands your notes, slides, and PDFs.

---

## 🚀 Features

- 📄 Upload and chat with your documents
- 🤖 Ask questions grounded in your uploaded materials
- 🧠 Combines document context with general AI knowledge
- ⚡ Simple, lightweight setup


---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/mattjacobs23/StudyBuddy.git
cd StudyBuddy
```

### 2. Install dependencies
First, install `uv` if you haven't already
```bash
pip install uv
```
### 3. Set up enviornment variables
Copy the `.env.template` to create a `.env` file
```
cd api 
cp .env.template .env
```
Edit the .env file to populate the required values:
- Ollama
    - Download ollama at https://ollama.com/download/
    - Follow instructions to pull and use models. E.g.:
        - Embedding model: `ollama pull nomic-embed-text`
        - Text model (LLM): `ollama pull gemma3:1b`

- Groq
    - Obtain an API key at https://console.groq.com/keys
    - Populate the `GROQ_API_KEY` with your Groq API key

Important: Do not commit your .env file to the repository, as it contain sensitive credentials. The .gitignore file excludes it by default.


### 4. Run the application
You can run the API and UI in seperate terminals:

- API
```bash
python run_api.py
```

- UI
```bash
python run_ui.py
```


---

## 💬 Usage

1. Upload documents you want to study or chat with
2. Ask questions about the content
3. Get answers grounded in your files

---

## 📄 Supported File Types

- PDF
- PNG
- JPG/JPEG

---

## 🔐 Environment Variables

If required, create a `.env` file:

```bash
OPENAI_API_KEY=your_api_key_here
```

