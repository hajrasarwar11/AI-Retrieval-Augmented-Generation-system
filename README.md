# 📚 AI Retrieval-Augmented Generation (RAG) System

An intelligent document question-answering system that uses advanced AI techniques to extract, index, and query information from PDF documents. This system leverages vector databases, embeddings, and large language models to provide accurate, context-aware answers to user questions.

## 🌟 Features

- **📄 PDF Document Processing**: Upload and process multiple PDF documents
- **🔍 Intelligent Chunking**: Automatic text splitting with semantic understanding
- **🧠 Vector Embeddings**: Generate high-quality embeddings using Sentence Transformers
- **💾 Vector Database Storage**: Store and retrieve embeddings efficiently using Qdrant
- **🤖 AI-Powered Q&A**: Get accurate answers using Google Gemini AI
- **🎯 Source Attribution**: Track and display source documents for each answer
- **🖥️ User-Friendly Interface**: Interactive Streamlit web interface
- **⚡ Fast API Backend**: RESTful API built with FastAPI
- **📊 Orchestration & Monitoring**: Built-in observability with Inngest
- **🔄 Real-time Processing**: Asynchronous document ingestion and query processing

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **FastAPI** | High-performance REST API framework |
| **Inngest** | Function orchestration and observability |
| **Google Gemini AI** | Large language model for answer generation |
| **Sentence Transformers** | Local embedding model (all-MiniLM-L6-v2) |
| **Qdrant** | Vector database for similarity search |
| **LlamaIndex** | PDF processing and text chunking |
| **Streamlit** | Interactive web frontend |
| **PyPDF2** | PDF text extraction |
| **scikit-learn** | TF-IDF vectorization (alternative embeddings) |

## 📁 Project Structure

```
AI-Retrieval-Augmented-Generation-system/
│
├── backend/                          # Backend API and logic
│   ├── main.py                       # FastAPI server + Inngest functions
│   ├── vector_db.py                  # Qdrant database client & operations
│   ├── data_loader.py                # PDF loading & embedding generation
│   ├── ingest.py                     # TF-IDF based ingestion (alternative)
│   ├── query.py                      # Query processing logic
│   ├── custom_types.py               # Pydantic type definitions
│   └── .env                          # Environment variables (API keys)
│
├── frontend/                         # Streamlit UI
│   ├── app.py                        # Full-featured user interface
│   └── simple_app.py                 # Simplified interface
│
├── data/                             # PDF storage folder
│   └── [Place your PDF files here]
│
├── quadrant_storage/                 # Vector database persistent storage
│   ├── collections/                  # Collection data
│   └── raft_state.json              # Database state
│
├── direct_ingest.py                  # Direct ingestion script (no UI)
├── reingest.py                       # Batch re-ingestion script
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## 📋 Prerequisites

Before running this project, ensure you have the following installed:

- **Python 3.8+** ([Download Python](https://www.python.org/downloads/))
- **Docker Desktop** ([Download Docker](https://www.docker.com/products/docker-desktop/)) - Required for Qdrant
- **Node.js** (Optional, for Inngest CLI) ([Download Node.js](https://nodejs.org/))
- **Git** (Optional, for cloning) ([Download Git](https://git-scm.com/))

## 🚀 Installation & Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/AI-Retrieval-Augmented-Generation-system.git
cd AI-Retrieval-Augmented-Generation-system/VS-Code
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Create a `.env` file in the `backend/` directory:

```bash
# backend/.env
GOOGLE_API_KEY=your-google-gemini-api-key-here
OPENAI_API_KEY=your-openai-api-key-here  # Optional, if using OpenAI
```

**How to get API keys:**
- **Google Gemini**: Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
- **OpenAI** (Optional): Visit [OpenAI Platform](https://platform.openai.com/api-keys)

### Step 5: Add PDF Documents

Place your PDF files in the `data/` folder:

```bash
data/
├── document1.pdf
├── document2.pdf
└── document3.pdf
```

## 🎯 How to Run the Project

### Method 1: Full Stack Setup (Recommended)

This runs all components: Qdrant database, FastAPI backend, Inngest orchestration, and Streamlit frontend.

#### 1️⃣ Start Qdrant Vector Database

```bash
# Windows (PowerShell)
docker run -d --name quadrant_rag_db -p 6333:6333 -v "${PWD}/quadrant_storage:/qdrant/storage" qdrant/qdrant

# Windows (Command Prompt)
docker run -d --name quadrant_rag_db -p 6333:6333 -v "%cd%/quadrant_storage:/qdrant/storage" qdrant/qdrant

# Linux/macOS
docker run -d --name quadrant_rag_db -p 6333:6333 -v "$(pwd)/quadrant_storage:/qdrant/storage" qdrant/qdrant
```

✅ **Verify:** Visit http://localhost:6333/dashboard

#### 2️⃣ Start FastAPI Backend Server

Open a new terminal:

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

✅ **Verify:** Visit http://127.0.0.1:8000/docs (FastAPI interactive documentation)

#### 3️⃣ Start Inngest Dev Server

Open another terminal:

```bash
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery
```

✅ **Verify:** Visit http://localhost:8288 (Inngest dashboard)

#### 4️⃣ Start Streamlit Frontend

Open another terminal:

```bash
streamlit run frontend/app.py
```

✅ **Verify:** The browser will automatically open http://localhost:8501

---

### Method 2: Direct Ingestion (Without UI)

For batch processing PDFs without the web interface:

```bash
python direct_ingest.py
```

This script will:
1. Scan the `data/` folder for PDFs
2. Process and chunk each document
3. Generate embeddings
4. Store vectors in Qdrant

---

### Method 3: TF-IDF Based Ingestion (Lightweight)

For faster ingestion using TF-IDF embeddings:

```bash
cd backend
python ingest.py
```

This creates embeddings using scikit-learn's TF-IDF vectorizer (no model downloads required).

## 📖 Usage Guide

### Using the Web Interface

1. **Start all services** as described in [Method 1](#method-1-full-stack-setup-recommended)

2. **Open the Streamlit app** at http://localhost:8501

3. **Upload PDFs** (Sidebar):
   - Click "Browse files" or drag & drop PDFs
   - Click **"Ingest PDF"** to process the document
   - Wait for confirmation message

4. **Ask Questions** (Main Chat Interface):
   - Type your question in the input box
   - Press Enter or click Send
   - View AI-generated answers with source citations

5. **Monitor Processing** (Optional):
   - Check Inngest dashboard: http://localhost:8288
   - View function execution logs, timing, and status

### Using the API Directly

#### Ingest a PDF:

```bash
curl -X POST "http://127.0.0.1:8000/api/inngest" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "rag/ingest_pdf",
    "data": {
      "pdf_path": "data/your-document.pdf",
      "source_id": "your-document"
    }
  }'
```

#### Query the RAG System:

```bash
curl -X POST "http://127.0.0.1:8000/api/inngest" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "rag/query",
    "data": {
      "query": "What is retrieval augmented generation?"
    }
  }'
```

## 🧪 Testing the System

### Quick Test

1. Add a sample PDF to the `data/` folder
2. Run direct ingestion:
   ```bash
   python direct_ingest.py
   ```
3. Start the Streamlit app:
   ```bash
   streamlit run frontend/app.py
   ```
4. Ask a question related to your PDF content

### Verify Qdrant Storage

Visit http://localhost:6333/dashboard and check:
- Collection name: `documents`
- Number of vectors stored
- Vector dimensions: 384

## 🐛 Troubleshooting

### Issue: Docker Container Not Starting

```bash
# Check if container already exists
docker ps -a

# Remove old container
docker rm -f quadrant_rag_db

# Restart with fresh container
docker run -d --name quadrant_rag_db -p 6333:6333 -v "${PWD}/quadrant_storage:/qdrant/storage" qdrant/qdrant
```

### Issue: Port Already in Use

```bash
# Find process using the port (Windows)
netstat -ano | findstr :8000

# Kill the process (Windows)
taskkill /PID <PID> /F

# Find process using the port (Linux/macOS)
lsof -ti:8000

# Kill the process (Linux/macOS)
kill -9 $(lsof -ti:8000)
```

### Issue: Missing API Key

Ensure `.env` file exists in `backend/` directory with valid API keys:

```bash
cd backend
type .env  # Windows
cat .env   # Linux/macOS
```

### Issue: Module Import Errors

```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Check Python version (should be 3.8+)
python --version
```

## 📊 Monitoring & Debugging

### Available Dashboards

| Dashboard | URL | Purpose |
|-----------|-----|---------|
| **FastAPI Docs** | http://127.0.0.1:8000/docs | API endpoints & testing |
| **Inngest** | http://localhost:8288 | Function execution logs |
| **Qdrant** | http://localhost:6333/dashboard | Vector database inspection |
| **Streamlit App** | http://localhost:8501 | User interface |

### Logs

- **FastAPI**: Check terminal where `uvicorn` is running
- **Streamlit**: Check terminal where `streamlit run` is running
- **Inngest**: View detailed logs at http://localhost:8288
- **Docker**: `docker logs quadrant_rag_db`

## 🔧 Configuration Options

### Embedding Model Configuration

Edit `backend/data_loader.py` to change the embedding model:

```python
# Current model (384 dimensions, fast)
EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

# Alternative models:
# EMBED_MODEL = SentenceTransformer('all-mpnet-base-v2')  # 768 dim, more accurate
# EMBED_MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # Multilingual
```

### Chunking Parameters

Edit `backend/data_loader.py`:

```python
splitter = SentenceSplitter(
    chunk_size=1000,      # Adjust chunk size
    chunk_overlap=200     # Adjust overlap
)
```

### LLM Model Configuration

Edit `backend/main.py`:

```python
# Current model
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# Alternative: More powerful model
# gemini_model = genai.GenerativeModel('gemini-pro')
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Your Name** - *Initial work* - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- FJWU - Artificial Intelligence Course (Dr. Irum Matloob)
- LlamaIndex for PDF processing tools
- Qdrant for vector database
- Google for Gemini AI
- Sentence Transformers for embedding models

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Contact: your.email@example.com

---

**Made with ❤️ for Assignment 3 - Artificial Intelligence Course**
- **Streamlit**: Frontend UI
