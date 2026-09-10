<div align="center">
  <img src="nexusai_banner.jpg" alt="NexusAI Banner" width="800">
</div>

# NexusAI CAP (Comprehensive AI Platform)

A production-grade, extensible AI ecosystem designed to unify the capabilities of 20+ Large Language Model (LLM) providers, advanced Retrieval-Augmented Generation (RAG) pipelines, and a robust 7-role agent architecture.

## 🌟 Overview

NexusAI CAP provides a single, cohesive framework for building, testing, and deploying autonomous agents. With built-in support for thousands of tools and Model Context Protocol (MCP) standards, you can easily integrate AI agents into any environment.

### Core Capabilities
- **Universal LLM Support**: Seamlessly integrate with OpenAI, Anthropic, Google Gemini, Groq, Mistral, Cohere, TogetherAI, and HuggingFace.
- **7-Role Agent Pipeline**: Build complex workflows using specialized agent roles out-of-the-box.
- **Advanced RAG**: Full integration with Faiss, ChromaDB, Sentence-Transformers, and various document parsers (PDF, DOCX, HTML, Markdown).
- **Extensive Tool Ecosystem**: Connects to over 2000+ tools for web scraping, file manipulation, Git operations, and system metrics.
- **Web Dashboard**: An integrated FastAPI web dashboard and WebSocket streaming capability for live agent monitoring.

## ⚙️ Prerequisites
- **Python**: 3.9, 3.10, 3.11, 3.12, or 3.13.

## 📦 Installation

NexusAI is highly modular. You can install only the dependencies you need:

1. Clone the repository:
   ```bash
   git clone https://github.com/Shivay00001/nexusaicap.git
   cd nexusaicap
   ```

2. Install with specific capabilities:
   ```bash
   # Install core system only
   pip install .

   # Install with specific LLM providers
   pip install ".[openai,anthropic]"

   # Install everything (RAG, Web, all LLMs, tools)
   pip install ".[all]"
   ```

## 💻 Usage

Once installed, use the unified CLI to initialize your projects, start the web dashboard, or interact with agents directly.

```bash
# Get CLI help
nexusai --help
```

## 🐳 Docker Deployment

To launch NexusAI without affecting your local environment, use the included Docker configuration:

```bash
docker compose up --build
```
*Note: Make sure you provide your API keys in the `.env` file before launching.*

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
