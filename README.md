# AI Deep Research Agent

An **AI-powered deep research system** built with **LangGraph**, **Google Custom Search API**, and **Gemini LLM models**.  
This project automates end-to-end research by generating a structured report with reliable web-sourced references.  

---

## Overview

Traditional search engines provide fragmented results, requiring manual filtering and summarization.  
This project solves that by combining **graph-based agents** with **LLM-driven synthesis** to deliver **comprehensive research reports** on any topic.

### What it does:
1. **Generates a research plan** – structures the report and identifies which sections need web search.  
2. **Runs web-based sub-agents** – performs targeted Google searches for each identified section.  
3. **Synthesizes findings** – analyzes results, cites references, and writes section drafts.  
4. **Compiles final report** – integrates all sections into a polished document with sources.  

---

## Repository Structure


```
├── Configuration.py # Configurations for API keys and settings
├── Prompts.py # Prompt templates used by agents
├── README.md # Documentation
├── invoke.py # Entry point script
├── main.py # Core research agent logic
└── requirements.txt # Dependencies
```

- **main.py** → Contains the core logic of the agent system.  
- **invoke.py** → Entry point. Users specify their research topic here.  
- **requirements.txt** → Python dependencies.  

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```
### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
```
### 3. Create a .env file for storing the keys
**You’ll need the following keys:**

*GOOGLE_API_KEY* → from Google AI Studio

*google_cse_api_key* -> from [Google Cloud console](https://console.cloud.google.com/)

*google_cse_id* -> from [Google Developers](https://programmablesearchengine.google.com/)


**Create a .env file**
```
GOOGLE_API_KEY=your_google_api_key
Google_cse_api_key= Google search api key
google_cse_id= ID of the Custom Search Engine
```
### 4. import for Building the Search client
```
from googleapiclient.discovery import build
```
----
## Usage

### 1. Open invoke.py and update the research topic you want to explore.


### 2. Run the entry point:
```
python invoke.py
```

### 3. The system will:

- Generate a research plan.

- Perform web searches.

- Write structured report sections with references.

- Save/print the final report.
---
## License

MIT License

---
## Author

Made by **Pritam Saha**
Email: [pritamsaha1109@gmail.com](mailto:pritamsaha1109@gmail.com)

---

