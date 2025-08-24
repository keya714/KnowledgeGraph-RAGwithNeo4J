# Knowledge Graph RAG with Neo4j

This project combines **Large Language Models (LLMs)** with **graph databases** to create a powerful **Retrieval-Augmented Generation (RAG)** pipeline.  
It enables users to query a **Neo4j knowledge graph** in **natural language**, automatically generates optimized **Cypher queries**, and visualizes results in real time. 

To access the project -> https://knowledge-graph-ragwithneo4j-test.streamlit.app

The project leverages:
- **Neo4j** for graph storage and querying.
- **Google Gemini** (via LangChain) for natural language understanding and Cypher generation.
- **Streamlit** for an interactive chatbot + graph visualization UI.
- **PyVis** for intuitive, dynamic network graph rendering.

---

## Features
- **Chatbot Interface** – Ask natural language questions and get intelligent responses.
- **Automated Cypher Generation** – LLM translates user queries into Neo4j Cypher queries.
- **Graph Visualization** – Explore query results as an interactive graph using PyVis.
- **Conversational Context** – Maintains chat history for context-aware responses.

---

## How It Works
1. User enters a **question** in natural language.
2. The **Gemini LLM (via LangChain)** generates a **Cypher query** based on the Neo4j schema.
3. The query is executed on the **Neo4j knowledge graph**.
4. Results are displayed as:
   - A **textual answer** (from the LLM).
   - An **interactive graph visualization** (via PyVis).
5. The chatbot maintains **conversational memory** for follow-up questions.

---

## Tech Stack
- **Streamlit** → UI for chatbot & visualization.
- **Neo4j** → Graph database backend.
- **LangChain** → LLM orchestration and Cypher generation.
- **Google Gemini** → LLM for natural language understanding.
- **PyVis** → Interactive graph visualization.

---

## Example Use Cases
- Querying knowledge graphs with natural language (no Cypher expertise required).
- Visualizing entity relationships in real-time.
- Building explainable RAG-powered assistants.
- Educational tool for graph databases + LLMs.

---

## Setup Notes
- Store Neo4j and Google API credentials securely in `.streamlit/secrets.toml`.
- The chatbot runs in **Streamlit** (`streamlit run app.py`).
- Customize for your **own Neo4j schema** to create domain-specific assistants.

---

## About This Project
An interactive **chatbot + graph explorer** that integrates **LLMs with Neo4j** for **knowledge-graph-powered RAG**. It empowers users to ask questions in plain English, see results as text + interactive graph, and bridges the gap between **natural language understanding** and **graph reasoning**.
