import streamlit as st
from neo4j import GraphDatabase
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

# Set up your Google API key as an environment variable
# If you don't have one, you can get it from Google AI Studio
os.environ["GOOGLE_API_KEY"] = "AIzaSyA5ApL2ltwbFn4GQShKTz_t35744kkq8dg"

# ====== Gemini LLM config ======
# Replace with your actual Gemini API key
llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # You can choose other Gemini models like "gemini-1.5-pro-latest"
    temperature=0,
)

URI = "neo4j+s://f130631d.databases.neo4j.io"   # Change if using Neo4j Aura
USER = "neo4j"
PASSWORD = "YWbr8AJPpup1G3tmlrgXp1iw0B_VKXRpk_JwPHrG-WI"

graph = Neo4jGraph(
    url=URI,
    username=USER,
    password=PASSWORD
)

# ====== LLM Response ======
def response(question):
    # ====== Prompt Template ======
    CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
    Instructions:
    Use only the provided relationship types and properties in the schema.
    Do not use any other relationship types or properties that are not provided.
    Schema:
    {schema}
    Note: Do not include any explanations or apologies in your responses.
    Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
    Do not include any text except the generated Cypher statement.
    Identify the main node, and return all the relationships and nodes connected to it.
    If no properties are provided, assume the nodes have only a property id.
    Please don't filter on relationships or connected nodes.

    Format the query as follows:
    MATCH p=(n:NodeLabel)-[r]-(m)
    WHERE n.id = 'value1'
    RETURN p

    The question is:
    {question}"""

    CYPHER_GENERATION_PROMPT = PromptTemplate(
        input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
    )

    # ====== Cypher QA Chain ======
    # Use the Gemini LLM
    chain = GraphCypherQAChain.from_llm(
        llm=llm_gemini,
        graph=graph,
        verbose=True,
        allow_dangerous_requests=True,
        return_intermediate_steps=True,
        cypher_prompt=CYPHER_GENERATION_PROMPT
    )

    # ====== Run Query ======
    response = chain.run(question)
    return response


def get_graph_data(limit=20):
    query = f"""
    MATCH (n)-[r]->(m)
    RETURN n, r, m
    LIMIT {limit}
    """
    with driver.session() as session:
        results = session.run(query)
        return results

def visualize_graph(results):
    G = nx.DiGraph()

    for record in results:
        n = record["n"]
        m = record["m"]
        r = record["r"]

        G.add_node(n.id, label=list(n.labels)[0], title=str(dict(n)))
        G.add_node(m.id, label=list(m.labels)[0], title=str(dict(m)))
        G.add_edge(n.id, m.id, label=r.type)

    net = Network(height="550px", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(G)

    net.save_graph("graph.html")
    return "graph.html"

# -----------------------------
# 🔹 Streamlit UI
# -----------------------------
st.set_page_config(layout="wide", page_title="Chatbot + Neo4j Visualization")

col1, col2 = st.columns([1, 2])



# -----------------------------
# 💬 Chatbot Section (Left)
# -----------------------------
with col1:
    st.header("💬 Chatbot")

    if "history" not in st.session_state:
        st.session_state.history = []

    user_input = st.text_input("Type your message:")

    if user_input:
        # 🔹 Replace this with real chatbot logic (LLM, Rasa, etc.)
        bot_response = f"🤖"+response(user_input)

        st.session_state.history.append(("You", user_input))
        st.session_state.history.append(("Bot", bot_response))

    # Display conversation history
    for speaker, text in st.session_state.history:
        if speaker == "You":
            st.markdown(f"🧑 {speaker}:** {text}")
        else:
            st.markdown(f"{speaker}:** {text}")

# -----------------------------
# 📊 Graph Section (Right)
# -----------------------------
with col2:
    st.header("📊 Neo4j Graph Visualization")

    if st.button("Load Graph"):
        results = get_graph_data(limit=100)
        graph_html = visualize_graph(results)

        with open(graph_html, "r", encoding="utf-8") as f:
            html_content = f.read()
            components.html(html_content, height=600, scrolling=True)