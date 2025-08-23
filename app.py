import streamlit as st
from neo4j import GraphDatabase # Use the native neo4j driver
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_community.graphs import Neo4jGraph # Keep this for the LLM chain
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

# Set up your Google API key as an environment variable
os.environ["GOOGLE_API_KEY"] = "AIzaSyA5ApL2ltwbFn4GQShKTz_t35744kkq8dg"

# ====== Gemini LLM config ======
llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

URI = "neo4j+s://f130631d.databases.neo4j.io"
USER = "neo4j"
PASSWORD = "YWbr8AJPpup1G3tmlrgXp1iw0B_VKXRpk_JwPHrG-WI"

# Create a graph driver for direct Neo4j interactions (visualization)
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

# Create a Neo4jGraph object for the LLM chain (from the community package)
# Note: You still need this for the LangChain GraphCypherQAChain
graph_for_llm = Neo4jGraph(
    url=URI,
    username=USER,
    password=PASSWORD
)

# ====== LLM Response ======
def response(question):
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

    chain = GraphCypherQAChain.from_llm(
        llm=llm_gemini,
        graph=graph_for_llm, # Use the graph object for the LLM
        verbose=True,
        allow_dangerous_requests=True,
        return_intermediate_steps=True,
        cypher_prompt=CYPHER_GENERATION_PROMPT
    )

    response = chain.run(question)
    return response

# Corrected get_graph_data function using the native driver's session
def get_graph_data(limit=20):
    query = f"""
    MATCH (n:Movie)-[r]-(m)
    RETURN n, r, m
    UNION ALL
    MATCH (n:Director)-[r]-(m)
    RETURN n, r, m
    UNION ALL
    MATCH (n:Actor)-[r]-(m)
    RETURN n, r, m
    LIMIT {limit}
    """
    with driver.session() as session:
        results = session.run(query)
        return results.data()

# Corrected visualize_graph function to handle results from the native driver
def visualize_graph(results):
    G = nx.DiGraph()

    for record in results:
        n = record["n"]
        r = record["r"]
        m = record["m"]

        # Use a unique property as a key for the node
        n_id = n.get('name') if n.get('name') else n.get('title')
        m_id = m.get('name') if m.get('name') else m.get('title')

        # The relationship is a tuple, so access its elements
        r_type = r[1]
        
        # Check if IDs are valid to avoid errors
        if not n_id or not m_id:
            continue

        n_labels = list(r[0].labels) if hasattr(r[0], 'labels') else []
        m_labels = list(r[2].labels) if hasattr(r[2], 'labels') else []
        
        # Get labels using the appropriate key for your schema.
        # This part requires adjusting based on what your relationships look like.
        n_label_text = 'Movie' if n.get('title') else ('Director' if n.get('name') else 'Node')
        m_label_text = 'Movie' if m.get('title') else ('Director' if m.get('name') else 'Node')

        # Create a formatted title for the tooltip
        n_title = ""
        if n_label_text == 'Movie':
            n_title = f"Title: {n.get('title', 'N/A')}\nYear: {n.get('year', 'N/A')}\nGenre: {n.get('genre', 'N/A')}"
        else:
            n_title = f"Name: {n.get('name', 'N/A')}"

        m_title = ""
        if m_label_text == 'Movie':
            m_title = f"Title: {m.get('title', 'N/A')}\nYear: {m.get('year', 'N/A')}\nGenre: {m.get('genre', 'N/A')}"
        else:
            m_title = f"Name: {m.get('name', 'N/A')}"
        
        G.add_node(n_id, label=n_label_text, title=n_title)
        G.add_node(m_id, label=m_label_text, title=m_title)
        G.add_edge(n_id, m_id, label=r_type)

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
        bot_response = f"🤖" + response(user_input)
        st.session_state.history.append(("You", user_input))
        st.session_state.history.append(("Bot", bot_response))

    for speaker, text in st.session_state.history:
        if speaker == "You":
            st.markdown(f"🧑 **{speaker}:** {text}")
        else:
            st.markdown(f"{speaker}:** {text}")

# -----------------------------
# 📊 Graph Section (Right)
# -----------------------------
with col2:
    st.header("📊 Neo4j Graph Visualization")

    if st.button("Load Graph"):
        results = get_graph_data(limit=10)
        graph_html = visualize_graph(results)

        if graph_html:
            with open(graph_html, "r", encoding="utf-8") as f:
                html_content = f.read()
                components.html(html_content, height=600, scrolling=True)
