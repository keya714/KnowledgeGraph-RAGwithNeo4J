import streamlit as st
from neo4j import GraphDatabase 
from pyvis.network import Network
import streamlit.components.v1 as components
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_community.graphs import Neo4jGraph 
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

# ====================
# Configuration
# ====================
# Use Streamlit secrets for secure credentials
# These should be stored in a .streamlit/secrets.toml file
# [neo4j]
# uri = "your_uri"
# user = "your_user"
# password = "your_password"
# [google]
# api_key = "your_api_key"

# Access secrets
# URI = st.secrets["neo4j"]["uri"]
# USER = st.secrets["neo4j"]["user"]
# PASSWORD = st.secrets["neo4j"]["password"]
# os.environ["GOOGLE_API_KEY"] = st.secrets["google"]["api_key"]

# Placeholder for your credentials
URI = "neo4j+s://f130631d.databases.neo4j.io"
USER = "neo4j"
PASSWORD = "YWbr8AJPpup1G3tmlrgXp1iw0B_VKXRpk_JwPHrG-WI"
os.environ["GOOGLE_API_KEY"] = "AIzaSyANdfOSp5L8a7cNU5ostjrHve5AoKNqGyo"


# ====== Gemini LLM config ======
llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

# Create a graph driver for direct Neo4j interactions (visualization)
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

# Create a Neo4jGraph instance for LLM interactions
graph_for_llm = Neo4jGraph(
    url=URI,
    username=USER,
    password=PASSWORD
)

# ====== LLM Response ======
def response(question, chat_history=[]):
    """Generates an LLM response and a Cypher query for graph visualization."""
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
    {query} 
    
    The Conversational History is:
    {chat_history}
    """

    CYPHER_GENERATION_PROMPT = PromptTemplate(
        input_variables=["schema", "query", "chat_history"], template=CYPHER_GENERATION_TEMPLATE
    )

    chain = GraphCypherQAChain.from_llm(
        llm=llm_gemini,
        graph=graph_for_llm,
        verbose=True,
        allow_dangerous_requests=True,
        return_intermediate_steps=True,
        cypher_prompt=CYPHER_GENERATION_PROMPT
    )

    response_dict = chain.invoke({'query': question, 'chat_history': chat_history})
    llm_answer = response_dict.get('result')
    
    # Safely get intermediate steps
    intermediate_steps_data = None
    if len(response_dict.get('intermediate_steps', [])) > 1:
        intermediate_steps_data = response_dict['intermediate_steps'][1].get('context')
    
    return llm_answer, intermediate_steps_data

def visualize_graph(data):
    """
    Visualizes a graph from a list of nodes and relationships.
    Returns the HTML content as a string.
    """
    g = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=True)
    g.barnes_hut()

    unique_nodes = set()
    for item in data:
        # Improved error handling for cases where 'p' is not as expected
        if not isinstance(item, dict) or 'p' not in item or not isinstance(item['p'], list) or len(item['p']) < 3:
            continue

        source_node_data = item['p'][0]
        relationship_type = item['p'][1]
        target_node_data = item['p'][2]

        # Robustly get node names, handling both dict and simple types
        source_node_name = source_node_data.get('id', str(source_node_data)) if isinstance(source_node_data, dict) else str(source_node_data)
        target_node_name = target_node_data.get('id', str(target_node_data)) if isinstance(target_node_data, dict) else str(target_node_data)

        if source_node_name not in unique_nodes:
            g.add_node(source_node_name, title=str(source_node_data))
            unique_nodes.add(source_node_name)

        if target_node_name not in unique_nodes:
            g.add_node(target_node_name, title=str(target_node_data))
            unique_nodes.add(target_node_name)
        
        g.add_edge(source_node_name, target_node_name, title=relationship_type, label=relationship_type)

    # Return the HTML content as a string
    return g.generate_html()

st.set_page_config(layout="wide", page_title="Chatbot + Neo4j Visualization")

col1, col2 = st.columns([1, 2])

# -----------------------------
# 💬 Chatbot Section (Left)
# -----------------------------
with col1:
    st.header("💬 Chatbot")

    if "history" not in st.session_state:
        st.session_state.history = []
    if "intermediate_steps_data" not in st.session_state:
        st.session_state.intermediate_steps_data = None
    
    if "llm_answer" not in st.session_state:
        st.session_state.llm_answer = None

    user_input = st.text_input("Type your message:")

    if user_input:
        st.session_state.llm_answer, st.session_state.intermediate_steps_data = response(
            question=user_input, 
            chat_history=st.session_state.history
        )
        bot_response = f"🤖 {st.session_state.llm_answer}"
        st.session_state.history.append(("You", user_input))
        st.session_state.history.append(("Bot", bot_response))
        

    # Corrected display logic
    for speaker, text in reversed(st.session_state.history):
        if speaker == "You":
            st.markdown(f"🧑 **{speaker}:** {text}")
        else:
            st.markdown(f"**{speaker}:** {text}")

# -----------------------------
# 📊 Graph Section (Right)
# -----------------------------
with col2:
    st.header("📊 Neo4j Graph Visualization")

    if st.button("Load Graph"):
        if st.session_state.intermediate_steps_data:
            # Generate the HTML content from the graph data
            html_content = visualize_graph(st.session_state.intermediate_steps_data)
            
            # Display the HTML content directly
            components.html(html_content, height=600, scrolling=True)
        else:
            st.info("Please ask a question first to generate the graph data.")

def print_world():
    print("Hello")
