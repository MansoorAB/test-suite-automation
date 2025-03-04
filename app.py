import streamlit as st
import pandas as pd
import chromadb
from chromadb.config import Settings
import os


def read_csv(input_file: str) -> pd.DataFrame:
    """Read CSV file using pandas."""
    return pd.read_csv(input_file)


def get_vector_db() -> chromadb.Client:
    """Get existing ChromaDB client."""
    # Get the root directory path
    root_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(root_dir, "chroma_db")
    
    # Initialize ChromaDB client with existing database
    client = chromadb.Client(Settings(
        persist_directory=db_path,
        anonymized_telemetry=False
    ))
    
    return client


def search_scenarios(
    client: chromadb.Client,
    query: str,
    n_results: int = 3
):
    """Search for similar scenarios using semantic search."""
    collection = client.get_collection("scenarios")
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    return results


def display_scenario_details(row: pd.Series):
    """Display scenario details in a formatted way."""
    st.subheader("Selected Scenario Details")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Feature:**")
        st.write(row['Feature'])
        
        st.write("**Packages:**")
        packages = row['Packages'].split(' ** ') if row['Packages'] != "NA" else []
        st.write("  " + " ".join(f"@{package}" for package in packages))
    
    with col2:
        st.write("**Scenario:**")
        st.write(row['Scenario'])
    
    # BDD Steps
    st.write("**BDD Steps:**")
    steps = row['BDD'].split(' ** ')
    for step in steps:
        st.write(f"- {step}")
    
    # Examples
    if row['Example'] != "NA":
        st.write("**Examples:**")
        example_parts = row['Example'].split('**')
        for part in example_parts:
            st.code(part.strip(), language="gherkin")


def main():
    st.set_page_config(
        page_title="Scenario Search",
        page_icon="��",
        layout="wide"
    )
    
    st.title("🔍 Scenario Search")
    
    # Initialize session state for selected scenario
    if 'selected_scenario' not in st.session_state:
        st.session_state.selected_scenario = None
    
    # Load data and get vector DB
    try:
        df = read_csv('output.csv')
        client = get_vector_db()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please run scenario_search.py first to create the vector database.")
        return
    
    # Search interface
    query = st.text_input("Enter your search query:", placeholder="e.g., How to add documents to a case?")
    
    if query:
        with st.spinner("Searching for similar scenarios..."):
            results = search_scenarios(client, query)
            
            # Display search results with checkboxes
            st.subheader("Search Results")
            for idx, (doc, distance) in enumerate(zip(
                results['documents'][0],
                results['distances'][0]
            )):
                # Find corresponding row in dataframe
                row = df[df['Scenario'] == doc].iloc[0]
                
                # Create a unique key for each checkbox
                checkbox_key = f"select_{idx}"
                
                # Display result with checkbox
                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.checkbox("Select", key=checkbox_key):
                        st.session_state.selected_scenario = row
                with col2:
                    st.write(f"**Result {idx + 1}** (Similarity: {1 - distance:.2f})")
                    st.write(f"Scenario: {row['Scenario']}")
    
    # Display selected scenario details
    if st.session_state.selected_scenario is not None:
        st.markdown("---")
        display_scenario_details(st.session_state.selected_scenario)


if __name__ == "__main__":
    main()