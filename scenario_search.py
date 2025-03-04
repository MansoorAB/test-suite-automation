import os
import pandas as pd
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Any


def read_csv(input_file: str) -> pd.DataFrame:
    """Read CSV file using pandas."""
    return pd.read_csv(input_file)


def setup_vector_db(df: pd.DataFrame) -> chromadb.Client:
    """Initialize ChromaDB and store scenario data with embeddings."""
    # Get the root directory path
    root_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(root_dir, "chroma_db")
    
    # Initialize ChromaDB client with absolute path
    client = chromadb.Client(Settings(
        persist_directory=db_path,
        anonymized_telemetry=False
    ))
    
    # Create or get collection
    collection = client.create_collection(
        name="scenarios",
        embedding_function=embedding_functions.DefaultEmbeddingFunction()
    )
    
    # Prepare documents for embedding (only Scenario column)
    documents = df['Scenario'].tolist()
    ids = [f"scenario_{idx}" for idx in range(len(documents))]
    
    # Add documents to collection
    collection.add(
        documents=documents,
        ids=ids
    )
    
    return client


def search_scenarios(
    client: chromadb.Client,
    query: str,
    n_results: int = 3
) -> List[Dict[str, Any]]:
    """Search for similar scenarios using semantic search."""
    collection = client.get_collection("scenarios")
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    return results


def print_search_results(
    results: List[Dict[str, Any]], 
    query: str,
    df: pd.DataFrame
) -> None:
    """Print search results in a formatted way."""
    print(f"\nSearch results for: {query}")
    print("-" * 50)
    
    for idx, (doc, distance) in enumerate(zip(
        results['documents'][0],
        results['distances'][0]
    )):
        # Find corresponding row in dataframe
        row = df[df['Scenario'] == doc].iloc[0]
        
        print(f"\nResult {idx + 1} (Similarity: {1 - distance:.2f})")
        print(f"Scenario: {row['Scenario']}")
        print(f"Feature: {row['Feature']}")
        print("Packages:")
        # Split packages by ** and print each with @ prefix on same line
        packages = row['Packages'].split(' ** ') if row['Packages'] != "NA" else []
        print("  " + " ".join(f"@{package}" for package in packages))
        print("BDD Steps:")
        # Split by ** and print each step on a new line
        steps = row['BDD'].split(' ** ')
        for step in steps:
            print(f"  - {step}")
        if row['Example'] != "NA":
            print("Examples:")
            # Split examples by ** and print each on a new line with proper indentation
            example_parts = row['Example'].split('**')
            for part in example_parts:
                print(f"           {part.strip()}")
        print("-" * 50)


def main():
    input_file = 'output.csv'
    
    # Read CSV file using pandas
    df = read_csv(input_file)
    print(f"Loaded {len(df)} scenarios from {input_file}")
    
    # Setup vector database
    client = setup_vector_db(df)
    print("Vector database initialized with scenarios")
    
    # Example searches
    queries = [
        "How to add documents to a case?",
        "How to use the welcome tour feature?",
        "What are the login steps?"
    ]
    
    for query in queries:
        results = search_scenarios(client, query)
        print_search_results(results, query, df)


if __name__ == "__main__":
    main()