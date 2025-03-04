import csv
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Any


def read_csv(input_file: str) -> List[Dict[str, Any]]:
    """Read CSV file and convert to list of dictionaries."""
    rows = []
    with open(input_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(row)
    return rows


def setup_vector_db(rows: List[Dict[str, Any]]) -> chromadb.Client:
    """Initialize ChromaDB and store scenario data with embeddings."""
    # Initialize ChromaDB client
    client = chromadb.Client(Settings(
        persist_directory="./chroma_db",
        anonymized_telemetry=False
    ))
    
    # Create or get collection
    collection = client.create_collection(
        name="scenarios",
        embedding_function=embedding_functions.DefaultEmbeddingFunction()
    )
    
    # Prepare documents for embedding
    documents = []
    metadatas = []
    ids = []
    
    for idx, row in enumerate(rows):
        # Combine relevant fields for semantic search
        search_text = f"{row['Scenario']} {row['BDD']}"
        documents.append(search_text)
        
        # Store metadata for retrieval
        metadatas.append({
            'feature': row['Feature'],
            'packages': row['Packages'],
            'scenario': row['Scenario'],
            'bdd': row['BDD'],
            'examples': row['Example']
        })
        
        ids.append(f"scenario_{idx}")
    
    # Add documents to collection
    collection.add(
        documents=documents,
        metadatas=metadatas,
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


def print_search_results(results: List[Dict[str, Any]], query: str) -> None:
    """Print search results in a formatted way."""
    print(f"\nSearch results for: {query}")
    print("-" * 50)
    
    for result in results['metadatas'][0]:
        print(f"\nScenario: {result['scenario']}")
        print(f"Feature: {result['feature']}")
        print(f"Packages: {result['packages']}")
        print("BDD Steps:")
        for step in result['bdd'].split(' ** '):
            print(f"  - {step}")
        if result['examples'] != "NA":
            print(f"Examples: {result['examples']}")
        print("-" * 50)


def main():
    input_file = 'output.csv'
    
    # Read CSV file
    rows = read_csv(input_file)
    print(f"Loaded {len(rows)} scenarios from {input_file}")
    
    # Setup vector database
    client = setup_vector_db(rows)
    print("Vector database initialized with scenarios")
    
    # Example searches
    queries = [
        "How to add documents to a case?",
        "How to use the welcome tour feature?",
        "What are the login steps?"
    ]
    
    for query in queries:
        results = search_scenarios(client, query)
        print_search_results(results, query)


if __name__ == "__main__":
    main()