import os
import pandas as pd
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Any
import json


def read_csv(input_file: str) -> pd.DataFrame:
    """Read CSV file using pandas."""
    df = pd.read_csv(input_file)
    
    # Convert JSON strings to Python lists
    for column in ['Packages', 'BDD', 'Example']:
        df[column] = df[column].apply(lambda x: json.loads(x) if pd.notna(x) else [])
    
    return df


def setup_vector_db(df: pd.DataFrame) -> chromadb.Client:
    """Initialize ChromaDB and store scenario data with embeddings."""
    # Get the root directory path
    root_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(root_dir, "chroma_db")
    
    print(f"Creating/accessing vector database at: {db_path}")
    
    # Create the directory if it doesn't exist
    os.makedirs(db_path, exist_ok=True)
    print(f"Database directory created/verified at: {db_path}")
    
    try:
        # Initialize ChromaDB client with absolute path
        client = chromadb.PersistentClient(path=db_path)
        print("Created persistent client")
        
        # Delete existing collection if it exists
        try:
            client.delete_collection("scenarios")
            print("Deleted existing scenarios collection")
        except:
            print("No existing scenarios collection found")
        
        # Create new collection
        collection = client.create_collection(
            name="scenarios",
            embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )
        print("Created new scenarios collection")
        
        # Prepare documents for embedding (only Scenario column)
        documents = df['Scenario'].tolist()
        ids = [f"scenario_{idx}" for idx in range(len(documents))]
        
        print(f"Adding {len(documents)} scenarios to the database")
        print("Documents to be added:", documents)
        
        # Add documents to collection
        collection.add(
            documents=documents,
            ids=ids
        )
        
        # Verify documents were added
        count = collection.count()
        print(f"Collection now contains {count} documents")
        
        print("Database setup complete")
        return client
        
    except Exception as e:
        print(f"Error setting up database: {str(e)}")
        raise


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
        # Print each package with @ prefix
        packages = row['Packages']
        print("  " + " ".join(f"@{package}" for package in packages))
        print("BDD Steps:")
        # Print each step on a new line
        steps = row['BDD']
        for step in steps:
            print(f"  - {step}")
        if row['Example'] and len(row['Example']) > 0:
            print("Examples:")
            # Print each example part on a new line with proper indentation
            for part in row['Example']:
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
    
    # Verify database exists
    root_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(root_dir, "chroma_db")
    if os.path.exists(db_path):
        print(f"Database directory exists at: {db_path}")
        print("Contents of database directory:")
        print(os.listdir(db_path))
    else:
        print("Warning: Database directory was not created!")
    
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