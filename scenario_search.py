import os
import pandas as pd
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Any
import json
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('scenario_search')


def read_csv(input_file: str) -> pd.DataFrame:
    """Read CSV file using pandas."""
    logger.info(f"Reading CSV file: {input_file}")
    try:
        df = pd.read_csv(input_file)
        
        # Convert JSON strings to Python lists
        for column in ['Packages', 'BDD', 'Example']:
            df[column] = df[column].apply(lambda x: json.loads(x) if pd.notna(x) else [])
        
        logger.info(f"Successfully loaded {len(df)} rows from CSV")
        return df
    except Exception as e:
        logger.error(f"Error reading CSV file: {str(e)}")
        raise


def setup_vector_db(df: pd.DataFrame) -> chromadb.Client:
    """Initialize ChromaDB and store scenario data with embeddings."""
    # Get the root directory path
    root_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(root_dir, "chroma_db")
    
    logger.info(f"Creating/accessing vector database at: {db_path}")
    
    # Create the directory if it doesn't exist
    os.makedirs(db_path, exist_ok=True)
    logger.info(f"Database directory created/verified at: {db_path}")
    
    try:
        # Initialize ChromaDB client with absolute path
        client = chromadb.PersistentClient(path=db_path)
        logger.info("Created persistent client")
        
        # Delete existing collection if it exists
        try:
            client.delete_collection("scenarios")
            logger.info("Deleted existing scenarios collection")
        except:
            logger.info("No existing scenarios collection found")
        
        # Create new collection
        collection = client.create_collection(
            name="scenarios",
            embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )
        logger.info("Created new scenarios collection")
        
        # Prepare documents for embedding (only Scenario column)
        documents = df['Scenario'].tolist()
        ids = [f"scenario_{idx}" for idx in range(len(documents))]
        
        logger.info(f"Adding {len(documents)} scenarios to the database")
        logger.debug(f"Documents to be added: {documents}")
        
        # Add documents to collection
        collection.add(
            documents=documents,
            ids=ids
        )
        
        # Verify documents were added
        count = collection.count()
        logger.info(f"Collection now contains {count} documents")
        
        logger.info("Database setup complete")
        return client
        
    except Exception as e:
        logger.error(f"Error setting up database: {str(e)}")
        raise


def search_scenarios(
    client: chromadb.Client,
    query: str,
    n_results: int = 3
) -> List[Dict[str, Any]]:
    """Search for similar scenarios using semantic search."""
    logger.info(f"Searching for: '{query}' (top {n_results} results)")
    
    collection = client.get_collection("scenarios")
    
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        logger.info(f"Found {len(results['documents'][0])} matching scenarios")
        return results
    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        raise


def print_search_results(
    results: List[Dict[str, Any]], 
    query: str,
    df: pd.DataFrame
) -> None:
    """Print search results in a formatted way."""
    logger.info(f"Displaying search results for: '{query}'")
    
    print(f"\nSearch results for: {query}")
    print("-" * 50)
    
    for idx, (doc, distance) in enumerate(zip(
        results['documents'][0],
        results['distances'][0]
    )):
        # Find corresponding row in dataframe
        try:
            row = df[df['Scenario'] == doc].iloc[0]
            
            similarity = 1 - distance
            logger.debug(f"Result {idx + 1}: '{doc}' (Similarity: {similarity:.2f})")
            
            print(f"\nResult {idx + 1} (Similarity: {similarity:.2f})")
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
        except IndexError:
            logger.warning(f"Could not find scenario '{doc}' in the dataframe")
            print(f"\nResult {idx + 1} (Similarity: {1 - distance:.2f})")
            print(f"Scenario: {doc}")
            print("(Additional details not available)")
            print("-" * 50)


def main():
    logger.info("Starting scenario search process")
    
    input_file = 'output.csv'
    
    try:
        # Read CSV file using pandas
        df = read_csv(input_file)
        logger.info(f"Loaded {len(df)} scenarios from {input_file}")
        
        # Setup vector database
        client = setup_vector_db(df)
        logger.info("Vector database initialized with scenarios")
        
        # Verify database exists
        root_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(root_dir, "chroma_db")
        if os.path.exists(db_path):
            logger.info(f"Database directory exists at: {db_path}")
            logger.debug(f"Contents of database directory: {os.listdir(db_path)}")
        else:
            logger.warning("Warning: Database directory was not created!")
        
        # Example searches
        queries = [
            "How to add documents to a case?",
            "How to use the welcome tour feature?",
            "What are the login steps?"
        ]
        
        logger.info(f"Running {len(queries)} example searches")
        
        for query in queries:
            results = search_scenarios(client, query)
            print_search_results(results, query, df)
        
        logger.info("Scenario search process completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()