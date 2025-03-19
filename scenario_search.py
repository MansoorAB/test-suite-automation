import os
# Set tokenizers parallelism before importing any ML libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import chromadb
from chromadb.config import Settings
from chromadb import Client
from chromadb.utils import embedding_functions
from typing import List, Dict, Any
import json
import logging
from datetime import datetime


def setup_logging(module_name: str) -> logging.Logger:
    """Setup logging with organized directory structure"""
    log_dir = f'./logs/{module_name}'
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%H%M%S')
    log_file = f'{log_dir}/{timestamp}.log'
    
    # Configure logging to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    print(f"\nLogging to: {os.path.abspath(log_file)}\n")
    return logging.getLogger(module_name)

logger = setup_logging('scenario_search')


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


class ScenarioSearch:
    def __init__(self):
        """Initialize ChromaDB client for scenario search"""
        os.makedirs("data/scenario_db", exist_ok=True)
        self.client = Client(Settings(
            persist_directory="./data/scenario_db"
        ))
        
        # Clear and recreate collection on each run
        try:
            self.client.delete_collection("scenarios")
        except:
            pass
            
        self.collection = self.client.create_collection(
            name="scenarios",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Load scenarios from feature files
        self.load_scenarios()

    def extract_scenarios(self, content: str) -> List[Dict[str, str]]:
        """Extract individual scenarios from feature file content"""
        scenarios = []
        current_scenario = []
        current_tags = ""
        
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('@'):
                current_tags = line
            elif line.startswith('Scenario:') or line.startswith('Scenario Outline:'):
                if current_scenario:
                    scenarios.append({
                        'tags': current_tags,
                        'content': '\n'.join(current_scenario)
                    })
                current_scenario = [line]
                current_tags = ""
            elif current_scenario and line:
                current_scenario.append(line)
                
        if current_scenario:
            scenarios.append({
                'tags': current_tags,
                'content': '\n'.join(current_scenario)
            })
            
        return scenarios

    def load_scenarios(self):
        """Load scenarios from feature files"""
        features_dir = "features"
        if not os.path.exists(features_dir):
            logger.warning(f"Features directory not found: {features_dir}")
            return
            
        feature_files = [f for f in os.listdir(features_dir) if f.endswith('.feature')]
        scenario_count = 0
        
        for file in feature_files:
            file_path = os.path.join(features_dir, file)
            with open(file_path, 'r') as f:
                content = f.read()
                scenarios = self.extract_scenarios(content)
                
                for scenario in scenarios:
                    self.collection.add(
                        documents=[scenario['content']],
                        metadatas=[{
                            "file": file,
                            "tags": scenario['tags']
                        }],
                        ids=[f"scenario_{scenario_count}"]
                    )
                    scenario_count += 1
                    
        logger.info(f"Loaded {scenario_count} scenarios from {len(feature_files)} feature files")

    def find_similar_scenarios(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Find similar scenarios based on query"""
        logger.info(f"Searching for scenarios similar to: {query}")
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        matches = []
        for i in range(len(results['ids'][0])):
            match = {
                'scenario': results['documents'][0][i],
                'file': results['metadatas'][0][i]['file'],
                'tags': results['metadatas'][0][i]['tags'],
                'similarity': 1 - results['distances'][0][i]
            }
            matches.append(match)
            
        logger.info(f"Found {len(matches)} matching scenarios")
        return matches


def main():
    """Test functionality"""
    logger.info("Starting scenario search test")
    searcher = ScenarioSearch()
    
    # Test queries
    test_queries = [
        """
        Scenario: User uploads PDF document
        Given user is on upload page
        When user selects a PDF file
        Then system validates the file
        """,
        
        """
        Scenario: Search documents by date
        Given user is on search page
        When user sets date range
        Then matching documents are displayed
        """
    ]
    
    for i, query in enumerate(test_queries, 1):
        logger.info(f"\nTest Query {i}:")
        logger.info("="*50)
        logger.info(query)
        
        matches = searcher.find_similar_scenarios(query)
        
        logger.info("\nMatching Scenarios:")
        logger.info("="*50)
        for match in matches:
            logger.info(f"\nFile: {match['file']}")
            logger.info(f"Tags: {match['tags']}")
            logger.info(f"Similarity: {match['similarity']:.2f}")
            logger.info("-"*30)
            logger.info(match['scenario'])


if __name__ == "__main__":
    main()