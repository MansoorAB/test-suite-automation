import os
import pandas as pd
import chromadb
import shutil
import json
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Union, Optional
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('scenario_api')

app = FastAPI(title="Scenario Search API", description="API for searching scenarios using vector similarity")

class QueryRequest(BaseModel):
    query: str
    n_results: Optional[int] = 3

class ScenarioResult(BaseModel):
    scenario: str
    similarity: float
    feature: str
    packages: Optional[List[str]]
    bdd_steps: List[str]
    example: Optional[List[str]]

class SearchResponse(BaseModel):
    results: List[ScenarioResult]

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

def setup_vector_db() -> chromadb.Client:
    """Set up ChromaDB client and ensure collection exists."""
    # Get the root directory path
    root_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(root_dir, "chroma_db")
    
    logger.info(f"Setting up vector database at: {db_path}")
    
    # clear and create the db directory
    if os.path.exists(db_path):
        logger.info(f"Removing existing database directory: {db_path}")
        shutil.rmtree(db_path)
    os.makedirs(db_path)
    logger.info(f"Created fresh database directory: {db_path}")
    
    # Initialize ChromaDB client
    try:
        client = chromadb.PersistentClient(path=db_path)
        logger.info("Created ChromaDB persistent client")
        
        # Create collection
        collection = client.create_collection("scenarios")
        logger.info("Created 'scenarios' collection")
        
        return client
    except Exception as e:
        logger.error(f"Error setting up vector database: {str(e)}")
        raise

def load_data_to_chroma(client: chromadb.Client, df: pd.DataFrame):
    """Load data from DataFrame to ChromaDB."""
    logger.info("Loading data to ChromaDB")
    collection = client.get_collection("scenarios")
    
    # Check if collection is empty
    if collection.count() == 0:
        # Prepare data for insertion
        ids = [f"scenario_{i}" for i in range(len(df))]
        documents = df['Scenario'].tolist()
        metadatas = []
        
        for _, row in df.iterrows():
            # Convert None values to empty strings to avoid ChromaDB error
            metadata = {
                "feature": row['Feature'] if pd.notna(row['Feature']) else "",
                "packages": json.dumps(row['Packages']),  # Store lists as JSON strings
                "bdd": json.dumps(row['BDD']),
                "example": json.dumps(row['Example'])
            }
            metadatas.append(metadata)
        
        logger.info(f"Adding {len(documents)} scenarios to ChromaDB")
        
        # Add data to collection
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"Successfully added {collection.count()} documents to ChromaDB")

def search_scenarios(
    client: chromadb.Client,
    query: str,
    n_results: int = 3
) -> Dict[str, Any]:
    """Search for similar scenarios using semantic search."""
    logger.info(f"Searching for: '{query}' (top {n_results} results)")
    collection = client.get_collection("scenarios")
    
    # Get total number of documents
    total_docs = collection.count()
    # Adjust n_results if it's greater than total documents
    n_results = min(n_results, total_docs)
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    logger.info(f"Found {len(results['documents'][0])} matching scenarios")
    return results

@app.post("/get_scenario", response_model=SearchResponse)
async def get_scenario(request: QueryRequest):
    logger.info(f"Received request: query='{request.query}', n_results={request.n_results}")
    
    try:
        # Read CSV data
        df = read_csv('output.csv')
        
        # Setup vector database
        client = setup_vector_db()
        
        # Load data to Chroma if needed
        load_data_to_chroma(client, df)
        
        # Search for scenarios
        results = search_scenarios(client, request.query, request.n_results)
        
        # Format response
        response_results = []
        for i in range(len(results['documents'][0])):
            scenario = results['documents'][0][i]
            distance = results['distances'][0][i]
            metadata = results['metadatas'][0][i]
            
            # Parse JSON strings back to lists
            packages = json.loads(metadata.get('packages', '[]'))
            bdd_steps = json.loads(metadata.get('bdd', '[]'))
            example = json.loads(metadata.get('example', '[]'))
            
            # Convert empty lists to None for optional fields
            packages = packages if packages else None
            example = example if example else None
            
            result = ScenarioResult(
                scenario=scenario,
                similarity=round(1 - distance, 2),  # Convert distance to similarity score
                feature=metadata.get('feature', ''),
                packages=packages,
                bdd_steps=bdd_steps,
                example=example
            )
            response_results.append(result)
        
        logger.info(f"Returning {len(response_results)} results")
        return SearchResponse(results=response_results)
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting FastAPI server")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 