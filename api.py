import os
import pandas as pd
import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Union, Optional
import uvicorn
import shutil
app = FastAPI(title="Scenario Search API", description="API for searching scenarios using vector similarity")

class QueryRequest(BaseModel):
    query: str
    n_results: Optional[int] = 3

class ScenarioResult(BaseModel):
    scenario: str
    similarity: float
    feature: str
    packages: Optional[str]
    bdd_steps: List[str]
    example: Optional[str]

class SearchResponse(BaseModel):
    results: List[ScenarioResult]

def read_csv(input_file: str) -> pd.DataFrame:
    """Read CSV file using pandas."""
    return pd.read_csv(input_file)

def setup_vector_db() -> chromadb.Client:
    """Set up ChromaDB client and ensure collection exists."""
    # Get the root directory path
    root_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(root_dir, "chroma_db")
    
    # clear and create the db directory
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    os.makedirs(db_path)
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=db_path)
    
    # Check if collection exists, create if not
    try:
        collection = client.get_collection("scenarios")
    except Exception:
        # Collection doesn't exist, create it
        collection = client.create_collection("scenarios")
    
    return client

def load_data_to_chroma(client: chromadb.Client, df: pd.DataFrame):
    """Load data from DataFrame to ChromaDB."""
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
                "packages": row['Packages'] if pd.notna(row['Packages']) else "",
                "bdd": row['BDD'] if pd.notna(row['BDD']) else "",
                "example": row['Example'] if pd.notna(row['Example']) else ""
            }
            metadatas.append(metadata)
        
        # Add data to collection
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

def search_scenarios(
    client: chromadb.Client,
    query: str,
    n_results: int = 3
) -> Dict[str, Any]:
    """Search for similar scenarios using semantic search."""
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
    
    return results

@app.post("/get_scenario", response_model=SearchResponse)
async def get_scenario(request: QueryRequest):
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
            
            # Parse BDD steps
            bdd_steps = []
            if metadata.get('bdd') and metadata['bdd'] != "":
                bdd_steps = metadata['bdd'].split(' ** ')
            
            # Convert empty strings back to None for the response
            packages = None if not metadata.get('packages') or metadata['packages'] == "" else metadata['packages']
            example = None if not metadata.get('example') or metadata['example'] == "" else metadata['example']
            
            result = ScenarioResult(
                scenario=scenario,
                similarity=round(1 - distance, 2),  # Convert distance to similarity score
                feature=metadata.get('feature', ''),
                packages=packages,
                bdd_steps=bdd_steps,
                example=example
            )
            response_results.append(result)
        
        return SearchResponse(results=response_results)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 