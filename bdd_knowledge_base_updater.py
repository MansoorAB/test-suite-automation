import os
import chromadb
import json
import logging
from datetime import datetime
from typing import List, Dict, Any

def setup_logging(module_name: str) -> logging.Logger:
    log_dir = f'./logs/{module_name}'
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%H%M%S')
    log_file = f'{log_dir}/{timestamp}.log'
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    return logging.getLogger(module_name)

logger = setup_logging('bdd_knowledge_base_updater')

class BDDKnowledgeBase:
    def __init__(self):
        """Initialize ChromaDB client for BDD step-implementation pairs"""
        # Create separate directory for BDD knowledge base
        os.makedirs("data/bdd_knowledge_db", exist_ok=True)
        
        self.client = chromadb.PersistentClient(path="data/bdd_knowledge_db")
        
        # Create or get collection for step-implementation pairs
        self.collection = self.client.get_or_create_collection(
            name="step_implementations",
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Initialized BDD knowledge base with {len(self.collection.get()['ids'])} step-implementation pairs")

    def update_knowledge_base(self, json_file: str):
        """Update the knowledge base with step-implementation pairs"""
        logger.info(f"Updating knowledge base from {json_file}")
        
        try:
            with open(json_file, 'r') as f:
                pairs = json.load(f)

            # Clear existing collection
            self.collection.delete()
            self.collection = self.client.get_or_create_collection(
                name="step_implementations",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Add new step-implementation pairs
            for scenario in pairs:
                for step in scenario["steps"]:
                    if step.get("java_implementation"):
                        step_text = step["step_text"]
                        normalized_step = self.normalize_step(step_text)
                        
                        self.collection.add(
                            documents=[normalized_step],
                            metadatas=[{
                                "original_step": step_text,
                                "normalized_step": normalized_step,
                                "pattern": step["java_implementation"]["pattern"],
                                "implementation": step["java_implementation"]["implementation"],
                                "feature_file": scenario["feature_file"],
                                "scenario_name": scenario["scenario"]
                            }],
                            ids=[f"step_{len(self.collection.get()['ids'])}"]
                        )
            
            logger.info(f"Successfully updated knowledge base with {len(self.collection.get()['ids'])} step-implementation pairs")
            
        except Exception as e:
            logger.error(f"Error updating knowledge base: {str(e)}")
            raise

    def normalize_step(self, step_text: str) -> str:
        """Normalize step text by replacing values with placeholders"""
        # Implementation from rag_bdd_implementation.py
        # ... (same normalization logic)
        pass

def main():
    """Daily update of BDD knowledge base"""
    logger.info("Starting daily BDD knowledge base update")
    
    kb = BDDKnowledgeBase()
    kb.update_knowledge_base("bdd_java_pairs.json")
    
    logger.info("Completed daily BDD knowledge base update")

if __name__ == "__main__":
    main() 