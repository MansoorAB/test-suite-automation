import os
# Set tokenizers parallelism before importing any ML libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import chromadb
from chromadb.config import Settings
import openai
from dotenv import load_dotenv

def setup_logging(module_name: str) -> logging.Logger:
    """Setup logging with organized directory structure"""
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
    
    # Print log file location to terminal
    print(f"\nLogging to: {os.path.abspath(log_file)}\n")
    
    return logging.getLogger(module_name)

logger = setup_logging('rag_bdd_implementation_gpt4')

class BDDImplementationRAG:
    def __init__(self):
        """Initialize ChromaDB and OpenAI"""
        # Initialize ChromaDB
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./data/chroma_db"
        ))
        self.collection = self.client.get_or_create_collection(
            name="bdd_steps",
            metadata={"description": "BDD steps and implementations"}
        )
        
        # Initialize OpenAI
        load_dotenv()
        openai.api_key = os.getenv('OPENAI_API_KEY')
        self.model = "gpt-4o-mini"
        
        logger.info("Initialized ChromaDB and OpenAI")

    def normalize_step(self, step_text: str) -> str:
        """Normalize step text by replacing quoted strings with placeholders"""
        quoted_strings = re.findall(r'"([^"]*)"', step_text)
        normalized = step_text
        
        for i, _ in enumerate(quoted_strings, 1):
            normalized = re.sub(r'"[^"]*"', f'QUOTED_TEXT{i}', normalized, 1)
        
        return normalized

    def load_bdd_java_pairs(self, json_file: str) -> None:
        """Load and index BDD-Java pairs from JSON file"""
        with open(json_file, 'r') as f:
            pairs = json.load(f)

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
        
        logger.info(f"Loaded BDD-Java pairs from {json_file}")

    def find_similar_step(self, step_text: str, similarity_threshold: float = 0.8) -> Optional[Dict]:
        """Find similar existing step based on normalized form"""
        normalized_step = self.normalize_step(step_text)
        logger.debug(f"Finding similar step for: {step_text}")
        logger.debug(f"Normalized form: {normalized_step}")

        results = self.collection.query(
            query_texts=[normalized_step],
            n_results=1
        )

        if results['distances'][0][0] < (1 - similarity_threshold):
            match_metadata = results['metadatas'][0][0]
            logger.info(f"Found similar step: {match_metadata['original_step']}")
            logger.info(f"Similarity score: {1 - results['distances'][0][0]:.2f}")
            return {
                'implementation': match_metadata['implementation'],
                'metadata': match_metadata,
                'similarity': 1 - results['distances'][0][0]
            }
        
        logger.info("No similar step found above threshold")
        return None

    def generate_implementation(self, step_text: str) -> Dict[str, Any]:
        """Generate a new implementation for a step"""
        prompt = f"""Generate a complete Java step definition implementation for this Cucumber step:
{step_text}

Requirements:
1. Use EXACTLY this regex pattern format: \"([^\"]*)\" for capturing quoted parameters
   Example: @When("^user clicks on \"([^\"]*)\" button$")
2. Include full implementation with WebDriver code
3. Follow this exact pattern format for all quoted parameters
4. Do not use {{{{string}}}} or any other format
5. Use descriptive method names based on the step text

Example:
For step: user enters "username" in login field
Implementation:
@When("^user enters \"([^\"]*)\" in login field$")
public void userEntersUsername(String username) {{
    WebElement loginField = driver.findElement(By.id("login-username"));
    wait.until(ExpectedConditions.elementToBeClickable(loginField));
    loginField.clear();
    loginField.sendKeys(username);
}}

Important: Always use \"([^\"]*)\" for parameters, never {{{{string}}}} or other formats.
Generate a complete implementation with actual code, not placeholder comments."""

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert in writing Selenium WebDriver step definitions in Java. Always use \"([^\"]*)\" for capturing parameters, never {{{string}}}."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        implementation = response.choices[0].message['content']
        return {
            'type': 'generated',
            'implementation': implementation.strip(),
            'similarity': 0.0
        }

    def get_implementation(self, step_text: str) -> Dict:
        """Get implementation for a step - either existing or generated"""
        similar = self.find_similar_step(step_text)
        
        if similar and similar['similarity'] >= 0.8:
            return {
                'type': 'existing',
                'implementation': similar['implementation'],
                'similarity': similar['similarity'],
                'pattern': similar['metadata']['pattern']
            }
        else:
            return self.generate_implementation(step_text)

def main():
    # Initialize RAG
    rag = BDDImplementationRAG()
    
    # Load existing pairs
    rag.load_bdd_java_pairs("bdd_java_pairs.json")
    
    # Test steps - 3 similar existing and 3 new patterns
    test_steps = [
        # Similar existing patterns
        'user logs in with username "test@example.com" and password "pass123"',
        'user selects document type "Legal Contract" from dropdown',
        'user adds team member "john.doe@test.com" as "Lead Attorney"',
        
        # New patterns
        'user exports document "contract.pdf" to format "docx"',
        'user schedules meeting for case "CASE-2024-001" on "2024-04-01" at "14:00"',
        'user configures auto-reminder for "weekly" with message "Update case status"'
    ]
    
    for step in test_steps:
        result = rag.get_implementation(step)
        logger.info("\n" + "="*80)
        logger.info(f"Step: {step}")
        logger.info(f"Type: {result['type']}")
        if result['type'] == 'existing':
            logger.info(f"Similarity: {result['similarity']:.2f}")
            logger.info(f"Pattern: {result['pattern']}")
        logger.info("Implementation:")
        logger.info(result['implementation'])

if __name__ == "__main__":
    main() 