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
import vertexai
from vertexai.generative_models import GenerativeModel

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
    return logging.getLogger(module_name)

logger = setup_logging('rag_bdd_implementation')

class BDDImplementationRAG:
    def __init__(self):
        """Initialize ChromaDB and Vertex AI"""
        # Initialize ChromaDB
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./data/chroma_db"
        ))
        self.collection = self.client.get_or_create_collection(
            name="bdd_steps",
            metadata={"description": "BDD steps and implementations"}
        )
        
        # Initialize Vertex AI
        vertexai.init(project=os.getenv("GOOGLE_CLOUD_PROJECT"), location="us-central1")
        self.model = GenerativeModel("gemini-1.5-pro-flash")
        
        logger.info("Initialized ChromaDB and Vertex AI")

    def normalize_step(self, step_text: str) -> str:
        """Normalize step text by replacing quoted strings with placeholders"""
        # Find all quoted strings
        quoted_strings = re.findall(r'"([^"]*)"', step_text)
        normalized = step_text
        
        # Replace each quoted string with a placeholder
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
                    
                    # Store in ChromaDB with normalized step as the document
                    self.collection.add(
                        documents=[normalized_step],  # Changed: store normalized step instead of implementation
                        metadatas=[{
                            "original_step": step_text,
                            "normalized_step": normalized_step,
                            "pattern": step["java_implementation"]["pattern"],
                            "implementation": step["java_implementation"]["implementation"],  # Moved implementation to metadata
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

        # Query ChromaDB using normalized step
        results = self.collection.query(
            query_texts=[normalized_step],
            n_results=1
        )

        if results['distances'][0][0] < (1 - similarity_threshold):
            match_metadata = results['metadatas'][0][0]
            logger.info(f"Found similar step: {match_metadata['original_step']}")
            logger.info(f"Similarity score: {1 - results['distances'][0][0]:.2f}")
            return {
                'implementation': match_metadata['implementation'],  # Changed: get implementation from metadata
                'metadata': match_metadata,
                'similarity': 1 - results['distances'][0][0]
            }
        
        logger.info("No similar step found above threshold")
        return None

    def generate_implementation(self, step_text: str) -> Dict[str, Any]:
        """Generate a new implementation for a step using true RAG approach"""
        # First, retrieve top N similar steps regardless of threshold
        normalized_step = self.normalize_step(step_text)
        
        # Get top 3 similar steps
        results = self.collection.query(
            query_texts=[normalized_step],
            n_results=3  # Retrieve top 3 similar steps
        )
        
        # Extract examples from results
        examples = []
        for i in range(min(3, len(results['metadatas'][0]))):
            metadata = results['metadatas'][0][i]
            similarity = 1 - results['distances'][0][i]
            
            # Only include reasonably similar examples with implementation
            if similarity > 0.5 and 'implementation' in metadata and 'pattern' in metadata:
                examples.append({
                    'step': metadata.get('original_step', ''),
                    'pattern': metadata.get('pattern', ''),
                    'implementation': metadata.get('implementation', ''),
                    'similarity': similarity
                })
        
        # Build a prompt that includes the examples
        examples_text = ""
        if examples:
            examples_text = "Here are some similar step implementations for reference:\n\n"
            for i, example in enumerate(examples, 1):
                examples_text += f"Example {i} (Similarity: {example['similarity']:.2f}):\n"
                examples_text += f"Step: {example['step']}\n"
                examples_text += f"Pattern: {example['pattern']}\n"
                examples_text += f"Implementation:\n{example['implementation']}\n\n"
        
        prompt = f"""Generate a complete Java step definition implementation for this Cucumber step:
{step_text}

Requirements:
1. Use EXACTLY this regex pattern format: \"([^\"]*)\" for capturing quoted parameters
   Example: @When("^user clicks on \"([^\"]*)\" button$")
2. Include full implementation with WebDriver code
3. Follow this exact pattern format for all quoted parameters
4. Do not use {{{{string}}}} or any other format
5. Use descriptive method names based on the step text
6. Maintain consistency with the example implementations provided

{examples_text}
For reference, here's a basic example:
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

        logger.debug(f"RAG Prompt with {len(examples)} examples:\n{prompt}")

        generation_config = {
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 1024,
        }

        response = self.model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings={"HARM_CATEGORY_DANGEROUS_CONTENT": "block_none"}
        )

        implementation = response.text
        return {
            'type': 'generated',
            'implementation': implementation.strip(),
            'similarity': 0.0,
            'examples_used': len(examples)
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
    
    # Track statistics
    stats = {
        'total': len(test_steps),
        'retrieved': 0,
        'generated': 0,
        'examples_used': []
    }
    
    for step in test_steps:
        result = rag.get_implementation(step)
        logger.info("\n" + "="*80)
        logger.info(f"Step: {step}")
        logger.info(f"Type: {result['type']}")
        
        if result['type'] == 'existing':
            logger.info(f"Similarity: {result['similarity']:.2f}")
            logger.info(f"Pattern: {result['pattern']}")
            stats['retrieved'] += 1
        else:
            logger.info(f"Examples used: {result.get('examples_used', 0)}")
            stats['generated'] += 1
            stats['examples_used'].append(result.get('examples_used', 0))
            
        logger.info("Implementation:")
        logger.info(result['implementation'])
    
    # Log statistics
    logger.info("\n" + "="*80)
    logger.info("IMPLEMENTATION STATISTICS")
    logger.info("="*80)
    logger.info(f"Total steps processed: {stats['total']}")
    logger.info(f"Retrieved implementations: {stats['retrieved']} ({stats['retrieved']/stats['total']*100:.1f}%)")
    logger.info(f"Generated implementations: {stats['generated']} ({stats['generated']/stats['total']*100:.1f}%)")
    
    if stats['generated'] > 0:
        avg_examples = sum(stats['examples_used']) / len(stats['examples_used'])
        logger.info(f"Average examples used per generation: {avg_examples:.1f}")
        logger.info(f"Examples distribution: {stats['examples_used']}")
    
    # Print log file location to terminal
    log_files = [f for f in os.listdir('./logs/rag_bdd_implementation') if f.endswith('.log')]
    if log_files:
        # Get the log file that was created in this run (most recent)
        current_time = datetime.now().strftime('%H%M%S')
        matching_logs = [f for f in log_files if f.startswith(current_time[:4])]
        if matching_logs:
            latest_log = sorted(matching_logs)[-1]
            log_path = os.path.abspath(f'./logs/rag_bdd_implementation/{latest_log}')
            print(f"\nDetailed results written to: {log_path}\n")
        else:
            print("\nCouldn't find the current log file.\n")

if __name__ == "__main__":
    main() 