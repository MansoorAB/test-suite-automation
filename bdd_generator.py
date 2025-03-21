import os
# Set tokenizers parallelism before importing any ML libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
from datetime import datetime
from typing import Dict, List
import vertexai
from vertexai.generative_models import GenerativeModel
from rag_bdd_implementation import BDDImplementationRAG

def setup_logging(module_name: str) -> logging.Logger:
    """Setup logging with organized directory structure"""
    log_dir = f'./logs/{module_name}'
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%H%M%S')
    log_file = f'{log_dir}/{timestamp}.log'
    
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

logger = setup_logging('bdd_generator')

class BDDGenerator:
    def __init__(self):
        """Initialize Vertex AI and RAG system"""
        vertexai.init(project=os.getenv("GOOGLE_CLOUD_PROJECT"), location="us-central1")
        self.model = GenerativeModel("gemini-1.5-pro-flash")
        self.rag = BDDImplementationRAG()
        self.rag.load_bdd_java_pairs("bdd_java_pairs.json")
        logger.info("Initialized Vertex AI and RAG system")

    def generate_step_implementation(self, step_text: str) -> Dict:
        """Get implementation for a step - either existing or generated"""
        logger.info(f"Processing step: {step_text}")
        result = self.rag.get_implementation(step_text)
        
        if result['type'] == 'existing':
            logger.info(f"Found existing implementation with similarity: {result['similarity']:.2f}")
        else:
            logger.info(f"Generated new implementation with {result.get('examples_used', 0)} examples")
        
        return result

    def generate_feature_file(self, acceptance_criteria: str, feature_name: str) -> str:
        """Generate BDD feature file from acceptance criteria"""
        logger.info(f"Generating BDD for feature: {feature_name}")
        logger.info("Acceptance Criteria:")
        logger.info(acceptance_criteria)

        prompt = f"""Convert this acceptance criteria into a Gherkin feature file following this exact format:

Feature: {feature_name} Feature

@smoke @regression
Scenario: [Descriptive scenario name]
  Given [precondition]
  When [action]
  And [additional action if needed]
  Then [expected result]
  And [additional result if needed]

Rules:
1. Start exactly with 'Feature: {feature_name} Feature'
2. Put @smoke @regression tags on a single line before each scenario
3. Do not include any 'As a user...' statements
4. Use clear, specific step descriptions
5. Include all data in double quotes
6. Indent steps with 2 spaces
7. Add multiple scenarios if needed for different test cases
8. Do not include any markdown code block markers

Return only the feature file content.

Acceptance Criteria:
{acceptance_criteria}"""

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

        feature_content = response.text
        logger.info("Generated Feature File:")
        logger.info(feature_content)
        return feature_content

    def generate_feature_and_steps(self, acceptance_criteria: str, feature_name: str, stats=None) -> tuple[str, str]:
        """Generate both feature file and step definitions"""
        # Initialize stats if not provided
        if stats is None:
            stats = {
                'total_steps': 0,
                'retrieved_steps': 0,
                'generated_steps': 0,
                'examples_used': []
            }
        
        # Generate feature file
        feature_content = self.generate_feature_file(acceptance_criteria, feature_name)
        
        # Extract steps from feature file
        steps = self._extract_steps(feature_content)
        stats['total_steps'] = len(steps)
        
        # Generate implementations for each step
        implementations = []
        for step in steps:
            result = self.generate_step_implementation(step)
            implementations.append(result['implementation'])
            
            # Track statistics
            if result['type'] == 'existing':
                stats['retrieved_steps'] += 1
            else:
                stats['generated_steps'] += 1
                if 'examples_used' in result:
                    stats['examples_used'].append(result['examples_used'])
        
        # Create Java file
        java_content = self._create_java_file(feature_name, implementations)
        
        return feature_content, java_content

    def _extract_steps(self, feature_content: str) -> List[str]:
        """Extract all steps from feature file content"""
        steps = []
        for line in feature_content.split('\n'):
            line = line.strip()
            if any(line.startswith(keyword) for keyword in ['Given ', 'When ', 'Then ', 'And ']):
                steps.append(line)
        return steps

    def _create_java_file(self, feature_name: str, implementations: List[str]) -> str:
        """Create a Java step definitions file"""
        class_name = feature_name.replace(' ', '') + 'Steps'
        
        java_content = f"""package stepdefinitions;

import io.cucumber.java.en.*;
import org.junit.Assert;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;

public class {class_name} {{
    
{chr(10).join(implementations)}

}}"""
        return java_content

    def save_feature_file(self, content: str, feature_name: str) -> str:
        """Save the generated feature file"""
        os.makedirs('./features', exist_ok=True)
        filename = f"./features/{feature_name.lower().replace(' ', '_')}.feature"
        
        with open(filename, 'w') as f:
            f.write(content)
        
        logger.info(f"Saved feature file to: {filename}")
        return filename

    def save_java_file(self, content: str, feature_name: str) -> str:
        """Save the generated Java step definitions file"""
        os.makedirs('./step_definitions', exist_ok=True)
        filename = f"./step_definitions/{feature_name.replace(' ', '')}Steps.java"
        
        with open(filename, 'w') as f:
            f.write(content)
        
        logger.info(f"Saved Java step definitions to: {filename}")
        return filename

def main():
    generator = BDDGenerator()
    
    # Example acceptance criteria
    test_cases = [
        {
            "name": "Document Upload",
            "criteria": """
1. User can upload PDF files up to 10MB
2. System validates file type and size
3. User can add metadata (title, type, tags)
4. System generates thumbnail for uploaded document
5. User receives confirmation after successful upload
6. System should show error for invalid files
            """
        }
    ]
    
    generated_files = []
    
    for test_case in test_cases:
        logger.info("\n" + "="*80)
        logger.info(f"Processing: {test_case['name']}")
        
        feature_content, java_content = generator.generate_feature_and_steps(
            test_case['criteria'],
            test_case['name']
        )
        
        feature_file = generator.save_feature_file(feature_content, test_case['name'])
        java_file = generator.save_java_file(java_content, test_case['name'])
        
        generated_files.extend([feature_file, java_file])
        logger.info("="*80)
    
    summary = "\nGenerated Files Summary:"
    summary += "\n" + "="*50
    for file in generated_files:
        summary += f"\n- {os.path.abspath(file)}"
    summary += "\n" + "="*50
    
    print(summary)
    logger.info(summary)

if __name__ == "__main__":
    main() 