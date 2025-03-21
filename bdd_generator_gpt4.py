import os
# Set tokenizers parallelism before importing any ML libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
from datetime import datetime
from typing import Dict, List, Any
import openai
from dotenv import load_dotenv
from rag_bdd_implementation_gpt4 import BDDImplementationRAG

# Load environment variables
load_dotenv()

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
    
    print(f"\nLogging to: {os.path.abspath(log_file)}\n")
    return logging.getLogger(module_name)

logger = setup_logging('bdd_generator_gpt4')

class BDDGenerator:
    def __init__(self):
        """Initialize OpenAI and RAG system"""
        openai.api_key = os.getenv('OPENAI_API_KEY')
        self.rag = BDDImplementationRAG()  # Initialize RAG system
        self.rag.load_bdd_java_pairs("bdd_java_pairs.json")  # Load existing implementations
        self.model = "gpt-4o-mini"  # Set the model attribute
        logger.info("Initialized OpenAI GPT-4 and RAG system")

    def generate_step_implementation(self, step_text: str) -> Dict:
        """Get implementation for a step - either existing or generated"""
        logger.info(f"Processing step: {step_text}")
        result = self.rag.get_implementation(step_text)
        
        if result['type'] == 'existing':
            logger.info(f"Found existing implementation with similarity: {result['similarity']:.2f}")
        else:
            logger.info(f"Generated new implementation with {result.get('examples_used', 0)} examples")
        
        return result

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
        feature_content = self.generate_feature_file(feature_name, acceptance_criteria)
        
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

    def save_java_file(self, content: str, feature_name: str) -> str:
        """Save the generated Java step definitions file"""
        # Create step_definitions directory if it doesn't exist
        os.makedirs('./step_definitions', exist_ok=True)
        
        # Create filename
        filename = f"./step_definitions/{feature_name.replace(' ', '')}Steps.java"
        
        with open(filename, 'w') as f:
            f.write(content)
        
        logger.info(f"Saved Java step definitions to: {filename}")
        return filename

    def generate_feature_file(self, feature_name: str, acceptance_criteria: str) -> Dict[str, Any]:
        """Generate a feature file from acceptance criteria"""
        prompt = f"""Convert this acceptance criteria into a Gherkin feature file.
                The feature file must be specific to these acceptance criteria and should not generate generic login scenarios.

                Feature: {feature_name}

                Requirements:
                1. Create scenarios that EXACTLY match the provided acceptance criteria
                2. Do not generate generic login scenarios
                3. Use specific steps that match the acceptance criteria
                4. Include all validations mentioned in acceptance criteria
                5. Tag scenarios with @smoke @regression

                Acceptance Criteria:
                {acceptance_criteria}

                Example format for reference:
                @smoke @regression
                Scenario: [Specific scenario from acceptance criteria]
                Given [specific precondition]
                When [specific action from acceptance criteria]
                And [additional specific action if needed]
                Then [specific expected result]
                And [additional verification if needed]

                Generate a complete .feature file with scenarios that match ONLY the provided acceptance criteria."""

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a BDD expert. \
                 Generate feature files that exactly match the provided acceptance criteria. Do not generate generic login scenarios."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        feature_content = response.choices[0].message['content']
        logger.info("Generated Feature File:")
        logger.info(feature_content)
        return feature_content

    def save_feature_file(self, content: str, feature_name: str) -> str:
        """Save the generated feature file"""
        # Create features directory if it doesn't exist
        os.makedirs('./features', exist_ok=True)
        
        # Create filename from feature name
        filename = f"./features/{feature_name.lower().replace(' ', '_')}.feature"
        
        with open(filename, 'w') as f:
            f.write(content)
        
        logger.info(f"Saved feature file to: {filename}")
        return filename

def main():
    generator = BDDGenerator()
    
    # Example acceptance criteria
    test_cases = [
        {
            "name": "Document Upload",
            "criteria": """
                As a user, I need to upload documents to the system
                Acceptance Criteria:
                1. User can upload PDF files up to 10MB
                2. System validates file type and size
                3. User can add metadata (title, type, tags)
                4. System generates thumbnail for uploaded document
                5. User receives confirmation after successful upload
                6. System should show error for invalid files
                            """
                        },
                        {
                            "name": "Search Functionality",
                            "criteria": """
                As a user, I need to search for documents
                Acceptance Criteria:
                1. User can search by document title, type, and content
                2. Search results show document title, type, and upload date
                3. User can filter results by date range
                4. System highlights matching terms in results
                5. Results are paginated with 10 items per page
                6. User can sort results by different fields
                            """
        }
    ]
    
    generated_files = []  # Track all generated files
    
    for test_case in test_cases:
        logger.info("\n" + "="*80)
        logger.info(f"Processing: {test_case['name']}")
        
        # Generate both feature and step definitions
        feature_content, java_content = generator.generate_feature_and_steps(
            test_case['criteria'],
            test_case['name']
        )
        
        # Save both files
        feature_file = generator.save_feature_file(feature_content, test_case['name'])
        java_file = generator.save_java_file(java_content, test_case['name'])
        
        generated_files.extend([feature_file, java_file])
        logger.info("="*80)
    
    # Print and log summary
    summary = "\nGenerated Files Summary:"
    summary += "\n" + "="*50
    for file in generated_files:
        summary += f"\n- {os.path.abspath(file)}"
    summary += "\n" + "="*50
    
    print(summary)
    logger.info(summary)

if __name__ == "__main__":
    main() 