import os
import re
import json
import glob
from pathlib import Path
from collections import defaultdict
import logging
from typing import Dict, List, Any, Optional, Tuple
from vertexai.generative_models import GenerativeModel
import vertexai
from google.cloud import aiplatform
from config import VertexAIConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('extract_bdd_java_pairs')


class BDDJavaExtractor:
    def __init__(self, feature_dir: str, java_dir: str, output_file: str = "bdd_java_pairs.json"):
        self.feature_dir = feature_dir
        self.java_dir = java_dir
        self.output_file = output_file
        self.step_definitions: Dict[str, Dict[str, Any]] = {}
        self.bdd_java_pairs: List[Dict[str, Any]] = []
        
        # Initialize Vertex AI configuration
        self.vertex_config = VertexAIConfig()
        
        # Initialize Vertex AI
        try:
            vertexai.init(
                project=self.vertex_config.project_id,
                location=self.vertex_config.location
            )
            logger.info("Successfully initialized Vertex AI")
        except Exception as e:
            logger.error(f"Error initializing Vertex AI: {str(e)}")
            raise
        
        logger.info(f"Initialized BDDJavaExtractor with feature_dir={feature_dir}, java_dir={java_dir}")
        
    def extract_java_step_definitions(self) -> None:
        """Extract all step definitions from Java files"""
        logger.info(f"Scanning Java files in {self.java_dir}...")
        
        try:
            java_files = glob.glob(f"{self.java_dir}/**/*.java", recursive=True)
            logger.info(f"Found {len(java_files)} Java files")
            
            # Regex patterns for step annotations
            step_pattern = re.compile(r'@(?:Given|When|Then|And|But)\s*\(\s*"(.*?)"\s*\)', re.DOTALL)
            
            for java_file in java_files:
                logger.debug(f"Processing {java_file}")
                try:
                    with open(java_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                        # Find all step definition annotations
                        step_matches = step_pattern.finditer(content)
                        step_count = 0
                        
                        for step_match in step_matches:
                            step_text = step_match.group(1)
                            start_pos = step_match.end()
                            
                            # Find the method signature that follows the annotation
                            method_sig_pattern = re.compile(r'public\s+void\s+\w+\s*\([^)]*\)\s*(?:throws\s+[^{]+)?\s*\{', re.DOTALL)
                            method_sig_match = method_sig_pattern.search(content[start_pos:])
                            
                            if method_sig_match:
                                method_sig = method_sig_match.group(0)
                                method_start = start_pos + method_sig_match.end()
                                
                                # Find the matching closing brace by counting braces
                                open_braces = 1
                                close_pos = method_start
                                
                                while open_braces > 0 and close_pos < len(content):
                                    if content[close_pos] == '{':
                                        open_braces += 1
                                    elif content[close_pos] == '}':
                                        open_braces -= 1
                                    close_pos += 1
                                
                                if open_braces == 0:
                                    # Extract the complete method body including nested structures
                                    method_body = content[method_start:close_pos-1].strip()
                                    
                                    # Include the method signature in the implementation
                                    full_implementation = method_sig + "\n" + method_body + "\n}"
                                    
                                    # Convert regex pattern in step to a more readable format
                                    readable_step = self._convert_regex_to_readable(step_text)
                                    self.step_definitions[readable_step] = {
                                        "pattern": step_text,
                                        "implementation": full_implementation,
                                        "file": os.path.basename(java_file)
                                    }
                                    step_count += 1
                        
                        logger.debug(f"Extracted {step_count} step definitions from {java_file}")
                except Exception as e:
                    logger.error(f"Error processing Java file {java_file}: {str(e)}")
            
            logger.info(f"Extracted {len(self.step_definitions)} step definitions in total")
        except Exception as e:
            logger.error(f"Error extracting Java step definitions: {str(e)}")
            raise
    
    def _convert_regex_to_readable(self, step_text: str) -> str:
        """Convert regex patterns in step text to a more readable format"""
        logger.debug(f"Converting regex pattern: {step_text}")
        
        # Replace common regex patterns with placeholders
        readable = step_text
        # Replace quoted capture groups
        readable = re.sub(r'\"([^\"]*?)\"', '"PARAM"', readable)
        # Replace unquoted capture groups
        readable = re.sub(r'\(\.\*?\)', 'PARAM', readable)
        readable = re.sub(r'\([^)]+\)', 'PARAM', readable)
        
        logger.debug(f"Converted to: {readable}")
        return readable
    
    def extract_feature_steps(self) -> None:
        """Extract steps from feature files and match with Java implementations"""
        logger.info(f"Scanning feature files in {self.feature_dir}...")
        
        try:
            feature_files = glob.glob(f"{self.feature_dir}/**/*.feature", recursive=True)
            logger.info(f"Found {len(feature_files)} feature files")
            
            step_pattern = re.compile(r'^\s*(?:Given|When|Then|And|But)\s+(.+)$', re.MULTILINE)
            scenario_pattern = re.compile(r'^\s*Scenario:(.+)$', re.MULTILINE)
            
            for feature_file in feature_files:
                logger.debug(f"Processing {feature_file}")
                try:
                    with open(feature_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                        # Extract scenarios
                        scenarios = scenario_pattern.finditer(content)
                        scenario_count = 0
                        
                        for scenario_match in scenarios:
                            scenario_title = scenario_match.group(1).strip()
                            scenario_start = scenario_match.end()
                            
                            # Find the next scenario or end of file
                            next_scenario = scenario_pattern.search(content, scenario_start)
                            scenario_end = next_scenario.start() if next_scenario else len(content)
                            
                            scenario_content = content[scenario_start:scenario_end]
                            
                            # Extract steps in this scenario
                            steps = step_pattern.finditer(scenario_content)
                            scenario_steps = []
                            
                            for step_match in steps:
                                step_text = step_match.group(1).strip()
                                
                                # Try to find matching Java implementation
                                java_impl = self._find_matching_implementation(step_text)
                                
                                scenario_steps.append({
                                    "step_text": step_text,
                                    "java_implementation": java_impl
                                })
                            
                            if scenario_steps:
                                self.bdd_java_pairs.append({
                                    "feature_file": os.path.basename(feature_file),
                                    "scenario": scenario_title,
                                    "steps": scenario_steps
                                })
                                scenario_count += 1
                        
                        logger.debug(f"Extracted {scenario_count} scenarios from {feature_file}")
                except Exception as e:
                    logger.error(f"Error processing feature file {feature_file}: {str(e)}")
            
            logger.info(f"Extracted {len(self.bdd_java_pairs)} scenarios with steps in total")
        except Exception as e:
            logger.error(f"Error extracting feature steps: {str(e)}")
            raise
    
    def convert_cucumber_to_regex(self, pattern: str) -> str:
        """Convert a Cucumber step pattern to Python regex pattern"""
        logger.debug(f"Original pattern: {pattern}")
        
        # Handle start/end markers
        if not pattern.startswith('^'):
            pattern = '^' + pattern
        if not pattern.endswith('$'):
            pattern = pattern + '$'
        
        # Split the pattern into parts (text and capture groups)
        parts = []
        current_pos = 0
        
        # Find all capture groups and their surrounding text
        while True:
            # Find next capture group
            capture_match = re.search(r'"(\[\^"\]\*)"', pattern[current_pos:])
            if not capture_match:
                # Add remaining text
                if current_pos < len(pattern):
                    parts.append(('text', pattern[current_pos:]))
                break
            
            # Add text before capture group
            if capture_match.start() > 0:
                parts.append(('text', pattern[current_pos:current_pos + capture_match.start()]))
            
            # Add capture group
            parts.append(('capture', capture_match.group(0)))
            current_pos += capture_match.end()
        
        # Build regex pattern
        regex_parts = []
        for part_type, part in parts:
            if part_type == 'text':
                # Escape special regex characters in text parts
                regex_parts.append(re.escape(part))
            else:
                # Replace capture group with wildcard
                regex_parts.append('"[^"]*"')
        
        pattern = ''.join(regex_parts)
        logger.debug(f"Converted pattern: {pattern}")
        return pattern
    
    def _find_matching_implementation(self, step_text: str) -> Optional[Dict[str, Any]]:
        """Find the Java implementation that matches this step text using Gemini"""
        logger.debug(f"Finding implementation for step: {step_text}")
        
        # Create a few-shot prompt with examples that better match your patterns
        few_shot_examples = """
        BDD Step: user verifies "Welcome to InSight!" popup and click "Next"
        Available Pattern: ^user verifies "([^"]*)" popup and click "([^"]*)"$
        ✓ MATCHES - because the text between quotes in BDD matches with ([^"]*) in pattern
        
        BDD Step: user verifies popup title "Sidebar Menu" and click "Next"
        Available Pattern: ^user verifies popup title "([^"]*)" and click "([^"]*)"$
        ✓ MATCHES - because both the fixed text and parameterized parts align
        
        BDD Step: user verifies popup title "Tasks" and click "Next"
        Available Pattern: ^user verifies popup title "([^"]*)" and click "([^"]*)"$
        ✓ MATCHES - same pattern structure with different parameter values
        
        BDD Step: user clicks on save button from drop down
        Available Pattern: ^user clicks on save button from drop down$
        ✓ MATCHES - exact text match with no parameters
        """
        
        # Create the prompt for the current step
        prompt = f"""
        You are a pattern matching expert. Given a BDD step and available Cucumber step definition patterns, find the EXACT matching pattern.

        Rules for matching:
        1. Fixed text parts must match exactly (case-sensitive)
        2. Text between quotes in BDD step (like "Welcome") matches with ([^"]*) in pattern
        3. Pattern must account for entire step text, not partial matches
        4. Pattern should start with ^ and end with $
        5. Return ONLY the matching pattern, no explanation

        Examples:
        {few_shot_examples}

        Available patterns:
        {json.dumps([impl_data["pattern"] for impl_data in self.step_definitions.values()], indent=2)}

        BDD Step to match: {step_text}

        Return the exact matching pattern or null if no match found. Return ONLY the pattern, nothing else:"""
        
        try:
            # Create model instance
            model = GenerativeModel("gemini-1.5-pro")
            
            # Set up generation config for precise output
            generation_config = {
                "temperature": 0.1,  # Very low temperature for deterministic output
                "top_p": 0.1,       # More focused sampling
                "top_k": 1,         # Most likely token only
                "max_output_tokens": 100,
            }
            
            # Generate response
            response = model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings={"HARM_CATEGORY_DANGEROUS_CONTENT": "block_none"}
            )
            
            predicted_pattern = response.text.strip()
            logger.debug(f"LLM predicted pattern: {predicted_pattern}")
            
            # Handle null response
            if predicted_pattern.lower() == "null":
                logger.debug("LLM found no matching pattern")
                return None
            
            # Find the implementation data for the predicted pattern
            for impl_data in self.step_definitions.values():
                if impl_data["pattern"] == predicted_pattern:
                    logger.debug(f"Found match using LLM: {predicted_pattern}")
                    return impl_data
            
            logger.debug(f"No implementation found for predicted pattern: {predicted_pattern}")
            return None
            
        except Exception as e:
            logger.error(f"Error using LLM for pattern matching: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def save_to_json(self) -> None:
        """Save the extracted pairs to a JSON file"""
        logger.info(f"Saving extracted data to {self.output_file}")
        
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(self.bdd_java_pairs, f, indent=2)
            logger.info(f"Saved {len(self.bdd_java_pairs)} BDD-Java pairs to {self.output_file}")
        except Exception as e:
            logger.error(f"Error saving to JSON file: {str(e)}")
            raise
    
    def extract_and_save(self) -> None:
        """Run the full extraction process"""
        logger.info("Starting extraction process")
        
        try:
            self.extract_java_step_definitions()
            self.extract_feature_steps()
            self.save_to_json()
            
            # Print statistics
            total_steps = sum(len(scenario["steps"]) for scenario in self.bdd_java_pairs)
            matched_steps = sum(
                1 for scenario in self.bdd_java_pairs 
                for step in scenario["steps"] 
                if step["java_implementation"]
            )
            
            match_percentage = matched_steps/total_steps*100 if total_steps > 0 else 0
            
            logger.info(f"\nExtraction complete!")
            logger.info(f"Total scenarios: {len(self.bdd_java_pairs)}")
            logger.info(f"Total steps: {total_steps}")
            logger.info(f"Steps with matched Java implementation: {matched_steps} ({match_percentage:.1f}%)")
        except Exception as e:
            logger.error(f"Error during extraction process: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract BDD-Java pairs from feature files and step definitions')
    parser.add_argument('--feature-dir', required=True, help='Directory containing feature files')
    parser.add_argument('--java-dir', required=True, help='Directory containing Java step definition files')
    parser.add_argument('--output', default='bdd_java_pairs.json', help='Output JSON file path')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level')
    
    args = parser.parse_args()
    
    # Set log level based on argument
    logger.setLevel(getattr(logging, args.log_level))
    
    logger.info(f"Starting extraction with arguments: {args}")
    
    extractor = BDDJavaExtractor(args.feature_dir, args.java_dir, args.output)
    extractor.extract_and_save() 