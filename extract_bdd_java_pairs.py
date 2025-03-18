import os
import re
import json
import glob
from pathlib import Path
from collections import defaultdict

class BDDJavaExtractor:
    def __init__(self, feature_dir, java_dir, output_file="bdd_java_pairs.json"):
        self.feature_dir = feature_dir
        self.java_dir = java_dir
        self.output_file = output_file
        self.step_definitions = {}
        self.bdd_java_pairs = []
        
    def extract_java_step_definitions(self):
        """Extract all step definitions from Java files"""
        print(f"Scanning Java files in {self.java_dir}...")
        java_files = glob.glob(f"{self.java_dir}/**/*.java", recursive=True)
        
        # Regex patterns for step annotations
        step_pattern = re.compile(r'@(?:Given|When|Then|And|But)\s*\(\s*"(.+?)"\s*\)')
        method_pattern = re.compile(r'public\s+void\s+\w+\s*\([^)]*\)\s*\{(.*?)\}', re.DOTALL)
        
        for java_file in java_files:
            print(f"Processing {java_file}")
            with open(java_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                # Find all step definition annotations and their methods
                step_matches = step_pattern.finditer(content)
                for step_match in step_matches:
                    step_text = step_match.group(1)
                    # Find the method implementation that follows the annotation
                    start_pos = step_match.end()
                    method_match = method_pattern.search(content[start_pos:])
                    
                    if method_match:
                        method_body = method_match.group(1).strip()
                        # Convert regex pattern in step to a more readable format
                        readable_step = self._convert_regex_to_readable(step_text)
                        self.step_definitions[readable_step] = {
                            "pattern": step_text,
                            "implementation": method_body,
                            "file": os.path.basename(java_file)
                        }
        
        print(f"Extracted {len(self.step_definitions)} step definitions")
    
    def _convert_regex_to_readable(self, step_text):
        """Convert regex patterns in step text to a more readable format"""
        # Replace common regex patterns with placeholders
        readable = step_text
        # Replace quoted capture groups
        readable = re.sub(r'\"([^\"]*?)\"', '"PARAM"', readable)
        # Replace unquoted capture groups
        readable = re.sub(r'\(\.\*?\)', 'PARAM', readable)
        readable = re.sub(r'\([^)]+\)', 'PARAM', readable)
        return readable
    
    def extract_feature_steps(self):
        """Extract steps from feature files and match with Java implementations"""
        print(f"Scanning feature files in {self.feature_dir}...")
        feature_files = glob.glob(f"{self.feature_dir}/**/*.feature", recursive=True)
        
        step_pattern = re.compile(r'^\s*(?:Given|When|Then|And|But)\s+(.+)$', re.MULTILINE)
        scenario_pattern = re.compile(r'^\s*Scenario:(.+)$', re.MULTILINE)
        
        for feature_file in feature_files:
            print(f"Processing {feature_file}")
            with open(feature_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                # Extract scenarios
                scenarios = scenario_pattern.finditer(content)
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
        
        print(f"Extracted {len(self.bdd_java_pairs)} scenarios with steps")
    
    def _find_matching_implementation(self, step_text):
        """Find the Java implementation that matches this step text"""
        # Try direct match first
        for pattern, impl_data in self.step_definitions.items():
            # Convert regex pattern to Python regex and try to match
            java_pattern = impl_data["pattern"]
            
            # Replace Cucumber expression placeholders with regex
            regex_pattern = java_pattern
            regex_pattern = regex_pattern.replace("\\", "\\\\")  # Escape backslashes
            regex_pattern = regex_pattern.replace("(", "\\(").replace(")", "\\)")  # Escape parentheses
            regex_pattern = regex_pattern.replace("[", "\\[").replace("]", "\\]")  # Escape brackets
            regex_pattern = regex_pattern.replace(".", "\\.")  # Escape dots
            regex_pattern = regex_pattern.replace("*", "\\*")  # Escape asterisks
            regex_pattern = regex_pattern.replace("+", "\\+")  # Escape plus signs
            regex_pattern = regex_pattern.replace("?", "\\?")  # Escape question marks
            regex_pattern = regex_pattern.replace("{", "\\{").replace("}", "\\}")  # Escape braces
            regex_pattern = regex_pattern.replace("$", "\\$")  # Escape dollar signs
            regex_pattern = regex_pattern.replace("^", "\\^")  # Escape carets
            
            # Replace capture groups with wildcards for matching
            regex_pattern = re.sub(r'\\"\\\([^\\\)]+\\\)\\"', '".*?"', regex_pattern)
            regex_pattern = re.sub(r'\\\([^\\\)]+\\\)', '.*?', regex_pattern)
            
            try:
                if re.match(f"^{regex_pattern}$", step_text):
                    return impl_data
            except re.error:
                # If regex compilation fails, try a simpler approach
                pass
        
        # If no direct match, try a more fuzzy approach
        step_words = set(re.findall(r'\b\w+\b', step_text.lower()))
        best_match = None
        best_score = 0
        
        for pattern, impl_data in self.step_definitions.items():
            pattern_words = set(re.findall(r'\b\w+\b', pattern.lower()))
            intersection = step_words.intersection(pattern_words)
            
            if len(intersection) > best_score:
                best_score = len(intersection)
                best_match = impl_data
        
        # Return the best match if it's reasonably good
        if best_score >= len(step_words) * 0.5:
            return best_match
        
        return None
    
    def save_to_json(self):
        """Save the extracted pairs to a JSON file"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.bdd_java_pairs, f, indent=2)
        print(f"Saved {len(self.bdd_java_pairs)} BDD-Java pairs to {self.output_file}")
    
    def extract_and_save(self):
        """Run the full extraction process"""
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
        
        print(f"\nExtraction complete!")
        print(f"Total scenarios: {len(self.bdd_java_pairs)}")
        print(f"Total steps: {total_steps}")
        print(f"Steps with matched Java implementation: {matched_steps} ({matched_steps/total_steps*100:.1f}%)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract BDD-Java pairs from feature files and step definitions')
    parser.add_argument('--feature-dir', required=True, help='Directory containing feature files')
    parser.add_argument('--java-dir', required=True, help='Directory containing Java step definition files')
    parser.add_argument('--output', default='bdd_java_pairs.json', help='Output JSON file path')
    
    args = parser.parse_args()
    
    extractor = BDDJavaExtractor(args.feature_dir, args.java_dir, args.output)
    extractor.extract_and_save() 