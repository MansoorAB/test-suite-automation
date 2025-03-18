import csv
import re
import json
import os
import glob
import logging
from typing import List, Dict, Any


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('feature_to_csv')


def parse_feature_file(file_path: str) -> List[List[Any]]:
    """
    Parse a single feature file and extract scenarios.
    
    Args:
        file_path: Path to the feature file
        
    Returns:
        List of rows containing extracted data
    """
    logger.info(f"Parsing feature file: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        return []
    
    # Extract feature name
    feature_match = re.search(r'Feature:\s*(.*)', content)
    feature_name = (
        feature_match.group(1).strip() if feature_match else "Unknown"
    )
    logger.debug(f"Extracted feature name: {feature_name}")
    
    # Split content into scenarios
    scenarios = content.split('Scenario')
    
    # Remove the first split which contains the Feature line
    scenarios = scenarios[1:]
    logger.info(f"Found {len(scenarios)} scenarios in {file_path}")
    
    rows = []
    for i, scenario in enumerate(scenarios):
        # Extract tags
        tags = re.findall(r'@(\w+)', scenario)
        packages = tags if tags else []
        
        # Split scenario into lines and process
        lines = scenario.split('\n')
        # Remove "Scenario:" or "Scenario Outline:" from the name
        scenario_name = lines[0].strip()
        scenario_name = re.sub(r'^Scenario(?:\s+Outline)?:\s*', '', scenario_name)
        # Remove any remaining colon and spaces
        scenario_name = re.sub(r'^:\s*', '', scenario_name)
        # Remove any remaining Outline: prefix
        scenario_name = re.sub(r'^Outline:\s*', '', scenario_name)
        
        logger.debug(f"Processing scenario {i+1}: {scenario_name}")
        
        # Extract BDD steps (Given/Then/When/And lines)
        bdd_steps = []
        for line in lines[1:]:
            line = line.strip()
            if line and any(line.startswith(keyword) for keyword in ['Given', 'Then', 'When', 'And', 'But']):
                # Remove any line breaks and extra spaces
                line = ' '.join(line.split())
                bdd_steps.append(line)
        
        logger.debug(f"Extracted {len(bdd_steps)} BDD steps")
        
        # Check for Examples section
        examples = []
        if "Examples:" in scenario:
            examples_section = scenario.split("Examples:")[1].strip()
            # Get both header and data rows
            example_lines = examples_section.split('\n')
            if len(example_lines) >= 2:
                header = example_lines[0].strip()
                data = example_lines[1].strip()
                examples = [header, data]
                logger.debug("Extracted examples section")
        
        rows.append([
            feature_name,
            json.dumps(packages),  # Store as JSON string
            scenario_name,
            json.dumps(bdd_steps),  # Store as JSON string
            json.dumps(examples)   # Store as JSON string
        ])
    
    return rows


def process_feature_files(directory: str) -> List[List[Any]]:
    """
    Process all feature files in the given directory.
    
    Args:
        directory: Directory containing feature files
        
    Returns:
        List of rows containing extracted data from all files
    """
    logger.info(f"Searching for feature files in {directory}")
    
    # Find all .feature files in the directory
    feature_files = glob.glob(os.path.join(directory, "**", "*.feature"), recursive=True)
    
    if not feature_files:
        logger.warning(f"No feature files found in {directory}")
        return []
    
    logger.info(f"Found {len(feature_files)} feature files")
    
    # Process each feature file
    all_rows = []
    for file_path in feature_files:
        rows = parse_feature_file(file_path)
        all_rows.extend(rows)
    
    logger.info(f"Extracted {len(all_rows)} scenarios in total")
    return all_rows


def write_to_csv(rows: List[List[Any]], output_file: str) -> None:
    """
    Write extracted data to CSV file.
    
    Args:
        rows: List of rows to write
        output_file: Path to output CSV file
    """
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Feature', 'Packages', 'Scenario', 'BDD', 'Example'])
            writer.writerows(rows)
        logger.info(f"Successfully wrote {len(rows)} rows to {output_file}")
    except Exception as e:
        logger.error(f"Error writing to CSV file {output_file}: {str(e)}")


def main():
    """Main function to process feature files and generate CSV."""
    logger.info("Starting feature file processing")
    
    # Directory containing feature files
    input_dir = 'data'
    output_file = 'output.csv'
    
    # Ensure the input directory exists
    if not os.path.exists(input_dir):
        logger.error(f"Input directory {input_dir} does not exist")
        logger.info("Creating data directory")
        os.makedirs(input_dir)
        logger.warning("No feature files to process. Please add .feature files to the data directory")
        return
    
    # Process all feature files
    rows = process_feature_files(input_dir)
    
    if not rows:
        logger.warning("No data extracted from feature files")
        return
    
    # Write to CSV
    write_to_csv(rows, output_file)
    
    logger.info("Feature file processing completed")


if __name__ == "__main__":
    main()