import csv
import re


def parse_feature_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Extract feature name
    feature_match = re.search(r'Feature:\s*(.*)', content)
    feature_name = (
        feature_match.group(1).strip() if feature_match else "Unknown"
    )
    
    # Split content into scenarios
    scenarios = content.split('Scenario')
    
    # Remove the first split which contains the Feature line
    scenarios = scenarios[1:]
    
    rows = []
    for scenario in scenarios:
        # Extract tags
        tags = re.findall(r'@(\w+)', scenario)
        packages = ' ** '.join(tags) if tags else "NA"
        
        # Split scenario into lines and process
        lines = scenario.split('\n')
        # Remove "Scenario:" or "Scenario Outline:" from the name
        scenario_name = lines[0].strip()
        scenario_name = re.sub(r'^Scenario(?:\s+Outline)?:\s*', '', scenario_name)
        # Remove any remaining colon and spaces
        scenario_name = re.sub(r'^:\s*', '', scenario_name)
        # Remove any remaining Outline: prefix
        scenario_name = re.sub(r'^Outline:\s*', '', scenario_name)
        
        # Extract BDD steps (Given/Then lines)
        bdd_steps = []
        for line in lines[1:]:
            line = line.strip()
            if line and (line.startswith('Given') or line.startswith('Then')):
                # Remove any line breaks and extra spaces
                line = ' '.join(line.split())
                bdd_steps.append(line)
        
        # Join BDD steps with separator
        bdd_text = ' ** '.join(bdd_steps)
        
        # Check for Examples section
        examples = "NA"
        if "Examples:" in scenario:
            examples_section = scenario.split("Examples:")[1].strip()
            # Get both header and data rows
            example_lines = examples_section.split('\n')
            if len(example_lines) >= 2:
                header = example_lines[0].strip()
                data = example_lines[1].strip()
                examples = f"{header}**{data}"
        
        rows.append([
            feature_name,
            packages,
            scenario_name,
            bdd_text,
            examples
        ])
    
    return rows


def write_to_csv(rows, output_file):
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Feature', 'Packages', 'Scenario', 'BDD', 'Example'])
        writer.writerows(rows)


def main():
    input_file = 'data/test.feature'
    output_file = 'output.csv'
    
    rows = parse_feature_file(input_file)
    write_to_csv(rows, output_file)
    print(f"CSV file has been generated: {output_file}")


if __name__ == "__main__":
    main()