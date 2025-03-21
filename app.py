import os
import json
import logging
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from bdd_generator_gpt4 import BDDGenerator
from scenario_search import ScenarioSearch

def print_with_timestamp(message: str):
    """Print message with timestamp to terminal"""
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{current_time}] {message}")

# Initial setup messages
print_with_timestamp("\n" + "="*80)
print_with_timestamp("Starting BDD Generator Application")
print_with_timestamp("="*80)

# Create logs directory if it doesn't exist
logs_dir = os.path.abspath('./logs/app')
if not os.path.exists(logs_dir):
    print_with_timestamp(f"Creating logs directory: {logs_dir}")
    os.makedirs(logs_dir, exist_ok=True)

# Create output directory if it doesn't exist
output_dir = os.path.abspath('./output')
if not os.path.exists(output_dir):
    print(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

# Set up logging with both file and console handlers
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(logs_dir, f'{timestamp}.log')

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Create and configure file handler
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Create and configure console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Get the logger and add both handlers
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

print_with_timestamp(f"Log file created at: {log_file}")
print_with_timestamp("="*80 + "\n")

logger.info("Application initialized")

# Initialize FastAPI
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize components
print("Initializing BDD Generator and Scenario Search...")
bdd_generator = BDDGenerator()
scenario_search = ScenarioSearch()
logger.info(f"Using {bdd_generator.model} for BDD generation")
print(f"Using {bdd_generator.model} for BDD generation")
print("Initialization complete")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/llm_info")
async def get_llm_info():
    return {"model": bdd_generator.model, "version": "2023-05"}

@app.post("/search_scenario")
async def search_scenario(request: dict):
    scenario = request.get("scenario", "")
    print(f"\nSearching for scenario: {scenario}")
    logger.info(f"Searching for scenario: {scenario}")
    
    matches = scenario_search.search(scenario)
    logger.info(f"Found {len(matches)} matches")
    print(f"Found {len(matches)} matches")
    
    return matches

@app.post("/generate_bdd")
async def generate_bdd(request: dict):
    criteria = request.get("criteria", "")
    feature_name = request.get("feature_name", "Feature")
    
    print_with_timestamp("\n" + "="*80)
    print_with_timestamp(f"Processing request for feature: {feature_name}")
    print_with_timestamp(f"Acceptance criteria: {criteria[:200]}...")
    print_with_timestamp("="*80)
    
    logger.info(f"Generating BDD for feature: {feature_name}")
    logger.info(f"Acceptance criteria: {criteria}")
    
    # Track statistics
    stats = {
        'total_steps': 0,
        'retrieved_steps': 0,
        'generated_steps': 0,
        'examples_used': []
    }
    
    # Generate feature file and step definitions
    print("Generating feature file and step implementations...")
    feature_content, java_content = bdd_generator.generate_feature_and_steps(
        criteria, 
        feature_name,
        stats=stats
    )
    print("Generation complete!")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'./output/{timestamp}_{feature_name.replace(" ", "_")}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {os.path.abspath(output_dir)}")
    
    # Save files
    feature_file_path = f'{output_dir}/{feature_name.replace(" ", "")}.feature'
    java_file_path = f'{output_dir}/{feature_name.replace(" ", "")}Steps.java'
    
    with open(feature_file_path, 'w') as f:
        f.write(feature_content)
    
    with open(java_file_path, 'w') as f:
        f.write(java_content)
    
    # Log file paths
    logger.info(f"Saved feature file to: {os.path.abspath(feature_file_path)}")
    logger.info(f"Saved Java step definitions to: {os.path.abspath(java_file_path)}")
    print(f"Saved feature file to: {os.path.abspath(feature_file_path)}")
    print(f"Saved Java step definitions to: {os.path.abspath(java_file_path)}")
    
    # Log statistics
    logger.info("\n" + "="*80)
    logger.info("IMPLEMENTATION STATISTICS")
    logger.info("="*80)
    logger.info(f"Total steps processed: {stats['total_steps']}")
    
    if stats['total_steps'] > 0:
        retrieval_pct = (stats['retrieved_steps'] / stats['total_steps']) * 100
        generation_pct = (stats['generated_steps'] / stats['total_steps']) * 100
        
        logger.info(f"Retrieved implementations: {stats['retrieved_steps']} ({retrieval_pct:.1f}%)")
        logger.info(f"Generated implementations: {stats['generated_steps']} ({generation_pct:.1f}%)")
        
        if stats['generated_steps'] > 0 and stats['examples_used']:
            avg_examples = sum(stats['examples_used']) / len(stats['examples_used'])
            logger.info(f"Average examples used per generation: {avg_examples:.1f}")
            logger.info(f"Examples distribution: {stats['examples_used']}")
    
    # Print summary to terminal
    print("\n" + "="*80)
    print(f"GENERATION COMPLETE: {feature_name}")
    print("="*80)
    print(f"Output directory: {os.path.abspath(output_dir)}")
    print(f"Feature file: {os.path.abspath(feature_file_path)}")
    print(f"Java file: {os.path.abspath(java_file_path)}")
    print(f"Log file: {log_file}")
    print("="*80)
    print(f"Total steps processed: {stats['total_steps']}")
    
    if stats['total_steps'] > 0:
        retrieval_pct = (stats['retrieved_steps'] / stats['total_steps']) * 100
        generation_pct = (stats['generated_steps'] / stats['total_steps']) * 100
        
        print(f"Retrieved implementations: {stats['retrieved_steps']} ({retrieval_pct:.1f}%)")
        print(f"Generated implementations: {stats['generated_steps']} ({generation_pct:.1f}%)")
        
        if stats['generated_steps'] > 0 and stats['examples_used']:
            avg_examples = sum(stats['examples_used']) / len(stats['examples_used'])
            print(f"Average examples used per generation: {avg_examples:.1f}")
    
    print("="*80)
    
    # Create summary.json
    summary = {
        "timestamp": timestamp,
        "feature_name": feature_name,
        "log_file": os.path.abspath(log_file),
        "feature_file": os.path.abspath(feature_file_path),
        "java_file": os.path.abspath(java_file_path),
        "stats": {
            "total_steps": stats['total_steps'],
            "retrieved_steps": stats['retrieved_steps'],
            "generated_steps": stats['generated_steps'],
            "retrieval_percentage": (stats['retrieved_steps'] / stats['total_steps'] * 100) if stats['total_steps'] > 0 else 0,
            "generation_percentage": (stats['generated_steps'] / stats['total_steps'] * 100) if stats['total_steps'] > 0 else 0,
            "examples_used": stats['examples_used'],
            "average_examples_per_generation": (sum(stats['examples_used']) / len(stats['examples_used'])) if stats['examples_used'] else 0
        }
    }
    
    summary_path = f'{output_dir}/summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print_with_timestamp(f"Created summary file: {os.path.abspath(summary_path)}")
    
    return {
        "feature_file": feature_content,
        "step_definitions": java_content,
        "stats": summary["stats"],
        "file_paths": {
            "feature_file": os.path.abspath(feature_file_path),
            "java_file": os.path.abspath(java_file_path),
            "summary_file": os.path.abspath(summary_path),
            "log_file": os.path.abspath(log_file)
        }
    }

if __name__ == "__main__":
    print("\nStarting FastAPI server...")
    logger.info("Starting FastAPI server")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False) 