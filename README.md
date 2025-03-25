BDD Generator
============

A tool that generates Behavior-Driven Development (BDD) feature files and step definitions using AI. Supports both GPT-4 and Vertex AI models for generation.

Features
--------
- Generates Gherkin feature files from acceptance criteria
- Creates corresponding Java step definitions
- Supports RAG (Retrieval-Augmented Generation) for step implementations
- Provides web interface for easy interaction
- Maintains logs and generates summary statistics
- Supports both GPT-4 and Vertex AI models

Prerequisites
------------
- Python 3.8+
- FastAPI
- OpenAI API key (for GPT-4)
- Google Cloud credentials (for Vertex AI)

Installation
-----------
1. Clone the repository:
   ```
   git clone [repository-url]
   cd bdd-generator
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables in .env file:
   ```
   OPENAI_API_KEY="your-openai-api-key"
   MODEL_TYPE="gpt-4"  # or "vertex" for Vertex AI
   ```

Usage
-----
1. Start the server:
   ```
   python app.py
   ```

2. Access the web interface:
   Open http://localhost:8000 in your browser

3. Enter your acceptance criteria and feature name in the web interface

4. Generated files will be available in:
   - Feature files: ./output/features/
   - Step definitions: ./output/step_definitions/
   - Logs: ./logs/app/
   - Summary: ./output/summary.json

Directory Structure
-----------------
```
/
├── app.py                          # Main FastAPI application
├── bdd_generator.py                # Vertex AI implementation
├── bdd_generator_gpt4.py          # GPT-4 implementation
├── rag_bdd_implementation.py       # RAG implementation for Vertex
├── rag_bdd_implementation_gpt4.py  # RAG implementation for GPT-4
├── scenario_search.py             # Search functionality
├── static/                        # Static web assets
├── templates/                     # HTML templates
├── output/                        # Generated files
│   ├── features/                 # Generated feature files
│   ├── step_definitions/         # Generated Java files
│   └── summary.json             # Generation statistics
└── logs/                         # Application logs
    └── app/                      # Log files
```

Module Info
----------
1. app.py
   - Main FastAPI application that handles HTTP requests and serves the web interface
   - Manages model selection between GPT-4 and Vertex AI
   - Handles file I/O operations and logging setup
   - Coordinates the generation process and returns results

2. bdd_generator.py / bdd_generator_gpt4.py
   - Core modules for BDD generation using Vertex AI and GPT-4 respectively
   - Convert acceptance criteria into Gherkin feature files
   - Manage step definition generation process
   - Track statistics for generated and retrieved steps

3. rag_bdd_implementation.py / rag_bdd_implementation_gpt4.py
   - Implement Retrieval-Augmented Generation (RAG) for step definitions
   - Maintain and query vector database of existing step implementations
   - Calculate similarity scores for step matching
   - Generate new step implementations using AI models with examples

4. scenario_search.py
   - Provides functionality to search through existing scenarios
   - Implements semantic search capabilities
   - Helps in finding similar existing implementations
   - Supports the RAG process by finding relevant examples

Configuration
------------
Model Selection:
- GPT-4: Set MODEL_TYPE="gpt-4" in .env
- Vertex AI: Set MODEL_TYPE="vertex" in .env

Logging:
- Log files are created with timestamps in ./logs/app/
- Console output includes timestamps for all operations
- Summary statistics are saved in ./output/summary.json

Output Files
-----------
1. Feature Files (.feature):
   - Located in ./output/features/
   - Contains Gherkin syntax scenarios

2. Step Definitions (.java):
   - Located in ./output/step_definitions/
   - Contains Java implementation of steps

3. Summary File (summary.json):
   - Contains statistics about generation
   - Includes file paths and step statistics
   - Shows retrieval vs. generation percentages

Troubleshooting
--------------
1. If logs directory is not created:
   - Ensure write permissions in the application directory
   - Check if the path is absolute in app.py

2. If model doesn't switch:
   - Verify .env file has correct MODEL_TYPE
   - Ensure environment variables are loaded

3. If generation fails:
   - Check API keys in .env
   - Verify network connectivity
   - Check log files for detailed error messages

Contributing
-----------
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
