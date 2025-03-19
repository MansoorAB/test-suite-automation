import os
import pandas as pd
import chromadb
import shutil
import json
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Union, Optional
import uvicorn
from scenario_search import ScenarioSearch
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('scenario_api')

# Configuration for LLM choice
USE_GPT4 = True  # Set to False to use Gemini

# Import appropriate BDD generator based on configuration
if USE_GPT4:
    from bdd_generator_gpt4 import BDDGenerator
    logger.info("Using GPT-4 for BDD generation")
else:
    from bdd_generator import BDDGenerator  # Gemini version
    logger.info("Using Gemini for BDD generation")

app = FastAPI(title="BDD Assistant", description="API for scenario search and BDD generation")

# Initialize components
scenario_searcher = ScenarioSearch()
bdd_generator = BDDGenerator()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class ScenarioRequest(BaseModel):
    scenario: str

class BDDRequest(BaseModel):
    criteria: str
    feature_name: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Pass LLM choice to template
    return templates.TemplateResponse("index.html", {
        "request": request,
        "using_gpt4": USE_GPT4
    })

@app.get("/llm_info")
async def get_llm_info():
    """Endpoint to get current LLM configuration"""
    return {
        "model": "GPT-4" if USE_GPT4 else "Gemini",
        "version": "gpt-4o-mini" if USE_GPT4 else "gemini-1.5-pro-flash"
    }

@app.post("/search_scenario")
async def search_scenario(request: ScenarioRequest):
    matches = scenario_searcher.find_similar_scenarios(request.scenario, n_results=3)
    return matches

@app.post("/generate_bdd")
async def generate_bdd(request: BDDRequest):
    try:
        feature_content, java_content = bdd_generator.generate_feature_and_steps(
            request.criteria,
            request.feature_name
        )
        return {
            'feature_file': feature_content,
            'step_definitions': java_content,
            'model_used': "GPT-4" if USE_GPT4 else "Gemini"
        }
    except Exception as e:
        logger.error(f"Error generating BDD: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    model_name = "GPT-4" if USE_GPT4 else "Gemini"
    logger.info(f"Starting FastAPI server using {model_name}")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 