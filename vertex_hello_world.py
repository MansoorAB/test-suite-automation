import os
import vertexai
from vertexai.generative_models import GenerativeModel

def init_vertex_ai():
    """Initialize Vertex AI with project details."""
    try:
        vertexai.init(
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location="us-central1"
        )
        print("Successfully initialized Vertex AI")
    except Exception as e:
        print(f"Error initializing Vertex AI: {str(e)}")
        raise

def generate_text(prompt: str) -> str:
    """Generate text using Vertex AI Gemini Pro."""
    try:
        # Create model instance
        model = GenerativeModel("gemini-1.5-pro")
        
        # Configure generation parameters
        generation_config = {
            "temperature": 0.2,    # Lower temperature for more focused output
            "top_p": 0.8,         # Nucleus sampling
            "top_k": 40,          # Top-k sampling
            "max_output_tokens": 1024,
        }
        
        # Generate content
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings={"HARM_CATEGORY_DANGEROUS_CONTENT": "block_none"}
        )
        
        return response.text
    except Exception as e:
        print(f"Error generating text: {str(e)}")
        raise

def main():
    # Initialize Vertex AI
    init_vertex_ai()
    
    # Test prompts
    prompts = [
        "Write a hello world message",
        "What is 2+2? Answer in one word.",
        "Write a one-line Python print statement"
    ]
    
    # Generate and print responses
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        response = generate_text(prompt)
        print(f"Response: {response}")

if __name__ == "__main__":
    main() 