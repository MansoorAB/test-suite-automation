import requests
import json
from pprint import pprint

def test_scenario_search(query: str, n_results: int = 3):
    """Test the scenario search API endpoint."""
    url = "http://localhost:8000/get_scenario"
    
    # Prepare request payload
    payload = {
        "query": query,
        "n_results": n_results
    }
    
    # Make POST request
    response = requests.post(url, json=payload)
    
    # Check if request was successful
    if response.status_code == 200:
        print(f"✅ Request successful (Status code: {response.status_code})")
        
        # Parse and display results
        results = response.json()
        print("\n📊 Search Results:")
        print("=" * 80)
        
        for i, result in enumerate(results["results"]):
            print(f"\n🔍 Result {i+1} (Similarity: {result['similarity']:.2f})")
            print(f"Scenario: {result['scenario']}")
            print(f"Feature: {result['feature']}")
            
            if result['packages']:
                print(f"Packages: {result['packages']}")
            
            if result['bdd_steps']:
                print("\nBDD Steps:")
                for step in result['bdd_steps']:
                    print(f"  - {step}")
            
            if result['example']:
                print("\nExample:")
                print(result['example'])
            
            print("-" * 80)
        
        # Return raw JSON for further processing if needed
        return results
    else:
        print(f"❌ Request failed (Status code: {response.status_code})")
        print(f"Error: {response.text}")
        return None

if __name__ == "__main__":
    # Example usage
    print("🚀 Testing Scenario Search API")
    print("=" * 80)
    
    # Test with a sample query
    query = "How to add documents to a case"
    print(f"Query: '{query}'\n")
    
    results = test_scenario_search(query)
    
    # Uncomment to see raw JSON response
    # print("\nRaw JSON response:")
    # pprint(results) 