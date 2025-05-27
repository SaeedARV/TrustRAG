"""
Example script showing how to use SimpleRAG with custom documents
"""

import os
import json
import sys
from simple_rag import SimpleRAG, batch_process_queries

def create_sample_corpus(output_path="sample_corpus.json"):
    """Create a sample corpus file to demonstrate the format"""
    corpus = {
        "custom_doc1": {
            "text": "TrustRAG is a system for making Retrieval-Augmented Generation more robust against adversarial attacks. It uses various defense mechanisms to filter out potentially harmful content."
        },
        "custom_doc2": {
            "text": "Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize rewards. It's used in robotics, game playing, and recommendation systems."
        },
        "custom_doc3": {
            "text": "K-means clustering is an unsupervised learning algorithm that groups similar data points together. In TrustRAG, it's used to identify and filter out adversarial content by finding outliers in the embedding space."
        },
        "custom_doc4": {
            "text": "Large Language Models (LLMs) like GPT-4 and Claude are trained on vast amounts of text data. They can generate human-like text but may hallucinate information not present in their training data."
        },
        "custom_doc5": {
            "text": "Prompt injection attacks try to manipulate an AI system by including malicious instructions in the input. These attacks can potentially make the AI ignore its safety constraints or produce harmful outputs."
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(corpus, f, indent=2)
    
    print(f"Sample corpus created at: {output_path}")
    return output_path

def main():
    # Create a custom output directory
    output_dir = "custom_rag_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we should create a sample corpus
    corpus_path = None
    if len(sys.argv) > 1:
        corpus_path = sys.argv[1]
        if not os.path.exists(corpus_path):
            print(f"Error: Corpus file {corpus_path} not found")
            sys.exit(1)
    else:
        # Create a sample corpus file
        corpus_path = create_sample_corpus()
    
    # Define some queries relevant to the custom corpus
    custom_queries = [
        "What is TrustRAG?",
        "Explain reinforcement learning",
        "How does k-means clustering work in TrustRAG?",
        "What are the limitations of large language models?",
        "What is a prompt injection attack?"
    ]
    
    # Process queries
    print(f"\nProcessing {len(custom_queries)} queries using corpus from: {corpus_path}")
    
    # Initialize RAG system with custom corpus
    rag = SimpleRAG(corpus_path=corpus_path, output_dir=output_dir)
    
    # Process each query individually to provide more detailed feedback
    for i, query in enumerate(custom_queries):
        print(f"\nProcessing query {i+1}: {query}")
        result = rag.process_query(query, k=3, query_id=f"custom_query_{i+1}")
        
        print(f"Results saved to:")
        print(f"  - JSON: {result['saved_paths']['json_path']}")
        print(f"  - Text: {result['saved_paths']['text_path']}")
        print(f"  - Pickle: {result['saved_paths']['pickle_path']}")
        
        # Display the formatted response
        print("\n" + "="*80)
        print(f"QUERY: {query}")
        print("="*80)
        print(result['formatted_response'])
    
    print(f"\nAll results have been saved to the {output_dir}/ directory")
    print("You can inspect the individual files or use the following to load the results in a Python script:")
    print(f"""
import pickle
import glob
import os

# Get the most recent results file
results_files = glob.glob('{output_dir}/*.pkl')
latest_file = max(results_files, key=os.path.getctime)

# Load the results
with open(latest_file, 'rb') as f:
    result = pickle.load(f)

# Print the query and response
print(result['query'])
print(result['formatted_response'])
""")

if __name__ == "__main__":
    main() 