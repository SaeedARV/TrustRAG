"""
Simple RAG implementation that returns raw retrieval results without LLM processing.
This script avoids the complex dependencies of the full TrustRAG system.
"""

import os
import json
import random
import pickle
from collections import defaultdict
from datetime import datetime

class SimpleRAG:
    def __init__(self, corpus_path=None, output_dir="rag_results"):
        # Create output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Sample corpus with pre-defined documents if no path provided
        self.corpus = {
            "doc1": {
                "text": "The capital of France is Paris. Paris is known as the City of Light and is famous for the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral."
            },
            "doc2": {
                "text": "Machine learning is a branch of artificial intelligence (AI) focused on building systems that learn from data. Common machine learning algorithms include linear regression, decision trees, and neural networks."
            },
            "doc3": {
                "text": "Python is a high-level programming language known for its readability and versatility. It's widely used in data science, web development, and automation."
            },
            "doc4": {
                "text": "Deep learning is a subset of machine learning that uses neural networks with multiple layers. It's particularly effective for tasks like image recognition and natural language processing."
            },
            "doc5": {
                "text": "The Eiffel Tower was completed in 1889 and stands 324 meters tall. It was built for the 1889 World's Fair in Paris and was initially criticized by many Parisians."
            },
            "doc6": {
                "text": "Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language. Key tasks in NLP include language translation, sentiment analysis, and text summarization."
            },
            "doc7": {
                "text": "RAG (Retrieval-Augmented Generation) is a technique that combines retrieval systems with generative AI models. It allows LLMs to access external knowledge not in their training data."
            },
            "doc8": {
                "text": "Transformer models, introduced in 2017, revolutionized NLP. They use attention mechanisms to process sequential data in parallel rather than sequentially, enabling better performance on many language tasks."
            }
        }

        # Load custom corpus if provided
        if corpus_path and os.path.exists(corpus_path):
            try:
                with open(corpus_path, 'r') as f:
                    self.corpus = json.load(f)
                print(f"Loaded corpus from {corpus_path} with {len(self.corpus)} documents")
            except Exception as e:
                print(f"Error loading corpus from {corpus_path}: {e}")
                print("Using default corpus instead")

    def basic_retrieval(self, query, k=3):
        """
        Very simple retrieval using keyword matching.
        In a real system, this would use proper embeddings and vector search.
        """
        scores = defaultdict(float)
        query_terms = query.lower().split()
        
        for doc_id, doc in self.corpus.items():
            doc_text = doc["text"].lower()
            # Simple term frequency scoring
            for term in query_terms:
                if term in doc_text:
                    scores[doc_id] += 1.0
                    # Bonus for terms appearing in sequence
                    if len(term) > 3 and f" {term} " in doc_text:
                        scores[doc_id] += 0.5
        
        # Sort by score
        sorted_docs = sorted([(doc_id, score) for doc_id, score in scores.items()], 
                            key=lambda x: x[1], reverse=True)
        
        # Get top-k documents
        top_k_docs = sorted_docs[:k]
        
        # If we don't have enough documents with matches, add some random ones
        if len(top_k_docs) < k:
            remaining_docs = [doc_id for doc_id in self.corpus.keys() if doc_id not in [d[0] for d in top_k_docs]]
            random_docs = random.sample(remaining_docs, min(k - len(top_k_docs), len(remaining_docs)))
            top_k_docs.extend([(doc_id, 0.1) for doc_id in random_docs])
        
        # Format results
        results = []
        for doc_id, score in top_k_docs:
            results.append({
                "doc_id": doc_id,
                "score": score,
                "content": self.corpus[doc_id]["text"]
            })
        
        return results
    
    def format_results(self, query, results):
        """Format the retrieval results as a RAG response"""
        response = f"Query: {query}\n\nRetrieved Documents:\n"
        
        for i, doc in enumerate(results):
            response += f"Document {i+1} [Score: {doc['score']:.2f}, ID: {doc['doc_id']}]:\n{doc['content']}\n\n"
        
        return response
    
    def save_results(self, query, results, formatted_response, query_id=None):
        """Save the retrieval results to files"""
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Use query ID if provided, otherwise use timestamp
        file_id = query_id if query_id else timestamp
        
        # Create a safe filename from the query
        safe_query = query.replace(" ", "_").replace("?", "").replace("!", "")[:30]
        filename_base = f"{file_id}_{safe_query}"
        
        # Save as JSON
        json_data = {
            "query": query,
            "timestamp": timestamp,
            "results": results
        }
        
        json_path = os.path.join(self.output_dir, f"{filename_base}.json")
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # Save as text
        text_path = os.path.join(self.output_dir, f"{filename_base}.txt")
        with open(text_path, 'w') as f:
            f.write(formatted_response)
        
        # Save as pickle (for programmatic access)
        pickle_path = os.path.join(self.output_dir, f"{filename_base}.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump({
                "query": query,
                "results": results,
                "formatted_response": formatted_response
            }, f)
        
        return {
            "json_path": json_path,
            "text_path": text_path,
            "pickle_path": pickle_path
        }
    
    def process_query(self, query, k=3, save=True, query_id=None):
        """Process a query and optionally save the results"""
        # Retrieve relevant documents
        results = self.basic_retrieval(query, k=k)
        
        # Format the RAG response
        rag_response = self.format_results(query, results)
        
        # Save results if requested
        saved_paths = None
        if save:
            saved_paths = self.save_results(query, results, rag_response, query_id)
        
        return {
            "query": query,
            "results": results,
            "formatted_response": rag_response,
            "saved_paths": saved_paths
        }

def batch_process_queries(queries, corpus_path=None, output_dir="rag_results", k=3):
    """Process a batch of queries and save the results"""
    # Initialize the RAG system
    rag_system = SimpleRAG(corpus_path=corpus_path, output_dir=output_dir)
    
    # Create a summary file
    summary_path = os.path.join(output_dir, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    all_results = []
    
    with open(summary_path, 'w') as summary_file:
        summary_file.write(f"RAG Processing Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        summary_file.write(f"Total queries: {len(queries)}\n\n")
        
        # Process each query
        for i, query in enumerate(queries):
            print(f"\nProcessing query {i+1}/{len(queries)}: {query}")
            
            # Process the query
            result = rag_system.process_query(query, k=k, query_id=f"query_{i+1}")
            all_results.append(result)
            
            # Write to summary file
            summary_file.write(f"Query {i+1}: {query}\n")
            if result['saved_paths']:
                summary_file.write(f"  - JSON: {os.path.basename(result['saved_paths']['json_path'])}\n")
                summary_file.write(f"  - Text: {os.path.basename(result['saved_paths']['text_path'])}\n")
                summary_file.write(f"  - Pickle: {os.path.basename(result['saved_paths']['pickle_path'])}\n")
            summary_file.write("\n")
            
            # Print the formatted response
            print("\n" + "="*80)
            print(f"QUERY {i+1}: {query}")
            print("="*80)
            print(result['formatted_response'])
    
    # Save all results in a single file
    all_results_path = os.path.join(output_dir, f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
    with open(all_results_path, 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"\nAll results saved to {output_dir}/")
    print(f"Summary file: {summary_path}")
    print(f"Combined results: {all_results_path}")
    
    return all_results

def main():
    # Define output directory
    output_dir = "rag_results"
    
    # Sample queries to test
    test_queries = [
        "What is machine learning?",
        "Tell me about Paris and the Eiffel Tower",
        "Explain Retrieval-Augmented Generation",
        "What is Python programming?",
        "How do transformer models work?",
        "What is capital of australia"
    ]
    
    # Process all queries and save results
    results = batch_process_queries(test_queries, output_dir=output_dir, k=3)
    
    # Show how to load the results
    print("\nTo load the saved results in another script:")
    print(f"import pickle")
    print(f"with open('{output_dir}/all_results_*.pkl', 'rb') as f:")
    print(f"    all_results = pickle.load(f)")
    print(f"for result in all_results:")
    print(f"    print(result['query'])")
    print(f"    print(result['formatted_response'])")

if __name__ == "__main__":
    main() 