# TrustRAG - Raw RAG Implementation

This folder contains a simple implementation of Retrieval-Augmented Generation (RAG) that returns raw retrieval results without LLM processing. This can be used to analyze and debug the retrieval components of your RAG system independently from the LLM response generation.

## Overview

The implementation provides:

1. A simple RAG system that retrieves documents based on query relevance
2. Saving of results in multiple formats (JSON, text, and pickle)
3. Tools for loading and analyzing saved results
4. Support for custom document collections

## Files

- `simple_rag.py`: Core RAG implementation with built-in corpus
- `custom_rag.py`: Example of using RAG with custom documents
- `load_results.py`: Tool for loading and viewing saved results

## Usage

### Basic RAG with Default Corpus

```bash
python simple_rag.py
```

This will run the RAG system with 5 sample queries against the default document corpus and save the results in the `rag_results/` directory.

### Custom RAG with Your Own Documents

```bash
python custom_rag.py [path_to_your_corpus.json]
```

If you don't provide a corpus file, a sample corpus will be created for demonstration.

### Loading and Viewing Results

```bash
python load_results.py [result_number]
```

This will display the available results and show details for the selected result (or the first result if none is specified).

## Document Format

The document corpus should be a JSON file with the following structure:

```json
{
  "doc_id1": {
    "text": "Document content goes here..."
  },
  "doc_id2": {
    "text": "Another document's content..."
  },
  ...
}
```

## Output Formats

Results are saved in three formats:

1. **Text (.txt)**: Human-readable formatted results
2. **JSON (.json)**: Structured data for programmatic access
3. **Pickle (.pkl)**: Python serialized objects for easy loading

## Integration with TrustRAG

This implementation is designed to work alongside the main TrustRAG system. To use it with TrustRAG:

1. Run your queries through this RAG system to get raw retrieval results
2. Analyze the results to understand retrieval behavior
3. Compare with TrustRAG's full pipeline results to identify discrepancies

You can also use the raw RAG option in TrustRAG itself:

```bash
python main_trustrag.py --defend_method=raw_rag --removal_method=none --eval_dataset=nq --model_name=TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

## Customizing the Retrieval Logic

The default implementation uses a simple keyword matching algorithm. To implement a more advanced retrieval system:

1. Modify the `basic_retrieval` method in the `SimpleRAG` class
2. Consider adding embedding-based retrieval using libraries like sentence-transformers
3. Implement filtering methods similar to TrustRAG's adversarial content detection

## Using the Results Programmatically

```python
import pickle

# Load results from a single query
with open('rag_results/query_1_What_is_machine_learning.pkl', 'rb') as f:
    result = pickle.load(f)
    
print(result['query'])
print(result['formatted_response'])

# Load all results
with open('rag_results/all_results_*.pkl', 'rb') as f:
    all_results = pickle.load(f)
    
for idx, result in enumerate(all_results):
    print(f"Query {idx+1}: {result['query']}")
```

## Benefits of Raw RAG

1. **Transparency**: See exactly what documents are retrieved without LLM post-processing
2. **Debugging**: Identify retrieval issues separately from generation issues
3. **Control**: Modify and filter retrieved documents before sending to an LLM
4. **Analysis**: Evaluate retrieval quality independently from response quality

## License

This code is part of the TrustRAG project and follows its licensing terms. 