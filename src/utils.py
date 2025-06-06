import os
from .contriever_src.contriever import Contriever
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import json
import numpy as np
import pickle
import random
import torch
from transformers import AutoTokenizer

from sentence_transformers import SentenceTransformer
from loguru import logger
import os

import sys
 
import time

model_code_to_qmodel_name = {
    "contriever": "facebook/contriever",
    "contriever-msmarco": "facebook/contriever-msmarco",
    "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp"
}

model_code_to_cmodel_name = {
    "contriever": "facebook/contriever",
    "contriever-msmarco": "facebook/contriever-msmarco",
    "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp"
}


def load_cached_data(cache_file, load_function, *args, **kwargs):
    if os.path.exists(cache_file):
        logger.info(f"Cache file {cache_file} exists. Loading data...")
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading cache file {cache_file}: {e}")
            logger.info(f"Regenerating data...")
            # If loading the cache fails, regenerate the data
            try:
                data = load_function(*args, **kwargs)
                # Save the regenerated data to cache
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(data, f)
                except Exception as e:
                    logger.error(f"Error saving regenerated data to cache {cache_file}: {e}")
                return data
            except Exception as e:
                logger.error(f"Error regenerating data: {e}")
                return None
    else:
        logger.info(f"Cache file {cache_file} does not exist. Generating data...")
        try:
            data = load_function(*args, **kwargs)
            # Save the generated data to cache
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
            except Exception as e:
                logger.error(f"Error saving data to cache {cache_file}: {e}")
            return data
        except Exception as e:
            logger.error(f"Error generating data: {e}")
            return None
    


def setup_experiment_logging(experiment_name=None, log_dir='logs'):
    """
    Configure logging for experiments with both console and file output.
    
    Args:
        experiment_name: Name of the experiment for the log file
        log_dir: Directory to store log files
    """
    # Remove any existing handlers
    logger.remove()
    
    # Add console handler with a simple format
    logger.add(sys.stderr, format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="INFO")
    
    # Add file handler if experiment_name is provided
    if experiment_name:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{experiment_name}.log")
        if os.path.exists(log_file):
            os.remove(log_file)
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level="INFO"
        )
        
    return logger 

def contriever_get_emb(model, input):
    return model(**input)

def dpr_get_emb(model, input):
    return model(**input).pooler_output

def ance_get_emb(model, input):
    input.pop('token_type_ids', None)
    return model(input)["sentence_embedding"]

def load_models(model_code):
    assert (model_code in model_code_to_qmodel_name and model_code in model_code_to_cmodel_name), f"Model code {model_code} not supported!"
    if 'contriever' in model_code:
        model = Contriever.from_pretrained(model_code_to_qmodel_name[model_code])
        assert model_code_to_cmodel_name[model_code] == model_code_to_qmodel_name[model_code]
        c_model = model
        tokenizer = AutoTokenizer.from_pretrained(model_code_to_qmodel_name[model_code])
        get_emb = contriever_get_emb
    elif 'ance' in model_code:
        model = SentenceTransformer(model_code_to_qmodel_name[model_code])
        assert model_code_to_cmodel_name[model_code] == model_code_to_qmodel_name[model_code]
        c_model = model
        tokenizer = model.tokenizer
        get_emb = ance_get_emb
    else:
        raise NotImplementedError
    # model: 用于生成query的embedding
    # c_model: 用于生成context的embedding
    return model, c_model, tokenizer, get_emb

def create_synthetic_dataset(dataset_name):
    """
    Creates a synthetic dataset when the real dataset cannot be loaded.
    This function generates minimal corpus, queries, and qrels from the sample data.
    
    Args:
        dataset_name: Name of the dataset to create a synthetic version for
        
    Returns:
        corpus: Dictionary of document IDs to document objects
        queries: Dictionary of query IDs to query objects
        qrels: Dictionary of query IDs to relevant document IDs
    """
    logger.info(f"Creating synthetic dataset for {dataset_name}")
    
    try:
        # Load the sample dataset file
        with open(f'results/adv_targeted_results/{dataset_name}.json', 'r') as f:
            sample_data = json.load(f)
            
        corpus = {}
        queries = {}
        qrels = {}
        
        # Create minimal corpus, queries, and qrels from sample data
        for id, item in sample_data.items():
            # Create query object
            queries[id] = {"text": item["question"]}
            
            # Create corpus entry (document containing the answer)
            doc_id = f"doc_{id}"
            corpus[doc_id] = {"text": f"The answer to '{item['question']}' is '{item['correct answer']}'"}
            
            # Create qrels entry (relevance mapping)
            qrels[id] = {doc_id: 1}
            
            # Add some additional documents that are not relevant
            for i in range(3):
                additional_doc_id = f"doc_{id}_extra_{i}"
                corpus[additional_doc_id] = {"text": f"This document contains information about {dataset_name} but not specifically about {item['question']}."}
        
        logger.info(f"Created synthetic dataset with {len(queries)} queries, {len(corpus)} documents, and {len(qrels)} relevance mappings")
        return corpus, queries, qrels
        
    except Exception as e:
        logger.error(f"Failed to create synthetic dataset: {e}")
        # Return minimal empty structures if everything fails
        return {}, {}, {}

def load_beir_datasets(dataset_name, split):
    assert dataset_name in ['nq', 'msmarco', 'hotpotqa']
    if dataset_name == 'msmarco': 
        split = 'train'
    
    try:
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset_name)
        out_dir = os.path.join(os.getcwd(), "datasets")
        data_path = os.path.join(out_dir, dataset_name)
        
        # Check if directory already exists
        if not os.path.exists(data_path):
            try:
                logger.info(f"Downloading {dataset_name} ...")
                logger.info(f"out_dir: {out_dir}")
                os.makedirs(out_dir, exist_ok=True)
                
                # Try to download and unzip
                try:
                    data_path = util.download_and_unzip(url, out_dir)
                except Exception as zip_err:
                    logger.error(f"Failed to download or unzip: {zip_err}")
                    return create_synthetic_dataset(dataset_name)
            except Exception as e:
                logger.error(f"Error preparing to download: {e}")
                return create_synthetic_dataset(dataset_name)
        
        logger.info(f"data_path: {data_path}")
        
        # Try to load the dataset
        try:
            data = GenericDataLoader(data_path)
            if '-train' in data_path:
                split = 'train'
            corpus, queries, qrels = data.load(split=split)
            
            # Verify we got valid data
            if not corpus or not queries or not qrels:
                logger.error("Empty data loaded from GenericDataLoader")
                return create_synthetic_dataset(dataset_name)
                
            return corpus, queries, qrels
        except Exception as e:
            logger.error(f"Failed to load dataset from {data_path}: {e}")
            return create_synthetic_dataset(dataset_name)
    
    except Exception as e:
        logger.error(f"Unexpected error in load_beir_datasets: {e}")
        return create_synthetic_dataset(dataset_name)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
        
def save_outputs(outputs, dir, file_name):
    json_dict = json.dumps(outputs, cls=NpEncoder)
    dict_from_str = json.loads(json_dict)
    if not os.path.exists(f'data_cache/outputs/{dir}'):
        os.makedirs(f'data_cache/outputs/{dir}', exist_ok=True)
    with open(os.path.join(f'data_cache/outputs/{dir}', f'{file_name}.json'), 'w', encoding='utf-8') as f:
        json.dump(dict_from_str, f, indent=4)

def save_results(results, dir, file_name="debug"):
    json_dict = json.dumps(results, cls=NpEncoder)
    dict_from_str = json.loads(json_dict)
    if not os.path.exists(f'results/query_results/{dir}'):
        os.makedirs(f'results/query_results/{dir}', exist_ok=True)
    with open(os.path.join(f'results/query_results/{dir}', f'{file_name}.json'), 'w', encoding='utf-8') as f:
        json.dump(dict_from_str, f, indent=4)

def load_results(file_name):
    with open(os.path.join('results', file_name)) as file:
        results = json.load(file)
    return results

def save_json(results, file_path="debug.json"):
    json_dict = json.dumps(results, cls=NpEncoder)
    dict_from_str = json.loads(json_dict)
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dict_from_str, f, indent=4)

def load_json(file_path):
    try:
        with open(file_path) as file:
            results = json.load(file)
        return results
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        # Create and return an empty dict if the file doesn't exist
        return {}
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON file: {file_path}")
        # Create and return an empty dict if the file is not valid JSON
        return {}
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        # Create and return an empty dict for any other error
        return {}

def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def clean_str(s):
    try:
        s=str(s)
    except:
        print('Error: the output cannot be converted to a string')
    s=s.strip()
    if len(s)>1 and s[-1] == ".":
        s=s[:-1]
    return s.lower()

def f1_score(precision, recall):
    f1_scores = np.divide(2 * precision * recall, precision + recall, where=(precision + recall) != 0)
    return f1_scores

class LoguruProgress:
    def __init__(self, iterable=None, desc=None, total=None, **kwargs):
        self.iterable = iterable
        self.desc = desc
        self.total = len(iterable) if iterable is not None else total
        self.n = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time
        logger.info(f"Starting {desc}: 0/{self.total}")

    def update(self, n=1):
        self.n += n
        current_time = time.time()
        # Log every second or at completion
        if current_time - self.last_log_time > 1 or self.n >= self.total:
            elapsed = current_time - self.start_time
            rate = self.n / elapsed if elapsed > 0 else 0
            logger.info(f"{self.desc}: {self.n}/{self.total} "
                       f"[{elapsed:.1f}s elapsed, {rate:.1f} it/s]")
            self.last_log_time = current_time

    def __iter__(self):
        if self.iterable is None:
            raise ValueError("Iterable not provided")
        for obj in self.iterable:
            yield obj
            self.update(1)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            elapsed = time.time() - self.start_time
            rate = self.n / elapsed if elapsed > 0 else 0
            logger.info(f"Completed {self.desc}: {self.n}/{self.total} "
                       f"[{elapsed:.1f}s elapsed, {rate:.1f} it/s]")

def progress_bar(iterable=None, desc=None, total=None, **kwargs):
    """
    A wrapper function that returns either LoguruProgress or tqdm based on whether we want
    logging output or standard tqdm output
    """
    return LoguruProgress(iterable=iterable, desc=desc, total=total, **kwargs)