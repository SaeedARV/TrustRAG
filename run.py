import os
import re

def run(test_params):
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
        print("Created logs directory")
        
    log_name = get_log_name(test_params)
    if os.path.exists(f'logs/{log_name}.log'):
        print(f"log {log_name}.log already exists")
        # load the log file
        with open(f'logs/{log_name}.log', 'r') as f:
            lines = f.readlines()
        # search if "Incorrect Answer Percentage:" in the log file with re
        if len(lines) > 0:
            incorrect_answer_percentage = re.search(r'Incorrect Answer Percentage:', lines[-1])
            if not incorrect_answer_percentage:
                print(f"log {log_name}.log is not complete, remove it")
                os.remove(f'logs/{log_name}.log')
            else:
                print(f"log {log_name}.log is complete")
                return
        else:
            print(f"log {log_name}.log is not complete, remove it")
            os.remove(f'logs/{log_name}.log')
    
    cmd = f"python main_trustrag.py"
    for key, value in test_params.items():
        if value is None:
            continue
        elif type(value) is str:
            cmd += f" --{key}=\"{value}\""
        else:
            cmd += f" --{key}={value}"

    cmd += f" --log_name={log_name}"
    print(cmd)
    os.system(cmd)
    return

def get_log_name(test_params):
    log_name = f"{test_params['eval_model_code']}_{test_params['eval_dataset']}_{test_params['attack_method']}_{test_params['defend_method']}_{test_params['removal_method']}_{test_params['adv_per_query']}_{test_params['model_name']}"
    if test_params['note'] is not None:
        log_name += f"_{test_params['note']}"
    return log_name


test_params = {
    'eval_model_code': "contriever",
    'eval_dataset': "nq", # ['nq','hotpotqa', 'msmarco']
    'split': "test",
    'query_results_dir': 'main',
    'model_name': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', # Using a smaller open-source model
    'top_k': 5,
    'gpu_id': 0,
    'attack_method': 'LM_targeted', # ['none', 'LM_targeted', 'hotflip', 'pia']
    'defend_method': 'conflict', # ['none', 'conflict', 'astute', 'instruct']
    'removal_method': 'none', # ['kmeans', 'kmeans_ngram', 'none']
    'adv_per_query': 3, # poison rate = adv_per_query / top_k
    'score_function': 'dot',
    'repeat_times': 10,
    'M': 10, # number of queries
    'seed': 12,
    'note': None
}


# Test with different datasets and open-source models
open_models = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "facebook/opt-350m",
    "facebook/opt-1.3b"
]

for dataset in ['nq', 'hotpotqa', 'msmarco']:
    for model in open_models:
        test_params['eval_dataset'] = dataset
        test_params['model_name'] = model
        run(test_params)

# Test with different defense methods
defense_methods = ['none', 'conflict', 'astute', 'instruct']
for method in defense_methods:
    test_params['defend_method'] = method
    test_params['model_name'] = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Use the smallest model for quick testing
    test_params['eval_dataset'] = 'nq'  # Use a single dataset for comparison
    run(test_params)

# Test with RL-based defense method
test_params['defend_method'] = 'conflict'  # Use conflict method with RL filtering
test_params['removal_method'] = 'rl'
test_params['model_name'] = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
test_params['eval_dataset'] = 'nq'
test_params['note'] = 'rl_defense'
run(test_params)

# Test RL-based method with different datasets
for dataset in ['nq', 'hotpotqa', 'msmarco']:
    test_params['eval_dataset'] = dataset
    test_params['removal_method'] = 'rl'
    test_params['note'] = f'rl_defense_{dataset}'
    run(test_params)