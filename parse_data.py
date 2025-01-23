import json
from tqdm import tqdm

def load_jsonl(file_path):
    with open(file_path, "r") as f:
        data = [json.loads(l) for l in f]
    return data

def parse_data(data):
    experiments = []
    for row in tqdm(data):
        text = row['text']
        # Find the last '\n' before '<<' in text
        description_end = text.rfind('\n', 0, text.find('<<'))
        description = text[:description_end]
        experiment = {'description': description, 'experiment': row['experiment'], 'participant': row['participant'], 'trials': []}
        rest = text[description_end + 1:]
        # print(f"Description: {description}")
        # Split prompt by '\n' and for each row, everything before << will be stimulus, between << and >> will be response, and after >> will be result
        trials = rest.split('\n')
        for trial in trials:
            stimulus_end = trial.find('<<')
            stimulus = trial[:stimulus_end] + '<<'
            response_end = trial.find('>>')
            response = trial[stimulus_end + 2:response_end]
            result = '>>' + trial[response_end + 2:]
            experiment['trials'].append({'stimulus': stimulus, 'response': response, 'result': result})
        experiments.append(experiment)
    with open('psych101_parsed.jsonl', 'w') as f:
        f.write(json.dumps(experiments))




jsonl = load_jsonl('./data/psych101.jsonl')
parse_data(jsonl)