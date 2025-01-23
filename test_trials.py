import json
from mlx_lm import load, generate


def dummy_csv():
    experiment = {
        'description': 'You will be shown several examples of geometric objects.\nYour task is to learn a rule that allows you to tell whether an object belongs to the E or K category.\nFor each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.\nYou will encounter four different problems with different rules.\n\nYou encounter a new problem with a new rule determining which objects belong to each category:\n',
        'trials': [
            {
                'stimulus': 'You see a big black square. You press <<',
                'response': 'K',
                'result': '>>. The correct category is K.'
            },
            {
                'stimulus': 'You see a small black triangle. You press <<',
                'response': 'K',
                'result': '>>. The correct category is E'
            },
            {
                'stimulus': 'You see a big white triangle. You press <<',
                'response': 'E',
                'result': '>>. The correct category is K.'
            }
        ]
    }

    # Write to jsonl
    with open('dummy_trial.jsonl', 'w') as f:
        f.write(json.dumps(experiment))

def load_model(model_path='centaur8b', adapter_path=None):
    model, tokenizer = load(model_path)#, adapter_path=adapter_path)
    return model, tokenizer


def test_prompt(model, tokenizer, experiment):
    accuracies = []
    prompt = experiment['description']
    for i, trial in enumerate(experiment['trials']):
        prompt = prompt + '\n' + trial['stimulus']
        print(f"Trial {i + 1}")
        print(f"Prompt: {prompt}")
        print(f"Result: {trial['result']}")
        response = generate(model, tokenizer, prompt=prompt, max_tokens=1)
        print(f"Real response: {trial['response']}")
        print(f"Model response: {response}")
        accuracy = response == trial['response']
        accuracies.append(accuracy)
        if len(accuracies) > 0:
            print(f'Accuracy so far: {sum(accuracies) / len(accuracies)}')
        prompt = prompt + trial['response'] + trial['result']

    print(f"Overall accuracy: {sum(accuracies) / len(accuracies)}")
    return accuracies


def test_trials(participant_json):
    # Read jsonl
    with open(participant_json, 'r') as f:
        experiments = json.load(f)
        for experiment in experiments:
            model, tokenizer = load_model(adapter_path='adapters')
            print(f'Experiment: {experiment["experiment"]}, Participant: {experiment["participant"]}')
            print(f'Description: {experiment["description"]}')
            test_prompt(model, tokenizer, experiment)

# dummy_csv()
# test_trials('dummy_trial.jsonl')
test_trials('psych101_parsed.jsonl')
