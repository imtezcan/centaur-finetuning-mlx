import json
from mlx_lm import load, generate


def load_model(model_path='centaur8b', adapter_path=None):
    model, tokenizer = load(model_path)#, adapter_path=adapter_path)
    return model, tokenizer


def test_prompt(model, tokenizer, experiment):
    accuracies = []
    prompt = experiment['description']
    for i, trial in enumerate(experiment['trials'][:15]):
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
    model, tokenizer = load_model()#adapter_path='adapters')
    with open(participant_json, 'r') as f:
        participants = json.load(f)
        results = []
        for participant in participants[50:55]:  # TODO only 5 participants for now
            print(f'Experiment: {participant["experiment"]}, Participant: {participant["participant"]}')
            print(f'Description: {participant["description"]}')
            participant_results = {'participant': participant["participant"], 'experiment': participant["experiment"],
                                   'accuracies': test_prompt(model, tokenizer, participant)}
            results.append(participant_results)

            with open('psych101_test_results.jsonl', 'a') as fr:
                fr.write(json.dumps(participant_results) + '\n')



# dummy_csv()
# test_trials('dummy_trial.jsonl')
test_trials('psych101_test_parsed.jsonl')
