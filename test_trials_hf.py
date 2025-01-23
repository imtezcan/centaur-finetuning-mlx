import json
import transformers
from unsloth import FastLanguageModel


def load_model_unsloth():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="marcelbinz/Llama-3.1-Centaur-8B-adapter",
        max_seq_length=32768,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def test_prompt(pipe, experiment):
    accuracies = []
    prompt = experiment['description']
    for i, trial in enumerate(experiment['trials']):
        prompt = prompt + '\n' + trial['stimulus']
        print(f"Trial {i + 1}")
        print(f"Prompt: {prompt}")
        print(f"Result: {trial['result']}")
        response = pipe(prompt)[0]['generated_text'][len(prompt):]
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
    model, tokenizer = load_model_unsloth()#adapter_path='adapters')
    pipe = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        trust_remote_code=True,
        pad_token_id=0,
        do_sample=True,
        temperature=1.0,
        max_new_tokens=1,
    )
    with open(participant_json, 'r') as f:
        participants = json.load(f)
        results = []
        for participant in participants[:1]:
            print(f'Experiment: {participant["experiment"]}, Participant: {participant["participant"]}')
            print(f'Description: {participant["description"]}')
            participant_results = {'participant': participant["participant"], 'experiment': participant["experiment"],
                                   'accuracies': test_prompt(pipe, participant)}
            results.append(participant_results)

            with open('psych101_test_results.jsonl', 'a') as fr:
                fr.write(json.dumps(participant_results))



# dummy_csv()
# test_trials('dummy_trial.jsonl')
test_trials('psych101_test_parsed.jsonl')
