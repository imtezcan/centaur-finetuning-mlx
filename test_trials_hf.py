import pandas
import json
import transformers
from unsloth import FastLanguageModel


def load_model_unsloth():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="marcelbinz/Llama-3.1-Centaur-70B-adapter",
        max_seq_length=32768,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def load_model_hf():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    checkpoint = "marcelbinz/Llama-3.1-Centaur-8B"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        device_map='auto',
        load_in_8bit=True,
        trust_remote_code=True
    )

    return model, tokenizer


def test_prompt_hf(model, tokenizer, experiment):
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

    accuracies = []
    prompt = experiment['description']
    for i, trial in enumerate(experiment['trials']):
        prompt = prompt + '\n' + trial['stimulus']
        print(f"Trial {i + 1}")
        print(f"Prompt: {prompt}")
        print(f"Result: {trial['result']}")
        # response = generate(model, tokenizer, prompt=trial['stimulus'], max_tokens=100)
        response = pipe(prompt)[0]['generated_text'][len(prompt):]
        print(f"Real response: {trial['response']}")
        print(f"Model response: {response}")
        accuracy = response == trial['response']
        print(f'Accuracy in trial {i + 1}: {accuracy}')
        accuracies.append(accuracy)
        prompt = prompt + trial['response'] + trial['result']

    print(f"Overall accuracy: {sum(accuracies) / len(accuracies)}")
    return accuracies


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


def test_trials_hf(participant_json):
    # Read jsonl
    with open(participant_json, 'r') as f:
        experiment = json.load(f)
        model, tokenizer = load_model_hf()
        print(f'Description: {experiment["description"]}')
        test_prompt_hf(model, tokenizer, experiment)


dummy_csv()
test_trials_hf('dummy_trial.jsonl')
