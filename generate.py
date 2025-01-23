import pandas as pd

# df = pd.read_json("hf://datasets/marcelbinz/Psych-101/prompts_training.jsonl", lines=True)
df = pd.read_json("hf://datasets/marcelbinz/Psych-101-test/prompts_testing_t1.jsonl", lines=True)

df.to_json('./data/psych101_test.jsonl', orient='records', lines=True)