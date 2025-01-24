import matplotlib.pyplot as plt
import pandas as pd
import json

def plot_accuracies(file_path, window_size=3):
    with open(file_path, 'r') as f:
        data = [json.loads(l) for l in f]
        for participant in data:
            experiment = participant['experiment']
            participant_id = participant['participant']
            accuracies = pd.Series(participant['accuracies'])
            # Calculate the cumulative sum
            acc_cumsum = accuracies.cumsum() / (pd.Series(range(window_size, len(accuracies) + 1)))
            # Apply moving average
            smoothed_acc = acc_cumsum.rolling(window=window_size).mean()

            plt.plot(smoothed_acc, label=f'{experiment} - {participant_id}')
        plt.xlabel('Trial')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy of Participants')
        plt.savefig('accuracies_50-55.png')
        plt.show()

plot_accuracies('psych101_test_results_50-55.jsonl', window_size=5)