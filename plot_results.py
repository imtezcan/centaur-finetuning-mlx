import matplotlib.pyplot as plt
import json
import numpy as np

def plot_accuracies(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        for participant in data:
            experiment = participant['experiment']
            participant_id = participant['participant']
            # Convert accuracies to cumulative sum
            acc_cumsum = np.cumsum(participant['accuracies']) / (np.arange(len(participant['accuracies'])) + 1)

            plt.plot(acc_cumsum, label=f'{experiment} - {participant_id}')
        plt.xlabel('Trial')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy of Participants')
        plt.show()

plot_accuracies('psych101_test_results_dummy.json')