# File: helper.py

import matplotlib.pyplot as plt

plt.ion()  # Enable interactive mode for matplotlib

def plot(scores, mean_scores):
    plt.clf()  # Clear the current figure
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score', color='blue')
    plt.plot(mean_scores, label='Mean Score', color='orange')
    plt.ylim(ymin=0)  # Set y-axis limits
    plt.legend()  # Show the legend
    
    # Add text annotations if we have data
    if len(scores) > 0:
        plt.text(len(scores) - 1, scores[-1], str(scores[-1]), color='blue', fontsize=12)
    if len(mean_scores) > 0:
        plt.text(len(mean_scores) - 1, mean_scores[-1], f'{mean_scores[-1]:.1f}', color='orange', fontsize=12)
    
    plt.draw()  # Draw the plot
    plt.pause(0.1)  # Pause to allow the plot to update