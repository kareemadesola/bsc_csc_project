import matplotlib.pyplot as plt


def plot_learning_curve(train_losses, val_losses, metric_name=None):
    """
    Plot the learning curve of a model during training.

    Parameters:
        train_losses (list): List of training losses or metric values for each epoch/iteration.
        val_losses (list): List of validation losses or metric values for each epoch/iteration.
        metric_name (str, optional): Name of the metric (e.g., "Accuracy", "Mean Squared Error"). Default is None.
    """

    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, label='Training', marker='o')
    plt.plot(epochs, val_losses, label='Validation', marker='o')

    plt.xlabel('Epochs')
    plt.ylabel('Loss' if metric_name is None else metric_name)
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.show()


# Example usage:
if __name__ == '__main__':
    train_losses = [0.8, 0.6, 0.4, 0.3, 0.2]
    val_losses = [1.0, 0.7, 0.5, 0.4, 0.3]
    plot_learning_curve(train_losses, val_losses, metric_name='MSE')
