# rallyrobopilot/ml/utils.py
import matplotlib.pyplot as plt

class LossPlotter:
    """A class to handle live plotting of training and validation loss."""
    def __init__(self):
        # No longer need plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.train_losses = []
        self.val_losses = []

        # Set up the plot aesthetics
        self.ax.set_xlabel("Epochs")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Training and Validation Loss Evolution")
        self.ax.grid(True)
        
    def update(self, train_loss, val_loss):
        """Appends new losses and redraws the plot."""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        # Clear the previous plot
        self.ax.cla()
        
        # Plot the new data
        self.ax.plot(self.train_losses, label='Training Loss', color='blue')
        self.ax.plot(self.val_losses, label='Validation Loss', color='orange')
        
        # Redraw the plot with updated aesthetics
        self.ax.set_xlabel("Epochs")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Training and Validation Loss Evolution")
        self.ax.legend()
        self.ax.grid(True)
        
        # --- MODIFIED LINES ---
        # Explicitly draw the canvas and flush events to update the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def save(self, filepath="loss_evolution.png"):
        """Saves the final plot to a file."""
        self.fig.savefig(filepath)
        print(f"Loss plot saved to {filepath}")