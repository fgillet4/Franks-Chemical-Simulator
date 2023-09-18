import matplotlib.pyplot as plt

def plot_arrows():
    # Create a new figure
    plt.figure()

    # Plotting arrows
    plt.arrow(0.3, 0.5, 0.4, 0, color='red', width=0.02, head_width=0.1, head_length=0.1, length_includes_head=True)
    plt.arrow(0.7, 0.5, -0.4, 0, color='blue', width=0.02, head_width=0.1, head_length=0.1, length_includes_head=True)

    # Configuring plot limits and displaying the plot
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')  # Turn off the axis
    plt.show()

plot_arrows()
