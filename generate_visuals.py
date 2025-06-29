import matplotlib.pyplot as plt

def plot_bankroll(history):
    plt.plot(history)
    plt.title("Bankroll Over Time")
    plt.xlabel("Bet #")
    plt.ylabel("Bankroll")
    plt.grid(True)
    plt.show()