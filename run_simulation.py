import random

def simulate_bankroll(start=10000, bets=100, winrate=0.55, kelly_fraction=0.01):
    bankroll = start
    history = []

    for i in range(bets):
        edge = (2 * winrate - 1)
        bet = bankroll * kelly_fraction * edge
        won = random.random() < winrate
        bankroll += bet if won else -bet
        history.append(bankroll)

    return history