class MatthewModel:
    def __init__(self):
        self.name = "Matthew Model"

    def predict_win_prob(self, team_stats):
        RS, RA = team_stats["runs_scored"], team_stats["runs_allowed"]
        return (RS ** 1.83) / ((RS ** 1.83) + (RA ** 1.83))