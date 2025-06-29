class SharpRecoilModel:
    def __init__(self):
        self.name = "Sharp Recoil Model"

    def detect_sharp_recoil(self, odds_data):
        return [game for game in odds_data if self._is_sharp(game)]

    def _is_sharp(self, game):
        try:
            changes = [abs(bookmaker['markets'][0]['outcomes'][0]['price'] -
                           bookmaker['markets'][0]['outcomes'][1]['price'])
                       for bookmaker in game['bookmakers']]
            return max(changes) > 20
        except:
            return False