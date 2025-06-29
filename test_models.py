from models.matthew_model import MatthewModel

def test_prediction():
    model = MatthewModel()
    prob = model.predict_win_prob({"runs_scored": 700, "runs_allowed": 650})
    assert 0 < prob < 1