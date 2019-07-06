from sklearn.ensemble import RandomForestClassifier

def random_forest():
    return RandomForestClassifier(n_estimators=100, oob_score=True)