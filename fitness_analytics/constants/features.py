class Features:
    def __init__(self, name, bodypart, complements):
        self.name = name
        self.bodypart = bodypart
        self.complements = complements
        self.side = ''
        self.results = []
        self.directions = []
        self.angles = []
        self.rep = []

    def assign_features(self, df):
        self.directions = df['direction'].tolist()
        self.rep = df['rep'].tolist()

    def return_json(self):
        output = {
            'direction': self.directions,
            'rep': self.rep
        }
        return output


class DerivedFeatures:
    def __init__(self):
        self.max_angles = {}
        self.min_angles = {}
        self.rep_time = {}
        self.rep_completion = {}
        self.total_time = 0
        self.total_reps = 0

    def return_json(self):
        output = {
            'max_angles': self.max_angles,
            'min_angles': self.min_angles,
            'rep_time': self.rep_time,
            'rep_completion': self.rep_completion,
            'total_time': self.total_time,
            'total_reps': self.total_reps
        }
        return output
