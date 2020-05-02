
class Result(object):

    def __init__(self):

        self.models = []

        self.labels = []
        self.accuracy = []

        self.total_fraud = 0
        self.false_pos = []
        self.false_neg = []
        self.correct_fraud = []
