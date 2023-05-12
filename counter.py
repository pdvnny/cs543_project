# Parker Dunn

class RecursiveCounter:
    def __init__(self):
        self.COUNT = 0

    def increment(self):
        self.COUNT += 1

    def __str__(self):
        return str(self.COUNT)

    def get_count(self):
        return self.COUNT

    def reset(self):
        self.COUNT = 0