

class StoreIntermediateCallback:

    def __init__(self):
        self.xs = []
        self.gs = []
    
    def __call__(self, x, g):
        self.xs.append(x)
        self.gs.append(g)
