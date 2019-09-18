import minpy

minpy.set_global_policy('only_numpy')


class Parameter:
    def __init__(self, o, A, M, a, alpha, b, lu):
        self.o = o
        self.A = A
        self.M = M
        self.a = a
        self.alpha = alpha
        self.b = b
        self.lu = lu