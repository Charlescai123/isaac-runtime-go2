import matlab
# import matlab.engine
from numpy import linalg as LA
import matplotlib.pyplot as plt
from numpy.linalg import pinv
import numpy as np
# import cvxpy as cp
import copy


class MatEngine:
    def __init__(self):
        self.mat_engine = None
        # self.matlab_engine_launch()
        # self.cvx_setup()

    def matlab_engine_launch(self, path="./robot/ha_teacher"):
        print("Launching Matlab Engine...")
        self.mat_engine = matlab.engine.start_matlab()
        self.mat_engine.cd(path)
        print("Matlab current working directory is ---->>>", self.mat_engine.pwd())

    def cvx_setup(self):
        self.engine.cd("./cvx/")
        print("Setting up the CVX Toolbox...")
        _ = self.engine.cvx_setup
        print("CVX Toolbox setup done.")
        self.engine.cd("..")


if __name__ == '__main__':
    As = np.array([[0, 1, 0, 0],
                   [0, 0, -1.42281786576776, 0.182898194776782],
                   [0, 0, 0, 1],
                   [0, 0, 25.1798795199119, 0.385056459685276]])

    Bs = np.array([[0,
                    0.970107410065162,
                    0,
                    -2.04237185222105]])

    Ak = np.array([[1, 0.0100000000000000, 0, 0],
                   [0, 1, -0.0142281786576776, 0.00182898194776782],
                   [0, 0, 1, 0.0100000000000000],
                   [0, 0, 0.251798795199119, 0.996149435403147]])

    Bk = np.array([[0,
                    0.00970107410065163,
                    0,
                    -0.0204237185222105]])

    sd = np.array([[0.234343490000000,
                    0,
                    -0.226448960000000,
                    0]])

    mat = MatEngine()
    K = mat.feedback_law(As, Bs, Ak, Bk, sd)
    print(K)
