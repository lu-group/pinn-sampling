import os
os.environ["DDEBACKEND"] = "tensorflow.compat.v1"
import numpy as np
from deepxde.backend import tf
import deepxde as dde
from scipy.integrate import solve_bvp


def main(P):
    NumDomain = 15

    def k(x):
        return 0.1 + np.exp(-0.5 * (x - 0.5) ** 2 / 0.15 ** 2)

    def fun(x, y):
        return np.vstack((y[1], 100 * (k(x) * y[0] + np.sin(2 * np.pi * x))))

    def bc(ya, yb):
        return np.array([ya[0], yb[0]])

    a = np.linspace(0, 1, 1000)
    b = np.zeros((2, a.size))

    res = solve_bvp(fun, bc, a, b)

    def sol(x):
        return res.sol(x)[0]

    def du(x):
        return res.sol(x)[1]
    l = 0.01

    def gen_traindata(num):
        xvals = np.linspace(0, 1, num)
        yvals = sol(xvals)

        return np.reshape(xvals, (-1, 1)), np.reshape(yvals, (-1, 1))

    geom = dde.geometry.Interval(0, 1)

    ob_x, ob_u = gen_traindata(8)
    observe_u = dde.PointSetBC(ob_x, ob_u, component=0)
    bc = dde.DirichletBC(geom, sol, lambda _, on_boundary: on_boundary, component=0)

    def pde(x, y):
        u = y[:, 0:1]
        k = y[:, 1:2]
        du_xx = dde.grad.hessian(y, x, component=0)
        return l * du_xx - k * u - tf.sin(2 * np.pi * x)

    data = dde.data.PDE(geom, pde, bcs=[bc, observe_u], num_domain=NumDomain-2, num_boundary=2,
                        train_distribution="pseudo", num_test=1000)

    net = dde.maps.PFNN([1, [20, 20], [20, 20], [20, 20], 2], "tanh", "Glorot uniform")
    model = dde.Model(data, net)

    xx = np.linspace(0, 1, 1001)[:, None]

    def l2_u(_, __):
        return dde.metrics.l2_relative_error(sol(xx), model.predict(xx)[:, 0:1])

    def l2_k(_, __):
        return dde.metrics.l2_relative_error(k(xx), model.predict(xx)[:, 1:2])

    resampler = dde.callbacks.PDEResidualResampler(period=P)
    model.compile("adam", lr=0.0001, metrics=[l2_u, l2_k])
    losshistory, train_state = model.train(epochs=50000, callbacks=[resampler])

    errors = losshistory.metrics_test.copy()
    errors = np.array(errors).reshape(-1, 2)
    error_u = errors[:, 0:1]
    error_k = errors[:, 1:2]
    print("L2 relative error of u:", error_u[-1])
    print("L2 relative error of k:", error_k[-1])
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    return error_u[-1], error_k[-1]


if __name__ == '__main__':
    main(P=1000)
