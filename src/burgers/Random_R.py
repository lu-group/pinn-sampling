import deepxde as dde
import numpy as np
from deepxde.backend import tf

dde.config.set_default_float("float64")


def gen_testdata():
    data = np.load("Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y


def main(P):
    NumDomain = 2000

    def pde(x, y):
        dy_x = dde.grad.jacobian(y, x, i=0, j=0)
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_t + y * dy_x - 0.01 / np.pi * dy_xx

    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    data = dde.data.TimePDE(geomtime, pde, [], num_domain=NumDomain, train_distribution='pseudo')

    net = dde.maps.FNN([2] + [64] * 3 + [1], "tanh", "Glorot normal")

    def output_transform(x, y):
        return -tf.sin(np.pi * x[:, 0:1]) + (1 - x[:, 0:1] ** 2) * (x[:, 1:]) * y

    net.apply_output_transform(output_transform)

    model = dde.Model(data, net)

    resampler = dde.callbacks.PDEResidualResampler(period=P)
    model.compile("adam", lr=1e-3)
    model.train(epochs=15000, callbacks=[resampler])
    model.compile("L-BFGS")
    losshistory, train_state = model.train()

    X, y_true = gen_testdata()
    y_pred = model.predict(X)
    error = dde.metrics.l2_relative_error(y_true, y_pred)
    print("L2 relative error:", error)
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    return error


if __name__ == '__main__':
    main(P=100)
