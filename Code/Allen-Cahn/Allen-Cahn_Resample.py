import deepxde as dde
import numpy as np
from deepxde.backend import tf
from scipy.io import loadmat

dde.config.set_default_float("float64")


def gen_testdata():
    data = loadmat("usol_D_0.001_k_5.mat")

    t = data["t"]

    x = data["x"]

    u = data["u"]

    dt = dx = 0.01

    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = u.flatten()[:, None]
    return X, y


def main(P):
    NumDomain = 2000

    def pde(x, y):
        u = y
        du_xx = dde.grad.hessian(y, x, i=0, j=0)
        du_t = dde.grad.jacobian(y, x, j=1)
        return du_t - 0.001 * du_xx + 5 * (u ** 3 - u)

    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    data = dde.data.TimePDE(geomtime, pde, [], num_domain=NumDomain, train_distribution='pseudo')

    net = dde.maps.FNN([2] + [64] * 3 + [1], "tanh", "Glorot normal")

    def output_transform(x, y):
        x_in = x[:, 0:1]
        t_in = x[:, 1:2]
        return t_in * (1 + x_in) * (1 - x_in) * y + tf.square(x_in) * tf.cos(np.pi * x_in)

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
