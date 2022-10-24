import deepxde as dde
import numpy as np
from deepxde.backend import tf

dde.config.set_default_float("float64")


def main(P):
    NumDomain = 4000

    def func(x):
        return np.sin(np.pi * x[:, 0:1]) * np.cos(2 * np.pi * x[:, 1:2]) + 0.5 * np.sin(4 * np.pi * x[:, 0:1]) * np.cos(
            8 * np.pi * x[:, 1:2])

    def pde(x, y):
        dy_tt = dde.grad.hessian(y, x, i=1, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_tt - 4 * dy_xx

    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    data = dde.data.TimePDE(geomtime, pde, [], num_domain=NumDomain, train_distribution='pseudo',
                            solution=func, num_test=10000)

    net = dde.nn.FNN([2] + [100] * 5 + [1], "tanh", "Glorot uniform")

    def output_transform(x, y):
        x_in = x[:, 0:1]
        t_in = x[:, 1:2]
        return 20 * y * x_in * (1 - x_in) * t_in ** 2 + tf.sin(np.pi * x_in) + 0.5 * tf.sin(4 * np.pi * x_in)

    net.apply_output_transform(output_transform)

    model = dde.Model(data, net)

    resampler = dde.callbacks.PDEResidualResampler(period=P)
    model.compile("adam", lr=1e-3, metrics=["l2 relative error"])
    model.train(epochs=15000, callbacks=[resampler])
    model.compile("L-BFGS", metrics=["l2 relative error"])
    losshistory, train_state = model.train()

    error = np.array(losshistory.metrics_test)[-1]
    print("L2 relative error:", error)
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    return error


if __name__ == '__main__':
    main(P=100)
