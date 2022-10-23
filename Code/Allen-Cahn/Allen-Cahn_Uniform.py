import deepxde as dde
import numpy as np
from deepxde.backend import tf
import skopt
from distutils.version import LooseVersion
from scipy.io import loadmat

dde.config.set_default_float("float64")


def quasirandom(n_samples, sampler):
    space = [(-1.0, 1.0), (0.0, 1.0)]
    if sampler == "LHS":
        sampler = skopt.sampler.Lhs(
            lhs_type="centered", criterion="maximin", iterations=1000
        )
    elif sampler == "Halton":
        sampler = skopt.sampler.Halton(min_skip=-1, max_skip=-1)
    elif sampler == "Hammersley":
        sampler = skopt.sampler.Hammersly(min_skip=-1, max_skip=-1)
    elif sampler == "Sobol":
        # Remove the first point [0, 0, ...] and the second point [0.5, 0.5, ...], which
        # are too special and may cause some error.
        if LooseVersion(skopt.__version__) < LooseVersion("0.9"):
            sampler = skopt.sampler.Sobol(min_skip=2, max_skip=2, randomize=False)
        else:
            sampler = skopt.sampler.Sobol(skip=0, randomize=False)
            return np.array(
                sampler.generate(space, n_samples + 2)[2:]
            )
    return np.array(sampler.generate(space, n_samples))


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


def main(NumDomain, method):
    def pde(x, y):
        u = y
        du_xx = dde.grad.hessian(y, x, i=0, j=0)
        du_t = dde.grad.jacobian(y, x, j=1)
        return du_t - 0.001 * du_xx + 5 * (u ** 3 - u)

    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    if method == 'Grid':
        data = dde.data.TimePDE(geomtime, pde, [], num_domain=NumDomain, train_distribution='uniform')
    elif method == 'Random':
        data = dde.data.TimePDE(geomtime, pde, [], num_domain=NumDomain, train_distribution='pseudo')
    elif method in ['LHS', 'Halton', 'Hammersley', 'Sobol']:
        sample_pts = quasirandom(NumDomain, method)
        data = dde.data.TimePDE(geomtime, pde, [], num_domain=0, train_distribution='uniform', anchors=sample_pts)

    net = dde.maps.FNN([2] + [64] * 3 + [1], "tanh", "Glorot normal")

    def output_transform(x, y):
        x_in = x[:, 0:1]
        t_in = x[:, 1:2]
        return t_in * (1 + x_in) * (1 - x_in) * y + tf.square(x_in) * tf.cos(np.pi * x_in)

    net.apply_output_transform(output_transform)

    model = dde.Model(data, net)

    model.compile("adam", lr=1e-3)
    model.train(epochs=15000)
    model.compile("L-BFGS")
    losshistory, train_state = model.train()

    X, y_true = gen_testdata()
    y_pred = model.predict(X)
    error = dde.metrics.l2_relative_error(y_true, y_pred)
    print("L2 relative error:", error)
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    return error


if __name__ == "__main__":
    main(NumDomain=5000, method='Grid')
    # main(NumDomain=5000, method='Random')
    # main(NumDomain=5000, method='LHS')
    # main(NumDomain=5000, method='Halton')
    # main(NumDomain=5000, method='Hammersley')
    # main(NumDomain=5000, method='Sobol')
