import deepxde as dde
import numpy as np
from deepxde.backend import tf
import skopt
from distutils.version import LooseVersion

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


def main(NumDomain, method):
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

    if method == 'Grid':
        data = dde.data.TimePDE(geomtime, pde, [], num_domain=NumDomain, train_distribution='uniform', solution=func, num_test=10000)
    elif method == 'Random':
        data = dde.data.TimePDE(geomtime, pde, [], num_domain=NumDomain, train_distribution='pseudo', solution=func, num_test=10000)
    elif method in ['LHS', 'Halton', 'Hammersley', 'Sobol']:
        sample_pts = quasirandom(NumDomain, method)
        data = dde.data.TimePDE(geomtime, pde, [], num_domain=0, train_distribution='uniform', anchors=sample_pts, solution=func, num_test=10000)

    net = dde.nn.FNN([2] + [100] * 5 + [1], "tanh", "Glorot uniform")

    def output_transform(x, y):
        x_in = x[:, 0:1]
        t_in = x[:, 1:2]
        return 20 * y * x_in * (1 - x_in) * t_in ** 2 + tf.sin(np.pi * x_in) + 0.5 * tf.sin(4 * np.pi * x_in)

    net.apply_output_transform(output_transform)

    model = dde.Model(data, net)

    model.compile("adam", lr=1e-3, metrics=["l2 relative error"])
    model.train(epochs=15000)
    model.compile("L-BFGS", metrics=["l2 relative error"])
    losshistory, train_state = model.train()

    error = np.array(losshistory.metrics_test)[-1]
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
