import deepxde as dde
import numpy as np
from deepxde.backend import tf
import skopt
from distutils.version import LooseVersion
from scipy.interpolate import griddata
from scipy.io import loadmat


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
    def gen_testdata():
        data = loadmat("KdV.mat")
        t = data["tt"].reshape(-1)
        x = data["x"].reshape(-1)
        # x = np.append(x, 1.)
        x = np.concatenate([x, np.array([1.])])
        y = data["uu"]
        y = np.concatenate([y, y[0:1, :]]).reshape(-1, 1)
        X = np.array([[a, b] for a in x for b in t])

        return X, y

    X_test, y_true = gen_testdata()

    def func(x_input):
        return griddata(X_test, y_true, x_input, method="cubic")

    lambda1 = tf.Variable(3, trainable=True, dtype=tf.float32)
    lambda2 = tf.Variable(3, trainable=True, dtype=tf.float32)

    def pde(x, y):
        dy_x = dde.grad.jacobian(y, x, i=0, j=0)
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        dy_xxx = dde.grad.jacobian(dy_xx, x, i=0, j=0)
        return dy_t + lambda1 * y * dy_x + lambda2 * dy_xxx

    def dirichlet(x, y):
        x_in = x[:, 0:1]
        t_in = x[:, 1:2]
        return 2 * t_in * y + tf.cos(np.pi * x_in)

    def periodic(x):
        return tf.concat([tf.math.cos(x[:, 0:1] * np.pi), tf.math.sin(x[:, 0:1] * np.pi),
                          tf.math.cos(2 * x[:, 0:1] * np.pi), tf.math.sin(2 * x[:, 0:1] * np.pi), x[:, 1:2]], 1)

    idx_x1 = np.arange(0, 513, 8)
    idx_x2 = np.arange(0, 513, 8)
    ob_x = np.concatenate([X_test.reshape((513, 201, 2))[idx_x1, 40, :], X_test.reshape((513, 201, 2))[idx_x2, 160, :]])
    ob_u = np.concatenate([y_true.reshape(513, 201, 1)[idx_x1, 40, :], y_true.reshape(513, 201, 1)[idx_x2, 160, :]])
    observe_u = dde.PointSetBC(ob_x, ob_u, component=0)

    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    if method == 'Grid':
        data = dde.data.TimePDE(geomtime, pde, [observe_u], num_domain=NumDomain, train_distribution='uniform',
                                solution=func, num_test=10000)
    elif method == 'Random':
        data = dde.data.TimePDE(geomtime, pde, [observe_u], num_domain=NumDomain, train_distribution='pseudo',
                                solution=func, num_test=10000)
    elif method in ['LHS', 'Halton', 'Hammersley', 'Sobol']:
        sample_pts = quasirandom(NumDomain, method)
        data = dde.data.TimePDE(geomtime, pde, [observe_u], num_domain=0, train_distribution='uniform',
                                solution=func, num_test=10000, anchors=sample_pts)

    net = dde.maps.FNN([2] + [100] * 3 + [1], "tanh", "Glorot normal")

    net.apply_feature_transform(periodic)
    net.apply_output_transform(dirichlet)

    model = dde.Model(data, net)
    epochs = 100000
    model.compile("adam", lr=0.001, metrics=["l2 relative error"], decay=("inverse time", 100000 // 5, 0.5), loss_weights=[1, 1000])
    # prediction of lambda1, saved in file "variables1.dat"
    variable1 = dde.callbacks.VariableValue([lambda1], period=1000, filename=f"variables1.dat", precision=8)
    # prediction of lambda2, saved in file "variables2.dat"
    variable2 = dde.callbacks.VariableValue([lambda2], period=1000, filename=f"variables2.dat", precision=8)

    losshistory, train_state = model.train(epochs=epochs, callbacks=[variable1, variable2])
    error = np.array(losshistory.metrics_test)[-1]
    print(f"L2 relative error: {error}")
    return error


if __name__ == "__main__":
    # main(NumDomain=600, method='Grid')
    # main(NumDomain=600, method='Random')
    # main(NumDomain=600, method='LHS')
    # main(NumDomain=600, method='Halton')
    main(NumDomain=600, method='Hammersley')
    # main(NumDomain=600, method='Sobol')
