import deepxde as dde
import numpy as np
from deepxde.backend import tf
from scipy.interpolate import griddata
from scipy.io import loadmat


def main(NumDomain):

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

    data = dde.data.TimePDE(geomtime, pde, [observe_u], num_domain=NumDomain, train_distribution='pseudo',
                            solution=func, num_test=10000)

    net = dde.maps.FNN([2] + [100] * 3 + [1], "tanh", "Glorot normal")

    net.apply_feature_transform(periodic)
    net.apply_output_transform(dirichlet)

    model = dde.Model(data, net)
    epochs = 50000
    model.compile("adam", lr=0.001, metrics=["l2 relative error"], decay=("inverse time", 100000 // 5, 0.5), loss_weights=[1, 1000])
    # prediction of lambda1, saved in file "variables1.dat"
    variable1 = dde.callbacks.VariableValue([lambda1], period=1000, filename=f"variables1.dat", precision=8)
    # prediction of lambda2, saved in file "variables2.dat"
    variable2 = dde.callbacks.VariableValue([lambda2], period=1000, filename=f"variables2.dat", precision=8)
    losshistory, train_state = model.train(epochs=epochs, callbacks=[variable1, variable2])

    for i in range(50):
        X = geomtime.random_points(100000)
        Y = np.abs(model.predict(X, operator=pde)).astype(np.float64)
        err_eq = Y / Y.mean() + 1
        err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
        X_ids = np.random.choice(
          a=len(X), size=NumDomain, replace=False, p=err_eq_normalized
        )
        X_selected = X[X_ids]
        data.replace_with_anchors(X_selected)
        losshistory, train_state = model.train(epochs=1000, callbacks=[variable1, variable2])
    error = losshistory.metrics_test.copy()
    error = np.array(error).reshape(-1, 1)
    np.savetxt(f'error_RAD.txt', error)
    return error


if __name__ == "__main__":
    main(NumDomain=600)
