import sys
import math
import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------------------ #

def read_dataset(in_fname, num_imgs, img_size=28):
    f = gzip.open(in_fname)
    f.read(16)

    buf = f.read(img_size * img_size * num_imgs)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_imgs, img_size*img_size)

    return np.asmatrix(data)

def read_labels(in_fname, num_labels, label_size=1):
    f = gzip.open(in_fname)
    f.read(8)

    buf = f.read(num_labels * label_size)
    labl = np.frombuffer(buf, dtype=np.uint8).astype(np.uint8)
    labl = labl.reshape(num_labels, label_size)

    return np.asmatrix(labl)

def read_thetas(in_fname='weights.pkl'):
    thetas = pd.read_pickle(in_fname)[0]
    return list(thetas)

def save_thetas(thetas, out_fname='weights.pkl'):
    thetas_df = pd.DataFrame(thetas)
    thetas_df.to_pickle(out_fname)

# ------------------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------------------ #

def normalize_features(X, mu=None, sigma=None):
    if mu is None or sigma is None:
        mu = np.mean(X, 0)

        sigma = np.asarray(np.max(X, 0)) - np.asarray(np.min(X, 0))
        sigma[sigma<1] = 1

        X = np.asmatrix(np.asarray(np.asarray(X)-np.asarray(mu))/np.asarray(sigma))
        return [X, mu, sigma]
    else:
        X = np.asmatrix(np.asarray(np.asarray(X)-np.asarray(mu))/np.asarray(sigma))
        return X

def shuffle_dataset(X, y):
    indices = np.arange(0, X.shape[0])
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    return [X, y]

# ------------------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------------------ #

def plot_img(x):
    x = np.asarray(x)[0]
    img = np.asarray(x.reshape(int(math.sqrt(x.shape[0])), int(math.sqrt(x.shape[0])), 1)).squeeze()
    plt.imshow(img, cmap='gray_r')
    plt.colorbar()

def plot_imgs(X, num_imgs_y, num_imgs_x):
    X = np.asarray(X)
    imgs = X.reshape(X.shape[0], int(math.sqrt(X.shape[1])), int(math.sqrt(X.shape[1])), 1)

    plt.figure()
    plt.suptitle('First {:0} images'.format(num_imgs_y*num_imgs_x))

    for y in range(num_imgs_y):
        for x in range(num_imgs_x):
            plt.subplot(num_imgs_y, num_imgs_x, y*num_imgs_x+x+1)
            
            img = np.asarray(imgs[y*num_imgs_x+x]).squeeze()
            plt.imshow(img, cmap='gray_r')
            plt.colorbar()

def plot_pastJ(pastJ):
    plt.plot(np.arange(0, len(pastJ)), pastJ, c='midnightblue', linestyle='-')

    plt.xlabel('Iteration [0;m[')
    plt.ylabel('Cost (J)')

    plt.legend(['J/iter. graph'])
    plt.title('Cost/Iteration Graph')

def plot_learning_curve(X, y, Xtest, ytest, thetas, lam, alpha, no_iters=25):
    training_curve = []
    testing_curve = []

    for m in range(1, len(X)+1):
        th_t, pastJ, pastJ_plt = miniBatchGradientDescent(X[:m], y[:m], thetas, lam, alpha, no_iters, m)

        training_curve.append(pastJ[-1])
        testing_curve.append(costFunction(Xtest, ytest, th_t, lam))

        plt.close(pastJ_plt)

    plt.plot(np.arange(1, len(X)+1), training_curve, c='lightcoral', linestyle='-')
    plt.plot(np.arange(1, len(X)+1), testing_curve, c='royalblue', linestyle='-')

    plt.xlabel('Training set size (m)')
    plt.ylabel('Cost (J)')

    plt.legend(['Training curve', 'Testing curve'])
    plt.title('Learning curve')

def plot_hidden_units(theta):
    theta = np.asarray(theta)

    for i in range(len(theta)):
        plt.subplot(math.ceil(math.sqrt(len(theta))), math.floor(math.sqrt(len(theta))), i+1)

        spx = math.floor(math.sqrt(len(np.asarray(theta[i])[1:])))
        img = np.asarray(theta[i])[1:].reshape(spx, spx)

        plt.imshow(img, cmap='gray')

# ------------------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------------------ #

def sigmoid(z):
    return np.asmatrix(1 / (1 + math.e ** -np.asarray(z)))

def predict(x, thetas, activation_func=sigmoid):
    # ----------------------------------------------------- #
    #   x                   ...         ( q x n )           #
    #   thetas              ...         list                #
    #   | - theta1          ...         ( s2-1 x s1 )       #
    #   | - ...                                             #
    #   | - theta(L-1)      ...         ( (sL)-1 x s(L-1) ) #
    # ----------------------------------------------------- #

    if type(x) != np.matrix:
        raise Exception('"x" should be of type "numpy.matrix" ... ')

    if type(thetas) != list:
        raise Exception('"thetas" should be of type "list" ... ')

    q = x.shape[0]
    al = np.asmatrix(np.vstack((np.ones(q), np.transpose(x))))                                          # ( s1 x q )

    for theta in thetas[:-1]:
        zl = theta * al
        al = np.vstack((np.ones(zl.shape[1]), activation_func(zl)))

    zl = thetas[-1] * al                                                                                # ( k x q )
    al = activation_func(zl)                                                                            # ( k x q )

    return al

def costFunction(X, y, thetas, lam):
    # ----------------------------------------------------- #
    #   X                   ...         ( m x n )           #
    #   y                   ...         ( m x k )           #
    #   thetas              ...         list                #
    #   | - theta1          ...         ( s2-1 x s1 )       #
    #   | - ...                                             #
    #   | - theta(L-1)      ...         ( (sL)-1 x s(L-1) ) #
    #   lam                 ...         scalar              #
    # ----------------------------------------------------- #

    m = X.shape[0]
    p = predict(X, thetas)

    J = (1/m) * np.sum(
                        -np.transpose(np.asarray(y)) * np.log(np.asarray(p)) - 
                        (1 - np.transpose(np.asarray(y))) * np.log(1 - np.asarray(p))
                    ) + (lam/(2*m)) * np.sum([
                        np.sum(np.asarray(theta[:,1:]) ** 2) for theta in thetas
                    ])

    return J

def computeGradient(X, y, thetas, lam, activation_func=sigmoid):
    # ----------------------------------------------------- #
    #   X                   ...         ( m x n )           #
    #   y                   ...         ( m x k )           #
    #   thetas              ...         list                #
    #   | - theta1          ...         ( s2-1 x s1 )       #
    #   | - ...                                             #
    #   | - theta(L-1)      ...         ( (sL)-1 x s(L-1) ) #
    #   lam                 ...         scalar              #
    # ----------------------------------------------------- #

    m = X.shape[0]

    als = [np.asmatrix(np.vstack((np.ones(m), np.transpose(X))))]
    zls = []

    for i in range(len(thetas)-1):
        theta = thetas[i]

        zls.append(theta * als[i])
        als.append(np.vstack((np.ones(zls[i].shape[1]), activation_func(zls[i]))))

    zls.append(thetas[-1] * als[-1])
    als.append(activation_func(zls[-1]))

    ds = [als[-1] - np.transpose(y)]

    for i in range(len(thetas)-1):
        ds.append(np.asmatrix(
                                np.asarray(np.transpose(thetas[-1-i][:,1:]) * ds[i]) *
                                np.asarray(als[-2-i][1:]) * 
                                np.asarray(1 - als[-2-i][1:])
                            ))
    
    Ds = []
    for i in range(len(thetas)):
        delta = ds[i] * np.transpose(als[-2-i])
        Ds.append((1/m) * (delta) + lam * np.hstack((np.zeros((delta.shape[0],1)), thetas[-1-i][:,1:])))

    return list(reversed(Ds))

def miniBatchGradientDescent(X, y, thetas, lam, alpha, no_iters, batch_size):
    pastJ = [costFunction(X[:batch_size], y[:batch_size], thetas, lam)]

    plt.ion()
    plt.show()

    pastJ_plt = plt.figure()
    past_plt = plt.plot(np.arange(0,no_iters+1), pastJ + list(np.full(no_iters+1-len(pastJ), 0)), linestyle='-.')[0]
    
    plt.title('Mini Batch Gradient Descent (iteration 0)')
    plt.xlabel('Iterations [0;m]')
    plt.ylabel('Cost (J)')
    plt.legend(['J/iteration plot'])

    plt.pause(0.001)

    try:
        for i in range(no_iters):
            X, y = shuffle_dataset(X, y)
            print(' [gradient-descent]: Iteration #%d ... ' % i)

            grads = computeGradient(X[:batch_size], y[:batch_size], thetas, lam)

            for j in range(len(grads)):
                thetas[j] = thetas[j] - alpha * grads[j]

            pastJ.append(costFunction(X, y, thetas, lam))
            print('\t~-> J = %f' % pastJ[-1])

            past_plt.set_ydata(pastJ + list(np.full(no_iters+1-len(pastJ), 0)))
            plt.title('Mini Batch Gradient Descent (iteration {:0})'.format(i+1))

            plt.draw()
            plt.pause(0.001)
    except KeyboardInterrupt:
        pass

    plt.ioff()
    return [thetas, pastJ, pastJ_plt]

def optimizeLambda(X, y, Xval, yval, thetas, lam, alpha, no_iters=30, batch_size=50,
                    lambdas=[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]):
    minl = None
    minc = sys.maxsize

    for l in lambdas:
        print(' [opt-lambda]: Testing lambda=%f ... ' % l)

        th_t, pastJ, pastJ_plt = miniBatchGradientDescent(X, y, thetas.copy(), l, alpha, no_iters, batch_size)
        plt.close(pastJ_plt)
        Jval = costFunction(Xval, yval, th_t, l)

        if Jval < minc:
            minl = l
            minc = Jval

    return minl

def computeAccuracy(X, y, thetas):
    preds = predict(X, thetas)
    return np.sum(np.abs(np.asarray(preds) - np.asarray(np.transpose(y))))/X.shape[0]

# ------------------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------------------ #

def main():
    # -------------------------------------------------------- #
    # --- LOAD DATASET --------------------------------------- #
    # -------------------------------------------------------- #

    X = read_dataset('train-images-idx3-ubyte.gz', 60000)
    y = read_labels('train-labels-idx1-ubyte.gz', 60000)
    Xtest = read_dataset('t10k-images-idx3-ubyte.gz', 10000)
    ytest = read_labels('t10k-labels-idx1-ubyte.gz', 10000)

    labl_dict = ['T-Shirt/top', 'Trousers', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # -------------------------------------------------------- #
    # --- PREPARE DATASET ------------------------------------ #
    # -------------------------------------------------------- #

    X, y = shuffle_dataset(X, y)
    Xtest, ytest = shuffle_dataset(Xtest, ytest)

    X, mu, sigma = normalize_features(X)
    Xtest = normalize_features(Xtest, mu, sigma)

    Xval = np.asmatrix(X[round(X.shape[0]*0.8):])
    yval = np.asmatrix(y[round(y.shape[0]*0.8):])
    X = np.asmatrix(X[:round(X.shape[0]*0.8)])
    y = np.asmatrix(y[:round(y.shape[0]*0.8)])

    y = np.where((np.ones(y.shape)*np.arange(0,10)) == y, 1, 0)
    yval = np.where((np.ones(yval.shape)*np.arange(0,10)) == yval, 1, 0)
    ytest = np.where((np.ones(ytest.shape)*np.arange(0,10)) == ytest, 1, 0)

    # -------------------------------------------------------- #
    # --- BUILD MODEL ---------------------------------------- #
    # -------------------------------------------------------- #

    n = 28*28
    s1 = n+1
    s2 = 100+1
    s3 = 50+1
    k = 10

    theta1 = np.asmatrix(np.random.rand(s2-1, s1)*2-1)
    theta2 = np.asmatrix(np.random.rand(s3-1, s2)*2-1)
    theta3 = np.asmatrix(np.random.rand(k, s3)*2-1)

    thetas = [theta1, theta2, theta3]

    # -------------------------------------------------------- #
    # --- LOAD MODEL ----------------------------------------- #
    # -------------------------------------------------------- #

    yN = input('Do you want to load the stored weights? [y/N] ')

    if yN in ['y', 'Y', 'j', 'J']:
        thetas = read_thetas()
    else:
        print('Ok! Beginning to learn from scratch ...')

    # -------------------------------------------------------- #
    # --- LEARNING PARAMETERS -------------------------------- #
    # -------------------------------------------------------- #

    lam = 0.03
    alpha = 0.1
    no_iters = 10000
    batch_size = 50

    # -------------------------------------------------------- #
    # --- LEARNING ------------------------------------------- #
    # -------------------------------------------------------- #

    lam = optimizeLambda(X, y, Xval, yval, thetas.copy(), lam, alpha)
    thetas, pastJ = miniBatchGradientDescent(X, y, thetas.copy(), lam, alpha, no_iters, batch_size)

    # -------------------------------------------------------- #
    # --- EVALUATION ----------------------------------------- #
    # -------------------------------------------------------- #

    print('Optimal Lambda: %f ... ' % lam)
    print('Final cost: %f ... ' % costFunction(Xtest, ytest, thetas.copy(), lam))
    print('Final accuracy: %f ... ' % computeAccuracy(Xtest, ytest, thetas.copy()))

    # -------------------------------------------------------- #
    # --- VISUALIZATION -------------------------------------- #
    # -------------------------------------------------------- #

    plot_pastJ(pastJ)

    plt.figure()
    plot_learning_curve(X[:200], y[:200], Xtest[:200], ytest[:200], thetas.copy(), lam, alpha)

    plt.figure()
    plot_hidden_units(thetas[0])

    plt.show()

    # -------------------------------------------------------- #
    # --- SAVING --------------------------------------------- #
    # -------------------------------------------------------- #

    yN = input('Do you want to save the learned weights? [y/N] ')

    if yN in ['y', 'Y', 'j', 'J']:
        save_thetas(thetas)
    else:
        print('Alright! It\'s your choice...')

    # -------------------------------------------------------- #
    # --- TESTING -------------------------------------------- #
    # -------------------------------------------------------- #

    try:
        while True:
            rand_ind = np.random.randint(0, Xtest.shape[0])
            plot_img(Xtest[rand_ind])

            pred = predict(Xtest[rand_ind], thetas)

            print('Prediction:\t\t%s' % labl_dict[np.where(np.asarray(np.transpose(pred))[0] == np.max(pred))[0][0]])
            print('Label:\t\t\t%s' % labl_dict[np.where(ytest[rand_ind] == 1)[0][0]])
            print('-'*15)

            plt.show()
    except KeyboardInterrupt:
        print('[+] Bye!')

    # -------------------------------------------------------- #
    # --- FIN ------------------------------------------------ #
    # -------------------------------------------------------- #

if __name__ == '__main__':
    main()