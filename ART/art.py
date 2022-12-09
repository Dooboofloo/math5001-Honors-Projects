import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def genMeasureMat(rows, cols):
    stripWidth = 0.02 # hyperparameter describing how close the center of the pixel must be to the line in order to be counted

    pixelWidth = int(np.sqrt(cols))
    theta = np.linspace(0, np.pi, rows, endpoint=False)
    
    r = np.zeros((rows, cols))

    for i in range(rows):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"Row {i+1} of {rows}")
    
        w = np.array([np.cos(theta[i]), np.sin(theta[i])])

        for j in range(cols):
            pixelCenter = np.array([((2 * (j % pixelWidth) + 1) / pixelWidth) - 1, 1 - ((2 * (j // pixelWidth) + 1) / pixelWidth)])

            proj = np.dot(pixelCenter, w) * w

            distFromLine = np.linalg.norm(pixelCenter - proj)

            if (distFromLine <= stripWidth):
                r[i][j] = 1
            else:
                r[i][j] = 0
    
    return r


if __name__ == '__main__':
    # inputRadon = np.loadtxt('./radon/SheppLoganRadon.csv', delimiter=',')
    inputRadon = np.loadtxt('./radon/radon995.csv', delimiter=',')

    numAngles = inputRadon.shape[0]
    numSamples = inputRadon.shape[1]

    m = numAngles * numSamples
    n = numAngles ** 2

    # The input radon data, represented as a vector
    # shape (m × 1)
    b = inputRadon.reshape(m)

    # The measurement matrix
    # shape: (m × n) = (numAngles * numSamples × numAngles^2)
    # A = genMeasureMat(m, n)
    # np.savetxt('mm45x50.csv', A, delimiter=',')
    A = np.loadtxt('mm45x50.csv', delimiter=',')
    # A = np.loadtxt('mm90x50.csv', delimiter=',')

    # coefficient matrix (unknowns)
    # to become output image
    # shape: (n × 1) = (numAngles^2 × 1)
    x = np.zeros(n)

    # => measurementMatrix * x has shape (m × 1) = (numAngles * numSamples × 1) = shape(b)

    # ===== Kaczmarz Iteration =====

    lambdaK = 0.5 # Relaxation parameter
    
    numBatches = 1000
    for k in range(m * numBatches):
        if k % (m+1) == 0:
            print(f"Batch {k // m + 1}/{numBatches}")
        i = k % m # index
        # x = x + ((lambdaK * (b[i] - np.inner(A[i], x)) / np.inner(A[i], A[i])) * A[i].transpose())

        alpha = np.dot(A[i], A[i].T)
        scalarb = b[i]
        scalarAidotx = np.dot(A[i], x)
        x = x + (1 / alpha)*(scalarb - scalarAidotx)*(A[i].T)

    plt.imshow(x.reshape((numAngles, numAngles)))

    plt.show()
    