import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift, ifft

def applyFilter(radon):
    '''Applies ramp filter to input radon data'''

    numAngles, numSamples = radon.shape

    w = np.linspace(-np.pi, np.pi, numSamples)
    r = abs(w) # ramp filter

    filt = fftshift(r) # prepare filter

    filtRadon = np.zeros((numAngles, numSamples))
    for i in range(numAngles):
        # filter radon data row by row
        projfft = fft(radon[i]) # Fourier Transform

        filtProj = projfft * filt # Convolution Theorem!

        filtRadon[i] = np.real(ifft(filtProj)) # discard complex part of inverse FFT and add it to filtRadon
    
    return filtRadon

def backproject(radon):
    '''Project sinogram back onto the image plane'''
    
    numAngles, imageLen = radon.shape

    # Theta values each row of inputRadon corresponds to
    theta = np.linspace(0, np.pi, numAngles, endpoint=True)

    # The matrix to become the reconstructed image
    reconMatrix = np.zeros((imageLen, imageLen))

    # Shift coordinates to be centered at (0, 0)
    x = np.arange(imageLen) - (imageLen / 2)
    y = x.copy()
    X, Y = np.meshgrid(x, y)  # Set up 2D grid

    # Back Projection
    for n in range(numAngles):
        Xrot = X * np.sin(theta[n]) - Y * np.cos(theta[n]) # rotate
        XrotCor = np.round( Xrot + imageLen / 2 ) # correct back to centered coordinate system
        XrotCor = XrotCor.astype('int') # convert all entries to integers

        # Initialize projection matrix
        projMatrix = np.zeros((imageLen, imageLen))

        m0, m1 = np.where( (XrotCor >= 0) & (XrotCor < imageLen) ) # Cut off bits of the image outside of the border

        projMatrix[m0, m1] = radon[n][XrotCor[m0, m1]] # backproject in-bounds data
        reconMatrix += projMatrix # add data to reconMatrix
    
    # Normalize reconstruction matrix
    reconMatrix = (reconMatrix - reconMatrix.min()) / (reconMatrix.max() - reconMatrix.min())

    # return normalized reconstruction matrix and flip it to correct orientation
    return np.fliplr(np.rot90(reconMatrix))

def fbp(radon):
    # Filter then backproject
    filteredRadon = applyFilter(radon)
    return backproject(filteredRadon)



if __name__ == '__main__':
    # Load original image (for comparison)
    inputImg = np.loadtxt('./image/squareImage.csv', delimiter=',')
    # inputImg = np.loadtxt('./image/SmileImage.csv', delimiter=',')

    # Normalize it
    inputImg = (inputImg - inputImg.min()) / (inputImg.max() - inputImg.min())

    # Load previously computed radon data
    inputRadon = np.loadtxt('./radon/squareRadon.csv', delimiter=',')
    # inputRadon = np.loadtxt('./radon/SmileRadon.csv', delimiter=',')


    # Filter and Back Project
    recovered = fbp(inputRadon)


    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(inputImg)
    ax2.imshow(recovered)
    ax3.imshow(inputImg - recovered)

    ax1.set_title('Original')
    ax2.set_title('FBP')
    ax3.set_title('Error')

    ax1.set_axis_off()
    ax2.set_axis_off()
    ax3.set_axis_off()

    plt.show()



    # below was stuff for machine learning project
    # for i in range(991, 1001):
    #     print('Image', i)
    #     inputImg = np.loadtxt(f'./image/image{i}.csv', delimiter=',')
    #     inputRadon = np.loadtxt(f'./radon/radon{i}.csv', delimiter=',')

    #     _, (ax1, ax2, ax3) = plt.subplots(1, 3)

    #     # had to flip the images upside down because the model was accidentally trained on flipped images lol
    #     ax1.imshow(np.flipud(inputImg))
    #     ax2.imshow( np.flipud(backproject(inputRadon)) )
    #     ax3.imshow( np.flipud(fbp(inputRadon)) )

    #     ax1.set_axis_off()
    #     ax2.set_axis_off()
    #     ax3.set_axis_off()

    #     ax1.set_title('Original')
    #     ax2.set_title('Unfiltered BP')
    #     ax3.set_title('FBP')

    
    #     plt.savefig(f'fbp{i}.png', dpi=400)
