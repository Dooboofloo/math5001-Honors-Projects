import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift, ifft

def arange2(start, stop=None, step=1):
    """#Modified version of numpy.arange which corrects error associated with non-integer step size"""
    if stop == None:
        a = np.arange(start)
    else: 
        a = np.arange(start, stop, step)
        if a[-1] > stop-step:   
            a = np.delete(a, -1)
    return a

def applyFilter(radon):
    # Filter approaches ramp filter as a becomes smaller
    a = 0.001

    numAngles, numSamples = radon.shape
    step = 2 * np.pi / numSamples

    # should probably modify this to be a linspace and avoid unnecessary arange2 function
    w = arange2(-np.pi, np.pi, step)
    if len(w) < numSamples:
        w = np.concatenate([w, [w[-1]+step]])
    
    rn1 = abs(2/a*np.sin(a*w/2))
    rn2 = np.sin(a*w/2)/(a*w/2)
    r = rn1*(rn2)**2

    filt = fftshift(r) # filt should be approximately abs(w)
    filtRadon = np.zeros((numAngles, numSamples))

    for i in range(numAngles):
        projfft = fft(radon[i])
        filtProj = projfft*filt # filter radon data
        filtRadon[i] = np.real(ifft(filtProj)) # discard complex part and add it to filtRadon
    
    return filtRadon

def backproject(radon):

    # Theta values each row of inputRadon corresponds to
    theta = np.linspace(0, 180, np.shape(inputRadon)[0], endpoint=False)

    imageLen = radon.shape[1]
    reconMatrix = np.zeros((imageLen, imageLen))

    x = np.arange(imageLen) - (imageLen / 2) # shift coordinates to be centered at (0, 0)
    y = x.copy()
    X, Y = np.meshgrid(x, y)

    # convert theta from degrees to rads
    theta = theta * np.pi / 180
    numAngles = len(theta)

    for n in range(numAngles):
        Xrot = X * np.sin(theta[n]) - Y * np.cos(theta[n]) # rotate
        XrotCor = np.round( Xrot + imageLen / 2 ) # correct back to centered coordinate system
        XrotCor = XrotCor.astype('int') # convert all entries to integers

        projMatrix = np.zeros((imageLen, imageLen))

        m0, m1 = np.where( (XrotCor >= 0) & (XrotCor < imageLen) ) # Cut off bits of the image outside of the border
        s = radon[n]

        projMatrix[m0, m1] = s[XrotCor[m0, m1]] # backproject in-bounds data
        reconMatrix += projMatrix # add data to reconMatrix
    
    return np.fliplr(np.rot90(reconMatrix))

def fbp(radon):
    filteredRadon = applyFilter(radon)
    return backproject(filteredRadon)



if __name__ == '__main__':
    # Load original image (for comparison)
    # inputImg = np.loadtxt('./image/image1.csv', delimiter=',')
    inputImg = np.loadtxt('./image/image1.csv', delimiter=',')
    # Load previously computed radon data
    # inputRadon = np.loadtxt('./radon/radon1.csv', delimiter=',')
    inputRadon = np.loadtxt('./radon/SheppLoganRadon.csv', delimiter=',')
    
    _, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(inputImg)
    ax2.imshow(fbp(inputRadon))

    ax1.set_title('Original')
    ax2.set_title('FBP')
    
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

    #     ax1.set_title('Original')
    #     ax2.set_title('Unfiltered BP')
    #     ax3.set_title('FBP')

    
    #     plt.savefig(f'fbp{i}.png', dpi=400)
