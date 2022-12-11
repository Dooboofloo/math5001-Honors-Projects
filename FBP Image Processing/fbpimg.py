import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from skimage.transform import radon, iradon, rescale
import os, sys

def normalize(input):
    return (input - input.min()) / (input.max() - input.min())

def normalizeToUINT8(input):
    nrmlzd = (input - input.min()) / (input.max() - input.min())
    nrmlzd = (nrmlzd * 255).astype(np.uint8)
    return nrmlzd

def toColor(input, color):
    colorIndex = 0 # default to red
    if color == 'green':
        colorIndex = 1
    if color == 'blue':
        colorIndex = 2

    output = np.zeros((input.shape[0], input.shape[1], 3))
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            output[i][j][colorIndex] = input[i][j]

    return output.astype(np.uint8)
    

# Method for playing with filters
def radonFilter(r, g, b):
    # Randomly shuffle column data
    # r = r[:, np.random.permutation(r.shape[1])]
    # g = g[:, np.random.permutation(g.shape[1])]
    # b = b[:, np.random.permutation(b.shape[1])]

    # Reflect sinograms vertically
    # r = np.flipud(r)
    # g = np.flipud(g)
    # b = np.flipud(b)

    # In each channel, shift every row over by an amount from 0 to 3 pixels
    # for i in range(r.shape[0]):
    #     r[i] = np.roll(r[i], i % 4)
    # for i in range(g.shape[0]):
    #     g[i] = np.roll(g[i], i % 4)
    # for i in range(b.shape[0]):
    #     b[i] = np.roll(b[i], i % 4)

    width = 0.65
    halfMaskWidth = int(r.shape[0] * width / 2)
    startIndex = (r.shape[0] // 2) - halfMaskWidth
    endIndex = (r.shape[0] // 2) + halfMaskWidth

    r = r.T
    g = g.T
    b = b.T

    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            if j < startIndex or j > endIndex:
                r[i][j] = 0
                g[i][j] = 0
                b[i][j] = 0
    
    r = r.T
    g = g.T
    b = b.T
    


    # thresh = 0.6
    # rThreshVal = ((r.max() - r.min()) * (1 - thresh)) + r.min()
    # gThreshVal = ((g.max() - g.min()) * (1 - thresh)) + g.min()
    # bThreshVal = ((b.max() - b.min()) * (1 - thresh)) + b.min()
    # r = np.asarray(r > rThreshVal) * r
    # g = np.asarray(g > gThreshVal) * g
    # b = np.asarray(b > bThreshVal) * b


    return r, g, b



if __name__ == '__main__':
    os.system('color')

    RESOLUTION = 512

    # Options
    saveImages = True
    savePlot = True
    showPlot = True

    filterRadon = False


    # ===== READ IN IMAGE =====
    if len(sys.argv) < 2 or not os.path.exists(f'./input/{sys.argv[1]}'):
        print("\033[31mPlease input the name of an image in './input'\033[0m")
        exit()
    
    imgName = sys.argv[1] # first passed in parameter

    print(f"\nLoading image ({imgName})...")

    im = iio.imread(f'./input/{imgName}')
    im = rescale(im, (RESOLUTION / im.shape[0], RESOLUTION / im.shape[1], 1)) # Rescale to appropriate size


    # ===== DO CALCULATIONS =====

    # Split into RGB channels
    r = im[:,:,0]
    g = im[:,:,1]
    b = im[:,:,2]

    # 0.43 ≈ (sqrt(2) - 1)... plus a bit or the corners pick up artifacts
    # the entire image must be within the imaging circle
    paddingWidth = int((im.shape[1] / 2) * 0.43)


    # Pad RGB channels for radon transform (does not affect final resolution)
    paddedR = np.pad(r, paddingWidth, mode='constant')
    paddedG = np.pad(g, paddingWidth, mode='constant')
    paddedB = np.pad(b, paddingWidth, mode='constant')

    theta = np.linspace(0., 180., max(paddedR.shape), endpoint=False)

    print("\nCalculating Radon transform...")

    # Calculate radon transform for each channel
    print("  \033[31m Red...")
    radonR = radon(paddedR, theta=theta)
    print("  \033[32m Green...")
    radonG = radon(paddedG, theta=theta)
    print("  \033[34m Blue...\033[0m")
    radonB = radon(paddedB, theta=theta)


    # Experimental optional filtering step
    if filterRadon:
        radonR, radonG, radonB = radonFilter(radonR, radonG, radonB)


    # Mix sinograms back together
    radonMixed = np.zeros((radonR.shape[0], radonR.shape[1], 3))
    for i in range(radonMixed.shape[0]):
        for j in range(radonMixed.shape[1]):
            radonMixed[i][j][0] = radonR[i][j]
            radonMixed[i][j][1] = radonG[i][j]
            radonMixed[i][j][2] = radonB[i][j]
    radonMixed = normalize(radonMixed)

    
    print("\nCalculating FBP...")

    # Calculate FBP of each channel (then crop back to original size)
    print("  \033[31m Red...")
    reconR = iradon(radonR, theta=theta, filter_name='ramp')[paddingWidth:-paddingWidth, paddingWidth:-paddingWidth]
    print("  \033[32m Green...")
    reconG = iradon(radonG, theta=theta, filter_name='ramp')[paddingWidth:-paddingWidth, paddingWidth:-paddingWidth]
    print("  \033[34m Blue...\033[0m ")
    reconB = iradon(radonB, theta=theta, filter_name='ramp')[paddingWidth:-paddingWidth, paddingWidth:-paddingWidth]

    # Mix FBPs back together
    reconMixed = np.zeros((reconR.shape[0], reconR.shape[1], 3))
    for i in range(reconMixed.shape[0]):
        for j in range(reconMixed.shape[1]):
            reconMixed[i][j][0] = reconR[i][j]
            reconMixed[i][j][1] = reconG[i][j]
            reconMixed[i][j][2] = reconB[i][j]
    reconMixed = normalize(reconMixed)

    print("\nCalculating Error...")

    # Calculate error in each channel
    errorR = abs(r - reconR)
    errorG = abs(g - reconG)
    errorB = abs(b - reconB)

    # Mix errors back together
    errorMixed = np.zeros((errorR.shape[0], errorR.shape[0], 3))
    for i in range(errorMixed.shape[0]):
        for j in range(errorMixed.shape[1]):
            errorMixed[i][j][0] = errorR[i][j]
            errorMixed[i][j][1] = errorG[i][j]
            errorMixed[i][j][2] = errorB[i][j]
    errorMixed = normalize(errorMixed)

    print("\nDone!")



    # ===== Save Images =====
    if saveImages:

        # Establish working directory
        dir = f'./processed/{imgName}'
        if not os.path.exists(dir):
            os.mkdir(dir)
        
        # Establish directory for original image
        originalDir = dir + '/image'
        if not os.path.exists(originalDir):
            os.mkdir(originalDir)

        iio.imwrite(f'{originalDir}/image_red.png', toColor(normalizeToUINT8(r), 'red'))
        iio.imwrite(f'{originalDir}/image_green.png', toColor(normalizeToUINT8(g), 'green'))
        iio.imwrite(f'{originalDir}/image_blue.png', toColor(normalizeToUINT8(b), 'blue'))
        iio.imwrite(f'{originalDir}/image.png', normalizeToUINT8(im))

        # Establish directory for sinograms
        sinogramDir = dir + '/sinograms'
        if not os.path.exists(sinogramDir):
            os.mkdir(sinogramDir)
        
        iio.imwrite(f'{sinogramDir}/sinogram_red.png', toColor(normalizeToUINT8(radonR), 'red'))
        iio.imwrite(f'{sinogramDir}/sinogram_green.png', toColor(normalizeToUINT8(radonG), 'green'))
        iio.imwrite(f'{sinogramDir}/sinogram_blue.png', toColor(normalizeToUINT8(radonB), 'blue'))
        iio.imwrite(f'{sinogramDir}/sinogram.png', normalizeToUINT8(radonMixed))

        # Establish directory for reconstructions
        reconDir = dir + '/reconstructions'
        if not os.path.exists(reconDir):
            os.mkdir(reconDir)

        iio.imwrite(f'{reconDir}/reconstruction_red.png', toColor(normalizeToUINT8(reconR), 'red'))
        iio.imwrite(f'{reconDir}/reconstruction_green.png', toColor(normalizeToUINT8(reconG), 'green'))
        iio.imwrite(f'{reconDir}/reconstruction_blue.png', toColor(normalizeToUINT8(reconB), 'blue'))
        iio.imwrite(f'{reconDir}/reconstruction.png', normalizeToUINT8(reconMixed))
        
        # Establish directory for errors
        errorDir = dir + '/error'
        if not os.path.exists(errorDir):
            os.mkdir(errorDir)
        
        iio.imwrite(f'{errorDir}/error_red.png', toColor(normalizeToUINT8(errorR), 'red'))
        iio.imwrite(f'{errorDir}/error_green.png', toColor(normalizeToUINT8(errorG), 'green'))
        iio.imwrite(f'{errorDir}/error_blue.png', toColor(normalizeToUINT8(errorB), 'blue'))
        iio.imwrite(f'{errorDir}/error.png', normalizeToUINT8(errorMixed))
        

    

    # ===== DISPLAY =====

    # Colormaps for displaying
    rcm = LinearSegmentedColormap.from_list('rcm', [(0, 0, 0), (1, 0, 0)], N=len(np.unique(r)))
    gcm = LinearSegmentedColormap.from_list('gcm', [(0, 0, 0), (0, 1, 0)], N=len(np.unique(g)))
    bcm = LinearSegmentedColormap.from_list('bcm', [(0, 0, 0), (0, 0, 1)], N=len(np.unique(b)))

    Rrcm = LinearSegmentedColormap.from_list('Rrcm', [(0, 0, 0), (1, 0, 0)], N=len(np.unique(radonR)))
    Rgcm = LinearSegmentedColormap.from_list('Rgcm', [(0, 0, 0), (0, 1, 0)], N=len(np.unique(radonG)))
    Rbcm = LinearSegmentedColormap.from_list('Rbcm', [(0, 0, 0), (0, 0, 1)], N=len(np.unique(radonB)))

    FBPrcm = LinearSegmentedColormap.from_list('FBPrcm', [(0, 0, 0), (1, 0, 0)], N=len(np.unique(reconR)))
    FBPgcm = LinearSegmentedColormap.from_list('FBPgcm', [(0, 0, 0), (0, 1, 0)], N=len(np.unique(reconG)))
    FBPbcm = LinearSegmentedColormap.from_list('FBPbcm', [(0, 0, 0), (0, 0, 1)], N=len(np.unique(reconB)))

    ERRrcm = LinearSegmentedColormap.from_list('ERRrcm', [(0, 0, 0), (1, 0, 0)], N=len(np.unique(errorR)))
    ERRgcm = LinearSegmentedColormap.from_list('ERRgcm', [(0, 0, 0), (0, 1, 0)], N=len(np.unique(errorG)))
    ERRbcm = LinearSegmentedColormap.from_list('ERRbcm', [(0, 0, 0), (0, 0, 1)], N=len(np.unique(errorB)))

    # Setting up plot
    fig, ((axr,axg,axb,axim), (axRr,axRg,axRb,axRim), (axFBPr, axFBPg, axFBPb, axFBPim), (axERRr, axERRg, axERRb, axERRim)) = plt.subplots(4,4, figsize=(8, 8))

    fig.suptitle(f'RGB Image Reconstruction by FBP ({imgName})')

    axr.set_title('Red Channel')
    axg.set_title('Green Channel')
    axb.set_title('Blue Channel')
    axim.set_title('Mixed')

    axr.text(-r.shape[0] / 10, r.shape[1] / 2, 'Original', fontsize='large', rotation='vertical', horizontalalignment='center', verticalalignment='center')
    axRr.text(-radonR.shape[0] / 10, radonR.shape[1] / 2, 'Sinograms', fontsize='large', rotation='vertical', horizontalalignment='center', verticalalignment='center')
    axFBPr.text(-reconR.shape[0] / 10, reconR.shape[1] / 2, 'Reconstructions', fontsize='large', rotation='vertical', horizontalalignment='center', verticalalignment='center')
    axERRr.text(-errorR.shape[0] / 10, errorR.shape[1] / 2, 'Error', fontsize='large', rotation='vertical', horizontalalignment='center', verticalalignment='center')

    axr.set_axis_off()
    axg.set_axis_off()
    axb.set_axis_off()
    axim.set_axis_off()

    axRr.set_axis_off()
    axRg.set_axis_off()
    axRb.set_axis_off()
    axRim.set_axis_off()

    axFBPr.set_axis_off()
    axFBPg.set_axis_off()
    axFBPb.set_axis_off()
    axFBPim.set_axis_off()

    axERRr.set_axis_off()
    axERRg.set_axis_off()
    axERRb.set_axis_off()
    axERRim.set_axis_off()
    

    # Plot all the stuff!

    # Plot image channels
    axr.imshow(r, cmap=rcm)
    axg.imshow(g, cmap=gcm)
    axb.imshow(b, cmap=bcm)
    axim.imshow(im)

    # Plot radon channels
    axRr.imshow(radonR, cmap=Rrcm)
    axRg.imshow(radonG, cmap=Rgcm)
    axRb.imshow(radonB, cmap=Rbcm)
    axRim.imshow(radonMixed)

    # Plot FBP channels
    axFBPr.imshow(reconR, cmap=FBPrcm)
    axFBPg.imshow(reconG, cmap=FBPgcm)
    axFBPb.imshow(reconB, cmap=FBPbcm)
    axFBPim.imshow(reconMixed)

    # Plot error channels
    axERRr.imshow(errorR, cmap=ERRrcm)
    axERRg.imshow(errorG, cmap=ERRgcm)
    axERRb.imshow(errorB, cmap=ERRbcm)
    axERRim.imshow(errorMixed)

    plt.subplots_adjust(0.1, 0.012, 0.9, 0.9, 0.036, 0)

    # Save plot
    if savePlot:
        dir = f'./processed/{imgName}'
        if not os.path.exists(dir):
            os.mkdir(dir)
        
        plt.savefig(f'{dir}/{imgName}_processed.svg', format='svg')
    
    if showPlot:
        # Show the plot
        plt.show()

    
