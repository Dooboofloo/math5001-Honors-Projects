import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from skimage.transform import radon, iradon, rescale
import os

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
    


# All images in ./input/ are 3024x3024

if __name__ == '__main__':
    os.system('color')

    RESOLUTION = 128
    saveImages = False
    showPlot = True
    savePlot = True


    # ===== READ IN IMAGE ===== 
    imgName = 'mcnutt.jpg'

    print(f"\nLoading image ({imgName})...")

    im = iio.imread(f'./input/{imgName}')
    im = rescale(im, (RESOLUTION / im.shape[0], RESOLUTION / im.shape[1], 1)) # Rescale to appropriate size


    # ===== DO CALCULATIONS =====

    # Split into RGB channels
    r = im[:,:,0]
    g = im[:,:,1]
    b = im[:,:,2]

    # 0.43 ≈ sqrt(2) - 1... plus a bit
    # the entire image must be within the imaging circle
    paddingWidth = int((im.shape[1] / 2) * 0.43)


    # Pad RGB channels for radon transform (does not effect final resolution)
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

    if showPlot:

        # Colormaps for displaying
        rcm = LinearSegmentedColormap.from_list('rcm', [(0, 0, 0), (1, 0, 0)], N=len(np.unique(r)))
        gcm = LinearSegmentedColormap.from_list('gcm', [(0, 0, 0), (0, 1, 0)], N=len(np.unique(g)))
        bcm = LinearSegmentedColormap.from_list('bcm', [(0, 0, 0), (0, 0, 1)], N=len(np.unique(b)))

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
        axRr.imshow(radonR, cmap=rcm)
        axRg.imshow(radonG, cmap=gcm)
        axRb.imshow(radonB, cmap=bcm)
        axRim.imshow(radonMixed)

        # Plot FBP channels
        axFBPr.imshow(reconR, cmap=rcm)
        axFBPg.imshow(reconG, cmap=gcm)
        axFBPb.imshow(reconB, cmap=bcm)
        axFBPim.imshow(reconMixed)

        # Plot error channels
        axERRr.imshow(errorR, cmap=rcm)
        axERRg.imshow(errorG, cmap=gcm)
        axERRb.imshow(errorB, cmap=bcm)
        axERRim.imshow(errorMixed)

        plt.subplots_adjust(0.1, 0.012, 0.9, 0.9, 0.036, 0)

        # Save plot
        if savePlot:
            dir = f'./processed/{imgName}'
            if not os.path.exists(dir):
                os.mkdir(dir)
            
            plt.savefig(f'{dir}/{imgName}_processed.png')

        # Show the plot
        plt.show()

    
