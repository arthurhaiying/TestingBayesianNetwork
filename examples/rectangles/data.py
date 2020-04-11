import math
import random
import numpy as np
import train.data as data
import utils.visualize as visualize


""" generating noisy rectangle images with labels (tall, wide) """

# Returns labeled data for noisy rectangle images.
# For each clean rectangle image, constructs a number of noisy images (noisy_image_count),
# with each noisy image having up to noise_count noisy pixels.
# Noise is not added to the pixels that border the clean rectangle.
# Data is returned in column-based format.
def get(size,output,noisy_image_count=0,noise_count=0):
    assert output in ('label','height','width','row','col')
    image     = np.empty([size,size])
    evidence  = []
    marginals = []
    for (image,label) in __clean_rectangles(image,output):
        #visualize.image(image,label)
        lambdas  = image2lambdas(image)
        marginal = __label2distribution(label,size,output)
        evidence.append(lambdas)
        marginals.append(marginal)
        for image in __noisy_rectangles(image,noisy_image_count,noise_count):
            #visualize.image(image,label)
            lambdas = image2lambdas(image)
            evidence.append(lambdas)
            marginals.append(marginal)
    evidence  = data.evd_row2col(evidence)
    marginals = np.array(marginals) 
    return (evidence, marginals)
    

# Generates all images containing clean rectangles (no noise), together with their
# labels (tall, wide). On pixels contain 1., off pixels contain 0.
# Marks boundary pixels of the rectangle with grey .5 values (to avoid adding noise to).
def __clean_rectangles(image,output):
    size = image.shape[0]
    for r_ref in range(size):
        max_height = size-r_ref+1
        for c_ref in range(size):
            max_width = size-c_ref+1
            for width in range(1,max_width):
                for height in range(1,max_height):
                    # generate labeled image
                    # set label
                    if output == 'label':
                        if   width > height: label = 'wide'
                        elif width < height: label = 'tall'
                        else: continue # ambiguous
                    elif output == 'height': label = height
                    elif output == 'width' : label = width
                    elif output == 'row'   : label = r_ref
                    elif output == 'col'   : label = c_ref
                    else: assert False
                    # initialize image
                    image.fill(0.)
                    # set pixels
                    for r in range(r_ref,r_ref+height):
                        for c in range(c_ref,c_ref+width):
                            image[r,c] = 1.
                    # gray out rectangle boundaries (to avoid adding noise)
                    rs = r_ref-1
                    re = r_ref+height
                    cs = c_ref-1
                    ce = c_ref+width
                    __gray_out_col(image,cs,rs,re)
                    __gray_out_col(image,ce,rs,re)
                    __gray_out_row(image,rs,cs,ce)
                    __gray_out_row(image,re,cs,ce)
    
                    yield (image,label)
 
# Adds noise to an image that contains a clean rectangle.
# Pixels in the image contain 1. (on), 0. (off) or .5 (grey).
# Does not add noise to gray pixels.
# Does not add noise to more than 1/2 off pixels.
def __noisy_rectangles(image,noisy_image_count,noise_count):
    if noisy_image_count == 0: return
    size            = image.shape[0]
    consider_pixels = [] # candidates for adding noise to
    on_pixel_count  = 0
    for r in range(size):
        for c in range(size):
            if image[r,c] == 1.: on_pixel_count += 1
            if image[r,c] == 0.: consider_pixels.append((r,c))
    # adjust noise count, must be less than:
    # --number of on-pixels
    # --half number of off-pixels
    noise_count = min(noise_count,on_pixel_count-1,2*len(consider_pixels)//3)
    if noise_count == 0: return # otherwise, will return clean rectangles
    noisy_pixels = []
    for _ in range(noisy_image_count):
        for _ in range(noise_count):
            (r,c) = random.choice(consider_pixels)
            image[r,c] = 1. # adding noise
            noisy_pixels.append((r,c))
        yield image
        while noisy_pixels: # clear noise
            (r,c) = noisy_pixels.pop()
            image[r,c] = 0. 
                        
# grays out row r, in columns cs..ce
def __gray_out_row(image,r,cs,ce):
    size = image.shape[0]
    if r >= 0 and r < size:
        for c in range(max(cs,0),min(ce+1,size)):
            image[r,c] = .5

# grays out column c, in rows rs..re
def __gray_out_col(image,c,rs,re):
    size = image.shape[0]
    if c >= 0 and c < size:
        for r in range(max(rs,0),min(re+1,size)):
            image[r,c] = .5


# constants    
__dist      = {'tall':np.array([1.,0.]), 'wide':np.array([0.,1.])}
__on_pixel  = np.array([1.,0.]) # lambda for on pixel
__off_pixel = np.array([0.,1.]) # lambda for off pixel

def __label2distribution(label,size,output):
    if output == 'label': return __dist[label]
    distribution = np.zeros([size])
    if output in ('height','width'):
        distribution[label-1] = 1.
    else:
        assert output in ('row','col')
        distribution[label] = 1.
    return distribution

# must generate lambdas r, then c to match order of evidence nodes
def image2lambdas(image):
    # image contains gray pixels (.5)
    size    = image.shape[0]
    lambdas = []
    for r in range(size):
        for c in range(size):
            pixel = __on_pixel if image[r,c] == 1. else __off_pixel
            lambdas.append(pixel)
    return lambdas