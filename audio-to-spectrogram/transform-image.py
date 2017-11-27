from skimage import color
from skimage import io
import scipy.misc

# CONVERT RGB IMAGE TO GREYSCALE
img = color.rgb2gray(io.imread('shins.png'))
scipy.misc.imsave('shins-greyscale.png', img)




# TRIM WHITESPACE FROM IMAGE
from PIL import Image, ImageChops

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

im = Image.open("shins-greyscale.png")
im = trim(im)
im.show()
scipy.misc.imsave('shins-greyscale-trimmed.png', im)