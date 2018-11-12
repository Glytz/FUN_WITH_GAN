import os
import imageio
import re
import argparse
def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str)  # rows of the image
parser.add_argument("--img_duration", type=float, default=0.1)  # columns of the images
parser.add_argument("--output", type=str, default="giffmaker.gif")  # columns of the images
param = parser.parse_args()

png_dir = param.input
dir_list = os.listdir(png_dir)
sort_nicely(dir_list)
images = []
for file_name in dir_list:
    if file_name.endswith('.png'):
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))
imageio.mimsave(param.output, images, duration=param.img_duration)