from einops import rearrange
import numpy as np


# parse_shape
# reduce
# repeat


# suppose we have a set of 32 images in "h w c" format (height-width-channel)
images = [np.random.randn(30, 40, 3) for _ in range(32)]

# stack along first (batch) axis, output is a single array
rearrange(images, 'b h w c -> b h w c').shape
#(32, 30, 40, 3)

# concatenate images along height (vertical axis), 960 = 32 * 30
rearrange(images, 'b h w c -> (b h) w c').shape
#(960, 40, 3)

# concatenated images along horizontal axis, 1280 = 32 * 40
rearrange(images, 'b h w c -> h (b w) c').shape
#(30, 1280, 3)

# reordered axes to "b c h w" format for deep learning
rearrange(images, 'b h w c -> b c h w').shape
#(32, 3, 30, 40)

# flattened each image into a vector, 3600 = 30 * 40 * 3
rearrange(images, 'b h w c -> b (c h w)').shape
#(32, 3600)

# split each image into 4 smaller (top-left, top-right, bottom-left, bottom-right), 128 = 32 * 2 * 2
rearrange(images, 'b (h1 h) (w1 w) c -> (b h1 w1) h w c', h1=2, w1=2).shape
#(128, 15, 20, 3)

# space-to-depth operation
rearrange(images, 'b (h h1) (w w1) c -> b h w (c h1 w1)', h1=2, w1=2).shape
#(32, 15, 20, 12)