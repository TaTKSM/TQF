import numpy as np
from numba import jit
from typing import Union
from pathlib import Path




@jit(nopython=True)
def create_r_slice(a: float, size: int, seed: int) -> np.ndarray:
    """
    Generate a noisy rectangle dataset at a specified input feature a.
    """
    assert size >= 1
    assert 0. <= a <= 1.
    np.random.seed(seed)
    dat = np.zeros((size, 2), dtype=np.float32)
    
    # edge lengths: up & down = 2 * (1 - a), left & right = 2 * a
    p_top    = 0.5 * (1.0 - a)
    p_bottom = 0.5 * (1.0 - a)
    p_right  = 0.5 * a
    # p_left   = 0.5 * a

    # edge is selected according to cumulative probability
    c1 = p_top
    c2 = c1 + p_bottom
    c3 = c2 + p_right
    # c4 = c3 + p_left (= 1.0)

    for i in range(size):

        u = np.random.uniform(0.0, 1.0)
        t = np.random.uniform(-1.0, 1.0)

        if u < c1:
            # upper edge: y = +a, x in [-(1-a), +(1-a)]
            x = t * (1.0 - a)
            y = + a

        elif u < c2:
            # lower edge: y = -a
            x = t * (1.0 - a)
            y = - a

        elif u < c3:
            # right edge: x = +(1-a), y in [-a, +a]
            x = + (1.0 - a)
            y = t * a

        else:
            # left edge: x = -(1-a)
            x = - (1.0 - a)
            y = t * a

        dat[i, 0] = x
        dat[i, 1] = y
    
    sigma = 0.08
    dat += sigma * np.random.randn(size, 2) 
    th = 0.25 * np.pi
    O = np.array([[np.cos(th), np.sin(th)], [- np.sin(th), np.cos(th)]], dtype=np.float32)
    dat = dat @ O
    
    return dat



@jit(nopython=True)
def sig(x: np.ndarray) -> np.ndarray:

    def sigmoid(x):
        return 0.5 * (1. + np.tanh(x / 2.))

    if x.ndim == 1:
        xc = np.mean(x)
    else:
        xc = np.sum(x, axis=1) / x.shape[1]

    Z = 1.5
    
    return sigmoid(Z * xc)



@jit(nopython=True)
def create_r(size: int, seed: int, num_features: int) -> tuple:
    """
    Generate a noisy rectangle dataset.
    """
    assert size >= 1
    assert num_features >= 1
    np.random.seed(seed)
    xs = np.random.uniform(low=-2, high=2, size=(size, num_features))
    a_s = sig(xs)
    dat = np.zeros((size, 2))

    for i, a in enumerate(a_s):
        dat[i] = create_r_slice(size=1, seed=seed + i, a=a).ravel()

    return xs.astype(np.float32), dat.astype(np.float32)



def create_character_data(char: str) -> np.ndarray:
    """
    Create a point cloud dataset that represents a given character.
    """
    from PIL import Image, ImageDraw, ImageFont

    # image size
    IMSIZE = 200
    img_size = (IMSIZE, IMSIZE)
    img = Image.new("L", img_size, "white")
    draw = ImageDraw.Draw(img)

    # このファイル (gendata.py) と同じフォルダにある arial.ttf を使う
    font_path = Path(__file__).with_name("arial.ttf")
    font = ImageFont.truetype(str(font_path), 120)

    # location adjustment
    bbox = draw.textbbox((0, 0), char, font=font)
    x = (img_size[0] - (bbox[2] - bbox[0])) // 2
    y = (img_size[1] - (bbox[3] - bbox[1])) // 2
    draw.text((x, y), char, font=font, fill=0)    # 0=black, 255=white

    # convert to numpy array (white=255, black=0)
    array_img = np.array(img)
    array_img = np.array(array_img <= 0, dtype=int)

    dat = []
    for i in range(IMSIZE):
        for j in range(IMSIZE):
            if array_img[i, j] > 0:
                dat.append([j, -i])
                
    dat = np.array(dat, dtype=np.float32)
    from sklearn.preprocessing import StandardScaler
    dat = StandardScaler().fit_transform(dat)

    return dat.astype(np.float32)



def create_c_slice(a: float, char_data: np.ndarray) -> np.ndarray:
    """
    Create a character dataset at a specified rotation angle a.
    """
    assert 0. <= a <= 1.
    assert char_data.ndim == 2 and char_data.shape[1] == 2

    DAT = char_data + np.array([0., 1.], dtype=np.float32)
    th = np.pi * 0.5 * a
    O = np.array([[np.cos(th), - np.sin(th)], [np.sin(th), np.cos(th)]], dtype=np.float32)
    dat = DAT @ O

    return dat



def create_c(size: int, seed: int, dim_used: int, dim_useless: int, char: str) -> tuple:
    """
    Create a character datast with some informationless (noise) features.
    """
    assert size >= 1
    assert dim_used >= 1 and dim_useless >= 0

    RNG = np.random.default_rng(seed)
    xs = RNG.uniform(low=-2, high=2, size=(size, dim_used)).astype(np.float32)
    a_s = sig(xs)
    cdat = create_character_data(char=char)
    dat = np.array([[RNG.permutation(create_c_slice(a=a, char_data=cdat))[0]] for a in a_s], dtype=np.float32).squeeze()
    assert np.shape(dat) == (size, 2), np.shape(dat)

    if dim_useless > 0:
        xs_noise = RNG.uniform(low=-2, high=2, size=(size, dim_useless)).astype(np.float32)
        xs = np.c_[xs, xs_noise]
    
    assert np.shape(xs) == (size, dim_used + dim_useless), np.shape(xs)

    return xs, dat

