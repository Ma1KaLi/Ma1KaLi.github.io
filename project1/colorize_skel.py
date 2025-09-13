# CS180: Project 1

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import time
import skimage as sk
import skimage.io as skio
from skimage.util import img_as_ubyte

def to_float01(im):
    return sk.img_as_float32(im)

def split_bgr(stack):
    H_total, W = stack.shape[:2]
    H = H_total // 3
    stack = stack[:3*H, :]

    b = stack[0:H, :]
    g = stack[H:2*H, :]
    r = stack[2*H:3*H, :]
    return b, g, r

def _overlap_slices(h, w, dy, dx):
    # vertical
    if dy >= 0:
        ref_y = slice(dy, h)
        mov_y = slice(0,  h - dy)
    else:
        ref_y = slice(0,  h + dy)
        mov_y = slice(-dy, h)
    # horizontal
    if dx >= 0:
        ref_x = slice(dx, w)
        mov_x = slice(0,  w - dx)
    else:
        ref_x = slice(0,  w + dx)
        mov_x = slice(-dx, w)
    return ref_y, ref_x, mov_y, mov_x

def _crop_interior_same(a, b, border):
    h, w = a.shape
    r = int(h * border); c = int(w * border)
    if r == 0 and c == 0:
        return a, b
    return a[r:h-r, c:w-c], b[r:h-r, c:w-c]

def ssd_score(a, b):
    # ignoring the sqrt as minimizing the sqrt of the sum is the same as minimizing the sum itself
    d = a - b
    return float(np.sum(d * d))

def ncc_score(a, b):
    af = a.astype(np.float32).ravel()
    bf = b.astype(np.float32).ravel()
    af -= af.mean()
    bf -= bf.mean()
    na = np.linalg.norm(af)
    nb = np.linalg.norm(bf)
    return float(np.dot(af, bf) / (na * nb))

def align_bruteforce_center(moving, ref, center, win=15, metric='ncc', border=0.05):
    # ssd: minimize SSD
    # ncc: maximize NCC
    h, w = ref.shape
    cy, cx = center
    best_dy, best_dx = cy, cx
    if metric == 'ssd':
        best_score = np.inf
        better = lambda s, best: s < best
    elif metric == 'ncc':
        best_score = -np.inf
        better = lambda s, best: s > best

    for dy in range(cy - win, cy + win + 1):
        for dx in range(cx - win, cx + win + 1):
            ry, rx, my, mx = _overlap_slices(h, w, dy, dx)
            A = ref[ry, rx]
            B = moving[my, mx]
            if A.size == 0 or B.size == 0:
                continue

            A, B = _crop_interior_same(A, B, border)

            if A.size == 0 or B.size == 0:
                continue

            if metric == 'ssd':
                score = ssd_score(A, B)
            else:  # 'ncc'
                score = ncc_score(A, B)

            if better(score, best_score):
                best_score = score
                best_dy, best_dx = dy, dx

    shifted = np.roll(np.roll(moving, best_dy, axis=0), best_dx, axis=1)
    return shifted, (best_dy, best_dx), best_score

#------------------------Pyramid-Speedup------------------------
def down2x_mean(im):
    h, w = im.shape
    hh = (h // 2) * 2 
    ww = (w // 2) * 2
    im = im[:hh, :ww]
    return 0.25 * (im[0::2, 0::2] + im[1::2, 0::2] + im[0::2, 1::2] + im[1::2, 1::2])

def build_pyramid(im, min_side=100):
    pyr = [im.astype(np.float32)]
    while min(pyr[-1].shape) > min_side:
        pyr.append(down2x_mean(pyr[-1]))
    return pyr

def align_pyramid_simple(moving_full, ref_full, min_side=100, win=10):
    # Build scoring images + pyramids
    ref_s, mov_s = _crop_interior_same(ref_full, moving_full, 0.05)
    pyr_r = build_pyramid(ref_s, min_side=min_side)
    pyr_m = build_pyramid(mov_s,  min_side=min_side)
    assert len(pyr_r) == len(pyr_m)
    L = len(pyr_r) - 1

    dy, dx = 0, 0
    _, (dy, dx), _ = align_bruteforce_center(pyr_m[L], pyr_r[L], center=(0,0), win=win)

    # Go finer
    for lev in range(L-1, -1, -1):
        dy *= 2; dx *= 2
        _, (dy, dx), _ = align_bruteforce_center(pyr_m[lev], pyr_r[lev], center=(dy,dx), win=win)

    # Apply final shift
    shifted_full = np.roll(np.roll(moving_full, dy, axis=0), dx, axis=1)
    return shifted_full, (dy, dx)

#------------------------Generating Small Scale Results------------------------
if __name__ == '__main__':
    imname_small = ['monastery.jpg', 'cathedral.jpg', 'tobolsk.jpg']
    outname_small = ['out_monastery.jpg', 'out_cathedral.jpg', 'out_tobolsk.jpg']

    num_pic = 0
    for i in imname_small: 
        # load & normalize
        plate = skio.imread(i)
        plate = to_float01(plate)

        # split
        b, g, r = split_bgr(plate)

        # choose hyperparameters
        metric = 'ncc'      
        win = 15 
        border = 0.05 

        # align
        ag, disp_g, score_g = align_bruteforce_center(g, b, win=win, center = (0, 0), metric=metric, border=border)
        ar, disp_r, score_r = align_bruteforce_center(r, b, win=win, center = (0, 0), metric=metric, border=border)
        print(f"[{metric.upper()}] G→B displacement (dy,dx): {disp_g}, score: {score_g:.6f}")
        print(f"[{metric.upper()}] R→B displacement (dy,dx): {disp_r}, score: {score_r:.6f}")

        # stack as RGB
        im_out = np.dstack([ar, ag, b])
        im_out = np.clip(im_out, 0, 1).astype(np.float32)
        im_out_u8 = img_as_ubyte(im_out)
        skio.imsave(outname_small[num_pic], im_out_u8)
        print(f"Saved: {outname_small[num_pic]}")
        num_pic += 1

        skio.imshow(im_out)
        skio.show()

#------------------------Generating Pyramid Results------------------------

if __name__ == '__main__':
    imname_pyramid  = ['church.tif', 'emir.tif', 'harvesters.tif', 'icon.tif', 'italil.tif', 'lastochikino.tif', 
                       'lugano.tif', 'melons.tif', 'self_portrait.tif', 'siren.tif', 'three_generations.tif']
    outname_pyramid = ['out_church_pyr_simple.jpg', 'out_emir_pyr_simple.jpg', 'out_harvesters_pyr_simple.jpg', 
                       'out_icon_pyr_simple.jpg', 'out_italil_pyr_simple.jpg', 'out_lastochikino_pyr_simple.jpg',
                       'out_lugano_pyr_simple.jpg', 'out_melons_pyr_simple.jpg', 'out_self_portrait_pyr_simple.jpg',
                       'out_siren_pyr_simple.jpg', 'out_three_generations_pyr_simple.jpg']
    
    num_pic = 0
    for i in imname_pyramid:
        # load & normalize
        plate = skio.imread(i)
        plate = to_float01(plate)
        
        # split
        b, g, r = split_bgr(plate)

        # G -> B
        t0 = time.perf_counter()
        ag, disp_g = align_pyramid_simple(g, b, crop_frac=0.05, min_side=100, win=10)
        t1 = time.perf_counter()
        print(f"[Pyramid NCC] G→B displacement (dy,dx): {disp_g}, time: {t1 - t0:.3f}s")

        # R -> B
        t0 = time.perf_counter()
        ar, disp_r = align_pyramid_simple(r, b, crop_frac=0.05, min_side=100, win=10)
        t1 = time.perf_counter()
        print(f"[Pyramid NCC] R→B displacement (dy,dx): {disp_r}, time: {t1 - t0:.3f}s")

        # Stack and save
        im_out = np.dstack([ar, ag, b])
        im_out = np.clip(im_out, 0, 1).astype(np.float32)
        skio.imsave(outname_pyramid[num_pic], img_as_ubyte(im_out))
        print(f"Saved: {outname_pyramid[num_pic]}")
        num_pic += 1

        skio.imshow(im_out)
        skio.show()

#------------------------Special Revision for Emir.tif------------------------

def ncc_score_emir(a, b, eps=1e-8, std_min=1e-3):
    # zero-mean and normalize; guard low-texture
    a = a.astype(np.float32); b = b.astype(np.float32)
    av = a - a.mean(); bv = b - b.mean()
    if av.std() < std_min or bv.std() < std_min:
        return -1.0  # uninformative region -> very poor score
    na = np.linalg.norm(av) + eps
    nb = np.linalg.norm(bv) + eps
    return float(np.dot(av.ravel(), bv.ravel()) / (na * nb))

def grad_mag(I):
    gx = I[:, 1:] - I[:, :-1]
    gy = I[1:, :] - I[:-1, :]
    gx = np.pad(gx, ((0, 0), (0, 1)), mode='edge')
    gy = np.pad(gy, ((1, 0), (0, 0)), mode='edge')
    return np.sqrt(gx*gx + gy*gy).astype(np.float32)


def align_bruteforce_center_emir(moving, ref, center, win=15, border=0.05, expand_once=True):
    h, w = ref.shape
    cy, cx = center
    best_dy, best_dx = cy, cx
    best_score = -np.inf

    y0, y1 = cy - win, cy + win
    x0, x1 = cx - win, cx + win

    for dy in range(y0, y1 + 1):
        for dx in range(x0, x1 + 1):
            ry, rx, my, mx = _overlap_slices(h, w, dy, dx)
            A = ref[ry, rx]; B = moving[my, mx]
            if A.size == 0 or B.size == 0:
                continue
            # interior crop (fraction of current overlap)
            hh, ww = A.shape
            rr = int(hh * border); cc = int(ww * border)
            if rr*2 >= hh or cc*2 >= ww:
                continue
            A = A[rr:hh-rr, cc:ww-cc]
            B = B[rr:hh-rr, cc:ww-cc]
            if A.size == 0: 
                continue

            s = ncc_score_emir(A, B)
            if s > best_score:
                best_score = s
                best_dy, best_dx = dy, dx

    if expand_once and (
        abs(best_dy - cy) >= win or abs(best_dx - cx) >= win
    ):
        return align_bruteforce_center_emir(moving, ref, center=(cy, cx), win=win*2,
                                       border=border, expand_once=False)

    shifted = np.roll(np.roll(moving, best_dy, axis=0), best_dx, axis=1)
    return shifted, (best_dy, best_dx), best_score

def align_pyramid_simple_emir(moving_full, ref_full, crop_frac=0.05, min_side=100, win=10):
    ref_s, mov_s = _crop_interior_same(ref_full, moving_full, crop_frac)

    pyr_r = build_pyramid(ref_s, min_side=min_side)
    pyr_m = build_pyramid(mov_s,  min_side=min_side)
    L = len(pyr_r) - 1

    pyr_rg = [grad_mag(im) for im in pyr_r]
    pyr_mg = [grad_mag(im) for im in pyr_m]

    hC, wC = pyr_r[L].shape
    coarse_win = int(max(win, min(hC, wC) * 0.25))
    coarse_win = min(coarse_win, 80) 

    dy, dx = 0, 0
    _, (dy, dx), _ = align_bruteforce_center_emir(pyr_mg[L], pyr_rg[L],
                                             center=(0, 0), win=coarse_win,
                                             border=crop_frac, expand_once=True)
    # Go finer
    for lev in range(L-1, -1, -1):
        dy *= 2; dx *= 2
        cur_win = max(win, coarse_win >> (L - lev))
        _, (dy, dx), _ = align_bruteforce_center_emir(pyr_mg[lev], pyr_rg[lev],
                                                 center=(dy, dx), win=cur_win,
                                                 border=crop_frac, expand_once=True)
        refine_win = max(5, cur_win // 2)
        _, (dy, dx), _ = align_bruteforce_center_emir(pyr_m[lev], pyr_r[lev],
                                                 center=(dy, dx), win=refine_win,
                                                 border=crop_frac, expand_once=False)

    shifted_full = np.roll(np.roll(moving_full, dy, axis=0), dx, axis=1)
    return shifted_full, (dy, dx)

imname_pyramid_emir = 'emir.tif'
outname_pyramid_emir = 'out_emir_revision_pyr_simple.jpg'

plate = skio.imread(imname_pyramid_emir)
plate = to_float01(plate)
        
# split
b, g, r = split_bgr(plate)

# G -> B
t0 = time.perf_counter()
ag, disp_g = align_pyramid_simple_emir(g, b, crop_frac=0.05, min_side=100, win=10)
t1 = time.perf_counter()
print(f"[Pyramid NCC] G→B displacement (dy,dx): {disp_g}, time: {t1 - t0:.3f}s")

# R -> B
t0 = time.perf_counter()
ar, disp_r = align_pyramid_simple_emir(r, b, crop_frac=0.05, min_side=100, win=10)
t1 = time.perf_counter()
print(f"[Pyramid NCC] R→B displacement (dy,dx): {disp_r}, time: {t1 - t0:.3f}s")

# Stack and save
im_out = np.dstack([ar, ag, b])
im_out = np.clip(im_out, 0, 1).astype(np.float32)
skio.imsave(outname_pyramid_emir, img_as_ubyte(im_out))
print(f"Saved: {outname_pyramid_emir}")

skio.imshow(im_out)
skio.show()