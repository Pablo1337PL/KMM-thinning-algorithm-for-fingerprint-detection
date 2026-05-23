import numpy as np
from PIL import Image

class FingerprintProcessor:
    """
    A comprehensive toolkit for fingerprint image processing, enhancement, 
    skeletonization (KMM & Morphological), and minutiae extraction.
    """
    
    def __init__(self):
        """
        Initialize static lookup tables (LUTs) for the KMM algorithm upon class instantiation.
        This saves significant computation time by preventing recalculation on every image.
        """
        # 1. Deletion LUT for Phase C
        self._lut = np.zeros(256, dtype=bool)
        self._lut[[3, 5, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31, 48,
                   52, 53, 54, 55, 56, 60, 61, 62, 63, 65, 67, 69, 71, 77, 79, 80,
                   81, 83, 84, 85, 86, 87, 88, 89, 91, 92, 93, 94, 95, 97, 99, 101,
                   103, 109, 111, 112, 113, 115, 116, 117, 118, 119, 120, 121, 123, 124, 125, 126,
                   127, 131, 133, 135, 141, 143, 149, 151, 157, 159, 181, 183, 189, 191, 192, 193,
                   195, 197, 199, 205, 207, 208, 209, 211, 212, 213, 214, 215, 216, 217, 219, 220,
                   221, 222, 223, 224, 225, 227, 229, 231, 237, 239, 240, 241, 243, 244, 245, 246,
                   247, 248, 249, 251, 252, 253, 254, 255]] = True

        # 2. Connected Components LUT for Phase B
        _adjacent_neighbors = [[1, 3], [0, 2, 3, 4], [1, 4], [0, 1, 5, 6], 
                               [1, 2, 6, 7], [3, 6], [3, 4, 5, 7], [4, 6]]
        self._component_lut = np.zeros(256, dtype=np.uint8)
        
        for pattern in range(256):
            bits = [i for i in range(8) if (pattern >> i) & 1]
            if not bits: 
                continue
            
            bit_set, visited, max_size = set(bits), set(), 0
            for start_node in bits:
                if start_node in visited: 
                    continue
                queue, size = [start_node], 0
                visited.add(start_node)
                
                while queue:
                    node = queue.pop(0)
                    size += 1
                    for neighbor in _adjacent_neighbors[node]:
                        if neighbor in bit_set and neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                max_size = max(max_size, size)
            self._component_lut[pattern] = max_size

    
    # =========================================================================
    # 1. Morphological & Utility Operations
    # =========================================================================

    def _get_structuring_element(self, size: int, shape: str) -> np.ndarray:
        """Generates a morphological structuring element of a given size and shape."""
        size = max(3, size | 1) 
        se = np.zeros((size, size), dtype=np.uint8)
        c = size // 2
        
        if shape == 'ellipse':
            Y, X = np.ogrid[:size, :size]
            se[(Y - c) ** 2 + (X - c) ** 2 <= c ** 2] = 1
        elif shape == 'cross':
            se[c, :] = se[:, c] = 1
        elif shape == 'square':
            se[:] = 1
            
        return se

    def _to_gray(self, image) -> np.ndarray:
        """Converts an input image (PIL or Numpy) to a 2D grayscale numpy array."""
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                return np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
            return image
        return np.array(image.convert('L'))

    def _erode(self, img: np.ndarray, se: np.ndarray) -> np.ndarray:
        """Performs morphological erosion (local minimum) on the image."""
        pad = se.shape[0] // 2
        p = np.pad(img, pad, mode='edge')
        out = np.full_like(img, 255)
        for di, dj in np.argwhere(se):
            out = np.minimum(out, p[di:di + img.shape[0], dj:dj + img.shape[1]])
        return out

    def _dilate(self, img: np.ndarray, se: np.ndarray) -> np.ndarray:
        """Performs morphological dilation (local maximum) on the image."""
        pad = se.shape[0] // 2
        p = np.pad(img, pad, mode='edge')
        out = np.zeros_like(img)
        for di, dj in np.argwhere(se):
            out = np.maximum(out, p[di:di + img.shape[0], dj:dj + img.shape[1]])
        return out

    def _open(self, img: np.ndarray, se: np.ndarray) -> np.ndarray:
        """Performs morphological opening (erosion followed by dilation)."""
        return self._dilate(self._erode(img, se), se)

    def _close(self, img: np.ndarray, se: np.ndarray) -> np.ndarray:
        """Performs morphological closing (dilation followed by erosion)."""
        return self._erode(self._dilate(img, se), se)

    def _convolve2d(self, img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Applies a 2D linear convolution using Fast Fourier Transform (FFT)."""
        sh = tuple(int(2 ** np.ceil(np.log2(s + k - 1))) for s, k in zip(img.shape, kernel.shape))
        out = np.real(np.fft.ifft2(
            np.fft.fft2(img.astype(np.float64), sh) *
            np.fft.fft2(kernel.astype(np.float64), sh)
        ))
        ph, pw = kernel.shape[0] // 2, kernel.shape[1] // 2
        return out[ph: ph + img.shape[0], pw: pw + img.shape[1]]


    # =========================================================================
    # 2. Image Enhancement & Gabor Filtering
    # =========================================================================

    def normalize(self, gray: np.ndarray, M0: float = 100.0, V0: float = 100.0) -> np.ndarray:
        """Normalizes the image to a desired global mean (M0) and variance (V0)."""
        f = gray.astype(np.float64)
        M = f.mean()
        V = f.var()
        diff = f - M
        norm = np.where(
            diff >= 0,
            M0 + np.sqrt(V0 * (diff ** 2) / (V + 1e-8)),
            M0 - np.sqrt(V0 * (diff ** 2) / (V + 1e-8)),
        )
        return np.clip(norm, 0, 255).astype(np.uint8)


    def _get_orientation_map(self, norm: np.ndarray, block: int = 16, smooth_iter: int = 2) -> np.ndarray:
        f  = norm.astype(np.float64)
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
        gx = self._convolve2d(f, Kx)
        gy = self._convolve2d(f, Kx.T)

        Vx = 2.0 * gx * gy
        Vy = gx ** 2 - gy ** 2

        H, W = f.shape
        bVx  = np.zeros((H, W))
        bVy  = np.zeros((H, W))

        for r in range(0, H, block):
            for c in range(0, W, block):
                bVx[r:r + block, c:c + block] = Vx[r:r + block, c:c + block].mean()
                bVy[r:r + block, c:c + block] = Vy[r:r + block, c:c + block].mean()

        for _ in range(smooth_iter):
            bVx = self._convolve2d(bVx, np.ones((block, block), float) / (block * block))
            bVy = self._convolve2d(bVy, np.ones((block, block), float) / (block * block))

        return 0.5 * np.arctan2(bVx, bVy) + np.pi / 2.0


    def _gabor_kernel(self, size: int, theta: float, freq: float, sigma_perp: float, sigma_par: float) -> np.ndarray:
        """Generates a directional Gabor kernel based on spatial frequency and orientation."""
        h = size // 2
        x, y = np.meshgrid(np.arange(-h, h + 1), np.arange(-h, h + 1))
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        xr = -x * sin_t + y * cos_t
        yr =  x * cos_t + y * sin_t
        
        kernel = (np.exp(-0.5 * (xr ** 2 / sigma_perp ** 2 + yr ** 2 / sigma_par ** 2)) 
                  * np.cos(2.0 * np.pi * freq * xr))
        kernel -= kernel.mean()
        return kernel

    def gabor_enhance(self, gray: np.ndarray, n_angles: int, freq: float, ksize: int, sigma_perp: float, sigma_par: float) -> np.ndarray:
        """Enhances fingerprint ridges using a directional Gabor filter bank."""
        f = gray.astype(np.float64)
        f = (f.max() - f) / (f.max() - f.min() + 1e-8)

        angles = np.linspace(0, np.pi, n_angles, endpoint=False)
        kernels = [self._gabor_kernel(ksize, t, freq, sigma_perp, sigma_par) for t in angles]
        resps = np.stack([self._convolve2d(f, k) for k in kernels])

        orientation = self._get_orientation_map(gray)
        idx = np.round((orientation % np.pi) / np.pi * n_angles).astype(int) % n_angles

        rows = np.arange(gray.shape[0])[:, None]
        cols = np.arange(gray.shape[1])[None, :]
        picked = resps[idx, rows, cols]

        picked -= picked.min()
        picked /= (picked.max() + 1e-8)
        return picked

    def _compute_roi(self, gray: np.ndarray, block: int = 16, threshold_ratio: float = 0.2, morph_size: int = 33) -> np.ndarray:
        """Calculates the Region of Interest (ROI) mask based on local variance."""
        H, W = gray.shape
        roi_mask = np.zeros((H, W), dtype=np.uint8)
        f = gray.astype(float)
        
        global_std = f.std()
        threshold = threshold_ratio * global_std

        for r in range(0, H, block):
            for c in range(0, W, block):
                patch = f[r:r + block, c:c + block]
                if patch.size > 0 and patch.std() > threshold:
                    roi_mask[r:r + block, c:c + block] = 255

        morph_size = max(5, morph_size | 1)
        se = self._get_structuring_element(morph_size, 'ellipse')
        
        roi_mask = self._close(roi_mask, se)
        roi_mask = self._open(roi_mask, se)
        
        return roi_mask > 0


    # =========================================================================
    # 3. Main Pre-processing Pipeline
    # =========================================================================

    def preprocess_fingerprint(self,
                               image,
                               threshold: int = 124,
                               freq: float = 0.1,
                               n_angles: int = 16,
                               ksize: int = 17,
                               sigma_perp: float = 2.0,
                               sigma_par: float = 2.5,
                               morph_size: int = 3, 
                               roi_block: int = 8,
                               threshold_ratio: float = 0.2) -> np.ndarray:
        """
        Executes the full preprocessing pipeline: Normalization -> Gabor Enhancement 
        -> Binarization -> ROI Masking.
        """
        ksize = max(5, ksize | 1) 
        
        gray = self._to_gray(image)
        normalized = self.normalize(gray)

        enhanced = self.gabor_enhance(normalized, n_angles=n_angles, freq=freq, 
                                      ksize=ksize, sigma_perp=sigma_perp, sigma_par=sigma_par)
        enhanced_u8 = (enhanced * 255).astype(np.uint8)

        if morph_size > 1:
            morph_size = max(3, morph_size | 1)
            se = self._get_structuring_element(morph_size, 'ellipse')
            enhanced_u8 = self._open(self._close(enhanced_u8, se), se)

        binary = np.where(enhanced_u8 >= threshold, np.uint8(0), np.uint8(255))

        roi_mask = self._compute_roi(normalized, block=roi_block, threshold_ratio=threshold_ratio, 
                                     morph_size=max(15, roi_block * 4 | 1))
        
        binary[~roi_mask] = 255
        return binary


    # =========================================================================
    # 4. Skeletonization Methods
    # =========================================================================

    def apply_kmm(self, binary: np.ndarray) -> np.ndarray:
        """
        Applies the KMM (Saeed, Rybnik, Tabedzki, Adamski) skeletonization algorithm 
        to reduce ridge thickness to a single pixel.
        """
        img = (binary == 0).astype(np.uint8)
        _pad_kwargs = dict(mode='constant', constant_values=0)

        while True:
            previous_img = img.copy()
            p = np.pad(img, 1, **_pad_kwargs)

            # Phase A: Border and corner classification
            edges = ((p[:-2, 1:-1] == 0) | (p[2:, 1:-1] == 0) | 
                     (p[1:-1, :-2] == 0) | (p[1:-1, 2:] == 0))
            img[(img == 1) & edges] = 2

            corners = ((p[:-2, :-2] == 0) | (p[:-2, 2:] == 0) | 
                       (p[2:, :-2] == 0) | (p[2:, 2:] == 0))
            img[(img == 1) & corners] = 3

            # Phase B: Deletion of specific contour pixels
            contour = (img == 2) | (img == 3)
            pc = np.pad(contour.astype(np.uint8), 1, **_pad_kwargs)
            pattern_b = (pc[:-2, :-2]       | pc[:-2, 1:-1] * 2  | pc[:-2, 2:] * 4 |
                         pc[1:-1, :-2] * 8  | pc[1:-1, 2:] * 16  | pc[2:, :-2] * 32 |
                         pc[2:, 1:-1] * 64  | pc[2:, 2:] * 128).astype(np.uint8)
                         
            max_component = self._component_lut[pattern_b]
            img[contour & (max_component >= 2) & (max_component <= 4)] = 0

            # Phase C: Sequential deletion via LUT mapping
            H, W = img.shape
            for N in (2, 3):
                for r, c in np.argwhere(img == N):
                    w = 0
                    if r > 0:
                        if c > 0 and img[r-1, c-1]: w += 128
                        if           img[r-1, c  ]: w += 1
                        if c < W-1 and img[r-1, c+1]: w += 2
                    if c > 0 and img[r, c-1]: w += 64
                    if c < W-1 and img[r, c+1]: w += 4
                    if r < H-1:
                        if c > 0 and img[r+1, c-1]: w += 32
                        if           img[r+1, c  ]: w += 16
                        if c < W-1 and img[r+1, c+1]: w += 8
                    img[r, c] = 0 if self._lut[w] else 1

            # Check convergence
            if np.array_equal(img, previous_img):
                break

        return np.where(img == 1, np.uint8(0), np.uint8(255))

    def skeletonize(self, binary: np.ndarray) -> np.ndarray:
        """
        Applies mathematical morphology (erosion & dilation) to compute 
        the image skeleton (Zhang-Suen alternative).
        """
        current = np.where(binary == 0, np.uint8(255), np.uint8(0))
        current = np.pad(current, 1, mode='constant', constant_values=0)
        se = self._get_structuring_element(3, 'cross')
        skeleton = np.zeros_like(current)

        while current.max() > 0:
            eroded = self._erode(current, se)
            opened = self._dilate(eroded, se)
            residual = np.clip(current.astype(np.int16) - opened.astype(np.int16), 0, 255).astype(np.uint8)
            skeleton = np.maximum(skeleton, residual)
            current = eroded

        skeleton = skeleton[1:-1, 1:-1]
        return np.where(skeleton > 0, np.uint8(0), np.uint8(255))


    # =========================================================================
    # 5. Minutiae Detection & Visualization
    # =========================================================================

    def detect_minutiae(self, skeleton: np.ndarray, margin: int = 12) -> dict:
        """
        Detects ridge endings and bifurcations using the Crossing Number method.
        """
        s = (skeleton == 0).astype(np.int16)
        H, W = s.shape
        p = np.pad(s, 1, mode='constant', constant_values=0)

        # 8 neighbors collected in clockwise order
        neighbors = np.stack([
            p[:-2, :-2], p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:],
            p[2:, 2:], p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2]
        ], axis=0)

        diff_sum = np.zeros((H, W), dtype=np.int16)
        for k in range(8):
            diff_sum += np.abs(neighbors[k] - neighbors[(k + 1) % 8])
            
        cn = (diff_sum // 2).astype(np.uint8)

        mask = np.zeros((H, W), dtype=bool)
        mask[margin:H - margin, margin:W - margin] = True
        mask &= (s == 1)

        terminations = [tuple(pt) for pt in np.argwhere(mask & (cn == 1))]
        bifurcations = [tuple(pt) for pt in np.argwhere(mask & (cn == 3))]

        return {'terminations': terminations, 'bifurcations': bifurcations}





    # better detection ??? less false positives from broken lines
    # added __ so we dont have 2 methods with the same name
    def __detect_minutiae(self, skeleton: np.ndarray, margin: int = 12, spurious_dist: float = 5.0) -> dict:
        """
        Detects ridge endings and bifurcations using the Crossing Number method,
        with an added post-processing step to remove spurious/false minutiae 
        caused by broken ridges.
        """
        import math
        
        s = (skeleton == 0).astype(np.int16)
        H, W = s.shape
        p = np.pad(s, 1, mode='constant', constant_values=0)

        # 8 neighbors collected in clockwise order
        neighbors = np.stack([
            p[:-2, :-2], p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:],
            p[2:, 2:], p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2]
        ], axis=0)

        diff_sum = np.zeros((H, W), dtype=np.int16)
        for k in range(8):
            diff_sum += np.abs(neighbors[k] - neighbors[(k + 1) % 8])
            
        cn = (diff_sum // 2).astype(np.uint8)

        mask = np.zeros((H, W), dtype=bool)
        mask[margin:H - margin, margin:W - margin] = True
        mask &= (s == 1)

        raw_terminations = [tuple(pt) for pt in np.argwhere(mask & (cn == 1))]
        bifurcations = [tuple(pt) for pt in np.argwhere(mask & (cn == 3))]

        # --- Post-Processing: Remove Spurious Terminations ---
        # If two terminations are extremely close to each other, they are likely 
        # a single broken ridge rather than two actual minutiae.
        valid_terminations = []
        skip_indices = set()
        
        for i, t1 in enumerate(raw_terminations):
            if i in skip_indices:
                continue
                
            is_spurious = False
            for j in range(i + 1, len(raw_terminations)):
                if j in skip_indices:
                    continue
                t2 = raw_terminations[j]
                
                # Calculate Euclidean distance
                dist = math.sqrt((t1[0] - t2[0])**2 + (t1[1] - t2[1])**2)
                
                if dist < spurious_dist:
                    # Both are deemed spurious (a broken line)
                    is_spurious = True
                    skip_indices.add(j)
                    break
                    
            if not is_spurious:
                valid_terminations.append(t1)

        return {'terminations': valid_terminations, 'bifurcations': bifurcations}


    def draw_minutiae(self, skeleton: np.ndarray, minutiae: dict, radius: int = 5) -> np.ndarray:
        """
        Overlays the detected minutiae onto the skeleton image in RGB format.
        Colors ONLY the skeleton lines within a certain radius of the minutiae points.
        Terminations = Red lines, Bifurcations = Blue lines.
        """
        # Convert grayscale skeleton (0=ridge, 255=bg) to a 3-channel RGB image
        rgb_image = np.stack([skeleton, skeleton, skeleton], axis=-1).copy()
        H, W = skeleton.shape
        Y, X = np.ogrid[:H, :W]

        # Helper function to create a circular mask
        def _circle(center, r):
            cy, cx = center
            return (Y - cy) ** 2 + (X - cx) ** 2 <= r ** 2

        # Create a boolean mask of the skeleton itself (where the ridges are black/0)
        skeleton_mask = (skeleton == 0)

        # 1. Color the terminations (Ridge Endings) -> RED
        for pt in minutiae.get('terminations', []):
            circle_mask = _circle(pt, radius)
            # Combine the circle area with the actual skeleton pixels
            line_mask = circle_mask & skeleton_mask
            
            # Apply Red color (R=255, G=0, B=0) to the black pixels
            rgb_image[line_mask, 0] = 255
            rgb_image[line_mask, 1] = 0
            rgb_image[line_mask, 2] = 0

        # 2. Color the bifurcations -> BLUE
        for pt in minutiae.get('bifurcations', []):
            circle_mask = _circle(pt, radius)
            # Combine the circle area with the actual skeleton pixels
            line_mask = circle_mask & skeleton_mask
            
            # Apply Blue color (R=0, G=0, B=255) to the black pixels
            rgb_image[line_mask, 0] = 0
            rgb_image[line_mask, 1] = 0
            rgb_image[line_mask, 2] = 255

        return rgb_image.astype(np.uint8)


    # old method for drawing - makes circles
    def __draw_minutiae(self, skeleton: np.ndarray, minutiae: dict, radius: int = 4) -> np.ndarray:
        """
        Overlays the detected minutiae onto the skeleton image in RGB format.
        Terminations = Red, Bifurcations = Blue.
        """
        rgb_image = np.stack([skeleton, skeleton, skeleton], axis=-1).copy()
        H, W = skeleton.shape
        Y, X = np.ogrid[:H, :W]

        def _circle(center, r):
            cy, cx = center
            return (Y - cy) ** 2 + (X - cx) ** 2 <= r ** 2

        for pt in minutiae.get('terminations', []):
            m = _circle(pt, radius)
            rgb_image[m, 0], rgb_image[m, 1], rgb_image[m, 2] = 255, 0, 0

        for pt in minutiae.get('bifurcations', []):
            m = _circle(pt, radius)
            rgb_image[m, 0], rgb_image[m, 1], rgb_image[m, 2] = 0, 0, 255

        return rgb_image.astype(np.uint8)