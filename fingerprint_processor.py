import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

class IrisProcessor:
    """Klasa logiczna zawierająca algorytmy z instrukcji."""
    
    @staticmethod
    def to_grayscale(img):
        if len(img.shape) == 3:
            return np.dot(img[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
        return img

    @staticmethod
    def calculate_base_threshold(gray_img):
        h, w = gray_img.shape
        return np.sum(gray_img) / (h * w)

    @staticmethod
    def binarize(gray_img, threshold):
        return (gray_img < threshold).astype(np.uint8) * 255
    
    @staticmethod
    def filter_min_max(img, size, mode='min'):
        pad_w = size // 2
        pad_img = np.pad(img, pad_w, mode='edge')
        windows = sliding_window_view(pad_img, window_shape=(size, size))
        
        if mode == 'min':
            return np.min(windows, axis=(-2, -1)).astype(np.uint8)
        else:
            return np.max(windows, axis=(-2, -1)).astype(np.uint8)
        
    @staticmethod
    def apply_morphology(img, operation, size):
        if "Brak" in operation:
            return img
        elif "Usuń rzęsy" in operation:
            tmp = IrisProcessor.filter_min_max(img, size, 'max')
            return IrisProcessor.filter_min_max(tmp, size, 'min')
        elif "Zalej refleksy" in operation:
            tmp = IrisProcessor.filter_min_max(img, size, 'min')
            return IrisProcessor.filter_min_max(tmp, size, 'max')
        elif "Tylko powiększ czarne" in operation:
            return IrisProcessor.filter_min_max(img, size, 'min')
        elif "Tylko powiększ białe" in operation:
            return IrisProcessor.filter_min_max(img, size, 'max')
        return img

    @staticmethod
    def find_center_and_radius_via_projections(binary_img):
        THRESHOLD_RATIO = 0.9
        
        inverted = np.where(binary_img == 0, 1, 0)
        
        proj_y = np.sum(inverted, axis=1)
        proj_x = np.sum(inverted, axis=0)
        
        def get_bounds(proj):
            max_val = np.max(proj)
            if max_val == 0:
                return 0, 0
            
            threshold = max_val * THRESHOLD_RATIO * 0.2
            valid_indices = np.where(proj > threshold)[0]
            if len(valid_indices) == 0:
                return 0, 0
                
            return valid_indices[0], valid_indices[-1]
            
        top, bottom = get_bounds(proj_y)
        left, right = get_bounds(proj_x)
        
        radius_x = (right - left) / 2
        radius_y = (bottom - top) / 2
        radius = int((radius_x + radius_y) / 2)
        
        max_y = np.max(proj_y)
        y_indices = np.where((proj_y <= max_y) & (proj_y > max_y * THRESHOLD_RATIO))[0]
        center_y = int(np.mean(y_indices)) 
        
        proj_x = np.sum(inverted, axis=0)
        max_x = np.max(proj_x)
        x_indices = np.where((proj_x <= max_x) & (proj_x > max_x * THRESHOLD_RATIO))[0]
        center_x = int(np.mean(x_indices))

        return center_x, center_y, radius

    @staticmethod
    def find_center_and_radius_via_n_projections(binary_img, n_angles=7):
        """
        Wyznacza środek i promień źrenicy wykonując n projekcji pod różnymi kątami.
        Wersja używająca czystego NumPy i rzutowania współrzędnych (bez obracania obrazu).
        """
        threshold_1 = 0.33 # Procent piku histogramu, poniżej którego uznajemy, że to szum
        threshold_2 = 0.1 # Procent piku projekcji, poniżej którego uznajemy, że to szum
        
        inverted = np.where(binary_img == 0, 1, 0).astype(np.uint8)
        h, w = inverted.shape
        
        intersection_mask = np.ones((h, w), dtype=np.uint8)
        
        y_grid, x_grid = np.indices((h, w))
        
        active_y, active_x = np.where(inverted == 1)
        
        if len(active_x) == 0:
            return 0, 0, 0
            
        angles = np.linspace(0, np.pi, n_angles, endpoint=False)
        
        for angle in angles:
            nx = np.cos(angle)
            ny = np.sin(angle)
            
            p_active = active_x * nx + active_y * ny
            
            min_p, max_p = np.floor(p_active.min()), np.ceil(p_active.max())
            bins = np.arange(min_p, max_p + 2)
            hist, edges = np.histogram(p_active, bins=bins)
            
            max_val = np.max(hist)
            if max_val == 0: continue
            
            threshold = max_val * threshold_1
            valid_indices = np.where(hist > threshold)[0]
            
            if len(valid_indices) == 0: continue
                
            p_min = edges[valid_indices[0]]
            p_max = edges[valid_indices[-1] + 1]
            
            P_grid = x_grid * nx + y_grid * ny
            
            stripe_mask = (P_grid >= p_min) & (P_grid <= p_max)
            
            intersection_mask &= stripe_mask.astype(np.uint8)

        final_pixels = inverted & intersection_mask
        
        proj_y = np.sum(final_pixels, axis=1)
        proj_x = np.sum(final_pixels, axis=0)
        
        def get_final_bounds(proj):
            max_val = np.max(proj)
            if max_val == 0: return 0, 0
            valid_indices = np.where(proj > max_val * threshold_2)[0]
            if len(valid_indices) == 0: return 0, 0
            return valid_indices[0], valid_indices[-1]
            
        top, bottom = get_final_bounds(proj_y)
        left, right = get_final_bounds(proj_x)
        
        center_y = int((top + bottom) / 2)
        center_x = int((left + right) / 2)
        
        radius_y = (bottom - top) / 2
        radius_x = (right - left) / 2
        radius = int((radius_x + radius_y) / 2)
        
        return center_x, center_y, radius

    @staticmethod
    def draw_crosshair_and_circle(img, x, y, r, cross_size=20, color=(255, 0, 0)):
        if len(img.shape) == 2:
            img_color = np.stack([img, img, img], axis=-1)
        else:
            img_color = img.copy()
            
        h, w = img_color.shape[:2]
        
        x_start = max(0, x - cross_size)
        x_end = min(w, x + cross_size)
        img_color[y, x_start:x_end] = color
        
        y_start = max(0, y - cross_size)
        y_end = min(h, y + cross_size)
        img_color[y_start:y_end, x] = color
        
        y_idx, x_idx = np.ogrid[:h, :w]
        
        dist_from_center_sq = (x_idx - x)**2 + (y_idx - y)**2
        
        ring_thickness = 2
        ring_mask = (dist_from_center_sq <= r**2) & (dist_from_center_sq >= (max(0, r - ring_thickness))**2)
        
        img_color[ring_mask] = color
        
        return img_color
    
    @staticmethod
    def find_iris_radius(gray_img, cx, cy, pupil_radius):
        """
        Wyznacza promień tęczówki analizując wyłącznie poziomy pas przechodzący przez środek źrenicy.
        Ignoruje zakłócenia od powiek i rzęs z góry i z dołu.
        """
        h, w = gray_img.shape
        
        # Wycinamy poziomy pasek o wysokości 20 pikseli wokół środka źrenicy
        strip_height = 10 
        y_start = max(0, cy - strip_height)
        y_end = min(h, cy + strip_height)
        
        # Pobieramy pasek i uśredniamy go pionowo, aby uzyskać jeden stabilny ciąg wartości (1D)
        horizontal_strip = gray_img[y_start:y_end, :]
        profile = np.mean(horizontal_strip, axis=0)
        
        # Obliczamy gradient (pochodną) - czyli różnicę jasności między sąsiednimi pikselami.
        # W miejscu przejścia ciemnej tęczówki w jasną twardówkę będzie skok.
        gradient = np.abs(np.diff(profile))
        
        ignore_margin = int(pupil_radius * 1.2)
        safe_left = max(0, cx - ignore_margin)
        safe_right = min(w - 1, cx + ignore_margin)
        gradient[safe_left:safe_right] = 0
        
        # największy skok po lewej stronie
        left_half = gradient[:cx]
        left_edge_x = np.argmax(left_half) if len(left_half) > 0 else 0
        
        # największy skok po prawej stronie
        right_half = gradient[cx:]
        right_edge_x = cx + np.argmax(right_half) if len(right_half) > 0 else 0
        
        r_left = cx - left_edge_x
        r_right = right_edge_x - cx
        
        # uśredniamy wynik
        iris_radius = int((r_left + r_right) / 2)
        
        # zabezpieczenie przed błędem: tęczówka nie może być poza zdjęciem
        if iris_radius < pupil_radius:
            iris_radius = pupil_radius + 20 # Wartość domyślna awaryjna
            
        return iris_radius

    @staticmethod
    def unwrap_iris(image, cx, cy, r_pupil, r_iris, width=128, height=64):
        """
        Przekształca pierścień tęczówki w prostokąt (normalizacja Daugmana).
        Zgodnie z literaturą pomija obszary z góry i dołu (powieki/rzęsy),
        pobierając wyłącznie bezpieczne wycinki z lewej i prawej strony.
        """
        if len(image.shape) == 3:
            unwrapped = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            unwrapped = np.zeros((height, width), dtype=np.uint8)

        # Prawa strona oka: od -45 stopni (-pi/4) do 45 stopni (pi/4)
        # Lewa strona oka: od 135 stopni (3*pi/4) do 225 stopni (5*pi/4)
        half_w = width // 2
        theta_right = np.linspace(-np.pi / 4, np.pi / 4, half_w)
        theta_left = np.linspace(3 * np.pi / 4, 5 * np.pi / 4, width - half_w)
        
        thetas = np.concatenate([theta_right, theta_left])
        rhos = np.linspace(0, 1, height)

        theta_grid, rho_grid = np.meshgrid(thetas, rhos)

        r_grid = r_pupil + rho_grid * (r_iris - r_pupil)

        x_grid = cx + r_grid * np.cos(theta_grid)
        y_grid = cy + r_grid * np.sin(theta_grid)

        x_grid = np.clip(np.round(x_grid), 0, image.shape[1] - 1).astype(int)
        y_grid = np.clip(np.round(y_grid), 0, image.shape[0] - 1).astype(int)

        unwrapped = image[y_grid, x_grid]
        
        return unwrapped
    
    @staticmethod
    def generate_iris_code(unwrapped_img, f=0.1, n_bands=8, n_points=128):
        """
        Wyznacza 2048-bitowy kod tęczówki za pomocą algorytmu Daugmana.
        Oczekuje na wejściu znormalizowanego obrazu (np. 64x128 pikseli).
        """
        if len(unwrapped_img.shape) == 3:
            gray = np.dot(unwrapped_img[...,:3], [0.299, 0.587, 0.114])
        else:
            gray = unwrapped_img.astype(float)

        h, w = gray.shape
        band_h = h // n_bands  

        sigma = 0.5 * np.pi * f
        
        x = np.arange(-n_points // 2, n_points // 2)
        
        gabor_real = np.exp(-(x**2) / (sigma**2)) * np.cos(2 * np.pi * f * x)
        gabor_imag = np.exp(-(x**2) / (sigma**2)) * (-np.sin(2 * np.pi * f * x))

        y = np.arange(band_h)
        center_y = (band_h - 1) / 2.0
        sigma_y = band_h / 4.0 
        
        gauss_window = np.exp(-((y - center_y)**2) / (2 * sigma_y**2))
        gauss_window /= np.sum(gauss_window) 

        iris_code_bits = []

        for i in range(n_bands):
            band = gray[i * band_h : (i + 1) * band_h, :]
            
            if w != n_points:
                indices = np.linspace(0, w - 1, n_points).astype(int)
                band_sampled = band[:, indices]
            else:
                band_sampled = band

            signal_1d = np.dot(band_sampled.T, gauss_window)
            
            # Usuwamy składową stałą, by sygnał wahał się wokół zera!
            signal_1d = signal_1d - np.mean(signal_1d)

            # mieniamy z mode='wrap' na mode='reflect', bo mamy ucięte boki i zszyte na środku.
            pad_w = len(gabor_real) // 2
            sig_padded = np.pad(signal_1d, pad_w, mode='reflect')

            # Używamy mode='valid', aby pozbyć się dodanego paddingu
            res_real = np.convolve(sig_padded, gabor_real, mode='valid')
            res_imag = np.convolve(sig_padded, gabor_imag, mode='valid')

            # Wyrównanie długości
            res_real = res_real[:n_points]
            res_imag = res_imag[:n_points]

            bit1 = (res_imag < 0).astype(np.uint8)
            bit2 = (res_real < 0).astype(np.uint8)

            iris_code_bits.append(bit1)
            iris_code_bits.append(bit2)

        return np.array(iris_code_bits, dtype=np.uint8)

    @staticmethod
    def visualize_iris_code(code_array):
        """Tworzy czarno-biały obrazek kodu dla UI."""
        return (code_array * 255).astype(np.uint8)
    
    @staticmethod
    def calculate_hamming_distance(code1, code2):
        """
        Oblicza odległość Hamminga zgodnie z literaturą.
        d = (1/N) * suma(C_i XOR C'_i)
        """
        b1 = (code1 > 128).astype(np.uint8)
        b2 = (code2 > 128).astype(np.uint8)
        
        h = min(b1.shape[0], b2.shape[0])
        w = min(b1.shape[1], b2.shape[1])
        b1 = b1[:h, :w]
        b2 = b2[:h, :w]
        
        diff = np.bitwise_xor(b1, b2)
        
        distance = np.sum(diff) / diff.size
        return distance