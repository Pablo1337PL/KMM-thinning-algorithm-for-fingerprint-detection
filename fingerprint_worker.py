import numpy as np
import time
from PIL import Image
from PyQt5.QtCore import QThread, pyqtSignal

from fingerprint_processor import FingerprintProcessor

class FingerprintWorker(QThread):
    finished = pyqtSignal(object)

    def __init__(self, image, step, params):
        super().__init__()
        self.image = image
        self.step = step
        self.params = params
        self.is_cancelled = False
        self.proc = FingerprintProcessor()

    def run(self):
        img = self.image.copy()

        try:
            if self.step == 0:
                pass  # Oryginał
            elif self.step >= 1:
                # Krok 1: Skala szarości
                pil_img = Image.fromarray(img)
                img = self.proc._to_gray(pil_img)
                
                if self.step >= 2:
                    # Krok 2: Normalizacja
                    img = self.proc.normalize(img)
                    
                    if self.step >= 3:
                        # Krok 3: Segmentacja (Maska ROI)
                        roi_block = self.params.get('roi_block', 8)
                        t_ratio = self.params.get('threshold_ratio', 0.2) 
                        
                        maska = self.proc._compute_roi(img, block=roi_block, 
                                                       threshold_ratio=t_ratio, 
                                                       morph_size=max(15, roi_block * 4 | 1))
                        
                        if self.step == 3:
                            # WIZUALIZACJA: Nałożenie maski na znormalizowany obraz
                            # Kopiujemy obraz i wszystkie piksele poza maską (~maska) ustawiamy na 0 (czarny)
                            wizualizacja_roi = img.copy()
                            wizualizacja_roi[~maska] = 0
                            img = wizualizacja_roi
                                
                        if self.step >= 4:
                            # Krok 4: Mapa Orientacji
                            ori = self.proc._get_orientation_map(img)
                            
                            if self.step == 4:
                                # Wizualizacja mapy orientacji (przeskalowana do skali szarości)
                                img = (ori / np.pi * 255).astype(np.uint8)
                                
                            if self.step >= 5:
                                # Krok 5: Filtracja Gabora
                                freq = self.params.get('freq', 0.1)
                                n_angles = self.params.get('n_angles', 16)
                                ksize = self.params.get('ksize', 17)
                                ksize = max(5, ksize | 1) # Wymuszenie nieparzystości
                                sigma_perp = self.params.get('sigma_perp', 1.2)
                                sigma_par = self.params.get('sigma_par', 1.5)
                                
                                gabor_img = self.proc.gabor_enhance(img, n_angles=n_angles, freq=freq, 
                                                                    ksize=ksize, sigma_perp=sigma_perp, sigma_par=sigma_par)
                                
                                if self.step == 5:
                                    # Wizualizacja wyniku Gabora
                                    img = (gabor_img * 255).astype(np.uint8)
                                    
                                if self.step >= 6:
                                    # # Krok 6: Binaryzacja progowa z nałożeniem maski
                                    threshold = self.params.get('threshold', 124)
                                    wzmU8 = (gabor_img * 255).astype(np.uint8)
                                    
                                    # # Opcjonalne morfologiczne czyszczenie szumu przed binaryzacją
                                    # morph_size = self.params.get('morph_size', 3)
                                    # if morph_size > 1:
                                    #     se = self.proc._get_structuring_element(morph_size, 'ellipse')
                                    #     wzmU8 = self.proc._open(self.proc._close(wzmU8, se), se)

                                    # bin_img = np.where(wzmU8 >= threshold, np.uint8(0), np.uint8(255))
                                    
                                    bin_img = np.where(wzmU8 >= threshold, np.uint8(0), np.uint8(255))
                                    morph_size = self.params.get('morph_size', 3)
                                    if morph_size > 1:
                                        # Wymuszenie liczby nieparzystej i całkowitej
                                        morph_size = max(3, int(morph_size) | 1)
                                        se = self.proc._get_structuring_element(morph_size, 'ellipse')
                                        bin_img = self.proc._open(self.proc._close(bin_img, se), se)

                                    bin_img[~maska] = 255 # Wyczyszczenie tła poza maską
                                    
                                    if self.step == 6:
                                        img = bin_img

                                    if self.step >= 7:
                                        # Krok 7: Szkieletyzacja z pomiarem czasu!
                                        t0 = time.perf_counter()
                                        szkielet_kmm = self.proc.apply_kmm(bin_img.copy())
                                        czas_kmm = time.perf_counter() - t0
                                        
                                        t0 = time.perf_counter()
                                        szkielet_morf = self.proc.skeletonize(bin_img.copy())
                                        czas_morf = time.perf_counter() - t0
                                        
                                        H = szkielet_kmm.shape[0]
                                        separator_2d = np.zeros((H, 5), dtype=np.uint8) 
                                        
                                        if self.step == 7:
                                            img = np.hstack((szkielet_kmm, separator_2d, szkielet_morf))
                                            
                                        if self.step >= 8:
                                            # Krok 8: Poprawa połączeń ORAZ czyszczenie szumu (kropek)
                                            szkielet_kmm = self.proc.connect_broken_lines(szkielet_kmm, max_dist=10.0, max_angle_diff=45.0)
                                            szkielet_kmm = self.proc.prune_skeleton(szkielet_kmm, num_iter=1)
                                            
                                            szkielet_morf = self.proc.connect_broken_lines(szkielet_morf, max_dist=10.0, max_angle_diff=45.0)
                                            szkielet_morf = self.proc.prune_skeleton(szkielet_morf, num_iter=1)
                                            
                                            if self.step == 8:
                                                # Wyświetlamy połączone i wyczyszczone szkielety (czarno-białe)
                                                img = np.hstack((szkielet_kmm, separator_2d, szkielet_morf))

                                            if self.step >= 9:
                                                # Krok 9: Detekcja minucji
                                                minucje_kmm = self.proc.detect_minutiae(szkielet_kmm, roi_mask=maska)
                                                img_min_kmm = self.proc.draw_minutiae(szkielet_kmm, minucje_kmm)
                                                
                                                minucje_morf = self.proc.detect_minutiae(szkielet_morf, roi_mask=maska)
                                                img_min_morf = self.proc.draw_minutiae(szkielet_morf, minucje_morf)
                                                
                                                if self.step == 9 or self.step == 10:
                                                    separator_3d = np.zeros((H, 5, 3), dtype=np.uint8)
                                                    img = np.hstack((img_min_kmm, separator_3d, img_min_morf))
                                                    
                                                if self.step == 10:
                                                    # Krok 10: Przygotowanie pakietu statystyk dla okienka
                                                    stats_text = (
                                                        f"ALGORYTM KMM\n"
                                                        f"Czas wykonania: {czas_kmm:.4f} sekund\n"
                                                        f"Zakończenia: {len(minucje_kmm['terminations'])}\n"
                                                        f"Bifurkacje: {len(minucje_kmm['bifurcations'])}\n"
                                                        f"Suma minucji: {len(minucje_kmm['terminations']) + len(minucje_kmm['bifurcations'])}\n\n"
                                                        f"ALGORYTM MORFOLOGICZNY\n"
                                                        f"Czas wykonania: {czas_morf:.4f} sekund\n"
                                                        f"Zakończenia: {len(minucje_morf['terminations'])}\n"
                                                        f"Bifurkacje: {len(minucje_morf['bifurcations'])}\n"
                                                        f"Sma minucji: {len(minucje_morf['terminations']) + len(minucje_morf['bifurcations'])}"
                                                    )
                                                    # Wysyłamy obraz ORAZ tekst jednocześnie!
                                                    self.finished.emit((img, stats_text))
                                                    return  # Kończymy, żeby nie wysłać podwójnie poniżej

            # Zabezpieczenie przed typami float
            if type(img) is np.ndarray and img.dtype == np.float64:
                 img = img.astype(np.uint8)

            if not self.is_cancelled:
                self.finished.emit(img)
                
        except Exception as e:
            print(f"Błąd w wątku przetwarzania: {e}")