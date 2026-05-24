import os
import time
import csv
from collections import defaultdict
from PIL import Image
import numpy as np
from fingerprint_processor import FingerprintProcessor

def run_batch_tests(data_dir="data", output_csv="wyniki_testow.csv"):
    proc = FingerprintProcessor()
    wyniki_zgrupowane = defaultdict(list)
    
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';') 
        writer.writerow(['Plik', 'Reka', 'Palec', 'Proba',
                         'KMM - Czas (s)', 'KMM - Zakonczenia', 'KMM - Bifurkacje', 'KMM - Suma Minucji',
                         'Morf - Czas (s)', 'Morf - Zakonczenia', 'Morf - Bifurkacje', 'Morf - Suma Minucji'])
        
        if not os.path.exists(data_dir):
            print(f"Błąd: Nie znaleziono katalogu '{data_dir}'.")
            return
            
        pliki = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.bmp', '.jpg', '.jpeg'))]
        print(f"Znaleziono {len(pliki)} obrazów. Rozpoczynam testy...\n")
        
        for plik in pliki:
            sciezka = os.path.join(data_dir, plik)
            n = os.path.splitext(plik)[0]
            reka = n[0] if len(n) > 0 else '?'; palec = n[1] if len(n) > 1 else '?'; proba = n[2] if len(n) > 2 else '?'
            
            try:
                pil_img = Image.open(sciezka).convert('RGB')
                img_array = np.array(pil_img)
                
                # Pre-processing DOKŁADNIE zsynchronizowany z main.py i fingerprint_worker.py
                gray_img = proc._to_gray(img_array)
                norm_img = proc.normalize(gray_img)
                
                # Używamy threshold_ratio=0.33 i morph_size=33 (zgodnie z workerem) do maski
                roi_mask = proc._compute_roi(norm_img, block=8, threshold_ratio=0.33, morph_size=33) 
                
                # Parametry Gabora i binaryzacji zgodne z okienkiem
                bin_img = proc.preprocess_fingerprint(
                    img_array, 
                    threshold=128, 
                    sigma_perp=1.2, 
                    sigma_par=1.5, 
                    morph_size=3, 
                    threshold_ratio=0.33
                )
                
                # ==================== TEST 1: KMM ====================
                start_kmm = time.perf_counter()
                szkielet_kmm = proc.apply_kmm(bin_img.copy())
                szkielet_kmm = proc.connect_broken_lines(szkielet_kmm, max_dist=10.0, max_angle_diff=45.0)
                szkielet_kmm = proc.prune_skeleton(szkielet_kmm, num_iter=1)
                minucje_kmm = proc.detect_minutiae(szkielet_kmm, roi_mask=roi_mask) 
                czas_kmm = time.perf_counter() - start_kmm
                
                k_term = len(minucje_kmm.get('terminations', []))
                k_bif = len(minucje_kmm.get('bifurcations', []))
                k_suma = k_term + k_bif
                
                # ==================== TEST 2: Morfologia ====================
                start_morf = time.perf_counter()
                szkielet_morf = proc.skeletonize(bin_img.copy())
                szkielet_morf = proc.connect_broken_lines(szkielet_morf, max_dist=10.0, max_angle_diff=45.0)
                szkielet_morf = proc.prune_skeleton(szkielet_morf, num_iter=1)
                minucje_morf = proc.detect_minutiae(szkielet_morf, roi_mask=roi_mask)
                czas_morf = time.perf_counter() - start_morf
                
                m_term = len(minucje_morf.get('terminations', []))
                m_bif = len(minucje_morf.get('bifurcations', []))
                m_suma = m_term + m_bif
                
                wyniki_zgrupowane[(reka, palec)].append({
                    'c_kmm': czas_kmm, 'k_t': k_term, 'k_b': k_bif, 'k_s': k_suma,
                    'c_morf': czas_morf, 'm_t': m_term, 'm_b': m_bif, 'm_s': m_suma
                })
                
                writer.writerow([plik, reka, palec, proba, 
                                 f"{czas_kmm:.4f}".replace('.', ','), k_term, k_bif, k_suma,
                                 f"{czas_morf:.4f}".replace('.', ','), m_term, m_bif, m_suma])
                print(f"Przetworzono: {plik: <10} | KMM: {k_suma: <4} min | Morf: {m_suma: <4} min")
                
            except Exception as e: 
                print(f"Błąd pliku {plik}: {e}")
        
        writer.writerow([])
        writer.writerow([])
        writer.writerow(['PODSUMOWANIE (ŚREDNIE Z PRÓB)'])
        writer.writerow(['Reka', 'Palec', 'KMM - Czas', 'KMM - Zak', 'KMM - Bif', 'KMM - Suma', 'Morf - Czas', 'Morf - Zak', 'Morf - Bif', 'Morf - Suma'])
        
        for (reka, palec) in sorted(wyniki_zgrupowane.keys()):
            pomiary = wyniki_zgrupowane[(reka, palec)]
            ile = len(pomiary)
            writer.writerow([reka, palec,
                             f"{sum(p['c_kmm'] for p in pomiary)/ile:.4f}".replace('.', ','), 
                             f"{sum(p['k_t'] for p in pomiary)/ile:.2f}".replace('.', ','), 
                             f"{sum(p['k_b'] for p in pomiary)/ile:.2f}".replace('.', ','), 
                             f"{sum(p['k_s'] for p in pomiary)/ile:.2f}".replace('.', ','),
                             f"{sum(p['c_morf'] for p in pomiary)/ile:.4f}".replace('.', ','), 
                             f"{sum(p['m_t'] for p in pomiary)/ile:.2f}".replace('.', ','), 
                             f"{sum(p['m_b'] for p in pomiary)/ile:.2f}".replace('.', ','), 
                             f"{sum(p['m_s'] for p in pomiary)/ile:.2f}".replace('.', ',')])
                             
    print(f"\nGotowe! Wyniki zapisane do pliku {output_csv}.")

if __name__ == "__main__":
    run_batch_tests()