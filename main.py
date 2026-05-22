import sys, os
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QGridLayout,
                             QHBoxLayout, QPushButton, QLabel, QSlider, QFileDialog, QComboBox,
                             QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

from przegladarka_obrazow import PrzegladarkaObrazow
from fingerprint_processor import IrisProcessor
from fingerprint_worker import IrisWorker

class SaveWorker(QThread):
    success = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, file_path, save_path, process_func):
        super().__init__()
        self.file_path = file_path
        self.save_path = save_path
        self.process_func = process_func 

    def run(self):
        """Ta metoda uruchamia się w tle, gdy wywołamy .start()"""
        try:
            pil_img = Image.open(self.file_path)
            pil_img = pil_img.convert('RGB')
            img = np.array(pil_img)
            
            img = self.process_func(img) 
            
            Image.fromarray(img).save(self.save_path)
            
            self.success.emit()
            
        except Exception as e:
            self.error.emit(str(e))

class IrisMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Segmentacja tęczówki")
        self.current_step = 0
        self.original_image = None
        self.current_processed_image = None
        self.current_file_path = ""
        self.setup_ui()

    def setup_ui(self):
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
        
        # --- Pasek ładowania ---
        top_layout = QHBoxLayout()
        self.btn_load = QPushButton("Wczytaj obraz")
        self.btn_load.clicked.connect(self.load_image)
        top_layout.addWidget(self.btn_load)

        self.btn_save = QPushButton("Zapisz kod (BMP)")
        self.btn_save.clicked.connect(self.save_iris_code)
        self.btn_save.setVisible(False) # Domyślnie ukryty
        top_layout.addWidget(self.btn_save)


        top_layout.addStretch()
        layout.addLayout(top_layout)
        
        # --- Przeglądarka i Projekcje ---
        self.grid_widget = QWidget()
        self.grid = QGridLayout(self.grid_widget)
        self.grid.setSpacing(0)

        self.viewer = PrzegladarkaObrazow()
        self.viewer.visible_rect_changed.connect(self.update_projections_from_rect)
        self.viewer.setMinimumSize(600, 400)

        self.grid.addWidget(self.viewer, 1, 1)

        self.grid.setColumnStretch(1, 1)
        self.grid.setRowStretch(1, 1)
        
        
        layout.addWidget(self.grid_widget)

        # --- Suwak Źrenicy ---
        self.control_layout = QHBoxLayout()
        self.lbl_param = QLabel("Parametr x_I (Źrenica):")
        self.slider_x = QSlider(Qt.Horizontal)
        self.slider_x.setRange(1, 50)
        self.slider_x.setValue(40)
        self.slider_x.setTickPosition(QSlider.TicksBelow)
        self.lbl_slider_val = QLabel("4.0")
        
        self.control_layout.addWidget(self.lbl_param)
        self.control_layout.addWidget(self.slider_x)
        self.control_layout.addWidget(self.lbl_slider_val)
        layout.addLayout(self.control_layout)
        self.slider_x.valueChanged.connect(self.on_slider_changed)

        # --- Suwak Tęczówki ---
        self.control_layout_iris = QHBoxLayout()
        self.lbl_param_iris = QLabel("Parametr x_P (Tęczówka):")
        self.slider_x_iris = QSlider(Qt.Horizontal)
        self.slider_x_iris.setRange(1, 100)
        self.slider_x_iris.setValue(15)
        self.slider_x_iris.setTickPosition(QSlider.TicksBelow)
        self.lbl_slider_val_iris = QLabel("1.5")
        
        self.control_layout_iris.addWidget(self.lbl_param_iris)
        self.control_layout_iris.addWidget(self.slider_x_iris)
        self.control_layout_iris.addWidget(self.lbl_slider_val_iris)
        layout.addLayout(self.control_layout_iris)
        self.slider_x_iris.valueChanged.connect(self.on_slider_iris_changed)
        
        # --- Suwak Częstotliwości (Gabor) ---
        self.control_layout_gabor = QHBoxLayout()
        self.lbl_param_f = QLabel("Częstotliwość f (Falka Gabora):")
        self.slider_f = QSlider(Qt.Horizontal)
        self.slider_f.setRange(1, 314) # Odpowiada od 0.01 do 3.14 (około Pi)
        self.slider_f.setValue(50)     # Domyślnie 0.10
        self.slider_f.setTickPosition(QSlider.TicksBelow)
        self.lbl_slider_val_f = QLabel("0.50")
        
        self.control_layout_gabor.addWidget(self.lbl_param_f)
        self.control_layout_gabor.addWidget(self.slider_f)
        self.control_layout_gabor.addWidget(self.lbl_slider_val_f)
        layout.addLayout(self.control_layout_gabor)
        self.slider_f.valueChanged.connect(self.on_slider_f_changed)

        self.set_controls_visible(visible_pupil=False, visible_iris=False, visible_gabor=False)

        # --- Morfologia ---
        self.morph_widget = QWidget()
        self.morph_layout = QVBoxLayout(self.morph_widget)
        opcje = ["Brak", "Usuń rzęsy (Max -> Min)", "Zalej refleksy (Min -> Max)", "Tylko powiększ czarne (Min)", "Tylko powiększ białe (Max)"]
        
        row1 = QHBoxLayout()
        self.combo_morph_1 = QComboBox()
        self.combo_morph_1.addItems(opcje)
        self.combo_morph_1.setCurrentIndex(1) 
        # <=>
        # self.combo_morph_1.setCurrentText("Usuń rzęsy (Max -> Min)")
        self.slider_morph_1 = QSlider(Qt.Horizontal)
        self.slider_morph_1.setRange(1, 15)
        self.slider_morph_1.setValue(3)
        self.lbl_morph_1 = QLabel("Rozmiar: 7")
        row1.addWidget(QLabel("Krok A:")); row1.addWidget(self.combo_morph_1); row1.addWidget(self.slider_morph_1); row1.addWidget(self.lbl_morph_1)
        
        row2 = QHBoxLayout()
        self.combo_morph_2 = QComboBox()
        self.combo_morph_2.addItems(opcje)
        self.combo_morph_2.setCurrentIndex(2)
        self.slider_morph_2 = QSlider(Qt.Horizontal)
        self.slider_morph_2.setRange(1, 15)
        self.slider_morph_2.setValue(3)
        self.lbl_morph_2 = QLabel("Rozmiar: 7")
        row2.addWidget(QLabel("Krok B:")); row2.addWidget(self.combo_morph_2); row2.addWidget(self.slider_morph_2); row2.addWidget(self.lbl_morph_2)

        self.morph_layout.addLayout(row1)
        self.morph_layout.addLayout(row2)
        layout.addWidget(self.morph_widget)
        self.morph_widget.setVisible(False)
        
        self.combo_morph_1.currentIndexChanged.connect(self.on_morph_changed)
        self.slider_morph_1.valueChanged.connect(self.on_morph_changed)
        self.combo_morph_2.currentIndexChanged.connect(self.on_morph_changed)
        self.slider_morph_2.valueChanged.connect(self.on_morph_changed)

        # --- PANEL MORFOLOGII TĘCZÓWKI
        self.morph_widget_iris = QWidget()
        self.morph_layout_iris = QVBoxLayout(self.morph_widget_iris)
        
        row3 = QHBoxLayout()
        self.combo_morph_3 = QComboBox()
        self.combo_morph_3.addItems(opcje)
        self.combo_morph_3.setCurrentIndex(1) 
        self.slider_morph_3 = QSlider(Qt.Horizontal)
        self.slider_morph_3.setRange(1, 40)
        self.slider_morph_3.setValue(1)
        self.lbl_morph_3 = QLabel("Rozmiar: 3")
        row3.addWidget(QLabel("Krok A (Tęczówka):")); row3.addWidget(self.combo_morph_3); row3.addWidget(self.slider_morph_3); row3.addWidget(self.lbl_morph_3)
        
        row4 = QHBoxLayout()
        self.combo_morph_4 = QComboBox()
        self.combo_morph_4.addItems(opcje)
        self.combo_morph_3.setCurrentIndex(2) 
        self.slider_morph_4 = QSlider(Qt.Horizontal)
        self.slider_morph_4.setRange(1, 40)
        self.slider_morph_4.setValue(3)
        self.lbl_morph_4 = QLabel("Rozmiar: 7")
        row4.addWidget(QLabel("Krok B (Tęczówka):")); row4.addWidget(self.combo_morph_4); row4.addWidget(self.slider_morph_4); row4.addWidget(self.lbl_morph_4)

        self.morph_layout_iris.addLayout(row3)
        self.morph_layout_iris.addLayout(row4)
        layout.addWidget(self.morph_widget_iris)
        self.morph_widget_iris.setVisible(False)
        
        self.combo_morph_3.currentIndexChanged.connect(self.on_morph_iris_changed)
        self.slider_morph_3.valueChanged.connect(self.on_morph_iris_changed)
        self.combo_morph_4.currentIndexChanged.connect(self.on_morph_iris_changed)
        self.slider_morph_4.valueChanged.connect(self.on_morph_iris_changed)

        # --- PANEL PORÓWNAWCZY (KROK 10) ---
        self.comp_panel = QWidget()
        comp_layout = QHBoxLayout(self.comp_panel)
        
        # Lewa strona: Obrazy jeden pod drugim
        left_img_layout = QVBoxLayout()
        self.lbl_code1 = QLabel("Kod 1 (brak)"); self.lbl_code1.setAlignment(Qt.AlignCenter)
        self.lbl_code2 = QLabel("Kod 2 (brak)"); self.lbl_code2.setAlignment(Qt.AlignCenter)
        self.btn_load_c1 = QPushButton("Wczytaj Kod 1"); self.btn_load_c1.clicked.connect(lambda: self.load_code_to_compare(1))
        self.btn_load_c2 = QPushButton("Wczytaj Kod 2"); self.btn_load_c2.clicked.connect(lambda: self.load_code_to_compare(2))
        
        left_img_layout.addWidget(self.btn_load_c1); left_img_layout.addWidget(self.lbl_code1)
        left_img_layout.addWidget(self.btn_load_c2); left_img_layout.addWidget(self.lbl_code2)
        comp_layout.addLayout(left_img_layout)
        
        # Prawa strona: Informacje
        self.info_panel = QLabel("Wczytaj dwa kody,\naby porównać...")
        self.info_panel.setStyleSheet("font-size: 16px; font-weight: bold; border: 1px solid gray; padding: 10px;")
        comp_layout.addWidget(self.info_panel)
        
        layout.addWidget(self.comp_panel)
        self.comp_panel.setVisible(False)
        
        # Zmienne przechowujące wczytane kody
        self.loaded_code1 = None
        self.loaded_code2 = None

        # --- Nawigacja ---
        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("← Wstecz")
        self.btn_next = QPushButton("Dalej →")
        self.btn_next.setEnabled(False)
        self.lbl_step = QLabel(f"Krok: {self.current_step} - Wczytywanie")
        
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.lbl_step)
        nav_layout.addWidget(self.btn_next)
        layout.addLayout(nav_layout)
        
        self.btn_next.clicked.connect(self.next_step)
        self.btn_prev.clicked.connect(self.prev_step)
        
        self.setCentralWidget(main_widget)
        self.resize(800, 600)

    # --- Metody Logiczne UI ---
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Wybierz obraz oka", "", "Images (*.png *.jpg *.bmp)")
        if file_path:
            self.current_file_path = file_path
            pil_img = Image.open(file_path).convert('RGB')
            self.original_image = np.array(pil_img)
            self.current_step = 0
            self.btn_next.setEnabled(True)
            self.process()

    def next_step(self):
        self.current_step += 1
        self.process()

    def prev_step(self):
        if self.current_step > 0:
            self.current_step -= 1
            self.process()

    def set_controls_visible(self, visible_pupil, visible_iris=False, visible_gabor=False):
        self.lbl_param.setVisible(visible_pupil)
        self.slider_x.setVisible(visible_pupil)
        self.lbl_slider_val.setVisible(visible_pupil)

        self.lbl_param_iris.setVisible(visible_iris)
        self.slider_x_iris.setVisible(visible_iris)
        self.lbl_slider_val_iris.setVisible(visible_iris)

        self.lbl_param_f.setVisible(visible_gabor)
        self.slider_f.setVisible(visible_gabor)
        self.lbl_slider_val_f.setVisible(visible_gabor)

    def set_morph_visible(self, visible):
        self.morph_widget.setVisible(visible)

    def on_slider_changed(self, value):
        self.lbl_slider_val.setText(f"{value / 10.0:.1f}")
        if self.current_step >= 2: self.process()

    def on_slider_iris_changed(self, value):
        self.lbl_slider_val_iris.setText(f"{value / 10.0:.1f}")
        if self.current_step >= 5: self.process()

    def on_slider_f_changed(self, value):
        self.lbl_slider_val_f.setText(f"{value / 100.0:.2f}")
        if self.current_step >= 9: self.process()

    def on_morph_changed(self):
        self.lbl_morph_1.setText(f"Rozmiar: {self.slider_morph_1.value() * 2 + 1}")
        self.lbl_morph_2.setText(f"Rozmiar: {self.slider_morph_2.value() * 2 + 1}")
        if self.current_step >= 3: self.process()

    def on_morph_iris_changed(self):
        sz3 = self.slider_morph_3.value() * 2 + 1
        sz4 = self.slider_morph_4.value() * 2 + 1
        self.lbl_morph_3.setText(f"Rozmiar: {sz3}")
        self.lbl_morph_4.setText(f"Rozmiar: {sz4}")
        if self.current_step >= 6: self.process()

    def process(self):
        if self.original_image is None: return

        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.is_cancelled = True
            self.worker.wait()

        self.btn_next.setEnabled(False)
        self.btn_prev.setEnabled(False)

        params = {
            'x_param': self.slider_x.value() / 10.0,
            'x_param_iris': self.slider_x_iris.value() / 10.0,
            'op1': self.combo_morph_1.currentText(),
            'sz1': self.slider_morph_1.value() * 2 + 1,
            'op2': self.combo_morph_2.currentText(),
            'sz2': self.slider_morph_2.value() * 2 + 1,
            # tęczówka
            'op3': self.combo_morph_3.currentText(),
            'sz3': self.slider_morph_3.value() * 2 + 1,
            'op4': self.combo_morph_4.currentText(),
            'sz4': self.slider_morph_4.value() * 2 + 1,

            'f_frequency': self.slider_f.value() / 100.0
        }

        self.worker = IrisWorker(self.original_image, self.current_step, params)
        self.worker.finished.connect(self.on_process_finished)
        self.worker.start()

    def on_process_finished(self, processed_img):
        self.current_processed_image = processed_img
        self.viewer.wyswietl_obraz_numpy(processed_img)
        
        tytuly_krokow = [
            "Krok 0: Oryginał", 
            "Krok 1: Skala szarości", 
            "Krok 2: Detekcja źrenicy",
            "Krok 3: "
        ]
        self.lbl_step.setText(tytuly_krokow[self.current_step] if self.current_step < len(tytuly_krokow) else f"Krok {self.current_step}")
        
        self.set_controls_visible(
            visible_pupil=(self.current_step in [2, 3, 4]), 
            visible_iris=(self.current_step in [5, 6, 7]),
            visible_gabor=(self.current_step == 9))

        self.morph_widget.setVisible(self.current_step == 3)
        self.morph_widget_iris.setVisible(self.current_step == 6)
        
        # if self.current_step == 4:
        #     self.proj_gora.setVisible(True); self.proj_boczna.setVisible(True)
        #     gray_img = IrisProcessor.to_grayscale(processed_img)
        #     inverted = np.where(gray_img < 128, 255, 0).astype(np.uint8)    
        #     self.proj_gora.update_plot(inverted, rgb_mode=False)
        #     self.proj_boczna.update_plot(inverted, rgb_mode=False)
        # else:
        #     self.proj_gora.setVisible(False); self.proj_boczna.setVisible(False)
        
        # self.btn_save.setVisible(self.current_step == 9)

        is_step_10 = (self.current_step == 10)
        self.comp_panel.setVisible(is_step_10)
        self.grid_widget.setVisible(not is_step_10)


        self.btn_prev.setEnabled(self.current_step > 0)
        self.btn_next.setEnabled(self.current_step < 10)

    def update_projections_from_rect(self, x, y, w, h):
        if not self.current_processed_image is None and self.current_step == 4:
            widoczny_fragment = self.current_processed_image[y:y+h, x:x+w]
            gray = IrisProcessor.to_grayscale(widoczny_fragment)
            inverted = np.where(gray < 128, 255, 0).astype(np.uint8)
    
    def create_process_func(self):
        """
        Tworzy funkcję przeprowadzającą obraz przez wszystkie kroki algorytmu,
        używając parametrów z suwaków aktualnych w momencie kliknięcia 'Zapisz'.
        """
        params = {
            'x_I': max(0.01, self.slider_x.value() / 10.0),
            'x_P': max(0.01, self.slider_x_iris.value() / 10.0),
            'op1': self.combo_morph_1.currentText(), 'sz1': self.slider_morph_1.value() * 2 + 1,
            'op2': self.combo_morph_2.currentText(), 'sz2': self.slider_morph_2.value() * 2 + 1,
            'op3': self.combo_morph_3.currentText(), 'sz3': self.slider_morph_3.value() * 2 + 1,
            'op4': self.combo_morph_4.currentText(), 'sz4': self.slider_morph_4.value() * 2 + 1,
            'f': self.slider_f.value() / 100.0
        }

        def _process(img):
            img_gray = IrisProcessor.to_grayscale(img)
            P = IrisProcessor.calculate_base_threshold(img_gray)
            
            thr_I = P / params['x_I']
            pupil_bin = np.where(img_gray < thr_I, 0, 255).astype(np.uint8)
            pupil_bin = IrisProcessor.apply_morphology(pupil_bin, params['op1'], params['sz1'])
            pupil_bin = IrisProcessor.apply_morphology(pupil_bin, params['op2'], params['sz2'])
            
            cx, cy, r_pupil = IrisProcessor.find_center_and_radius_via_n_projections(pupil_bin)
            
            thr_P = P / params['x_P']
            iris_bin = np.where(img_gray < thr_P, 0, 255).astype(np.uint8)
            iris_bin = IrisProcessor.apply_morphology(iris_bin, params['op3'], params['sz3'])
            iris_bin = IrisProcessor.apply_morphology(iris_bin, params['op4'], params['sz4'])
            r_iris = IrisProcessor.find_iris_radius(iris_bin, cx, cy, r_pupil)
            
            unwrapped = IrisProcessor.unwrap_iris(img, cx, cy, r_pupil, r_iris, width=128, height=64)
            
            code = IrisProcessor.generate_iris_code(unwrapped, f=params['f'])
            return IrisProcessor.visualize_iris_code(code)

        return _process

    def save_iris_code(self):
        if not self.current_file_path:
            return

        # 1. Przygotowanie ścieżek
        base_name = os.path.basename(self.current_file_path)
        name_without_ext = os.path.splitext(base_name)[0]

        # Pobierz 3 katalogi powyżej pliku
        file_dir = os.path.dirname(self.current_file_path)
        path_parts = []
        current_path = file_dir
        for _ in range(3):
            parent_dir = os.path.basename(current_path)
            if parent_dir:
                path_parts.insert(0, parent_dir)
            current_path = os.path.dirname(current_path)
        
        # Stwórz nową nazwę pliku: dir0_dir1_dir2_name_coded.bmp
        if len(path_parts) >= 3 and path_parts[0] == 'MMU-Iris-Database':
            new_filename = f"{path_parts[0]}_{path_parts[1]}_{path_parts[2]}_{name_without_ext}_coded.bmp"
        else:
            # Fallback, jeśli nie ma 3 katalogów
            new_filename = f"{name_without_ext}_coded.bmp"
        
        save_dir = "coded_iris"
        os.makedirs(save_dir, exist_ok=True) # Tworzy folder, jeśli nie istnieje
        
        save_path = os.path.join(save_dir, new_filename)

        # 2. Zabezpieczenie przed klikaniem
        self.btn_save.setEnabled(False)
        self.btn_save.setText("Zapisywanie...")

        # 3. Uruchomienie Workera
        process_function = self.create_process_func()
        self.save_worker = SaveWorker(self.current_file_path, save_path, process_function)
        
        self.save_worker.success.connect(lambda: self.on_save_success(save_path))
        self.save_worker.error.connect(self.on_save_error)
        self.save_worker.start()



    def load_code_to_compare(self, slot):
        path, _ = QFileDialog.getOpenFileName(self, "Wybierz kod tęczówki", "coded_iris", "Images (*.bmp)")
        if path:
            img = np.array(Image.open(path).convert('L'))
            qimg = QImage(img.data, img.shape[1], img.shape[0], img.shape[1], QImage.Format_Grayscale8)
            pix = QPixmap.fromImage(qimg).scaled(400, 100, Qt.KeepAspectRatio)
            
            if slot == 1:
                self.loaded_code1 = img
                self.lbl_code1.setPixmap(pix)
            else:
                self.loaded_code2 = img
                self.lbl_code2.setPixmap(pix)
                
            self.run_comparison_logic()

    def run_comparison_logic(self):
        if self.loaded_code1 is not None and self.loaded_code2 is not None:
            dist = IrisProcessor.calculate_hamming_distance(self.loaded_code1, self.loaded_code2)
            
            # Próg decyzyjny 0.3 zgodnie z literaturą
            threshold = 0.3
            is_same = dist < threshold
            
            result_text = f"ODLEGŁOŚĆ HAMMINGA: {dist:.4f}\n\n"
            result_text += f"PRÓG DECYZYJNY: {threshold}\n\n"
            result_text += "WYNIK: " + ("ZGODNY (To ta sama osoba)" if is_same else "NIEZGODNY (Różne osoby)")
            
            color = "green" if is_same else "red"
            self.info_panel.setText(result_text)
            self.info_panel.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {color}; border: 2px solid {color}; padding: 10px;")


    def on_save_success(self, save_path):
        self.btn_save.setEnabled(True)
        self.btn_save.setText("Zapisz kod (BMP)")
        QMessageBox.information(self, "Sukces", f"Zapisano kod tęczówki pomyślnie:\n{save_path}")

    def on_save_error(self, err_msg):
        self.btn_save.setEnabled(True)
        self.btn_save.setText("Zapisz kod (BMP)")
        QMessageBox.critical(self, "Błąd", f"Wystąpił błąd podczas zapisywania:\n{err_msg}")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = IrisMainWindow()
    win.show()
    sys.exit(app.exec())
