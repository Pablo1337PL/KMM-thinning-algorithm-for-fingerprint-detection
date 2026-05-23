import sys, os
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QGridLayout,
                             QHBoxLayout, QPushButton, QLabel, QSlider, QFileDialog, QComboBox,
                             QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

from przegladarka_obrazow import PrzegladarkaObrazow
from fingerprint_processor import FingerprintProcessor
from fingerprint_worker import FingerprintWorker

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

class MainWindow(QMainWindow):
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
        
        top_layout = QHBoxLayout()
        self.btn_load = QPushButton("Wczytaj skan (BMP/PNG)")
        self.btn_load.clicked.connect(self.load_image)
        top_layout.addWidget(self.btn_load)

        self.btn_save = QPushButton("Zapisz wynik")
        # self.btn_save.clicked.connect(self.save_result)
        self.btn_save.setVisible(False) 
        top_layout.addWidget(self.btn_save)
        
        top_layout.addStretch()
        layout.addLayout(top_layout)
        
        
        self.grid_widget = QWidget()
        self.grid = QGridLayout(self.grid_widget)
        self.grid.setSpacing(0)

        self.viewer = PrzegladarkaObrazow()
        self.viewer.setMinimumSize(600, 400)
        self.grid.addWidget(self.viewer, 1, 1)

        self.grid.setColumnStretch(1, 1)
        self.grid.setRowStretch(1, 1)
        
        layout.addWidget(self.grid_widget)

        
        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("← Wstecz")
        self.btn_next = QPushButton("Dalej →")
        self.btn_next.setEnabled(False)
        self.lbl_step = QLabel(f"Krok: {self.current_step} - Oczekiwanie na obraz")
        self.lbl_step.setAlignment(Qt.AlignCenter)
        
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.lbl_step, 1) # ,1 dodaje rozciąganie do środka
        nav_layout.addWidget(self.btn_next)
        
        layout.addLayout(nav_layout)
        
        self.btn_next.clicked.connect(self.next_step)
        self.btn_prev.clicked.connect(self.prev_step)
        
        self.setCentralWidget(main_widget)
        self.resize(800, 600)
        
    
    # Metody UI
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

    def process(self):
        if self.original_image is None: return

        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.is_cancelled = True
            self.worker.wait()

        self.btn_next.setEnabled(False)
        self.btn_prev.setEnabled(False)

        params = {
            'threshold': 128, # próg dla kroku 6
            'threshold_ratio': 0.33 # próg dla kroku 3 (segmentacja ROI)
        }

        self.worker = FingerprintWorker(self.original_image, self.current_step, params)
        self.worker.finished.connect(self.on_process_finished)
        self.worker.start()

    def on_process_finished(self, processed_img):
        self.current_processed_image = processed_img
        self.viewer.wyswietl_obraz_numpy(processed_img)
        
        tytuly_krokow = [
            "Krok 0: Oryginał", 
            "Krok 1: Skala szarości", 
            "Krok 2: Normalizacja",
            "Krok 3: Segmentacja (Maska ROI)",
            "Krok 4: Mapa Orientacji",
            "Krok 5: Filtr Gabora",
            "Krok 6: Binaryzacja",
            "Krok 7: Szkieletyzacja (LEWO: KMM | PRAWO: Morfologiczna)",
            "Krok 8: Detekcja Minucji (LEWO: KMM | PRAWO: Morfologiczna)"
        ]
        
        if self.current_step < len(tytuly_krokow):
            self.lbl_step.setText(tytuly_krokow[self.current_step])
        else:
            self.lbl_step.setText(f"Krok {self.current_step}")
        
        # Aktywacja przycisków
        self.btn_prev.setEnabled(self.current_step > 0)
        self.btn_next.setEnabled(self.current_step < 8) # 8 - ilość kroków
    
    
    def on_save_success(self, save_path):
        self.btn_save.setEnabled(True)
        self.btn_save.setText("Zapisz kod (BMP)")
        QMessageBox.information(self, "Sukces", f"Zapisano pomyślnie:\n{save_path}")

    def on_save_error(self, err_msg):
        self.btn_save.setEnabled(True)
        self.btn_save.setText("Zapisz kod (BMP)")
        QMessageBox.critical(self, "Błąd", f"Wystąpił błąd podczas zapisywania:\n{err_msg}")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
