import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from fingerprint_processor import IrisProcessor

class IrisWorker(QThread):
    finished = pyqtSignal(object)

    def __init__(self, image, step, params):
        super().__init__()
        self.image = image
        self.step = step
        self.params = params
        self.is_cancelled = False

    def run(self):
        img = self.image.copy()

        if self.step == 0:
            pass
        elif self.step >= 1:
            img = IrisProcessor.to_grayscale(img)
            P = IrisProcessor.calculate_base_threshold(img)
            
            if self.step >= 2:
                x_param = self.params.get('x_param', 1.0)
                if x_param <= 0: x_param = 0.1 
                threshold_I = P / x_param
                img = np.where(img < threshold_I, 0, 255).astype(np.uint8)

                if self.step >= 3:
                    op1 = self.params.get('op1', 'Brak')
                    sz1 = self.params.get('sz1', 3)
                    img = IrisProcessor.apply_morphology(img, op1, sz1)
                    
                    op2 = self.params.get('op2', 'Brak')
                    sz2 = self.params.get('sz2', 3)
                    img = IrisProcessor.apply_morphology(img, op2, sz2)

                    if self.step >= 4:
                        # cx, cy = IrisProcessor.find_center_via_projections(img)
                        # img = IrisProcessor.draw_crosshair(img, cx, cy, size=30, color=(255, 0, 0))

                        cx, cy, pupil_radius = IrisProcessor.find_center_and_radius_via_n_projections(img)
                        img_cross = IrisProcessor.draw_crosshair_and_circle(img, cx, cy, pupil_radius, cross_size=30, color=(255, 0, 0))
                        
                        if self.step == 4:
                            img = img_cross

                        if self.step >= 5:
                            img_gray = IrisProcessor.to_grayscale(self.image.copy())
                            x_param_iris = self.params.get('x_param_iris', 1.0)
                            if x_param_iris <= 0: x_param_iris = 0.1 
                            threshold_P = P / x_param_iris
                            
                            img_iris_bin = np.where(img_gray < threshold_P, 0, 255).astype(np.uint8)
                            if self.step == 5:
                                img = img_iris_bin

                            if self.step >= 6:
                                op3 = self.params.get('op3', 'Brak')
                                sz3 = self.params.get('sz3', 3)
                                img_iris_bin = IrisProcessor.apply_morphology(img_iris_bin, op3, sz3)
                                
                                op4 = self.params.get('op4', 'Brak')
                                sz4 = self.params.get('sz4', 3)
                                img_iris_bin = IrisProcessor.apply_morphology(img_iris_bin, op4, sz4)

                                if self.step == 6:
                                    img = img_iris_bin
                                
                                
                                if self.step >= 7:
                                    iris_radius = IrisProcessor.find_iris_radius(img_iris_bin, cx, cy, pupil_radius)
                                    
                                    if self.step == 7:
                                        color_display = self.image.copy()
                                        color_display = IrisProcessor.draw_crosshair_and_circle(color_display, cx, cy, pupil_radius, color=(255, 0, 0))
                                        color_display = IrisProcessor.draw_crosshair_and_circle(color_display, cx, cy, iris_radius, color=(0, 255, 0))
                                        img = color_display

                                    
                                    if self.step >= 8:
                                        unwrapped = IrisProcessor.unwrap_iris(
                                            self.image, cx, cy, pupil_radius, iris_radius, width=360, height=60
                                        )
                                        if self.step == 8:
                                            img = unwrapped
                                        
                                        if self.step >= 9:
                                            f_val = self.params.get('f_frequency', 0.5)
                                            
                                            code = IrisProcessor.generate_iris_code(unwrapped, f=f_val)
                                            img = IrisProcessor.visualize_iris_code(code)
        if not self.is_cancelled:
            self.finished.emit(img)