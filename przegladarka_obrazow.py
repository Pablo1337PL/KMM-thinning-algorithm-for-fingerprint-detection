import numpy as np
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal

class PrzegladarkaObrazow(QGraphicsView):
    pixel_hovered = pyqtSignal(int, int, object)
    
    visible_rect_changed = pyqtSignal(int, int, int, int)

    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        
        self.obecny_pixmap = None
        self.obecny_obraz_numpy = None

        self.horizontalScrollBar().valueChanged.connect(self.emit_visible_rect)
        self.verticalScrollBar().valueChanged.connect(self.emit_visible_rect)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.emit_visible_rect()

    def wheelEvent(self, event):
        """Obsługa przybliżania/oddalania (Zoom)."""
        if self.obecny_pixmap is None:
            return

        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor
        zoom_factor = zoom_in_factor if event.angleDelta().y() > 0 else zoom_out_factor

        self.scale(zoom_factor, zoom_factor)
        
        self.emit_visible_rect()

    def wyswietl_obraz_numpy(self, img_array):
        img_array = np.ascontiguousarray(img_array)
        
        if len(img_array.shape) == 2:
            # Jeśli to skala szarości/binarka, powielamy 1 kanał 3 razy żeby udawał RGB
            img_array = np.stack([img_array, img_array, img_array], axis=-1)
        
        self.obecny_obraz_numpy = img_array
        
        height, width, channel = img_array.shape
        bytes_per_line = channel * width
        
        q_img = QImage(img_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.obecny_pixmap = QPixmap.fromImage(q_img.copy())
        
        self.scene.clear()
        self.scene.addPixmap(self.obecny_pixmap)
        self.setSceneRect(self.scene.itemsBoundingRect())
        
        self.emit_visible_rect()

    def emit_visible_rect(self):
        """Oblicza, jaki fragment obrazu widać aktualnie na ekranie."""
        if self.obecny_pixmap is None or self.obecny_obraz_numpy is None:
            return
            
        rect = self.mapToScene(self.viewport().rect()).boundingRect()
        img_rect = self.scene.itemsBoundingRect()
        
        rect = rect.intersected(img_rect)

        x = int(max(0, rect.x()))
        y = int(max(0, rect.y()))
        w = int(max(1, min(self.obecny_obraz_numpy.shape[1] - x, rect.width())))
        h = int(max(1, min(self.obecny_obraz_numpy.shape[0] - y, rect.height())))

        self.visible_rect_changed.emit(x, y, w, h)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if self.obecny_obraz_numpy is not None:
            pos = self.mapToScene(event.pos())
            x, y = int(pos.x()), int(pos.y())
            h, w = self.obecny_obraz_numpy.shape[:2]
            
            if 0 <= x < w and 0 <= y < h:
                pixel = self.obecny_obraz_numpy[y, x]
                self.pixel_hovered.emit(x, y, pixel)