from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QApplication, QMainWindow

from src.src_recognizer_ui import Ui_MainWindow
from util import util
from net.net_recognizer import NN

import cv2
import numpy as np
import sys
import torch


class MyCore(QMainWindow, Ui_MainWindow):
	def __init__(self):
		super().__init__()
		self.setupUi(self)
		self.setWindowIcon(QIcon("../static/recognizer/logo.png"))

		util.pixmap(self.label_canvas, Qt.GlobalColor.white)
		util.pixmap(self.label_thumbnail, Qt.GlobalColor.black)
		self.label_canvas.mousePressEvent = self.mouse_press
		self.label_canvas.mouseMoveEvent = self.mouse_move

		self.model = NN().to(NN.DEVICE)
		self.model.load_state_dict(torch.load("../model/model_recognizer.pt", map_location=NN.DEVICE))
		self.model.eval()

	def keyPressEvent(self, event):
		if event.key() == Qt.Key.Key_Return:
			self.recognize()

	def mouse_press(self, event):
		self.label_canvas.pos = event.pos()

	def mouse_move(self, event):
		pixmap = self.label_canvas.pixmap()
		with QPainter(pixmap) as painter:
			painter.setPen(QPen(Qt.GlobalColor.black, 24))
			painter.drawLine(self.label_canvas.pos, event.pos())
		self.label_canvas.setPixmap(pixmap)
		self.label_canvas.pos = event.pos()

		thumbnail = self.preview()
		self.label_thumbnail.setPixmap(QPixmap(QImage(thumbnail, *thumbnail.shape, QImage.Format.Format_Indexed8)))

	def recognize(self):
		thumbnail = self.preview()
		tensor_feature = torch.tensor(thumbnail).view(1, 1, *thumbnail.shape).float().to(NN.DEVICE)
		with torch.no_grad():
			prediction = self.model(tensor_feature)
		self.label_result.setText(f"识别结果: {prediction.argmax().item()}")
		util.pixmap(self.label_canvas, Qt.GlobalColor.white)

	def preview(self):
		img = self.label_canvas.pixmap().toImage()
		bits = img.constBits()
		bits.setsize(img.sizeInBytes())
		binary_img = 255 - np.frombuffer(bits, dtype=np.uint8).reshape((400, 400, 4))[..., 0]
		return cv2.resize(binary_img, (28, 28), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
	app = QApplication(sys.argv)
	my_core = MyCore()
	my_core.setFixedSize(my_core.window().size())
	my_core.show()
	sys.exit(app.exec())
