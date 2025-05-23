# Form implementation generated from reading ui file 'E:/MyWorkspace/Python/project-ml/src/vis_recognizer_ui.ui'
#
# Created by: PyQt6 UI code generator 6.7.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 601)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setContentsMargins(-1, 0, -1, -1)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(30)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_timer = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_timer.setObjectName("label_timer")
        self.horizontalLayout.addWidget(self.label_timer)
        self.label_epoch = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_epoch.setObjectName("label_epoch")
        self.horizontalLayout.addWidget(self.label_epoch)
        self.label_step = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_step.setObjectName("label_step")
        self.horizontalLayout.addWidget(self.label_step)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.widget_loss = QtWidgets.QWidget(parent=self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_loss.sizePolicy().hasHeightForWidth())
        self.widget_loss.setSizePolicy(sizePolicy)
        self.widget_loss.setObjectName("widget_loss")
        self.verticalLayout.addWidget(self.widget_loss)
        self.widget_accuracy = QtWidgets.QWidget(parent=self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_accuracy.sizePolicy().hasHeightForWidth())
        self.widget_accuracy.setSizePolicy(sizePolicy)
        self.widget_accuracy.setObjectName("widget_accuracy")
        self.verticalLayout.addWidget(self.widget_accuracy)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "手写数字识别-训练平台"))
        self.label_timer.setText(_translate("MainWindow", "Time spent: 00:00:00"))
        self.label_epoch.setText(_translate("MainWindow", "Current epoch: 0"))
        self.label_step.setText(_translate("MainWindow", "Current step: 0"))
