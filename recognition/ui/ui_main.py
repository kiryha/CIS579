# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ui_main.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_Recognizer(object):
    def setupUi(self, Recognizer):
        if not Recognizer.objectName():
            Recognizer.setObjectName(u"Recognizer")
        Recognizer.resize(280, 429)
        self.centralwidget = QWidget(Recognizer)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.splitter = QSplitter(self.centralwidget)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Horizontal)
        self.label = QLabel(self.splitter)
        self.label.setObjectName(u"label")
        self.splitter.addWidget(self.label)
        self.linAlfa = QLineEdit(self.splitter)
        self.linAlfa.setObjectName(u"linAlfa")
        self.linAlfa.setMaximumSize(QSize(150, 16777215))
        self.splitter.addWidget(self.linAlfa)

        self.verticalLayout.addWidget(self.splitter)

        self.splitter_2 = QSplitter(self.centralwidget)
        self.splitter_2.setObjectName(u"splitter_2")
        self.splitter_2.setOrientation(Qt.Horizontal)
        self.label_2 = QLabel(self.splitter_2)
        self.label_2.setObjectName(u"label_2")
        self.splitter_2.addWidget(self.label_2)
        self.linIterations = QLineEdit(self.splitter_2)
        self.linIterations.setObjectName(u"linIterations")
        self.linIterations.setMaximumSize(QSize(150, 16777215))
        self.splitter_2.addWidget(self.linIterations)

        self.verticalLayout.addWidget(self.splitter_2)

        self.btnTeach = QPushButton(self.centralwidget)
        self.btnTeach.setObjectName(u"btnTeach")
        self.btnTeach.setMinimumSize(QSize(0, 35))

        self.verticalLayout.addWidget(self.btnTeach)

        self.line = QFrame(self.centralwidget)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.verticalLayout.addWidget(self.line)

        self.labImage = QLabel(self.centralwidget)
        self.labImage.setObjectName(u"labImage")

        self.verticalLayout.addWidget(self.labImage)

        self.btnLoadImage = QPushButton(self.centralwidget)
        self.btnLoadImage.setObjectName(u"btnLoadImage")
        self.btnLoadImage.setMinimumSize(QSize(0, 35))

        self.verticalLayout.addWidget(self.btnLoadImage)

        self.btnRecognize = QPushButton(self.centralwidget)
        self.btnRecognize.setObjectName(u"btnRecognize")
        self.btnRecognize.setMinimumSize(QSize(0, 35))

        self.verticalLayout.addWidget(self.btnRecognize)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)

        Recognizer.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(Recognizer)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 280, 21))
        Recognizer.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(Recognizer)
        self.statusbar.setObjectName(u"statusbar")
        Recognizer.setStatusBar(self.statusbar)

        self.retranslateUi(Recognizer)

        QMetaObject.connectSlotsByName(Recognizer)
    # setupUi

    def retranslateUi(self, Recognizer):
        Recognizer.setWindowTitle(QCoreApplication.translate("Recognizer", u"Number Recognizer", None))
        self.label.setText(QCoreApplication.translate("Recognizer", u"Alfa: ", None))
        self.linAlfa.setText(QCoreApplication.translate("Recognizer", u"0.1", None))
        self.label_2.setText(QCoreApplication.translate("Recognizer", u"Iterations: ", None))
        self.linIterations.setText(QCoreApplication.translate("Recognizer", u"500", None))
        self.btnTeach.setText(QCoreApplication.translate("Recognizer", u"Teach Model", None))
        self.labImage.setText("")
        self.btnLoadImage.setText(QCoreApplication.translate("Recognizer", u"Load Image", None))
        self.btnRecognize.setText(QCoreApplication.translate("Recognizer", u"Recognize Image Number", None))
    # retranslateUi
