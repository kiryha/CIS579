"""
Build a neuron network for handwritten digit recognition using existing labeled data sets.
Data source: www.kaggle.com/c/digit-recognizer
Data explanation: www.wikiwand.com/en/mnist_database

## Data
Source data is a set of 28x28 pixels images of handwritten digits (784 pixel values)
CSV files rows: digits tata (0, 1, 2, 3 ... 9), 42 000 digit examples
CSV file columns: label (digit), pixel 0, pixel 1, ... pixel 783

## Process
3 parts of training:
    - forward propagation
    - backward propagation
    - update parameters
"""
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PySide2 import QtWidgets, QtCore, QtGui
from ui import ui_main

from PIL import Image
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MatplotlibWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(MatplotlibWidget, self).__init__(parent)
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.axis('off')
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.canvas)

    def update_plot(self, image):
        self.ax.clear()
        self.ax.imshow(image, interpolation='nearest')
        self.ax.axis('off')
        self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)  # remove margins
        self.canvas.draw()


class Recognizer(QtWidgets.QMainWindow, ui_main.Ui_Recognizer):
    def __init__(self):
        super(Recognizer, self).__init__()
        self.setupUi(self)

        # Setup plot image display
        self.plot_widget = MatplotlibWidget(self)
        self.layImages.addWidget(self.plot_widget)
        # Show random noise in UI
        current_image = np.random.rand(100, 100)
        self.update_plot(current_image)

        # Model
        self.data = None  # MNIST csv
        self.W1_path = f'{root}/data/model/W1.csv'
        self.W2_path = f'{root}/data/model/W2.csv'
        self.b1_path = f'{root}/data/model/b1.csv'
        self.b2_path = f'{root}/data/model/b2.csv'
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.load_model()

        # UI calls
        self.btnLoadImage.clicked.connect(self.load_image)
        self.btnTeach.clicked.connect(self.teach_model)
        self.btnRecognize.clicked.connect(self.recognize)

    def load_model(self):
        """
        Load trained model if it exists
        """

        print('Loading train.csv...')

        # Load MNIST
        data_file = f"{root}/data/mnist/train.csv"
        data = pd.read_csv(data_file)
        self.data = np.array(data)

        # Split into DEV and TRAIN sets
        rows, columns = self.data.shape
        data_dev = self.data[0:1000].T
        data_train = self.data[1000:rows].T

        self.Y_dev = data_dev[0]
        self.X_dev = data_dev[1:columns]
        self.X_dev = self.X_dev / 255.

        self.Y_train = data_train[0]  # 784 items
        self.X_train = data_train[1:columns]
        self.X_train = self.X_train / 255.

        print('The train.csv loaded!')

        if not os.path.exists(self.W1_path):
            return

        print('Loading data...')
        self.W1 = np.loadtxt(self.W1_path, delimiter=',')
        self.W2 = np.loadtxt(self.W2_path, delimiter=',')
        self.b1 = np.loadtxt(self.b1_path, delimiter=',')
        self.b2 = np.loadtxt(self.b2_path, delimiter=',')
        print('Data loaded!')

    # Image Display
    def update_plot(self, image):
        self.plot_widget.update_plot(image)

    # ML functions
    def init_parameters(self):
        W1 = np.random.rand(10, 784) - 0.5
        b1 = np.random.rand(10, 1) - 0.5
        W2 = np.random.rand(10, 10) - 0.5
        b2 = np.random.rand(10, 1) - 0.5

        return W1, b1, W2, b2

    def rel_u(self, Z):

        return np.maximum(Z, 0)

    def rel_u_derivative(self, Z):

        return Z > 0

    def softmax(self, Z):

        A = np.exp(Z) / sum(np.exp(Z))

        return A

    def one_hot(self, Y):

        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T

        return one_hot_Y

    def get_predictions(self, A2):

        return np.argmax(A2, 0)

    def get_accuracy(self, predictions, Y):

        print(predictions, Y)

        return np.sum(predictions == Y) / Y.size

    # Propagation
    def forward_propagation(self, W1, b1, W2, b2, X):

        Z1 = W1.dot(X) + b1
        A1 = self.rel_u(Z1)
        Z2 = W2.dot(A1) + b2
        A2 = self.softmax(Z2)

        return Z1, A1, Z2, A2

    def backward_propagation(self, Z1, A1, Z2, A2, W1, W2, X, Y):

        m = Y.size
        one_hot_Y = self.one_hot(Y)
        dZ2 = A2 - one_hot_Y
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2)
        dZ1 = W2.T.dot(dZ2) * self.rel_u_derivative(Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1)

        return dW1, db1, dW2, db2

    def update_parameters(self, W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):

        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1
        W2 = W2 - alpha * dW2
        b2 = b2 - alpha * db2

        return W1, b1, W2, b2

    def gradient_descent(self, alpha, iterations):

        W1, b1, W2, b2 = self.init_parameters()

        for i in range(iterations):
            Z1, A1, Z2, A2 = self.forward_propagation(W1, b1, W2, b2, self.X_train)
            dW1, db1, dW2, db2 = self.backward_propagation(Z1, A1, Z2, A2, W1, W2, self.X_train, self.Y_train)
            W1, b1, W2, b2 = self.update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

            if i % 10 == 0:
                print(f"Iteration: {i}")
                predictions = self.get_predictions(A2)
                print(f'Accuracy: {self.get_accuracy(predictions, self.Y_train)}')

        return W1, b1, W2, b2

    # Recognition
    def make_predictions(self, X, W1, b1, W2, b2):

        _, _, _, A2 = self.forward_propagation(W1, b1, W2, b2, X)
        predictions = self.get_predictions(A2)

        return predictions

    def recognize_custom(self):
        """
        Recognize custom JPG
        """

        image_path = f'{root}/data/custom_images/1_01.jpg'
        img = Image.open(image_path)
        current_image = np.array(img)  # convert image to numpy array
        self.update_plot(current_image)

        prediction = self.make_predictions(current_image, self.W1, self.b1, self.W2, self.b2)

        # Report
        message = f'Number recognized as {prediction[0]}'
        print(message)
        self.statusbar.showMessage(message)

    def recognize_mnist(self, index):
        """
        Recognize image from DEV set by image data index
        """

        # Get image data from MNIST
        current_image = self.X_dev[:, index, None]
        prediction = self.make_predictions(self.X_dev[:, index, None], self.W1, self.b1, self.W2, self.b2)
        label = self.Y_dev[index]

        # Report
        message = f'Number {label} recognized as {prediction[0]}'
        print(message)
        self.statusbar.showMessage(message)

        # Show image in UI
        current_image = current_image.reshape((28, 28)) * 255
        self.update_plot(current_image)

    # UI calls
    def load_image(self):
        """
        Load custom jpg
        """

        image_path = f'{root}/data/custom_images/1_01.jpg'
        img = Image.open(image_path)
        current_image = np.array(img)  # convert image to numpy array
        self.update_plot(current_image)

    def teach_model(self):

        self.statusbar.showMessage('Teaching model...')

        alfa = float(self.linAlfa.text())
        iterations = int(self.linIterations.text())
        self.W1, self.b1, self.W2, self.b2 = self.gradient_descent(alfa, iterations)

        # Save data to CSV files
        np.savetxt(self.W1_path, self.W1, delimiter=',')
        np.savetxt(self.W2_path, self.W2, delimiter=',')
        np.savetxt(self.b1_path, self.b1, delimiter=',')
        np.savetxt(self.b2_path, self.b2, delimiter=',')

        self.statusbar.showMessage('Model trained and saved to files!')

    def recognize(self):

        # TODO: implement switch to rcognize from DEV or custom image

        # Recognize image from MNIST
        self.recognize_mnist(int(self.linIndex.text()))

        # Recognize custom image
        # self.recognize_custom()


if __name__ == "__main__":

    root = os.path.dirname(os.path.abspath(__file__))

    app = QtWidgets.QApplication([])
    recognizer = Recognizer()
    # recognizer.setWindowIcon(QtGui.QIcon('{0}/icons/split_smart.ico'.format(root)))
    recognizer.show()
    app.exec_()
