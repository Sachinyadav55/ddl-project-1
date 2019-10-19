

#installing requirement : PyQt5
#pip install PyQt5==5.9.2

import sys,subprocess,argparse
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import pyqtSlot
from os import listdir
from os.path import isfile, join
import time



image = '00030.jpeg'
global index
index = 0

class App(QWidget):

    def __init__(self):
        super().__init__()
        parser = argparse.ArgumentParser(usage="python3 Label_Images.py <Path File> \nThis program is used to Label positive/negative instances from the dataset", description="")
        parser.add_argument('-v', '--version', action='version',version='%(prog)s 1.0', help="Show program's version number and exit.")
        parser.add_argument('DirectoryofImages',type =str, help='Path of the folder containing the images')
        args = parser.parse_args()

        global Directory
        global onlyfiles
        global image1
        global index
        Directory = args.DirectoryofImages
        onlyfiles = [f for f in listdir(Directory) if isfile(join(Directory, f))]
        onlyfiles.sort()
        print(onlyfiles)
        image1 = str(Directory)+str(onlyfiles[index])
        self.title = 'Labeling Images for Deep Learning '+ str( onlyfiles[index])
        self.left = 100
        self.top = 100
        self.width = 800
        self.height = 600
        self.initUI(image1)
        p1 = subprocess.Popen('mkdir Positive', shell=True)
        p1.communicate()
        p1 = subprocess.Popen('mkdir Negative', shell=True)
        p1.communicate()
            
    
    def initUI(self,image):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        label = QLabel(self)
        pixmap = QPixmap(image)
        label.setPixmap(pixmap)
        self.resize(pixmap.width(),pixmap.height())

        button = QPushButton('I see positive', self)
        button.setToolTip('I see a fish')
        button.move(20,70)
        button.clicked.connect(self.on_click)

        button = QPushButton('I see negative', self)
        button.setToolTip('This is an example button')
        button.move(150,70)
        button.clicked.connect(self.on_click2)

        button = QPushButton('Exit', self)
        button.setToolTip('This is an example button')
        button.move(80,200)
        button.clicked.connect(self.exit)

        self.show()
        
    @pyqtSlot()
    def on_click(self):
        print('PyQt5 button click')
        global index
        global onlyfiles
        image1 = str(Directory)+str(onlyfiles[index])
        subprocess.call(['cp ./'+ str(image1) + ' ./Positive' ],shell=True)
        self.close()
        if index == len(onlyfiles)-1:
            print('End of list reached Exiting')
            sys.exit()
        index = index + 1
        self.title = 'Labeling Images for Deep Learning '+ str(onlyfiles[index])
        image1 = str(Directory)+str(onlyfiles[index])
        self.initUI(image1)

    def on_click2(self):
        print('PyQt5 button click2')
        global index
        global onlyfiles
        image1 = str(Directory)+str(onlyfiles[index])
        subprocess.call(['cp ./'+str(image1)+ ' ./Negative' ],shell=True)
        self.close()
        if index == len(onlyfiles)-1:
            print('End of list reached Exiting')
            sys.exit()
        index =index + 1
        self.title = 'Labeling Images for Deep Learning '+ str(onlyfiles[index])
        image1 = str(Directory)+str(onlyfiles[index])
        self.initUI(image1)

    def exit(self):
        print('Exiting')
        self.close()
    
if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
