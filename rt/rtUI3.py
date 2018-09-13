from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from functools import partial
import os
import sys


from moldescriptors import get_features
from main import make_preds

class Stream(QtCore.QObject):
    newText = QtCore.pyqtSignal(str)

    def write(self, text):
        self.newText.emit(str(text))

class getFeatures(QThread):
    update_progress = pyqtSignal(float)
    
    def __init__(self,input_smiles,output_predictions,input_library,gui_object):
        QThread.__init__(self)
        self.input_smiles = input_smiles
        self.output_predictions = output_predictions
        self.input_library = input_library
        self.gui_object = gui_object
    
    def __del__(self):
        self.wait()

    def run(self):
        infile = open(self.input_smiles)
        if len(infile.readline().split(",")) > 2:
            get_features(infile_name=self.input_smiles,outfile_name=self.output_predictions,
                    library_file=self.input_library,id_index=0,mol_index=1,time_index=2,gui_object=self) #.gui_object
        else:
            get_features(infile_name=self.input_smiles,outfile_name=self.output_predictions,
                    library_file=self.input_library,id_index=0,mol_index=1,time_index=None,gui_object=self) #.gui_object
    
    def update_progress2(self,perc):
        self.update_progress.emit(perc)

class getPredictions(QThread):
    def __init__(self,reference_infile,pred_infile,outfile,gui_object,outfile_modname="",num_jobs=0):
        QThread.__init__(self)
        self.reference_infile = reference_infile
        self.pred_infile = pred_infile
        self.outfile = outfile
        self.gui_object = gui_object
        self.outfile_modname = outfile_modname
        self.num_jobs = num_jobs
    
    def __del__(self):
        self.wait()

    def run(self):
        make_preds(reference_infile=self.reference_infile,pred_infile=self.pred_infile,outfile=self.outfile,k=os.path.split(self.outfile_modname)[-1],outfile_modname=self.outfile_modname,num_jobs=self.num_jobs)
        
class OutLog:
    def __init__(self, edit, out=None, color=None):
        """(edit, out=None, color=None) -> can write stdout, stderr to a
        QTextEdit.
        edit = QTextEdit
        out = alternate stream ( can be the original sys.stdout )
        color = alternate color (i.e. color stderr a different color)
        """
        self.edit = edit
        self.out = None
        self.color = color

    def write(self, m):
        if self.color:
            tc = self.edit.textColor()
            self.edit.setTextColor(self.color)

        self.edit.moveCursor(QtGui.QTextCursor.End)
        self.edit.insertPlainText(m)

        if self.color:
            self.edit.setTextColor(tc)

        if self.out:
            self.out.write(m)


class Ui_Dialog(object):
    def __init__(self):
        sys.stdout = Stream(newText=self.onUpdateText)
    
    def onUpdateText(self, text):
        """Write console output to text widget."""
        cursor = self.textEdit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.textEdit.setTextCursor(cursor)
        self.textEdit.ensureCursorVisible()
    
    def browse_for_file(self,set_obj,use_dir=False):
        if use_dir:  
            dlg = QFileDialog() #QFileDialog.getExistingDirectory(QFileDialog.Directory)
            dlg.setFileMode(QFileDialog.Directory)
            dlg.setOption(QFileDialog.ShowDirsOnly, True)
        else: dlg = QFileDialog()
        #dlg.setFileMode(QFileDialog.AnyFile)
        #dlg.setFilter(["Text files (*.txt)"])
        #filenames = QStringList()
        
        if dlg.exec_(): filenames = dlg.selectedFiles()
        try: set_obj.setText(filenames[0])
        except: pass

    def run_feature_extraction(self):
        self.run_feat_extract.setEnabled(False)
        self.run_retention_pred.setEnabled(False)
        
        self.progressbar_feat.setProperty("value", 0)

        input_smiles = self.inputfield_smiles.displayText()
        input_library = self.inputfield_library.displayText()
        output_predictions = self.inputfield_feat_out.displayText()
        
        self.worker = getFeatures(input_smiles,output_predictions,input_library,self)
        self.worker.start()
        self.worker.update_progress.connect(self.updateProgressBar)
        #QtGui.QMainWindow.connect(self.worker,QtCore.SIGNAL("FEAT_PERC"),updateProgressBar)

        self.run_feat_extract.setEnabled(True)
        self.run_retention_pred.setEnabled(True)

    def run_pred(self):
        self.run_feat_extract.setEnabled(False)
        self.run_retention_pred.setEnabled(False)
        
        #self.progressbar_pred.setProperty("value", 0)

        outfile = self.inputfield_out_predict.displayText()
        reference_infile = self.inputfield_train.displayText()
        pred_infile = self.inputfield_predict.displayText()
        outfile_modname = self.inputfield_out_predict_2.displayText()
        num_jobs = self.inputfield_out_predict_3.displayText()

        self.worker = getPredictions(reference_infile,pred_infile,outfile,self,outfile_modname=outfile_modname,num_jobs=num_jobs)
        self.worker.start()

        self.run_feat_extract.setEnabled(True)
        self.run_retention_pred.setEnabled(True)

    def updateProgressBar(self,val):
        self.progressbar_feat.setProperty("value", val)

    def setupUi(self, Dialog):
        Dialog.setObjectName("CALLC")
        Dialog.resize(447, 689)
        self.horizontalLayoutWidget = QtWidgets.QWidget(Dialog)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 650, 421, 31))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        
        self.run_retention_pred = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.run_retention_pred.setObjectName("run_retention_pred")
        self.horizontalLayout.addWidget(self.run_retention_pred)
        self.run_retention_pred.clicked.connect(self.run_pred)
        
        self.gridLayoutWidget = QtWidgets.QWidget(Dialog)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(10, 60, 421, 161))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(1)
        self.gridLayout.setObjectName("gridLayout")
        
        self.label_3 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 4, 0, 1, 2)
        
        self.browse_feat_out = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.browse_feat_out.setObjectName("browse_feat_out")
        self.gridLayout.addWidget(self.browse_feat_out, 5, 0, 1, 1)
        
        self.label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 2)
        
        self.inputfield_smiles = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.inputfield_smiles.setObjectName("inputfield_smiles")
        self.gridLayout.addWidget(self.inputfield_smiles, 1, 1, 1, 1)
        
        self.inputfield_library = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.inputfield_library.setObjectName("inputfield_library")
        self.gridLayout.addWidget(self.inputfield_library, 3, 1, 1, 1)
        
        self.browse_smiles = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.browse_smiles.setObjectName("browse_smiles")
        self.gridLayout.addWidget(self.browse_smiles, 1, 0, 1, 1)
        
        self.inputfield_feat_out = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.inputfield_feat_out.setObjectName("inputfield_feat_out")
        self.gridLayout.addWidget(self.inputfield_feat_out, 5, 1, 1, 1)
        
        self.browse_library = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.browse_library.setObjectName("browse_library")
        self.gridLayout.addWidget(self.browse_library, 3, 0, 1, 1)
        
        self.label_2 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 2)
        
        self.progressbar_feat = QtWidgets.QProgressBar(self.gridLayoutWidget)
        self.progressbar_feat.setProperty("value", 0)
        self.progressbar_feat.setObjectName("progressbar_feat")
        self.gridLayout.addWidget(self.progressbar_feat, 6, 0, 1, 2)
        
        self.gridLayoutWidget_2 = QtWidgets.QWidget(Dialog)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(10, 320, 421, 191))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setSpacing(1)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.browse_out_predict = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        
        self.browse_out_predict.setObjectName("browse_out_predict")
        self.gridLayout_2.addWidget(self.browse_out_predict, 5, 0, 1, 1)
        
        self.browse_train = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.browse_train.setObjectName("browse_train")
        self.gridLayout_2.addWidget(self.browse_train, 1, 0, 1, 1)
        
        self.browse_predict = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.browse_predict.setObjectName("browse_predict")
        self.gridLayout_2.addWidget(self.browse_predict, 3, 0, 1, 1)
        
        self.label_4 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_4.setObjectName("label_4")
        self.gridLayout_2.addWidget(self.label_4, 4, 0, 1, 2)
        
        self.label_6 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_6.setObjectName("label_6")
        self.gridLayout_2.addWidget(self.label_6, 0, 0, 1, 2)
        
        self.inputfield_out_predict = QtWidgets.QLineEdit(self.gridLayoutWidget_2)
        self.inputfield_out_predict.setObjectName("inputfield_out_predict")
        self.gridLayout_2.addWidget(self.inputfield_out_predict, 5, 1, 1, 1)
        
        self.label_5 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_5.setObjectName("label_5")
        self.gridLayout_2.addWidget(self.label_5, 2, 0, 1, 2)
        
        self.inputfield_train = QtWidgets.QLineEdit(self.gridLayoutWidget_2)
        self.inputfield_train.setObjectName("inputfield_train")
        self.gridLayout_2.addWidget(self.inputfield_train, 1, 1, 1, 1)
        
        self.inputfield_predict = QtWidgets.QLineEdit(self.gridLayoutWidget_2)
        self.inputfield_predict.setObjectName("inputfield_predict")
        self.gridLayout_2.addWidget(self.inputfield_predict, 3, 1, 1, 1)
        
        self.label_9 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_9.setObjectName("label_9")
        self.gridLayout_2.addWidget(self.label_9, 6, 0, 1, 2)
        
        self.browse_out_predict_2 = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.browse_out_predict_2.setObjectName("browse_out_predict_2")
        self.gridLayout_2.addWidget(self.browse_out_predict_2, 7, 0, 1, 1)
        
        self.inputfield_out_predict_2 = QtWidgets.QLineEdit(self.gridLayoutWidget_2)
        self.inputfield_out_predict_2.setObjectName("inputfield_out_predict_2")
        self.gridLayout_2.addWidget(self.inputfield_out_predict_2, 7, 1, 1, 1)
        
        self.inputfield_out_predict_3 = QtWidgets.QLineEdit(self.gridLayoutWidget_2)
        self.inputfield_out_predict_3.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.inputfield_out_predict_3.setObjectName("inputfield_out_predict_3")
        
        self.gridLayout_2.addWidget(self.inputfield_out_predict_3, 10, 1, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_10.setObjectName("label_10")
        
        self.gridLayout_2.addWidget(self.label_10, 10, 0, 1, 1)
        self.line = QtWidgets.QFrame(Dialog)
        self.line.setGeometry(QtCore.QRect(10, 260, 421, 16))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        
        self.line_2 = QtWidgets.QFrame(Dialog)
        self.line_2.setGeometry(QtCore.QRect(10, 300, 421, 16))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        
        self.line_3 = QtWidgets.QFrame(Dialog)
        self.line_3.setGeometry(QtCore.QRect(10, 40, 421, 16))
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        
        self.line_4 = QtWidgets.QFrame(Dialog)
        self.line_4.setGeometry(QtCore.QRect(10, 0, 421, 16))
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(Dialog)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(10, 220, 421, 41))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        
        self.label_7 = QtWidgets.QLabel(Dialog)
        self.label_7.setGeometry(QtCore.QRect(10, 10, 421, 41))
        self.label_7.setObjectName("label_7")
        
        self.label_8 = QtWidgets.QLabel(Dialog)
        self.label_8.setGeometry(QtCore.QRect(10, 270, 421, 41))
        self.label_8.setObjectName("label_8")
        
        self.run_feat_extract = QtWidgets.QPushButton(Dialog)
        self.run_feat_extract.setGeometry(QtCore.QRect(10, 230, 419, 23))
        self.run_feat_extract.setObjectName("run_feat_extract")
        self.run_feat_extract.clicked.connect(self.run_feature_extraction)
        
        self.textEdit = QtWidgets.QTextEdit(Dialog)
        self.textEdit.setGeometry(QtCore.QRect(10, 520, 421, 121))
        self.textEdit.setObjectName("textEdit")
        
        self.browse_smiles.clicked.connect(partial(self.browse_for_file,self.inputfield_smiles,False))
        self.browse_feat_out.clicked.connect(partial(self.browse_for_file,self.inputfield_feat_out,False))
        self.browse_library.clicked.connect(partial(self.browse_for_file,self.inputfield_library,False))
        
        self.browse_predict.clicked.connect(partial(self.browse_for_file,self.inputfield_predict,False))
        self.browse_train.clicked.connect(partial(self.browse_for_file,self.inputfield_train,False))
        self.browse_out_predict.clicked.connect(partial(self.browse_for_file,self.inputfield_out_predict,False))
        self.browse_out_predict_2.clicked.connect(partial(self.browse_for_file,self.inputfield_out_predict_2,False))
        
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("CALLC", "CALLC"))
        self.run_retention_pred.setText(_translate("Dialog", "Run"))
        self.label_3.setText(_translate("Dialog", "Outputfile"))
        self.browse_feat_out.setText(_translate("Dialog", "Browse"))
        self.label.setText(_translate("Dialog", "Input smiles"))
        self.browse_smiles.setText(_translate("Dialog", "Browse"))
        self.browse_library.setText(_translate("Dialog", "Browse"))
        self.label_2.setText(_translate("Dialog", "Input library"))
        self.browse_out_predict.setText(_translate("Dialog", "Browse"))
        self.browse_train.setText(_translate("Dialog", "Browse"))
        self.browse_predict.setText(_translate("Dialog", "Browse"))
        self.label_4.setText(_translate("Dialog", "Outputfile predictions"))
        self.label_6.setText(_translate("Dialog", "Inputfile train molecules"))
        self.label_5.setText(_translate("Dialog", "Inputfile molecules to predict retention time"))
        self.label_9.setText(_translate("Dialog", "Outputfile models"))
        self.browse_out_predict_2.setText(_translate("Dialog", "Browse"))
        self.inputfield_out_predict_3.setText(_translate("Dialog", "4"))
        self.label_10.setText(_translate("Dialog", "Number of threads to start"))
        self.label_7.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-size:12pt;\">Feature extraction</span></p></body></html>"))
        self.label_8.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-size:12pt;\">Retention time prediction</span></p></body></html>"))
        self.run_feat_extract.setText(_translate("Dialog", "Run"))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    
    sys.exit(app.exec_())