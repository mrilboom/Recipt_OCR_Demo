# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("出租车发票检测系统")
        MainWindow.resize(482, 262)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.frame)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_1 = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(10)
        self.label_1.setFont(font)
        self.label_1.setTextFormat(QtCore.Qt.MarkdownText)
        self.label_1.setObjectName("label_1")
        self.gridLayout_3.addWidget(self.label_1, 0, 0, 1, 1)
        self.browse2 = QtWidgets.QPushButton(self.frame)
        self.browse2.setObjectName("browse2")
        self.gridLayout_3.addWidget(self.browse2, 1, 2, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.gridLayout_3.addWidget(self.label_2, 1, 0, 1, 1)
        self.text2 = QtWidgets.QLineEdit(self.frame)
        self.text2.setObjectName("text2")
        self.gridLayout_3.addWidget(self.text2, 1, 1, 1, 1)
        self.text1 = QtWidgets.QLineEdit(self.frame)
        self.text1.setObjectName("text1")
        self.gridLayout_3.addWidget(self.text1, 0, 1, 1, 1)
        self.sub_bt = QtWidgets.QPushButton(self.frame)
        self.sub_bt.setObjectName("sub_bt")
        self.gridLayout_3.addWidget(self.sub_bt, 2, 1, 1, 1)
        self.browse1 = QtWidgets.QPushButton(self.frame)
        self.browse1.setObjectName("browse1")
        self.gridLayout_3.addWidget(self.browse1, 0, 2, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_3, 0, 0, 1, 1)
        self.gridLayout_5.addLayout(self.gridLayout_4, 0, 0, 1, 1)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.checkBox10 = QtWidgets.QCheckBox(self.frame)
        self.checkBox10.setObjectName("checkBox10")
        self.gridLayout.addWidget(self.checkBox10, 2, 2, 1, 1)
        self.checkBox3 = QtWidgets.QCheckBox(self.frame)
        self.checkBox3.setObjectName("checkBox3")
        self.gridLayout.addWidget(self.checkBox3, 0, 3, 1, 1)
        self.checkBox7 = QtWidgets.QCheckBox(self.frame)
        self.checkBox7.setObjectName("checkBox7")
        self.gridLayout.addWidget(self.checkBox7, 1, 3, 1, 1)
        self.checkBox2 = QtWidgets.QCheckBox(self.frame)
        self.checkBox2.setObjectName("checkBox2")
        self.gridLayout.addWidget(self.checkBox2, 0, 2, 1, 1)
        self.checkBox5 = QtWidgets.QCheckBox(self.frame)
        self.checkBox5.setObjectName("checkBox5")
        self.gridLayout.addWidget(self.checkBox5, 1, 1, 1, 1)
        self.checkBox8 = QtWidgets.QCheckBox(self.frame)
        self.checkBox8.setObjectName("checkBox8")
        self.gridLayout.addWidget(self.checkBox8, 1, 4, 1, 1)
        self.checkBox6 = QtWidgets.QCheckBox(self.frame)
        self.checkBox6.setObjectName("checkBox6")
        self.gridLayout.addWidget(self.checkBox6, 1, 2, 1, 1)
        self.checkBox4 = QtWidgets.QCheckBox(self.frame)
        self.checkBox4.setObjectName("checkBox4")
        self.gridLayout.addWidget(self.checkBox4, 0, 4, 1, 1)
        self.checkBox0 = QtWidgets.QCheckBox(self.frame)
        self.checkBox0.setTristate(True)
        self.checkBox0.setObjectName("checkBox0")
        self.gridLayout.addWidget(self.checkBox0, 1, 5, 1, 1)
        self.checkBox11 = QtWidgets.QCheckBox(self.frame)
        self.checkBox11.setObjectName("checkBox11")
        self.gridLayout.addWidget(self.checkBox11, 2, 3, 1, 1)
        self.checkBox1 = QtWidgets.QCheckBox(self.frame)
        self.checkBox1.setObjectName("checkBox1")
        self.gridLayout.addWidget(self.checkBox1, 0, 1, 1, 1)
        self.checkBox9 = QtWidgets.QCheckBox(self.frame)
        self.checkBox9.setObjectName("checkBox9")
        self.gridLayout.addWidget(self.checkBox9, 2, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(10)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 0, 0, 2, 1)
        self.gridLayout_5.addLayout(self.gridLayout, 1, 0, 1, 1)
        self.gridLayout_2.addWidget(self.frame, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        # 槽连接部分
        for i in range(1, 12):
            eval("self.checkBox"+str(i)+".toggled.connect(self.get_check)")
        self.checkBox0.clicked.connect(self.set_check)

        self.browse1.clicked.connect(self.In_path)
        self.browse2.clicked.connect(self.Out_path)

        self.sub_bt.clicked.connect(self.run)

        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.setTabOrder(self.text1, self.text2)
        MainWindow.setTabOrder(self.text2, self.checkBox1)
        MainWindow.setTabOrder(self.checkBox1, self.checkBox2)
        MainWindow.setTabOrder(self.checkBox2, self.checkBox3)
        MainWindow.setTabOrder(self.checkBox3, self.checkBox4)
        MainWindow.setTabOrder(self.checkBox4, self.checkBox5)
        MainWindow.setTabOrder(self.checkBox5, self.checkBox6)
        MainWindow.setTabOrder(self.checkBox6, self.checkBox7)
        MainWindow.setTabOrder(self.checkBox7, self.checkBox8)
        MainWindow.setTabOrder(self.checkBox8, self.checkBox9)
        MainWindow.setTabOrder(self.checkBox9, self.checkBox10)
        MainWindow.setTabOrder(self.checkBox10, self.checkBox11)
        MainWindow.setTabOrder(self.checkBox11, self.checkBox0)
        MainWindow.setTabOrder(self.checkBox0, self.browse1)
        MainWindow.setTabOrder(self.browse1, self.browse2)
        MainWindow.setTabOrder(self.browse2, self.sub_bt)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_2.setText(_translate("MainWindow", "输出文件夹路径："))
        self.browse1.setText(_translate("MainWindow", "浏览"))
        self.browse2.setText(_translate("MainWindow", "浏览"))
        self.label_1.setText(_translate("MainWindow", "待识别图像文件夹路径："))
        self.sub_bt.setText(_translate("MainWindow", "提交检测"))
        self.label_5.setText(_translate("MainWindow", "识别项目："))
        self.checkBox1.setText(_translate("MainWindow", "发票代码"))
        self.checkBox2.setText(_translate("MainWindow", "发票号码"))
        self.checkBox6.setText(_translate("MainWindow", "上车"))
        self.checkBox5.setText(_translate("MainWindow", "日期"))
        self.checkBox3.setText(_translate("MainWindow", "车号"))
        self.checkBox4.setText(_translate("MainWindow", "证号"))
        self.checkBox8.setText(_translate("MainWindow", "单价"))
        self.checkBox7.setText(_translate("MainWindow", "下车"))
        self.checkBox10.setText(_translate("MainWindow", "等候"))
        self.checkBox9.setText(_translate("MainWindow", "里程"))
        self.checkBox11.setText(_translate("MainWindow", "金额"))
        self.checkBox0.setText(_translate("MainWindow", "全选/全不选"))


    def get_check(self):
        count = 0
        for i in range(1,12):
            if eval("self.checkBox"+str(i)).isChecked():
                count+=1
        if count==11:
            self.checkBox0.setCheckState(2)
        elif count==0:
            self.checkBox0.setCheckState(0)
        else:
            self.checkBox0.setCheckState(1)



    def set_check(self):
        state = self.checkBox0.checkState()
        # print(state)
        if state == 0:  # 2 全选，点一下之后为0，取消全选
            for i in range(1, 12):
                eval("self.checkBox" + str(i) + ".setCheckState(0)")
            self.checkBox0.setCheckState(0) # 处理结束后设置为空选状态
        else:   # 0  1  未选、半选，点一下之后为1，2
            for i in range(1, 12):
                eval("self.checkBox" + str(i) + ".setCheckState(2)")
            self.checkBox0.setCheckState(2) # 处理结束后设置为全选状态

    # 设置输入路径
    def In_path(self):
        dir_path = QFileDialog.getExistingDirectory(None, "choose directory", ".")
        self.text1.setText(dir_path)

    # 设置输出文件夹路径
    def Out_path(self):
        dir_path = QFileDialog.getExistingDirectory(None, "choose directory", ".")
        self.text2.setText(dir_path)


    # 开始识别的主函数入口
    def run(self):
        L = []
        # 复选框部分:
        for i in range(1,12):
            obj = eval("self.checkBox"+str(i))
            if obj.isChecked():
                L.append(obj.text()[-2:])
        print(L)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)  # 创建一个QApplication，也就是你要开发的软件app
    MainWindow = QtWidgets.QMainWindow()  # 创建一个QMainWindow，用来装载你需要的各种组件、控件
    ui = Ui_MainWindow()  # ui是你创建的ui类的实例化对象
    ui.setupUi(MainWindow)  # 执行类中的setupUi方法，方法的参数是第二步中创建的QMainWindow
    MainWindow.show()  # 执行QMainWindow的show()方法，显示这个QMainWindow
    sys.exit(app.exec_())  # 使用exit()或者点击关闭按钮退出QApplication