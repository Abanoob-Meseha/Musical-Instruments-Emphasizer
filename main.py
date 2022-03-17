from operator import index
from re import X
import sounddevice as sd
from PyQt5 import QtWidgets
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from pyqtgraph import *
from pyqtgraph import PlotWidget, PlotItem
import pyqtgraph as pg
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import pathlib
import numpy as np
from pyqtgraph.Qt import _StringIO
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtGui import QIcon
from matplotlib.figure import Figure
import pyqtgraph.exporters
import math 
#from DSPTASK3_MUSIC import Ui_MainWindow
import winsound
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer

from scipy.io import wavfile
from finalgui import Ui_MainWindow, MplCanvas
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq, irfft
import cmath
from mpl_toolkits.axes_grid1 import make_axes_locatable
from Guitar import The_String_of_Guitar


class MainWindow(QtWidgets.QMainWindow):
    The_Amplitude = 4096
    The_Sound_of_Guitar =[]
    The_gains_of_sliders = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    the_min_levels = [1000, 500, 0, 2000, 5000]
    the_max_levels = [2000, 1000, 500, 5000, 22050]
    the_sample_rate_of_drum = 11000
    the_sample_rate_of_guitar = 8000
    the_samplerate_of_piano = 44100  # Hz
    wavetable_size = 200
    isPaused = False
    The_Keys_of_Drums = ["DR1", "DR2"]
    The_Keys_of_Piano = ['W', "w", 'Q', 'q', 'Y', 'H', 'L',
                  'O', 'o', 'T', 't', 'X', "E", "e",
                  "u", 'R', "r", 'G', 'U', 'u', 'I', 'i', 'Weq', 'OiL', 'Wqw', 'tEi', 'rHu', 'GuT', 'YwX', 'tRi', 'IoY',
                  'LrU', 'uTe', 'LiY', 'TEt', 'eWi', 'HqT'
                  ]

    The_Strings_of_guitar = ["GUITAR_1", "GUITAR_2", "GUITAR_3", "GUITAR_4", "GUITAR_5"]
    the_base_freq = 261.63
    freqs = [98, 123, 147, 196, 240]
    the_unit_delay = 1000
    Move_to_the_right = 0
    The_step = 0
    prob_drum1 = 1.0
    prob_drum2 = 0.3
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.guitar_frequencyCopy = self.freqs
        self.the_note_freqs = {self.The_Keys_of_Piano[key_indx]: self.the_base_freq * pow(2, (key_indx / 37))
                           for key_indx in range(37)}

        self.instrumentsSettings = [self.ui.guiarFrequencySlider, self.ui.drumProbabilitySlider]
        self.instrumentsSettings[0].sliderReleased.connect(self.set_guiarFrequency)
        self.instrumentsSettings[1].sliderReleased.connect(self.set_drumProbability)

        #______________________________IMPORTANT INSTANCE VARIABLES__________________________________________________________#
        self.the_delays = [self.the_unit_delay * freq for freq in range(len(self.freqs))]
        self.the_stretch_factors = [2 * f / 98 for f in self.freqs]
        self.timer1 = QtCore.QTimer()
        self.startTimeIdx = 0
        self.zoomFactor = 1
        self.Timer = [self.timer1]
        self.GraphicsView=[self.ui.graphicsView_3]#ALL GRAPHIGSVIEW TO USE THEM WIH JUST INDEX
        self.white=mkPen(color=(255,255, 255))#white
        self.Color1=mkPen(color=(255, 0, 0))#RED
        self.Color2=mkPen(color=(0, 255, 0))#GREEN
        self.Color3=mkPen(color=(0, 0, 255))#BLUE
        self.Color4=mkPen(color=(255, 200, 200), style=QtCore.Qt.DotLine)#Dotted pale-red line
        self.Color5=mkPen(color=(200, 255, 200), style=QtCore.Qt.DotLine)#Dotted pale-green line
        self.Color6=mkPen(color=(200, 200, 255), style=QtCore.Qt.DotLine)## Dotted pale-blue line
        self.COLOR_Pen=[self.Color1,self.Color2,self.Color3,self.Color4,self.Color5,self.Color6]#STORE COLORS TO BE USED WITH INDEX
        self.The_Sliders = [self.ui.PIANO_SLIDER, self.ui.Bass_Guitar, self.ui.Bass_violin, self.ui.Trumpet, self.ui.electrophone]
        for x in range(len(self.The_Sliders)):
            self.make_the_sliders_work(x)




        
        #___________________________________________CONNECTING BUTTONS WITH THEIR FUNCTIONS_______________________________________#
        self.ui.Open.triggered.connect(lambda: self.open())
        self.ui.Clear.triggered.connect(lambda: self.clear())
        self.ui.VOLUME_SLIDER.valueChanged.connect(lambda: self.change_Volume())
        self.ui.PLAY.clicked.connect(lambda: self.play_msc())
        self.th = {}
        #-----------------------------------------------------------------------#
        self.ui.PIANO.clicked.connect(lambda: self.PIANO_PAGE())
        self.ui.DRUMS.clicked.connect(lambda: self.DRUMS_PAGE())
        self.ui.GUITAR.clicked.connect(lambda: self.GUITAR_PAGE())
        #---------------------------------------------------------------#
        self.ui.PIANO_W_7.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())
        self.ui.PIANO_W_8.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())
        self.ui.PIANO_W_9.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())
        self.ui.PIANO_W_10.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())
        self.ui.PIANO_W_11.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())
        self.ui.PIANO_W_12.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())
        self.ui.PIANO_W_13.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())
        self.ui.PIANO_W_14.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())

        self.ui.PIANO_W_15.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())
        self.ui.PIANO_W_16.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())
        self.ui.PIANO_W_17.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())
        self.ui.PIANO_W_18.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())
        self.ui.PIANO_W_19.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())
        self.ui.PIANO_W_20.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())
        self.ui.PIANO_W_21.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())
        self.ui.PIANO_W_22.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())

        self.ui.PIANO_B_1.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())
        self.ui.PIANO_B_2.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())
        self.ui.PIANO_B_3.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())
        self.ui.PIANO_B_4.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())
        self.ui.PIANO_B_5.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())
        self.ui.PIANO_B_6.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())
        self.ui.PIANO_B_7.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())
        self.ui.PIANO_B_8.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())
        self.ui.PIANO_B_9.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())
        self.ui.PIANO_B_10.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())

        self.ui.PIANO_B_11.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())
        self.ui.PIANO_B_12.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())
        self.ui.PIANO_B_13.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())
        self.ui.PIANO_B_14.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())
        self.ui.PIANO_B_15.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())

        self.ui.PIANO_W_6.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())
        self.ui.PIANO_W_1.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())
        self.ui.PIANO_W_2.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())
        self.ui.PIANO_W_3.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())
        self.ui.PIANO_W_4.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())
        self.ui.PIANO_W_5.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())

        #--------------------------------------------------------------#
        self.ui.GUITAR_1.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())
        self.ui.GUITAR_2.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())
        self.ui.GUITAR_3.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())
        self.ui.GUITAR_4.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())
        self.ui.GUITAR_5.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())
        #--------------------------------------------------------------#
        self.ui.DR1.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())
        self.ui.DR2.clicked.connect(lambda: self.Generating_The_Sounds_of_Musical_Instruments())
        



# TAB 1 functioms
    
    def open(self):        #open the music you need to listen
        for i in range(len(self.The_Sliders)):
            self.The_Sliders[i].setValue(1)
            self.The_gains_of_sliders[i] = 1.0           #reset all the sliders of equalizer to 1
        files_name = QFileDialog.getOpenFileName(self, 'Open only wav', os.getenv('HOME'), "wav(*.wav)")
        self.path = files_name[0]
        print(self.path)
        full_file_path = self.path
        self.url = QUrl.fromLocalFile(full_file_path)
        if pathlib.Path(self.path).suffix == ".wav":
            self.samplerate, self.data = wavfile.read(self.path)
            self.amplitude = np.int32((self.data))
            self.time = np.linspace(0., len(self.data) / self.samplerate, len(self.data))
            self.maxAmplitude = self.amplitude.max()
            self.minAmplitude = self.amplitude.min()
            self.endTimeIdx = int(self.samplerate * self.zoomFactor) - 1
            self.The_zeros = np.zeros(self.data.size)
            sd.play(self.data,self.samplerate)
            self.The_fft_of_the_signal = np.fft.rfft(self.data)
            self.new_fft_after_modification = np.copy(self.The_fft_of_the_signal)
            self.The_freq_of_fft = np.fft.rfftfreq(n=len(self.data), d=1 / self.samplerate)       #Get the Freq of fft
            self.plot()
            self.ui.PLAY.setText("STOP")
            self.The_Step = 0
        self.SPECTROGRAM(self.amplitude)

########
    def make_the_sliders_work(self, indx):
        self.The_Sliders[indx].sliderReleased.connect(lambda: self.getting_the_value_of_slider_and_the_new_signal(indx))
#######
    def getting_the_value_of_slider_and_the_new_signal(self, indx):
        self.The_slider_value = self.The_Sliders[indx].value()
        self.The_slider_value += 0.00001
        self.Controlling_The_Frequency(self.the_min_levels[indx], self.the_max_levels[indx], The_Level=indx, The_Gain=self.The_slider_value)

    def update_plot_data(self):  # ------------>>UPDATE THE VALUES FOR A LIVE SIGNAL<<
        self.Move_to_the_right = self.The_Step + 2
        self.The_Step += 0.11
        self.GraphicsView[0].plotItem.setXRange(self.The_Step, self.Move_to_the_right)
        if int(self.The_Step) == int(self.time[-1]):
            self.Timer[0].stop()
#########
    def Update_the_signal_after_signal_modification(self, bol_play=True):     # update our signal after changing the value of sliders of equalizer
        self.The_mid_of_x_axis = (self.The_Step + self.Move_to_the_right) / 2
        self.The_point_of_start = (self.The_mid_of_x_axis / (len(self.data) / self.samplerate)) * len(self.data)
        if (bol_play):
            sd.play(self.data[int(self.The_point_of_start):], self.samplerate)


    def plot(self):
        self.GraphicsView[0].clear()
        self.GraphicsView[0].setXRange(self.time[self.startTimeIdx], self.time[self.endTimeIdx])
        self.GraphicsView[0].setYRange(self.minAmplitude, self.maxAmplitude)
        self.pen = pg.mkPen(color=(121, 161, 60))
        self.GraphicsView[0].plot(self.time, self.amplitude, pen=self.pen)
        # self.GraphicsView[0].plotItem.showGrid(True, True, alpha=1)
        self.Timer[0].setInterval(100)
        if self.Move_to_the_right == 0:
            self.Timer[0].timeout.connect(lambda: self.update_plot_data())

        self.Timer[0].start()




    def SPECTROGRAM(self, data):#-------------------->>DRAW THE SPECTROGRAM<<
        for i in reversed(range(self.ui.verticalLayout_4.count())):
            self.ui.verticalLayout_4.itemAt(i).widget().deleteLater()
        self.ui.The_Spectro = MplCanvas(self.ui.verticalLayoutWidget_4, width=5, height=5, dpi=100)
        self.ui.verticalLayout_4.addWidget(self.ui.The_Spectro)
        spec, freqs, t, im = self.ui.The_Spectro.axes.specgram(data,Fs=self.samplerate,cmap='plasma')
        self.ui.The_Spectro.figure.colorbar(im).set_label('Intensity [dB]')
        self.ui.The_Spectro.draw()


#########
    def getting_the_ifft(self, signal_after_controlling_the_freq):
        bol_play = True
        self.data = np.fft.irfft(signal_after_controlling_the_freq)
        self.GraphicsView[0].clear()
        self.data = np.ascontiguousarray(self.data, dtype=np.int32)
        self.Normalized_data = self.data / self.data.max()
        self.data = (self.ui.VOLUME_SLIDER.value() / 100) * self.Normalized_data
        if self.ui.PLAY.text() == "PLAY":
            bol_play = False
        self.Update_the_signal_after_signal_modification(bol_play)
        self.GraphicsView[0].plot(self.time, self.data)
        self.SPECTROGRAM(self.data)




    def Controlling_The_Frequency(self, Min_Freq, Max_Freq, The_Level, The_Gain):           # comparing the frequency with the levels of frequency and change its value according to the value of the sliders
        self.new_fft_after_modification[(self.The_freq_of_fft >= Min_Freq) & (self.The_freq_of_fft <= Max_Freq)] = \
            self.new_fft_after_modification[(self.The_freq_of_fft >= Min_Freq) & (self.The_freq_of_fft <= Max_Freq)] / self.The_gains_of_sliders[The_Level]
        self.The_gains_of_sliders[The_Level] = The_Gain
        self.new_fft_after_modification[(self.The_freq_of_fft >= Min_Freq) & (self.The_freq_of_fft <= Max_Freq)] = \
            self.new_fft_after_modification[(self.The_freq_of_fft >= Min_Freq) & (self.The_freq_of_fft <= Max_Freq)] * self.The_gains_of_sliders[The_Level]

        self.getting_the_ifft(self.new_fft_after_modification)

#########

    def change_Volume(self):
        bol_play = True
        self.vol = self.ui.VOLUME_SLIDER.value() / 100
        if self.vol == 0:
            sd.play(self.The_zeros, self.samplerate)
        else:
            self.Normalized_data = self.data / self.data.max()
            self.data = (self.ui.VOLUME_SLIDER.value() / 100) * self.Normalized_data
            if self.ui.PLAY.text() == "PLAY":
                bol_play = False
            self.Update_the_signal_after_signal_modification(bol_play)

    def play_msc(self):
        self.Update_the_signal_after_signal_modification(False)
        if self.ui.PLAY.text() == "STOP":
            self.isPaused = True
            sd.stop()
            self.Timer[0].stop()
            self.ui.PLAY.setText("PLAY")
        else:
            self.isPaused = False
            self.Normalized_data = self.data / self.data.max()
            self.data = (self.ui.VOLUME_SLIDER.value() / 100) * self.Normalized_data
            sd.play(self.data[int(self.The_point_of_start):], self.samplerate)
            self.Timer[0].start()
            if int(self.The_Step) == int(self.time[-1]):
                self.Timer[0].stop()
                sd.stop()
            self.ui.PLAY.setText("STOP")



    def clear(self):#------------------------------>>CLEAR THE 2 GRAPHS<<
        sd.stop()
        self.Timer[0].stop()
        self.GraphicsView[0].clear()

#    TAB 2 Functions

    # _____________________________________________BUTTTONS FUNCTIONS_______________________________________________________#
    def PIANO_PAGE(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.HOME_PAGE)

    def DRUMS_PAGE(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.DRUMS_PAGE)

    def GUITAR_PAGE(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.GUITAR_PAGE)

        # ---------------------------------------------------------------------------------------------------#



    def get_the_data_of_chord(self, the_chords):
        the_chords = the_chords.split('-')
        chord_data = []
        for chord in the_chords:
            the_key_data = sum([self.getting_the_wave(self.the_note_freqs[piano_key]) for piano_key in list(chord)])
            chord_data.append(the_key_data)

        chord_data = np.concatenate(chord_data, axis=0)
        return chord_data.astype(np.int16)

    def getting_the_wave(self, freq, the_duration_time=0.5):
        wave_on_x_axis = np.linspace(0, the_duration_time, int(self.the_samplerate_of_piano * the_duration_time))
        The_Wave = self.The_Amplitude * np.sin(2 * np.pi * freq * wave_on_x_axis)
        return The_Wave

    def getting_the_data_of_song(self, The_Note_of_music):
        if len(self.ui.The_Main_Window.sender().objectName()) > 1:
            The_Generated_Tune = self.get_the_data_of_chord(self.ui.The_Main_Window.sender().objectName())
        else:
            The_Generated_Tune = [self.getting_the_wave(self.the_note_freqs[The_Note_of_music])]
            The_Generated_Tune = np.concatenate(The_Generated_Tune)

        The_Generated_Tune = The_Generated_Tune * (16300 / np.max(The_Generated_Tune))
        return The_Generated_Tune.astype(np.int16)

    def set_guiarFrequency(self):
        self.freqs = [-element * self.instrumentsSettings[0].value() for element in
                                 self.guitar_frequencyCopy]

    def get_the_sound_of_drums(self, The_wavetable_of_Drum, N_Samples, The_Prop):
        Samples = []
        The_Present_Sample = 0
        The_Previous_Sample = 0
        while len(Samples) < N_Samples:
            drawn_samples = np.random.binomial(1, The_Prop)
            sign = float(drawn_samples == 1) * 2 - 1
            The_wavetable_of_Drum[The_Present_Sample] = sign * 0.5 * (The_wavetable_of_Drum[The_Present_Sample] + The_Previous_Sample)
            Samples.append(The_wavetable_of_Drum[The_Present_Sample])
            The_Previous_Sample = Samples[-1]
            The_Present_Sample += 1
            The_Present_Sample = The_Present_Sample % The_wavetable_of_Drum.size
        return np.array(Samples)

    def set_drumProbability(self):
        self.prob_drum1 = 1

        self.prob_drum1 -= self.instrumentsSettings[1].value() / 10
        self.prob_drum2 = self.instrumentsSettings[1].value() / 10

    def Generating_The_Sounds_of_Musical_Instruments(self):
        if self.ui.The_Main_Window.sender().objectName() in self.The_Keys_of_Piano:
            The_Data_of_Key_or_Chord = self.getting_the_data_of_song(self.ui.The_Main_Window.sender().objectName())

            sd.play(The_Data_of_Key_or_Chord, self.the_samplerate_of_piano)

        elif self.ui.The_Main_Window.sender().objectName() in self.The_Keys_of_Drums:
            The_wavetable_of_Drum = np.ones(self.wavetable_size)
            if self.ui.The_Main_Window.sender().objectName() == "DR1":
                The_Data_of_Drum = self.get_the_sound_of_drums(The_wavetable_of_Drum, self.the_sample_rate_of_drum, self.prob_drum1)
            else:
                The_Data_of_Drum = self.get_the_sound_of_drums(The_wavetable_of_Drum, self.the_sample_rate_of_drum, self.prob_drum2)

            sd.play(The_Data_of_Drum, self.the_sample_rate_of_drum)

        elif self.ui.The_Main_Window.sender().objectName() in self.The_Strings_of_guitar:
            if self.ui.The_Main_Window.sender().objectName() == "GUITAR_1":
                self.choose_the_string_of_guitar(1)

            elif self.ui.The_Main_Window.sender().objectName() == "GUITAR_2":
                self.choose_the_string_of_guitar(2)

            elif self.ui.The_Main_Window.sender().objectName() == "GUITAR_3":
                self.choose_the_string_of_guitar(3)

            elif self.ui.The_Main_Window.sender().objectName() == "GUITAR_4":
                self.choose_the_string_of_guitar(4)

            elif self.ui.The_Main_Window.sender().objectName() == "GUITAR_5":
                self.choose_the_string_of_guitar(5)

            sd.play(self.The_Sound_of_Guitar, self.the_sample_rate_of_guitar)

    def choose_the_string_of_guitar(self,num):
        String = The_String_of_Guitar(self.freqs[num-1], self.the_delays[num-1], self.the_sample_rate_of_guitar,
                               self.the_stretch_factors[num-1])

        self.The_Sound_of_Guitar = [String.Getting_the_Sample() for sample in range(self.the_sample_rate_of_guitar)]






#---------------------------------END OF MAINWINDOW CLASS---------------------------------------------#


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())