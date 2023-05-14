from sys import byteorder
from array import array
import pyaudio # we installed with pip install pyaudio
import numpy as np
import scipy as sp
import time
from audio import *

aud=AudioRecorder()

aud.record()