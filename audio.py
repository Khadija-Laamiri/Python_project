from sys import byteorder
from array import array
import pyaudio # we installed with pip install pyaudio
import numpy as np
import scipy as sp
# import time
import os
import sounddevice as sd


class AudioRecorder:

    def __init__(self):
        self.rate=44100 #the audio signal is sampled 16000 times per second
        self.threshold=0.03 # silence threshold
        self.chunk_size=1024 #number of audio samples to be read at a time
        self.audio_format=pyaudio.paInt16
        self.audio_handler=pyaudio.PyAudio()
        self.channels=2 #create a PyAudio object that we can use to perform various audio operations
        
    #adjust the amplitude 
    def normalize_audio(self,data):
        
        #normalization factor that scales the data to a specific range, while preserving its relative shape and dynamics
        max_value = np.max(np.abs(data))
        normalization_factor = float(0.5) / max_value
        normalized_data = [i * normalization_factor for i in data]
        return normalized_data
    
    def is_silence(self,data):
        # test if an audio is silence
        if self.threshold > np.max(data):
            return True
        else:
            return False

    
    #removing any leading or trailing blanks...
    def trim(self,data):

        def trim_data(data):
            trimming_started=False
            trimmed_data=array('f')
            for i in data:
                if not trimming_started and self.threshold < abs(i):
                    trimming_started=True
                    trimmed_data.append(i)
                elif trimming_started:
                    trimmed_data.append(i)
            
            return trimmed_data
        #trim left side
        data = trim_data(data)
        data.reverse()

        #trim right side
        data=trim_data(data)
        data.reverse()

        return data
    
    def add_silence(self,data):
        #adds silence to the start and end to ensure that the entire audio signal is captured and precessed

        audio_with_silence=array('f',[0 for i in range(int(0.1*self.rate))]) 
        audio_with_silence.extend(data) #adds the audio data to the end of the silence array
        audio_with_silence.extend([0 for i in range(int(0.1*self.rate))]) # adds another 0.1s of silence to the end of the array
        return audio_with_silence

    #Records words from the microphone using pyaudio in paFloat32 format and 16000Hz sampling rate.
    #Returns data as an array of signed floats.
    def record(self):
        audio_stream = self.audio_handler.open(format=self.audio_format, channels=self.channels, rate=self.rate, input=True, frames_per_buffer=self.chunk_size)
        numSilent=0 #detect when the input audio has stopped
        recording_started=False # determine the beginning of the audio input stream
        recorded_audio = array('f')

        while 1: 
            #keep running until the recording complete
            data=audio_stream.read(self.chunk_size)

            # if byteorder=='big':
            #     data.byteswap() #important because audio data typically stored in little-endian byte order

            recorded_audio.extend(data)
            width=self.audio_handler.get_sample_size(self.audio_format)
            audio_silent = self.is_silence(data)
            if audio_silent and recording_started:
                numSilent+=1

            elif not audio_silent and not recording_started:
                recording_started=True
            
            if recording_started:# and numSilent > 30 
                #the user stopped speaking
                for i in range(5000):
                    print(i)
                break
        
        audio_stream.stop_stream()
        audio_stream.close()
        recorded_audio = self.normalize_audio(recorded_audio) #normalizes the audio
        recorded_audio = self.trim(recorded_audio) 
        recorded_audio = self.add_silence(recorded_audio)
        print(type(recorded_audio))

        sp.io.wavfile.write("audios/out.wav",self.rate,np.asarray(recorded_audio).astype(np.float32))
        return recorded_audio, width

    def terminate(self):
        self.audio_handler.terminate()




           



                
