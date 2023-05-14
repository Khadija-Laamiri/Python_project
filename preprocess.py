import fnmatch
import logging
import multiprocessing
import os
from multiprocessing.pool import Pool
import numpy as np
import random
import librosa

#the mel spectrogram of an oudia signal 
def calculatePowerSpectrogram(audio_data, samplerate, nbr_mels=128, nbr_fft=512, hop_length=160):
  #calcule de mel spectrogram and return a matrix of shape containing the mel spectrogram of te audio
  spect = librosa.feature.melspectrogram(audio_data, sr=samplerate, nbr_mels=nbr_mels, nbr_fft=nbr_fft, hop_length=hop_length)
  #convert a power spectrogram to a decibal scale using reference value
  spectLog = librosa.power_to_db(spect, ref=np.max)
  #normalize the mel spectrogram that was converted to db in the previous code
  spectNorm = (spectLog - np.mean(spectLog))/(np.std(spectLog))
  #shape: attribute of the numpy array returns a tuple of integers that represent the dimensions of the array.
  print((spectNorm.T).shape)
  return spectNorm.T
#This function takes in a sentence as input and returns a list of integer IDs that correspond to each letter in the
#  sentence. 
def sentenceToIds(sentence):
  return [letterToId(letter) for letter in sentence.lower()]
# the function letterToId to convert each letter to its corresponding ID.
# the function calculates the letter's ID by subtracting the ASCII value of 'a' from the ASCII value of the letter.
# For example, the letter 'a' has an ASCII value of 97, and 'a' - 'a' = 0, so the function returns ID 0 for the letter 'a'. Similarly, the letter 'b' has an ASCII value of 98, and 'b' - 'a' = 1, so the function returns ID 1 for the letter 'b'.
def letterToId(letter):
  if letter == ' ':
    return 27
  
  if letter == '\'':
    return 26
  
  return ord(letter) - ord('a')
# this function help us to search recursively  for a directory for a specific file pattern for eg (*.flac or *.trans.txt) 
# for audio and transcript files from LibriSpeech.
#Returns an iterator for the files found during the recursive traversal
# this function could be used to recursively traverse a directory containing audio files (e.g. WAV or MP3 files) that 
# contain speech, and then extract the text from those audio files using a speech-to-text transcription system.
def recursiveTraverse(directory, file_pattern):
  for root, dir_names, file_names in os.walk(directory):
    for filename in fnmatch.filter(file_names, file_pattern):
      #os.path.join(root, filename) is used to create the full path of each file that matches the file_pattern argument. 
      # This full file path is then yielded back to the caller using the yield keyword, which returns the value as a 
      # generator object that can be iterated over.
      yield os.path.join(root, filename)
class DatasetReader:
#  the class DatasetReader that reads and processes audio data for speech recognition tasks. 
#--init---     Initializes the DatasetReader instance with the path to the directory containing the data.
 def __init__(self, data_directory):
    self._data_directory = data_directory
    self._transcript_dict_cache = None
  # this method that returns a transcript dictionary which is built by calling the _build_transcript() method only 
  # if the _transcript_dict_cache attribute is not already set.
 def _transcript_dict(self):
    if not self._transcript_dict_cache:
      #_build_transcript: A method that builds (construire) a dictionary of transcripts, where the keys are audio IDs and the values
      #  are the corresponding transcripts converted to a list of integer IDs using a function sentenceToIds.
      self._transcript_dict_cache = self._build_transcript()
    return self._transcript_dict_cache
 #get_transcript_entries: A static method that extracts transcript entries from the transcript files (.trans.txt) in def _get_transcript_entries(transcript_directory): 
 def _get_transcript_entries(transcript_directory): 
    transcript_files = recursiveTraverse(transcript_directory, '*.trans.txt')
    for transcript_file in transcript_files:
      with open(transcript_file, 'r') as file:
        for line in file:
          #rstrip() method to remove any newline characters (\n) from the end of the line.
          line = line.rstrip('\n')
          #the split() method is used to split the line string into two parts, based on the first occurrence of the space
          #  character ' '. The second argument, 1, specifies the maximum number of splits to be performed on the string.
          # exemple :'Hello World', then splitted will be a list ['Hello', 'World']
          splitted = line.split(' ', 1)
          #the yield keyword is used to return each splitted line as a generator object.
          yield splitted
 #it provides a mapping between the spoken words and their corresponding text representations, which can be used to 
# improve the accuracy of the speech recognition system.
#it provides a mapping (correspondance)between the spoken words and their corresponding text representations, which can be used
#  to improve the accuracy of the speech recognition system.
 def create_transcript_mapping(self):
    transcript_dict = dict()
    for splitted in self._get_transcript_entries(self._data_directory):
      transcript_dict[splitted[0]] = sentenceToIds(splitted[1])
    return transcript_dict
 #method that extracts the audio ID from an audio file path. It does this by getting the base name of the file using the
 #  os.path.basename method, and then removes the file extension using os.path.splitext. The resulting audio ID is returned.
 def extract_audio_id(cls, audio_file):
    #For example, if the audio_file parameter is "/path/to/audio/file.wav", then the file_name variable would be "file.wav
    file_name = os.path.basename(audio_file)
    #The os.path.splitext function would then separate the file name into its name and extension components, resulting in
    #  a tuple ("file", ".wav"). Finally, the [0] index is used to extract the audio id "file" from the tuple.
    audio_id = os.path.splitext(file_name)[0]
    return audio_id
 #This function  takes an audio file path and a pre-processing function as input. It uses the librosa.load
 #  function to load the audio data and the sampling rate from the file. It then applies the given pre-processing function
 #  to the audio data and samplerate to obtain the audio fragments. Finally, it calls the _extract_audio_id function to
 #  extract the audio ID from the file path and returns both the audio ID and the audio fragments.
 def preprocess_audio_data(cls, audio_file, preprocess_fnc):
    audio_data, samplerate = librosa.load(audio_file)
    audio_fragments = preprocess_fnc(audio_data, samplerate)
    audio_id = cls.extract_audio_id(audio_file)
    return audio_id, audio_fragments  
 #save  the transcript [audio id and audio fragment] into an .npz file
 def prepare_audio_data_for_recognition(cls, audio_file, preprocess_fnc, transcript, out_directory):
   audio_id, audio_fragments = cls.preprocess_audio_data(audio_file, preprocess_fnc)
   # the np.savez function from the NumPy library to save the preprocessed audio data and its corresponding transcript to a NumPy .npz file.
   np.savez(out_directory + '/' + audio_id, audio_fragments=audio_fragments, transcript=transcript)    
 #This is a method that generates samples(echantillonage)for training a speech recognition model
 def generate_samples(self, directory, preprocess_fnc):
    #preprocess_fnc, which is a function that takes in raw audio data and returns preprocessed audio features.
    audio_files = list(recursiveTraverse(self._data_directory + '/' + directory, '*.flac'))
    transcript_dict = self._transcript_dict
    for audio_file in audio_files:
      audio_id, audio_fragments = self.preprocess_audio_data(audio_file, preprocess_fnc)
      if (audio_id in transcript_dict):
        yield audio_id, audio_fragments, transcript_dict[audio_id]
 # returns a directory path where preprocessed audio data is stored. 
 #the _get_directory method serves as a utility function for organizing the preprocessed audio data into subdirectories
 #  based on feature type and other factors.
 def get_directory(self, feature_type, sub_directory):
    #feature_type, which specifies the type of audio features to retrieve, and sub_directory, which specifies a 
    # subdirectory within the preprocessed data directory where the features should be stored.
    preprocess_directory = 'preprocessed'
    directory = self._data_directory + '/' + preprocess_directory + '/' + sub_directory
    
    return directory
 #This is a private method that serves as an error handler (gestion des erreurs)for the speech recognition project's preprocessing pipeline.
 def handle_preprocessing_error(cls, error: Exception):
    raise RuntimeError('Error during preprocessing') from error
 
 def store_samples(self, directory, preprocess_fnc):
    out_directory = self.get_directory(preprocess_fnc, directory)

    if not os.path.exists(out_directory):
      os.makedirs(out_directory)

    audio_files = list(recursiveTraverse(self._data_directory + '/' + directory, '*.flac'))

    with Pool(processes=multiprocessing.cpu_count()) as pool:
      transcript_dict = self._transcript_dict

      for audio_file in audio_files:
        audio_id = self._extract_audio_id(audio_file)

        if (audio_id in transcript_dict):
          transcript_entry = transcript_dict[audio_id]

        transform_args = (audio_file, preprocess_fnc, transcript_entry, out_directory)
        pool.apply_async(DatasetReader._transform_and_store_sample, transform_args, error_callback=self._preprocessing_error_callback)
#pool.close() is called to prevent any more tasks from being submitted to the pool
      pool.close()
#pool.join() is called to wait for all the worker processes to complete. This method blocks the main thread until all the
#  worker processes have finished executing      
      pool.join()      
#This function is used to load preprocessed audio samples from disk and return them as a generator.
#This function is used to load preprocessed audio samples from the disk.
 def load_samples(self, directory, max_size=False, loop_infinitely=False, limit_count=0, feature_type='mfcc'):
    load_directory = self._get_directory(feature_type, directory)

    if not os.path.exists(load_directory):
      raise ValueError('Directory {} does not exist'.format(load_directory))

    files = list(recursiveTraverse(load_directory, '*.npz'))
    random.shuffle(files)

    if limit_count:
      files = files[:limit_count]

    while True:
      for file in files:
        with np.load(file) as data:
          audio_length = data['audio_fragments'].shape[0]

          if not max_size or audio_length <= max_size:
            yield data['audio_fragments'], data['transcript']

          else:
            logging.warning('Audio snippet too long: {}'.format(audio_length))

      if not loop_infinitely:
        break

      random.shuffle(files) 
class Preprocess:

  def run(self):
    reader = DatasetReader('data')
    preprocess_fnc = calculatePowerSpectrogram
    reader.store_samples('train', preprocess_fnc)
    reader.store_samples('test', preprocess_fnc)
# class Preprocess with a method run() that performs some preprocessing on the audio data using the 
# calculatePowerSpectrogram function. It initializes a DatasetReader object with the data directory and then calls the 
# tore_samples method on it twice - once for the train directory and once for the test directory. The store_samples method
#  uses the calculatePowerSpectrogram function to preprocess the audio data and stores the preprocessed data in the
#  preprocessed directory within the respective train and test directories.