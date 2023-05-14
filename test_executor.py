from abc import abstractmethod, ABCMeta
from functools import partial
import tensorflow as tf
from speech_input import charge_plusieur_entre
from Speech_Model import create_default_model
from preprocess import DatasetReader
class ExecuteurDeTests(metaclass=ABCMeta):
    def __init__(self):
      self.lecteur = DatasetReader('data')
      self.taille_entree = self.determine_input_size()
      self.SaisieVocale = charge_plusieur_entre(self.input_size, 64, partial(self.CreerGenerateurDEchantillons, self.get_loader_limit_count()), self.get_max_steps())

    def DeterminerTailleEntree(self):
      return next(self.CreerGenerateurDEchantillons(limit_count=1))[0].shape[1]

    def ObtenirMaxEtapes (self):
      return None

    @abstractmethod
    def ObtenirNmbrLimiteChargeur (self) -> int:
      raise NotImplementedError('Le nombre limite de chargeur doit être mis en œuvre')

    @abstractmethod
    def CreerGenerateurDEchantillons (self, limit_count: int):
      raise NotImplementedError('La création du générateur d échantillons doit être implémentée')

    def DémarrerPipeline(self, sess, n_threads=1):
      coord = tf.train.Coordinator()
      tf.train.start_queue_runners(sess=sess, coord=coord)
      self.speech_input.start_threads(sess=sess, coord=coord, n_threads=n_threads)
      return coord

    def CreerModele (self, sess):
      model = create_default_model('evaluate', self.input_size, self.speech_input)
      model.restore(sess, 'train/best-weights')
    
      return model