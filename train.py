import os
import time
import numpy as np
from test_executor import ExecuteurDeTests
from speechModel import create_default_model
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class Train(ExecuteurDeTests):

    def CreerGenerateurDEchantillons(self, limit_count: int):
        self.limit = limit_count
  	
        return self.lecteur.load_samples("train", loop_infinitely=True,limit_count=limit_count,feature_type='power')

    def ObtenirNmbrLimiteChargeur(self) -> int:
        return self.limit

    def create_model(self, sess):
        model = create_default_model('train', self.taille_entree, self.SaisieVocale)
        model.restore_or_create(sess,'train/best-weights',1e-4) # checkpoint_directory="train/best-weights"
    
        return model

    def run(self):
        with tf.Session() as sess:

            model = self.create_model(sess)
            coord = self.DémarrerPipeline(sess, n_threads=2)
            step_time, loss = 0.0, 0.0
            current_step = 0
            previous_losses = []

            try:
                print('Begin training')

                while not coord.should_stop():

                    current_step += 1
                    is_checkpoint_step = current_step % 1 == 0

                    start_time = time.time()
                    print(1)
                    step_result = model.step(sess, summary=is_checkpoint_step)
                    avg_loss = step_result[0]
                    step_time += (time.time() - start_time) / 1000
                    loss += avg_loss / 1000

                    # save the checkpoint and print the stats

                    if is_checkpoint_step:
                        print(6)
                        global_step = model.global_step.eval()
                        print(7)

                        # prints the stats for the previous step
                        perplexity = np.exp(float(avg_loss)) if avg_loss < 300 else float("inf")
                        print("global step {:d} learning rate {:.4f} step-time {:.2f} average loss {:.2f} perplexity {:.2f}".format(global_step, model.learning_rate.eval(), step_time, avg_loss, perplexity))
                
                        # store the summary
                        summary = step_result[2]
                        print('!9')
                        model.summary_writer.add_summary(summary, global_step)
                        print(10)
                        previous_losses.append(loss)
                        print(11)

                        #save the checkpoint inside the weights directory for faster access later
                        checkpoint_path = 'train/best-weights' + '/' + "speech.ckpt"
                        print(checkpoint_path)
                        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                        print('Weights saved')
                        step_time, loss = 0.0, 0.0
                        print(13)


            except tf.errors.OutOfRangeError:
                print('Done training.')
            
            finally:
                coord.request_stop()

            coord.join()
