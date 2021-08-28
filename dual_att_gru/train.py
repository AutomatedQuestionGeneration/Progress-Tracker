import tensorflow as tf
from model import Encoder, DecoderWithDualAttention
import time
from loss import loss_function
def get_train_step_func(BATCH_SIZE, loss_object, optimizer):

  @tf.function
  def train_step(inp, targ, aux, enc_hidden1, enc_hidden2, encoder1, encoder2, decoder, targ_tokenizer):
    loss = 0

    with tf.GradientTape() as tape: # for automatic differentiation
      enc_output1, enc_hidden1 = encoder1(inp, enc_hidden1)
      enc_output2, enc_hidden2 = encoder2(aux, enc_hidden2)

      dec_hidden1 = enc_hidden1
      dec_hidden2 = enc_hidden2
      # enc_hidden -> 

      # Check whether we should concatenate dec_hidden and dec_hidden2 or keep it separate
      # dec_hidden2 = enc_hidden2

      dec_input = tf.expand_dims([targ_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

      # Teacher forcing - feeding the target as the next input
      for t in range(1, targ.shape[1]):
        # passing enc_output to the decoder
        predictions, dec_hidden1, dec_hidden2, _, _ = decoder(dec_input, dec_hidden1, dec_hidden2, enc_output1, enc_output2)
        # x, y, hidden1, hidden2, enc_output1, enc_output2

        loss += loss_function(targ[:, t], predictions, loss_object)

        # using teacher forcing
        dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder1.trainable_variables + encoder2.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss
    
  return train_step

def training_seq2seq(vocab_inp_size, embedding_dim, units, BATCH_SIZE, vocab_tar_size, 
                    epochs, attention1, attention2, dataset, validation_dataset, steps_per_epoch,
                    steps_per_epoch_val, targ_tokenizer, loss_object, optimizer):
  encoder1 = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
  encoder2 = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
  decoder = DecoderWithDualAttention(vocab_tar_size, embedding_dim, units, BATCH_SIZE, attention1, attention2)
  train_step_func = get_train_step_func(BATCH_SIZE, loss_object, optimizer)
  training_loss = []
  validation_loss = []

  for epoch in range(epochs):
    # print()
    # print(epoch)
    start = time.time()
    enc_hidden1 = encoder1.initialize_hidden_state()
    enc_hidden2 = encoder2.initialize_hidden_state()
    total_loss = 0

    for (batch, (aux, inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
    #   print(batch)
      batch_loss = train_step_func(inp, targ, aux, enc_hidden1, enc_hidden2, encoder1, encoder2, decoder, targ_tokenizer)
      # inp, targ, aux, enc_hidden, enc_hidden2, encoder, encoder2, decoder
      total_loss += batch_loss

      if batch % 100 == 0:
        print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss))
        
    enc_hidden1 = encoder1.initialize_hidden_state()
    enc_hidden2 = encoder2.initialize_hidden_state()
    total_val_loss = 0
    for (batch, (aux, inp, targ)) in enumerate(validation_dataset.take(steps_per_epoch)):
      val_loss = caculate_validation_loss(inp, targ, aux, enc_hidden1, enc_hidden2, encoder1, encoder2, decoder, targ_tokenizer, BATCH_SIZE, loss_object)
      total_val_loss += val_loss

    training_loss.append(total_loss / steps_per_epoch)
    validation_loss.append(total_val_loss / steps_per_epoch_val)
    print('Epoch {} Loss {:.4f} Validation Loss {:.4f}'.format(epoch + 1,
                                        training_loss[-1], validation_loss[-1]))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
  return encoder1, encoder2, decoder, training_loss, validation_loss

def caculate_validation_loss(inp, targ, aux, enc_hidden1, enc_hidden2, encoder1, encoder2, decoder, targ_tokenizer, BATCH_SIZE, loss_object):
  loss = 0
  enc_output1, enc_hidden1 = encoder1(inp, enc_hidden1)
  enc_output2, enc_hidden2 = encoder2(aux, enc_hidden1)
  dec_hidden1 = enc_hidden1
  dec_hidden2 = enc_hidden2
  dec_input = tf.expand_dims([targ_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

  # Teacher forcing - feeding the target as the next input
  # for t in range(1, targ.shape[1]):
  #   predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
  #   loss += loss_function(targ[:, t], predictions)
  #   dec_input = tf.expand_dims(targ[:, t], 1)

  # loss = loss / int(targ.shape[1])
  # return loss

  for t in range(1, targ.shape[1]):
        # passing enc_output to the decoder
        predictions, dec_hidden1, dec_hidden2, _, _ = decoder(dec_input, dec_hidden1, dec_hidden2, enc_output1, enc_output2)
        # x, y, hidden1, hidden2, enc_output1, enc_output2

        loss += loss_function(targ[:, t], predictions, loss_object)

        # using teacher forcing
        dec_input = tf.expand_dims(targ[:, t], 1)

  loss = loss / int(targ.shape[1])
  return loss