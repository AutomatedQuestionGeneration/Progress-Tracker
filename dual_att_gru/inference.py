import tensorflow as tf
import os
import numpy as np
from dataset import load_df, create_bifurcated_data, create_dataset_squad
from preprocess import load_dataset, preprocess_sentence
from sklearn.model_selection import train_test_split
from model import Encoder, DecoderWithDualAttention, BahdanauAttention
def translate(g_context, r_context, encoder1, encoder2, decoder, targ_tokenizer):
  attention_plot1 = np.zeros((max_length_targ, max_length_given))
  attention_plot2 = np.zeros((max_length_targ, max_length_reference))

  g_context = preprocess_sentence(g_context)

  given = [given_tokenizer.word_index[i] for i in g_context.split(' ')]
  given = tf.keras.preprocessing.sequence.pad_sequences([given],
                                                         maxlen=max_length_given,
                                                         padding='post')
  given = tf.convert_to_tensor(given)

  
  r_context = preprocess_sentence(r_context)
  
  ref = [reference_tokenizer.word_index[i] for i in r_context.split(' ')]
  ref = tf.keras.preprocessing.sequence.pad_sequences([ref],
                                                         maxlen=max_length_reference,
                                                         padding='post')
  ref = tf.convert_to_tensor(ref)


  result = ''

  hidden = [tf.zeros((1, units))]
  enc_out1, enc_hidden1 = encoder1(given, hidden)
  enc_out2, enc_hidden2 = encoder2(ref, hidden)

  dec_hidden1 = enc_hidden1
  dec_hidden2 = enc_hidden2
  dec_input = tf.expand_dims([targ_tokenizer.word_index['<start>']], 0)

  for t in range(max_length_targ):
    # predictions, dec_hidden, attention_weights = decoder(dec_input,
    #                                                      dec_hidden,
    #                                                      enc_out)

    predictions, dec_hidden1, dec_hidden2, _, _ = decoder(dec_input, dec_hidden1, dec_hidden2, enc_out1, enc_out2)

    predicted_id = tf.argmax(predictions[0]).numpy()

    result += targ_tokenizer.index_word[predicted_id] + ' '

    # until the predicted word is <end>.
    if targ_tokenizer.index_word[predicted_id] == '<end>':
      return result

    # the predicted ID is fed back into the model, no teacher forcing.
    dec_input = tf.expand_dims([predicted_id], 0)

  return result

if __name__=="__main__":
    output_path_prefix = "scratch/scratch6/gokul/adg_project_data"
    df = load_df(os.path.join("..", "Topic_Bifurcated_SQUAD1.csv"))
    df = df[:1000]
    reference_context, given_context, target_ques = create_bifurcated_data(df)
    r, g, q = create_dataset_squad(reference_context, given_context, target_ques)
    reference_tensor, given_tensor, target_tensor, reference_tokenizer, given_tokenizer, targ_tokenizer = load_dataset(r, g, q)
    max_length_targ, max_length_reference, max_length_given = target_tensor.shape[1], reference_tensor.shape[1], given_tensor.shape[1]
    reference_tensor_train, reference_tensor_val, given_tensor_train, given_tensor_val, target_tensor_train, target_tensor_val = train_test_split(reference_tensor, given_tensor, target_tensor)
    BUFFER_SIZE = len(reference_tensor_train)
    BATCH_SIZE = 32
    steps_per_epoch = len(reference_tensor_train)//BATCH_SIZE
    steps_per_epoch_val = len(reference_tensor_val)//BATCH_SIZE
    embedding_dim = 256  # for word embedding
    units = 1024  # dimensionality of the output space of RNN
    vocab_inp_size = len(given_tokenizer.word_index) + len(reference_tokenizer.word_index) + 1
    vocab_tar_size = len(targ_tokenizer.word_index)+1

    validation_dataset = tf.data.Dataset.from_tensor_slices((reference_tensor_val, given_tensor_val, target_tensor_val)).shuffle(BUFFER_SIZE)
    validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=True)
    
    attention1 = BahdanauAttention(units)
    attention2 = BahdanauAttention(units)
    # encoder_check = tf.keras.models.load_model('./encoder1')
    # encoder_check2 = tf.keras.models.load_model('./encoder2')
    # decoder_check = tf.keras.models.load_model('./decoder')
    # encoder_check.compile(optimizer="Adam", loss="mse", metrics=["mae"])
    # encoder_check2.compile(optimizer="Adam", loss="mse", metrics=["mae"])
    # decoder_check.compile(optimizer="Adam", loss="mse", metrics=["mae"])
    encoder_check = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
    encoder_check2 = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
    decoder_check = DecoderWithDualAttention(vocab_tar_size, embedding_dim, units, BATCH_SIZE, attention1, attention2)
    # 
    encoder_check.load_weights("./encoder1")
    print("="*20)
    encoder_check2.load_weights("./encoder2")
    decoder_check.load_weights("./decoder")
    ques_prediction = translate("is my country. ","Its a really and beautiful place to live. ", encoder_check, encoder_check2, decoder_check, targ_tokenizer)
    print(ques_prediction)