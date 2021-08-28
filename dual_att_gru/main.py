import pandas as pd
import tensorflow as tf
from dataset import load_df, create_bifurcated_data, create_dataset_squad
from loss import loss_function
from preprocess import load_dataset
from model import Encoder, DecoderWithDualAttention, BahdanauAttention
from sklearn.model_selection import train_test_split
from train import get_train_step_func, training_seq2seq, caculate_validation_loss
import os

if __name__=="__main__":
    output_path_prefix = "/scratch/scratch6/gokul/adg_project_data"
    df = load_df(os.path.join(output_path_prefix, "Topic_Bifurcated_SQUAD1.csv"))
    # df = load_df(os.path.join("..", "Topic_Bifurcated_SQUAD1.csv"))
    df = df[:1000]
    reference_context, given_context, target_ques = create_bifurcated_data(df)
    r, g, q = create_dataset_squad(reference_context, given_context, target_ques)
    print(g[-1])
    reference_tensor, given_tensor, target_tensor, reference_tokenizer, given_tokenizer, targ_tokenizer = load_dataset(r, g, q)
    max_length_targ, max_length_reference, max_length_given = target_tensor.shape[1], reference_tensor.shape[1], given_tensor.shape[1]
    reference_tensor_train, reference_tensor_val, given_tensor_train, given_tensor_val, target_tensor_train, target_tensor_val = train_test_split(reference_tensor, given_tensor, target_tensor)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    optimizer = tf.keras.optimizers.Adam()
    # Configuration 
    BUFFER_SIZE = len(reference_tensor_train)
    BATCH_SIZE = 32
    steps_per_epoch = len(reference_tensor_train)//BATCH_SIZE
    steps_per_epoch_val = len(reference_tensor_val)//BATCH_SIZE
    embedding_dim = 256  # for word embedding
    units = 1024  # dimensionality of the output space of RNN
    vocab_inp_size = len(given_tokenizer.word_index) + len(reference_tokenizer.word_index) + 1
    vocab_tar_size = len(targ_tokenizer.word_index)+1

    dataset = tf.data.Dataset.from_tensor_slices((reference_tensor_train, given_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    validation_dataset = tf.data.Dataset.from_tensor_slices((reference_tensor_val, given_tensor_val, target_tensor_val)).shuffle(BUFFER_SIZE)
    validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=True)



    epochs = 1

    attention1 = BahdanauAttention(units)
    attention2 = BahdanauAttention(units)
    print("Running seq2seq model with Bahdanau attention")
    encoder1_bah, encoder_bah2, decoder_bah, training_loss, validation_loss = training_seq2seq(vocab_inp_size, embedding_dim, units, BATCH_SIZE, vocab_tar_size, 
                                                                                epochs, attention1, attention2, dataset, validation_dataset, steps_per_epoch,
                                                                                steps_per_epoch_val, targ_tokenizer, loss_object, optimizer)
    encoder1_bah.save_weights(os.path.join(output_path_prefix, "encoder1"))
    encoder_bah2.save_weights(os.path.join(output_path_prefix, "encoder2"))
    decoder_bah.save_weights(os.path.join(output_path_prefix, "decoder"))
    
    # encoder1_bah.save(os.path.join(output_path_prefix, "encoder1")
    # encoder_bah2.save(os.path.join(output_path_prefix, "encoder2")
    # decoder_bah.save(os.path.join(output_path_prefix, "decoder")
    
    # tloss = np.vstack((tloss, training_loss))
    # vloss = np.vstack((vloss, validation_loss))