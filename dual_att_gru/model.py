import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,  # Whether to return the last output in the output sequence, or the full sequence. 
                                       return_state=True,  # Whether to return the last state in addition to the output.
                                       recurrent_initializer='glorot_uniform')
  
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state
  
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
  
    def call(self, query, values):
      # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        query_with_time_axis = tf.expand_dims(query, 1)
        values = query_with_time_axis
        # print("values.shape", values.shape)
        # print("query_with_time_axis.shape", query_with_time_axis.shape)
    
        # (64, 358, 1024) ->values.shape
        # (64, 1, 1024) -> query_with_time_axis.shape
    
        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(query_with_time_axis)))
        # print("self.W1(values).shape ->", self.W1(values).shape)
        # print("self.W2(query_with_time_axis).shape" , self.W2(query_with_time_axis).shape)
        # print("score.shape", score.shape)
        # (64, 358, 1024) -> self.W1(values).shape
        # (64, 1, 1024) -> self.W2(query_with_time_axis).shape
        # (64, 358, 1) -> score.shape
    
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
    
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        # print(context_vector.shape)
        # (64, 358, 1024)
    
        context_vector = tf.reduce_sum(context_vector, axis=1)
        # print(context_vector.shape)
        # (64, 1024)
        return context_vector, attention_weights
class DecoderWithDualAttention(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, attention_layer1 = None, attention_layer2 = None):
    super(DecoderWithDualAttention, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # used for attention
    self.attention1 = attention_layer1
    self.attention2 = attention_layer2

  def call(self, x, hidden1, hidden2, enc_output1, enc_output2):
    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)
    y = x
    attention_weights1 = None   
    
    if self.attention1:
      # enc_output shape == (batch_size, max_length, hidden_size)
      context_vector1, attention_weights1 = self.attention1(hidden1, enc_output1)
      # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
      # print(x.shape)
      x = tf.concat([tf.expand_dims(context_vector1, 1), x], axis=-1)
      # print(x.shape)
    
    attention_weights2 = None
    
    if self.attention2:
      # enc_output shape == (batch_size, max_length, hidden_size)
      context_vector2, attention_weights2 = self.attention2(hidden2, enc_output2)
      # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
      y = tf.concat([tf.expand_dims(context_vector2, 1), y], axis=-1)


    # passing the concatenated vector to the GRU
    output1, state1 = self.gru(x, initial_state = hidden1)
    output2, state2 = self.gru(y, initial_state = hidden2)

    # output shape == (batch_size * 1, hidden_size)
    output1 = tf.reshape(output1, (-1, output1.shape[2]))
    output2 = tf.reshape(output2, (-1, output2.shape[2]))

    output = tf.concat([output1, output2], axis = 1)

    # output shape == (batch_size, vocab)
    final_output = self.fc(output)

    return final_output, state1, state2, attention_weights1, attention_weights2
if __name__=="__main__":
    encoder = Encoder(config.vocab_inp_size, embedding_dim, units, BATCH_SIZE)

    # sample input
    sample_hidden = encoder.initialize_hidden_state()
    sample_output, sample_hidden = encoder(example_reference_batch, sample_hidden)
    print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
    print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))