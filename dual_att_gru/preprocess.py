import unicodedata
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split
# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
  
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
  
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",","¿")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
  
    # remove extra space
    w = w.strip()
  
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w

# Tokenize the sentence into list of words(integers) and pad the sequence to the same length
def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='')
    lang_tokenizer.fit_on_texts(lang)
  
    tensor = lang_tokenizer.texts_to_sequences(lang)
  
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                           padding='post')
    return tensor, lang_tokenizer
def load_dataset(reference, given, target):
    # creating cleaned input, output pairs
    #targ_lang, inp_lang = create_dataset(path, num_examples)
  
    reference_tensor, reference_tokenizer = tokenize(reference)
    given_tensor, given_tokenizer = tokenize(given)
    target_tensor, targ_lang_tokenizer = tokenize(target)

    return reference_tensor, given_tensor, target_tensor, reference_tokenizer, given_tokenizer, targ_lang_tokenizer

if __name__=="__main__":
    from dataset import load_df, create_bifurcated_data, create_dataset_squad
    df = load_df("../Topic_Bifurcated_SQUAD1.csv")
    df = df[:1000]
    reference_context, given_context, target_ques = create_bifurcated_data(df)
    r, g, q = create_dataset_squad(reference_context, given_context, target_ques)
    print(g[-1])
    reference_tensor, given_tensor, target_tensor, reference_tokenizer, given_tokenizer, targ_tokenizer = load_dataset(r, g, q)

    # Calculate max_length of the target tensors
    max_length_targ, max_length_reference, max_length_given = target_tensor.shape[1], reference_tensor.shape[1], given_tensor.shape[1]
    print(max_length_targ, max_length_reference, max_length_given)
    reference_tensor_train, reference_tensor_val, given_tensor_train, given_tensor_val, target_tensor_train, target_tensor_val = train_test_split(reference_tensor, given_tensor, target_tensor)
    print(given_tensor_train.shape, reference_tensor_train.shape)