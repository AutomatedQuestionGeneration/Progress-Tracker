import pandas as pd
from preprocess import preprocess_sentence
def load_df(df_path):
    df = pd.read_csv(df_path)
    return df
def create_bifurcated_data(df):
    reference_context = df["reference_context"].apply(lambda x: x).tolist()
    given_context = df["given_context"].apply(lambda x: x).tolist()
    target_ques = df["question"].apply(lambda x: x).tolist()
    return reference_context, given_context, target_ques
def create_dataset_squad(context_param, given_param, target_ques):
    reference_context = []
    given_context = []
    question = []
    for c,g,q in zip(context_param,given_param, target_ques):
        reference_context.append(preprocess_sentence(str(c)))
        given_context.append(preprocess_sentence(str(g)))
        question.append(preprocess_sentence(q))
    return tuple(reference_context), tuple(given_context), tuple(question)


if __name__=="__main__":
    df = load_data("../Topic_Bifurcated_SQUAD1.csv")
    reference_context, given_context, target_ques = create_bifurcated_data(df)
    # print(ref)
    r, g, q = create_dataset_squad(reference_context, given_context, target_ques)