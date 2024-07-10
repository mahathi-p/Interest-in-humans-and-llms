import pandas as pd
from typing import List, Union, Tuple
import spacy
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from edu_convokit import uptake_utils
from scipy.special import softmax
import logging
import re
from typing import Dict, List, Any
from scipy.special import softmax
from cleantext import clean
import torch
from transformers import BertTokenizer
from num2words import num2words

from utils import clean_str, clean_str_nopunct
from utils import MultiHeadModel, BertInputBuilder, get_num_words


UPTAKE_HF_MODEL_NAME = 'ddemszky/uptake-model'
UPTAKE_MIN_NUM_WORDS_SPEAKER_A = 3
HIGH_UPTAKE_THRESHOLD = 0.5
UPTAKE_MAX_INPUT_LENGTH = 120

def number_to_words(num):
    try:
        return num2words(re.sub(",", "", num))
    except:
        return num
    
clean_str_nopunct = lambda s: clean(s,
                            fix_unicode=True,  # fix various unicode errors
                            to_ascii=True,  # transliterate to closest ASCII representation
                            lower=True,  # lowercase text
                            no_line_breaks=True,  # fully strip line breaks as opposed to only normalizing them
                            no_urls=True,  # replace all URLs with a special token
                            no_emails=True,  # replace all email addresses with a special token
                            no_phone_numbers=True,  # replace all phone numbers with a special token
                            no_numbers=True,  # replace all numbers with a special token
                            no_digits=False,  # replace all digits with a special token
                            no_currency_symbols=False,  # replace all currency symbols with a special token
                            no_punct=True,  # fully remove punctuation
                            replace_with_url="<URL>",
                            replace_with_email="<EMAIL>",
                            replace_with_phone_number="<PHONE>",
                            replace_with_number=lambda m: number_to_words(m.group()),
                            replace_with_digit="0",
                            replace_with_currency_symbol="<CUR>",
                            lang="en"
                            )


clean_str = lambda s: clean(s,
                            fix_unicode=True,  # fix various unicode errors
                            to_ascii=True,  # transliterate to closest ASCII representation
                            lower=True,  # lowercase text
                            no_line_breaks=True,  # fully strip line breaks as opposed to only normalizing them
                            no_urls=True,  # replace all URLs with a special token
                            no_emails=True,  # replace all email addresses with a special token
                            no_phone_numbers=True,  # replace all phone numbers with a special token
                            no_numbers=True,  # replace all numbers with a special token
                            no_digits=False,  # replace all digits with a special token
                            no_currency_symbols=False,  # replace all currency symbols with a special token
                            no_punct=False,  # fully remove punctuation
                            replace_with_url="<URL>",
                            replace_with_email="<EMAIL>",
                            replace_with_phone_number="<PHONE>",
                            replace_with_number=lambda m: number_to_words(m.group()),
                            replace_with_digit="0",
                            replace_with_currency_symbol="<CUR>",
                            lang="en"
                            )


def initialize(path="."):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    input_builder = BertInputBuilder(tokenizer=tokenizer)
    model = MultiHeadModel.from_pretrained(path, head2size={"nsp": 2})
    model.to(device)
    model.eval()
    return input_builder, device, model

def get_clean_text(text, remove_punct=False):
    if remove_punct:
        return clean_str_nopunct(text)
    return clean_str(text)


class Annotator:
    """
        Annotator class for edu-convokit. Contains methods for annotating data.
    """
    def __init__(self):
        pass

    def _populate_analysis_unit(
            self,
            df: pd.DataFrame,
            analysis_unit: str,
            text_column: str,
            time_start_column: str,
            time_end_column: str,
            output_column: str,
            ) -> pd.DataFrame:
        """
        Populate output_column with number of words, sentences, or timestamps.
        """

        if analysis_unit == "words":
            df[output_column] = df[text_column].str.split().str.len()
        elif analysis_unit == "sentences":
            # Use nlp to split text into sentences
            nlp = spacy.load("en_core_web_sm")
            df[output_column] = df[text_column].apply(lambda x: len(list(nlp(x).sents)))
        elif analysis_unit == "timestamps":
            # Check type of time_start_column and time_end_column
            if df[time_start_column].dtype != "float64":
                df[time_start_column] = df[time_start_column].astype("float64")
            if df[time_end_column].dtype != "float64":
                df[time_end_column] = df[time_end_column].astype("float64")
            df[output_column] = df[time_end_column] - df[time_start_column]
        else:
            raise ValueError(f"Analysis unit {analysis_unit} not supported.")
        return df


def get_uptake_prediction(model, device, instance):
        instance["attention_mask"] = [[1] * len(instance["input_ids"])]
        for key in ["input_ids", "token_type_ids", "attention_mask"]:
            instance[key] = torch.tensor(instance[key]).unsqueeze(0)  # Batch size = 1
            instance[key] = instance[key].to(device)

        output = model(input_ids=instance["input_ids"],
                        attention_mask=instance["attention_mask"],
                        token_type_ids=instance["token_type_ids"],
                        return_pooler_output=False)

        return output

def get_uptake(
        df: pd.DataFrame,
        text_column: str,
        output_column: str,
        speaker_column: str, # Mandatory because we are interested in measuring speaker2's uptake of speaker1's words
        speaker1: Union[str, List[str]], # speaker1 is the student
        speaker2: Union[str, List[str]], # speaker2 is the teacher
        result_type: str = "binary", # raw: uptake score, binary: 1 if uptake score > threshold, 0 otherwise
    ) -> pd.DataFrame:
        """
        Get uptake predictions for a dataframe.
        Following the implementation here:
        https://huggingface.co/ddemszky/uptake-model/blob/main/handler.py

        Arguments:
            df (pd.DataFrame): dataframe to analyze
            text_column (str): name of column containing text to analyze
            output_column (str): name of column to store result
            speaker_column (str): name of column containing speaker names.
            speaker1 (str or list): speaker1 is the student
            speaker2 (str or list): speaker2 is the teacher
            result_type (str): raw or binary

        Returns:
            df (pd.DataFrame): dataframe with uptake predictions
        """

    #     logging.warning("""Note: This model was trained on teacher's uptake of student's utterances. So, speaker1 should be the student and speaker2 should be the teacher.
    # For more details on the model, see https://arxiv.org/pdf/2106.03873.pdf""")

    #     logging.warning("""Note: It's recommended that you merge utterances from the same speaker before running this model. You can do that with edu_convokit.text_preprocessing.merge_utterances_from_same_speaker.""")

        assert text_column in df.columns, f"Text column {text_column} not found in dataframe."
        assert speaker_column in df.columns, f"Speaker column {speaker_column} not found in dataframe."

        if output_column in df.columns:
            logging.warning(f"Target column {output_column} already exists in dataframe. Skipping.")
            return df

        if isinstance(speaker1, str):
            speaker1 = [speaker1]

        if isinstance(speaker2, str):
            speaker2 = [speaker2]

        # Uptake model is run slightly differently. So this is a separate function.
        input_builder, device, model = initialize(UPTAKE_HF_MODEL_NAME)

        predictions = []

        with torch.no_grad():
            for i, row in df.iterrows():
                if i == 0:
                    predictions.append(None)
                    continue

                s1 = df[speaker_column].iloc[i-1] ##
                s2 = df[speaker_column].iloc[i]
                textA = df[text_column].iloc[i-1]
                textB = df[text_column].iloc[i]

                # Skip if text is too short
                if len(textA.split()) < UPTAKE_MIN_NUM_WORDS_SPEAKER_A:
                    predictions.append(None)
                    continue

                if s1 in speaker1 and s2 in speaker2:
                    textA = get_clean_text(textA, remove_punct=False)
                    textB = get_clean_text(textB, remove_punct=False)

                    instance = input_builder.build_inputs([textA], textB,
                                                            max_length=UPTAKE_MAX_INPUT_LENGTH,
                                                            input_str=True)
                    output = get_uptake_prediction(model, device, instance)
                    uptake_score = softmax(output["nsp_logits"][0].tolist())[1]
                    if result_type == "binary":
                        uptake_score = 1 if uptake_score > HIGH_UPTAKE_THRESHOLD else 0

                    predictions.append(uptake_score)
                else:
                    predictions.append(None)
        df[output_column] = predictions

        return df

def get_second_speaker(first_role):
    if first_role=='student':
        second_role = 'teacher'
    if first_role=='teacher':
        second_role = 'student'
    return second_role


# Create separate DataFrames for first and second speakers and dialogues
def get_df_for_convid(conv_data):
    first_df = conv_data[['id','first_role', 'dial1']].rename(columns={'first_role': 'speaker', 'dial1': 'text'})
    second_df = conv_data[['id','second_role', 'dial2']].rename(columns={'second_role': 'speaker', 'dial2': 'text'})

    # Concatenate the DataFrames in the order of first, second, first, second, ...
    combined_df = pd.concat([first_df, second_df]).sort_index(kind='merge').reset_index(drop=True)

    combined_df = combined_df.dropna() 
    combined_df = combined_df.reset_index(drop=True)

   

    combined_df = get_uptake(
        df=combined_df, 
        text_column="text", 
        output_column="teacher_uptake_student", 
        speaker_column="speaker",
        # Conversation uptake is about how much the teacher builds on what the students say.
        # So, we want to specify the first speaker to be the students.
        speaker1="student",
        speaker2="teacher",
        result_type="raw"
    )

    combined_df = get_uptake(
        df=combined_df, 
        text_column="text", 
        output_column="student_uptake_teacher", 
        speaker_column="speaker",
        
        speaker1="teacher",
        speaker2="student",
        result_type="raw"
    )
    
    return combined_df