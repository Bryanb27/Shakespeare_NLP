#Imports
import re
import unicodedata
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import MWETokenizer
from textblob import TextBlob
import spacy
from gensim.utils import tokenize
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from collections import Counter
import pandas as pd

#Downloads
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#To calculate all the metrics
def calculate_metrics(tokens):
    token_counts = Counter(tokens)
    total_tokens = len(tokens)
    total_unique_tokens = len(token_counts)

    #Fix division by 0 error
    if total_unique_tokens == 0:
        avg_occurrences = 0
        avg_token_length = 0
    else:
        avg_occurrences = total_tokens / total_unique_tokens
        avg_token_length = sum(len(token) for token in tokens) / total_tokens
    
    return total_unique_tokens, avg_occurrences, avg_token_length

#Generate CSV
def generate_vocab_csv(tokens, file_name):
    token_counts = Counter(tokens)
    vocab_df = pd.DataFrame(list(token_counts.items()), columns=['Token', 'Count'])
    vocab_df[ 'Token_Length' ] = vocab_df['Token'].apply(len)
    vocab_df.to_csv(file_name, index=False)

#Read and Write files
def read_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        input_text = file.read()
    return input_text

def write_file(file_name, tokenized_text):
    with open(file_name, 'w', encoding='utf-8') as output_file:
        for token in tokenized_text:
            output_file.write(token + '\n')  

#A write method to fix a bug
def write_file_First(file_name, write_text):
    with open(file_name, 'w', encoding='utf-8') as output_file:
        for token in write_text:
            output_file.write(token + '')
        output_file.write('\n')

#A read method to fix a bug
def read_file_Lines(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        input_lines = [ line.strip() for line in file.readlines() ]
    return input_lines


#Funtion to stem text
def text_stemming():
    lemmatized_text = read_file_Lines("Shakespeare_Normalized_Tokenized_StopWord_Lemmatized.txt")
    stemmer01 = PorterStemmer()
    stemmer02 = SnowballStemmer('english')
    stemmed_Porter = [ stemmer01.stem(word) for word in lemmatized_text ]
    stemmed_Snowball = [ stemmer02.stem(word) for word in lemmatized_text ]
    write_file("Shakespeare_Normalized_Tokenized_StopWord_Lemmatized_Stemming01.txt", stemmed_Porter)
    write_file("Shakespeare_Normalized_Tokenized_StopWord_Lemmatized_Stemming02.txt", stemmed_Snowball)

#Function to lemmatizer text
def lemmatizer_function():
    nostopwords_text = read_file_Lines("Shakespeare_Normalized_Tokenized_StopWord.txt")
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = [ lemmatizer.lemmatize(word) for word in nostopwords_text ]
    write_file("Shakespeare_Normalized_Tokenized_StopWord_Lemmatized.txt", lemmatized_text)

#Funtion to remove stopword
def stopwords_function():
    tokenized_text = read_file_Lines("Shakespeare_Normalized_Tokenized02.txt")
    sw = stopwords.words('english')
    nostopwords_text = [ word for word in tokenized_text if not word in sw]
    write_file("Shakespeare_Normalized_Tokenized_StopWord.txt", nostopwords_text)

'''
   Here starts the token functions
   '''
def white_space_token(input_text):
    tokenized_text = input_text.split()
    write_file("Shakespeare_Normalized_Tokenized01.txt", tokenized_text)

def nltk_word(input_text):
    output = word_tokenize(input_text)
    write_file("Shakespeare_Normalized_Tokenized02.txt", output)

def nltk_tree_bank(input_text):
    t = TreebankWordTokenizer()
    output = t.tokenize(input_text)
    write_file("Shakespeare_Normalized_Tokenized03.txt", output)

def nltk_word_punctuation(input_text):
    output = wordpunct_tokenize(input_text)
    write_file("Shakespeare_Normalized_Tokenized04.txt", output)

def nltk_tweet(input_text):
    tk = TweetTokenizer()
    output = tk.tokenize(input_text)
    write_file("Shakespeare_Normalized_Tokenized05.txt", output)

def nltk_MWE(input_text):
    tk = MWETokenizer()
    output = tk.tokenize(input_text)
    write_file("Shakespeare_Normalized_Tokenized06.txt", output)

def text_blob(input_text):
    blob_object = TextBlob(input_text)
    write_file("Shakespeare_Normalized_Tokenized07.txt", blob_object.words)

def spaCy(input_text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(input_text)
    output = [token.text for token in doc]
    write_file("Shakespeare_Normalized_Tokenized08.txt", output)

def gensim(input_text):
    output = list(tokenize(input_text))
    write_file("Shakespeare_Normalized_Tokenized09.txt", output)

def keras(input_text):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([input_text])
    sequences = tokenizer.texts_to_sequences([input_text])[0]
    output = [ str(token) for token in sequences ]
    write_file("Shakespeare_Normalized_Tokenized10.txt", output)


def remove_accents(input_text):
    nfkd_form = unicodedata.normalize('NFKD', input_text)
    return ''.join([ c for c in nfkd_form if not unicodedata.combining(c) ])

#Funtion with tokenizers     
def tokenizing():
    #Read file
    input_text = read_file("Shakespeare_Normalized.txt")

    #Token functions
    white_space_token(input_text)
    nltk_word(input_text)
    nltk_tree_bank(input_text)
    nltk_word_punctuation(input_text)
    nltk_tweet(input_text)
    nltk_MWE(input_text)
    text_blob(input_text)
    spaCy(input_text)
    gensim(input_text)
    keras(input_text)

#Function to normalize text
def normalize_text():
    #Read file
    input_text = read_file("Shakespeare.txt")

    #lower case
    input_text = input_text.lower()

    #Accent and diacritic removal
    noaccents_text = remove_accents(input_text)

    #Acronym canonicalization
    nospecial_text = re.sub('\.{?!(\S[^. ])|\d}', '', noaccents_text)

    #Special Character removal
    pattern = r'[.,;!?\'\(\)#:-]'
    nospecial_text = re.sub(pattern, '', nospecial_text)
    nospecial_text = re.sub(r' +', ' ', nospecial_text)

    #Output file
    write_file_First("Shakespeare_Normalized.txt", nospecial_text)

# Start
normalize_text()
tokenizing()
stopwords_function()
lemmatizer_function()
text_stemming()

#For exercise 6
stemmed_Porter = read_file_Lines("Shakespeare_Normalized_Tokenized_StopWord_Lemmatized_Stemming01.txt")
stemmed_Snowball = read_file_Lines("Shakespeare_Normalized_Tokenized_StopWord_Lemmatized_Stemming02.txt")
lemmatized_text = read_file_Lines("Shakespeare_Normalized_Tokenized_StopWord_Lemmatized.txt")

generate_vocab_csv(stemmed_Porter, "Shakespeare_Vocabulary_Porter")
generate_vocab_csv(stemmed_Snowball, "Shakespeare_Vocabulary_Snowball")
generate_vocab_csv(lemmatized_text, "Shakespeare_Vocabulary_Lemmatized")

porter_metrics = calculate_metrics(stemmed_Porter)
snowball_metrics = calculate_metrics(stemmed_Snowball)
lemmatized_metrics = calculate_metrics(lemmatized_text)

with open('Shakespeare_Vocabulary_Analysis.txt', 'w') as file:
    file.write("Lemmatized Vocabulary:\n")
    file.write(f"Total Tokens: {lemmatized_metrics[0]}\n")
    file.write(f"Average Occurrences: {lemmatized_metrics[1]}\n")
    file.write(f"Average Token Length: {lemmatized_metrics[2]}\n\n")

    file.write("Snowball Stemmed Vocabulary:\n")
    file.write(f"Total Tokens: {snowball_metrics[0]}\n")
    file.write(f"Average Occurrences: {snowball_metrics[1]}\n")
    file.write(f"Average Token Length: {snowball_metrics[2]}\n\n")

    file.write("Porter Stemmed Vocabulary:\n")
    file.write(f"Total Tokens: {porter_metrics[0]}\n")
    file.write(f"Average Occurrences: {porter_metrics[1]}\n")
    file.write(f"Average Token Length: {porter_metrics[2]}\n")