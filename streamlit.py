import streamlit as st
import pandas as pd
import requests
import json
from datasets import DatasetDict
import numpy as np
import unicodedata
import difflib

# Load the dataset
@st.cache_resource
def load_data():
    dataset = DatasetDict.load_from_disk("filtered_dataset")
    return dataset

# Load the special characters JSON
def load_special_characters(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data["characters"]


if 'transcriptions' not in st.session_state:
    st.session_state.transcriptions = {}

if 'references' not in st.session_state:
    st.session_state.references = {}

if 'error_rates' not in st.session_state:
    st.session_state.error_rates = {}

if 'manuscript_data' not in st.session_state:
    st.session_state.manuscript_data = None


dataset = load_data()
special_characters = load_special_characters('/home/tclerice/Downloads/catmus-medieval.json')

def cer(transcription, reference):
    """Calculate the Character Error Rate (CER)"""
    errors = sum(1 for a, b in zip(unicodedata.normalize("NFD", transcription), reference) if a != b)
    errors += abs(len(transcription) - len(reference))
    return errors / len(reference)

def filter_data(dataset, language=None, century=None, script_type=None, genre=None):
    """Filter the dataset based on user selections"""
    filtered_data = dataset
    if language:
        filtered_data = filtered_data.filter(lambda x: x['language'] == language)
    if century:
        filtered_data = filtered_data.filter(lambda x: x['century'] == century)
    if script_type:
        filtered_data = filtered_data.filter(lambda x: x['script_type'] == script_type)
    if genre:
        filtered_data = filtered_data.filter(lambda x: x['genre'] == genre)
    return filtered_data


def format_differences(transcription, reference):
    """Format differences between transcription and reference text with colors"""
    diff = difflib.ndiff(unicodedata.normalize("NFD", transcription), reference)
    result = []
    for op in diff:
        if op.startswith('+'):
            result.append(f"<span style='color:red;'>{op[2]}</span>")
        elif op.startswith('-'):
            result.append(f"<span style='color:orange;'>{op[2]}</span>")
        else:
            result.append(op[2])
    return ''.join(result)


# Extract unique values for filter lists
def get_unique_values(dataset, column_name):
    train_values = dataset['train'].unique(column_name)
    dev_values = dataset['validation'].unique(column_name)
    test_values = dataset['test'].unique(column_name)
    all_values = set(train_values + dev_values + test_values)
    return sorted(list(all_values))


languages = get_unique_values(dataset, 'language')
centuries = get_unique_values(dataset, 'century')
script_types = get_unique_values(dataset, 'script_type')
genres = get_unique_values(dataset, 'genre')

# Streamlit App
st.title('Medieval Text Transcription Practice')


# Sidebar for filters
st.sidebar.title('Filter Options')
language = st.sidebar.selectbox('Language', options=[''] + languages)
century = st.sidebar.selectbox('Century', options=[''] + centuries)
script_type = st.sidebar.selectbox('Script Type', options=[''] + script_types)
genre = st.sidebar.selectbox('Genre', options=[''] + genres)
# Filter the dataset based on selections
filtered_data = filter_data(dataset['train'], language, century, script_type, genre)


if len(filtered_data) > 0:
    # Select 5 to 10 lines from the same manuscript

    if st.session_state.manuscript_data is None:
        # Select 5 to 10 lines from the same manuscript and store in session state
        manuscript_data = filtered_data.to_pandas()
        shelfmark = manuscript_data.sample(n=1).iloc[0]['shelfmark']
        manuscript_data = manuscript_data[manuscript_data.shelfmark == shelfmark].sample(n=5)
        st.session_state.manuscript_data = manuscript_data
    else:
        manuscript_data = st.session_state.manuscript_data
        shelfmark = manuscript_data.iloc[0]['shelfmark']
    st.write(f"Transcribe the following lines from manuscript: {shelfmark}")

    for index, row in manuscript_data.iterrows():
        line_key = f"line_{index+1}"
        
        with st.form(key=line_key):
            st.image(row['im']["bytes"], caption=f"Line {index+1}", use_column_width=True)
            if line_key not in st.session_state.transcriptions:
                st.session_state.transcriptions[line_key] = ""
            st.session_state.references[line_key] = row['text']

            # Text input for transcription
            st.session_state.transcriptions[line_key] = st.text_input(
                f"Transcription for line {index+1}", value=st.session_state.transcriptions[line_key], key=f"text_{line_key}"
            )

            # Button to check transcription
            submit_button = st.form_submit_button("Check Transcription")
            if submit_button:
                transcription = st.session_state.transcriptions[line_key]
                reference = st.session_state.references[line_key]
                error_rate = cer(transcription, reference)
                st.session_state.error_rates[line_key] = error_rate
                st.write(f"Correct: {reference}")
                st.markdown(f"Difference: {format_differences(transcription, reference)}", unsafe_allow_html=True)
                st.write(f"Character Error Rate for {line_key}: {error_rate:.2%}")
        
        # Display error rate if calculated
        if line_key in st.session_state.error_rates:
            error_rate = st.session_state.error_rates[line_key]
            st.write(f"Character Error Rate for {line_key}: {error_rate:.2%}")

    total_error_rate = sum(st.session_state.error_rates.values()) / len(st.session_state.error_rates) if st.session_state.error_rates else 0
    st.write(f"Total Character Error Rate: {total_error_rate:.2%}")




else:
    st.write("No data available with the selected filters.")


def display_virtual_keyboard(special_characters):
    st.sidebar.title("Virtual Keyboard")
    for char in special_characters:
        if st.sidebar.button(f"{char['legend']} ({char['character']})"):
            st.session_state.transcription += char['character']

# Initialize session state for transcription
if 'transcription' not in st.session_state:
    st.session_state.transcription = ""

display_virtual_keyboard(special_characters)
