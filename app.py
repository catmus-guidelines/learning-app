# app.py
import os
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import json
import difflib
import base64
from datasets import DatasetDict
import unicodedata
from jiwer import cer

app = Flask(__name__)

# Load the dataset
def load_data():
    dataset = DatasetDict.load_from_disk("filtered_dataset")
    return dataset

# Load the special characters JSON
def load_special_characters(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data["characters"]

# Extract unique values for filter lists
def get_unique_values(dataset, column_name):
    train_values = dataset['train'].unique(column_name)
    dev_values = dataset['validation'].unique(column_name)
    test_values = dataset['test'].unique(column_name)
    all_values = set(train_values + dev_values + test_values)
    return sorted(list(all_values))

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
    return filtered_data.to_pandas()

def format_differences(transcription, reference):
    """Format differences between transcription and reference text with colors"""
    diff = difflib.ndiff(transcription, reference)
    result = []
    for op in diff:
        if op.startswith('+'):
            result.append(f"<span style='color:#0e288e;'>{op[2].replace(' ', '[space]')}</span>")
        elif op.startswith('-'):
            result.append(f"<span style='color:#b32db5; text-decoration: line-through;'>{op[2]}</span>")
        else:
            result.append(op[2])
    return ''.join(result)

# Load the dataset and special characters
dataset = load_data()
special_characters = load_special_characters('/home/tclerice/Downloads/catmus-combining.json')

# Get unique values for filters
languages = get_unique_values(dataset, 'language')
centuries = get_unique_values(dataset, 'century')
script_types = get_unique_values(dataset, 'script_type')
genres = get_unique_values(dataset, 'genre')

@app.route('/')
def index():
    return render_template('index.html', languages=languages, centuries=centuries, script_types=script_types, genres=genres)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    language = request.form.get('language')
    century = request.form.get('century')
    script_type = request.form.get('script_type')
    genre = request.form.get('genre')

    filtered_data = filter_data(dataset['train'], language, century, script_type, genre)

    if len(filtered_data) > 0:
        shelfmark = filtered_data.sample(n=1).iloc[0]['shelfmark']
        mss_data = filtered_data[filtered_data.shelfmark == shelfmark].sample(n=5)
        mss_data.loc[:, 'im'] = mss_data["im"].apply(lambda row: base64.b64encode(row["bytes"]).decode('utf-8'))
        transcriptions = {}
        references = {}
        for index, row in mss_data.iterrows():
            line_key = f"line_{index+1}"
            references[line_key] = row['text']
            transcriptions[line_key] = ""

        return render_template('transcribe.html', shelfmark=shelfmark, manuscript_data=mss_data, transcriptions=transcriptions, references=references)

    else:
        return redirect(url_for('index'))

@app.route('/check_transcription', methods=['POST'])
def check_transcription():
    form = request.form.to_dict()

    reference = form["groundtruth"]
    transcription = unicodedata.normalize("NFD", form["transcription"])
    error_rate = cer(transcription, reference)*100
    diff = format_differences(transcription, reference)
    return render_template('cer.html', cer=error_rate, diff=diff)

if __name__ == '__main__':
    app.run(debug=True)
