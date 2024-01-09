from flask import Flask, render_template, request

import pandas as pd
import gensim
from gensim.models.doc2vec import Doc2Vec

app = Flask(__name__)

model = gensim.models.Word2Vec.load('model.model')
df = pd.read_csv("wiki_movie_plots_deduped.csv", sep=",")
placeholder = "Masukkan sinopsis film yang anda cari"
result = []

def pred(input):
    if input:
        input_doc = gensim.parsing.preprocessing.preprocess_string(input)
        input_doc_vector = model.infer_vector(input_doc)
        sims = model.docvecs.most_similar(positive=[input_doc_vector])

        result = []
        for i in sims:
            result.append(df['Title'].iloc[i[0]])

        return result
    else:
        return None

def update_placeholder(input):
    if input:
        placeholder = input
        return placeholder
    else:
        placeholder = "Masukkan sinopsis film yang anda cari"
        return placeholder

@app.route('/', methods=['GET'])
def index():
    user_input = request.args.get('user_input', '')
    placeholder = update_placeholder(user_input)
    result = pred(user_input)
    return render_template('index.html', result=result, placeholder=placeholder)

if __name__ == '__main__':
    app.run()