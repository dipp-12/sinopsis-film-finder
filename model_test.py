import pandas as pd

import gensim
from gensim.parsing.preprocessing import preprocess_documents
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

model = gensim.models.Word2Vec.load("model.model")
df = pd.read_csv("wiki_movie_plots_deduped.csv", sep=",")

input_doc = "Adaptation of the first of J.K. Rowling's popular children's novels about Harry Potter, a boy who learns on his eleventh birthday that he is the orphaned son of two powerful wizards and possesses unique magical powers of his own. He is summoned from his life as an unwanted child to become a student at Hogwarts, an English boarding school for wizards. There, he meets several friends who become his closest allies and help him discover the truth about his parents' mysterious deaths."
input_doc = gensim.parsing.preprocessing.preprocess_string(input_doc)
input_doc_vector = model.infer_vector(input_doc)
sims = model.docvecs.most_similar(positive = [input_doc_vector])

result = []
for i in sims:
    result.append(df['Title'].iloc[i[0]])

print(result)