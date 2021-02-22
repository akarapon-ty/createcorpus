import gensim

model = gensim.models.Word2Vec.load("wiki.th.text.model")

print(model.wv.similar_by_word("นก"))