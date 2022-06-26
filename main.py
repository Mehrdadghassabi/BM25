from typing import List, Tuple
from math import log
from hazm import *
import os


def calculate_doc_len(doc):
    count = 0
    for _ in doc.split():
        count = count + 1
    return count


def sort_order(e):
    return e[1]


def get_tf(doc, term):
    tf = 0
    for token in word_tokenize(doc):
        if token == term:
            tf = tf + 1

    return tf


class BM25:
    filesdir = []
    docs = []
    trainFilesNumber = 0

    def __init__(self):
        ...

    def read_train_files(self, dir):
        for (root, dirs, files) in os.walk(dir, topdown=True):
            for file in files:
                self.filesdir.append(os.path.join(root, file))

        self.trainFilesNumber = len(self.filesdir)
        normalizer = Normalizer()
        for dir in self.filesdir:
            f = open(dir, "r")
            self.docs.append(word_tokenize(normalizer.normalize(f.read())))

        return self

    def get_term_idf(self, term):
        dft = 0
        for i, doc in enumerate(self.docs):
            # print(i)
            if term in doc:
                dft = dft + 1
        a = (self.trainFilesNumber - dft + 0.5) / (dft + 0.5)
        return log(a, 2.718)

    def get_avgdl(self):
        totalsize = 0
        for doc in self.docs:
            totalsize = totalsize + len(doc)
        avgdl = totalsize / self.trainFilesNumber
        return avgdl

    def calculate_score(self, doc, query):
        k1 = 2.0
        b = 0.75
        init_score = 0
        # print(word_tokenize(query))
        for term in word_tokenize(query):
            surat = self.get_term_idf(term) * get_tf(doc, term) * (k1 + 1)
            makhraj1 = get_tf(doc, term)
            makhraj2 = k1 * (1 - b + b * calculate_doc_len(doc) / self.get_avgdl())
            # print(self.get_term_idf(term))
            val = surat / (makhraj1 + makhraj2)
            init_score = init_score + val

        return init_score

    def get_similar_docs(self, query) -> List[Tuple[str, int]]:
        docscore = []

        for i, dire in enumerate(self.filesdir):
            doc = open(dire, "r").read()
            t = (dire, self.calculate_score(doc, query))
            docscore.append(t)

        docscore.sort(key=sort_order, reverse=True)
        return docscore


def read_query():
    return input('query >> ')


if __name__ == "__main__":
    dataset_dir = "./Dataset_IR/Train"
    bm25 = BM25().read_train_files(dataset_dir)
    while True:
        query = read_query()
        lis = bm25.get_similar_docs(query)
        for i in range(len(lis)):
            print(lis[i])
