from textblob import TextBlob
import re
import os
import re
import math
import nltk
import timeit

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# from bigram_index import PhraseQuery, Corpus as BigramCorpus

stemmer = PorterStemmer()
stop_words = set((stopwords.words("english")) + [",", ":"])

search_time = 0

k1 = 0.5
b = 1


def normalise_query(query):
    """
    Performs Normalization of query by :
    1. Remove Special Characters
    2. Converts to lower case
    3. Removes stopwords

    Returns white-space seperated string
    """
    query = clean_line(query)
    tokens = word_tokenize(query.lower())
    filtered_tokens = [
        stemmer.stem(token) for token in tokens if token not in stop_words
    ]
    return " ".join(filtered_tokens)


def clean_line(line):
    """
    Removes Special Characters
    """
    line = re.sub(r"[_.]{2,}", "", line)
    line = re.sub(r"[\"]", "", line)
    line = re.sub(r"\w *\)", "", line)
    line = re.sub(r"^\d *\.", "", line)
    return line


def make_comparator(less_than):
    def compare(x, y):
        if less_than(x, y):
            return -1
        elif less_than(y, x):
            return 1
        else:
            return 0

    return compare


class Document:
    """

    Class to store Documents

    Attributes
    ----------
    docID : int
        Identifier of Document
    map_of_terms : dict {term : frequency}
        Dictionary to store frequency of terms in Document
    numberofTerms : int
        Number of Unique terms in Document
    """

    def __init__(self, map_of_terms, docId):
        """
        Parameters
        ----------
        docID : int
            Identifier of Document
        map_of_terms : dict {term : frequency}
            Dictionary to store frequency of terms in Document
        numberofTerms : int
            Number of Unique terms in Document
        """

        self.docId = docId
        self.map_of_terms = map_of_terms
        self.numberOfTerms = len(self.map_of_terms)


class BigramCorpus:
    def __init__(self, reader):
        self.reader = reader
        self.documents = {}
        self.average_document_length = 0
        self.build_corpus()

    def update_average(self, added_len):
        self.average_document_length = (
            self.average_document_length * len(self.documents) + added_len
        ) / (len(self.documents) + 1)

    def get_document(self, docId):
        return self.documents[docId]

    def build_corpus(self):
        current_document = -1
        while True:
            current_document += 1
            document_content = self.reader.get_next_document()
            if document_content == None:
                break
            passages = document_content.split("$$$")
            passage_number = 0
            for passage in passages:
                terms = passage.split()
                self.update_average(len(terms))
                map_terms = {}
                # for term in terms:
                #     if term in map_terms:
                #         map_terms[term] += 1
                #     else:
                #         map_terms[term] = 1
                for i in range(len(terms) - 1):
                    term = (terms[i], terms[i + 1])  # terms[i] + " " + terms[i + 1]
                    if term in map_terms:
                        map_terms[term] += 1
                    else:
                        map_terms[term] = 1
                docId = passage_number + 500 * current_document
                passage_number += 1
                # self.documents.append(Document(map_terms, docId))
                self.documents[docId] = Document(map_terms, docId)


class PhraseQuery:
    # phrase_queries = Array<String>
    # "contamin proper Nevada" property
    # Array<Tuple<string*>>
    def __init__(self, phrase_query):
        print("Got phrase query as", phrase_query)
        self.original_query = phrase_query
        self.phrase_query = normalise_query(phrase_query)
        terms = self.phrase_query.split()
        print("Phrase terms: ", terms)
        if len(terms) < 2:
            raise Exception("Phrase query must have at least 2 terms")
        self.bigrams = []
        for i in range(len(terms) - 1):
            self.bigrams.append((terms[i], terms[i + 1]))
        # self.phrase_queries.sort(key=lambda x: len(x), reverse=True)

    def get_bigrams(self):
        return self.bigrams

    def get_candidate_documents(self, bigram_index):
        return bigram_index.get_documents_for_query_AND(self.bigrams)


class Reader:
    """Class"""

    def __init__(self, path, original_files_dir):
        """
        Intialises class with given values.

        Args:
            path (str): path of the directory in which documents are stored.

            original_files_dir (dict {term : frequency}): Name of Original directory that contains  all documents
        """

        self.path = path
        self.original_files_dir = original_files_dir
        # add trailing / in original_files_dir

        if self.original_files_dir[-1] != "/":
            self.original_files_dir += "/"
        self.file_names = self.get_file_names()
        self.current_file_index = 0

    def reinit(self):
        self.current_file_index = 0

    def get_file_names(self):
        """Gets file names of files in the directory.

        Returns:
            list[str]: Returns sorted list of files in current directory given by path.
        """
        return sorted(os.listdir(self.path))

    def get_original_passage_filename(self, docId):
        """Fetches documents using documnet ID.

        Args:
            docId (int): Identifier for document.

        Returns:
            str: Returns filename with given docID
        """
        filename = self.file_names[docId // 500]
        return filename

    # def cache_pdfs(self, docIds):
    #     # self.cache = {}

    def get_original_passage_content(self, docId):
        """Fetches original passage using document ID

        Args:
            docId (int): Identifier for document.

        Returns:
            str: Returns complete passage with given docID
        """
        filename = self.file_names[docId // 500]
        with open(self.original_files_dir + filename) as f:
            passages = f.read().split("$$$")
            return passages[(docId % 500)]

    def get_next_document(self):
        """Gets next document in directory given by path where current file is given by current_file_index.

        Returns:
            str: Returns contents of next file in directory given by path.
        """
        if self.current_file_index >= len(self.file_names):
            return None
        with open(self.path + "/" + self.file_names[self.current_file_index]) as f:
            self.current_file_index += 1
            return f.read()


class Corpus:
    """

    Class to store corpus

    Attributes
    ----------
    average_document_length : int
        average length of all documents in the corpus
    documents : dict {docID : Document}
        Dictionary to map docID to instance of Document class
    """

    def __init__(self, reader):
        """Creates instance of Corpus class using Reader class instance.

        Args:
            reader (Reader):
        """
        self.documents = {}
        self.average_document_length = 0
        self.build_corpus(reader)

    def update_average(self, added_len):
        """Updates average document length after new document is added.

        Args:
            added_len (int): Length of new document that was added.
        """
        self.average_document_length = (
            self.average_document_length * len(self.documents) + added_len
        ) / (len(self.documents) + 1)

    def build_corpus(self, reader):
        """Builds corpus

        Args:
            reader (Reader): _description_
        """
        current_document = -1
        while True:
            current_document += 1
            document_content = reader.get_next_document()
            if document_content == None:
                break
            passages = document_content.split("$$$")
            passage_number = 0
            for passage in passages:
                terms = passage.split()
                self.update_average(len(terms))
                map_terms = {}
                for term in terms:
                    if term in map_terms:
                        map_terms[term] += 1
                    else:
                        map_terms[term] = 1
                docId = passage_number + 500 * current_document
                passage_number += 1
                # self.documents.append(Document(map_terms, docId))
                self.documents[docId] = Document(map_terms, docId)

    def get_document(self, docId):
        """Fetches documents using documnet id.

        Args:
            docId (int): unique identifier for the document

        Returns:
            Document: Returns document with given docID.
        """
        return self.documents[docId]


class InvertedIndex:
    def __init__(self, corpus):
        # Map <term, docId>
        self.index = {}
        self.corpus = corpus
        self.build_index(corpus)

    def add_document_to_index(self, document):
        docid = document.docId
        for term in document.map_of_terms:
            if term in self.index:
                self.index[term].append(docid)
            else:
                self.index[term] = [docid]

    def build_index(self, corpus):
        for document in corpus.documents:
            self.add_document_to_index(corpus.documents[document])

    def get_posting_list(self, term):
        if term in self.index:
            return self.index[term]
        else:
            return []

    # AND
    # 1 2 3 4 5
    # 2 3 5 7
    # 2 3 5
    def get_documents_for_query_AND(self, query_terms):
        # query_terms = query.split(" ")
        term_pointer = 1  # current
        result = self.get_posting_list(query_terms[0])
        while term_pointer < len(query_terms):
            ir = 0
            ic = 0
            result_temp = []
            posting_list = self.get_posting_list(query_terms[term_pointer])
            while ir < len(result) and ic < len(posting_list):
                if result[ir] < posting_list[ic]:
                    ir += 1
                elif result[ir] > posting_list[ic]:
                    ic += 1
                else:
                    result_temp.append(result[ir])
                    ir += 1
                    ic += 1
            result = result_temp
            term_pointer += 1
        return result

    # 1 2 4 5
    # 2 3 5 7
    # 1 2 4 5 7
    def get_documents_for_query_OR(self, query_terms):
        if query_terms == []:
            return []
        # query_terms = query.split(" ")
        term_pointer = 1  # current
        result = self.get_posting_list(query_terms[0])
        while term_pointer < len(query_terms):
            ir = 0
            ic = 0
            result_temp = []
            posting_list = self.get_posting_list(query_terms[term_pointer])
            while ir < len(result) and ic < len(posting_list):
                if result[ir] < posting_list[ic]:
                    result_temp.append(result[ir])
                    ir += 1
                elif result[ir] > posting_list[ic]:
                    result_temp.append(posting_list[ic])
                    ic += 1
                else:
                    result_temp.append(result[ir])
                    ir += 1
                    ic += 1
            while ir < len(result):
                result_temp.append(result[ir])
                ir += 1
            while ic < len(posting_list):
                result_temp.append(posting_list[ic])
                ic += 1
            result = result_temp
            term_pointer += 1
        return result

    def remove_documents_for_terms(self, query_terms, docs):
        return self.subtract(docs, self.get_documents_for_query_OR(query_terms))

    def subtract(self, list1, list2):
        i = 0
        j = 0
        result = []
        while i < len(list1) and j < len(list2):
            if list1[i] < list2[j]:
                result.append(list1[i])
                i += 1
            elif list1[i] > list2[j]:
                j += 1
            else:
                i += 1
                j += 1
        while i < len(list1):
            result.append(list1[i])
            i += 1
        return result

    def idf(self, query_term):
        N = len(self.corpus.documents)
        nqi = len(self.get_posting_list(query_term))
        return math.log((N + 1) / (nqi + 0.5))

    def BM25(self, document, query_terms, k1, b):
        """Ranks retrived documnets using BM25 Algorithm

        Args:
            document (Document): An instance of Documnet class for which rank is to be determined
            query_terms (list[str]): list of Normalised Query Terms
            k1 (int): k1 hyperparameter of BM25 Algorithm
            b (int): b hyperparameter of BM25 Algorithm

        Returns:
            float: Return BM25 score of the given document
        """
        score = 0
        # terms = query.split(" ")
        for term in query_terms:
            if term in document.map_of_terms:
                score += (
                    self.idf(term)
                    * document.map_of_terms[term]
                    * (k1 + 1)
                    / (
                        document.map_of_terms[term]
                        + k1
                        * (
                            1
                            - b
                            + b
                            * document.numberOfTerms
                            / self.corpus.average_document_length
                        )
                    )
                )
            else:
                pass
                # print(
                #     "Not found term" + str(term) + "in document" + str(document.docId)
                # )
        return score


class Query:
    """Class to store query"""

    def __init__(self, query):
        """Initialises class with following query.

        Args:
            query (str): Query to be initalised.
        """
        print("query is ", query)
        self.query = query
        self.was_corrected = False
        # self.query = self.spell_check()
        # self.query_terms = self.query.split(" ")
        # Remove ANDs
        self.and_terms = re.findall(r"\"\w+\"", self.query)
        self.phrases = re.findall(r"\"(\w+(\s+\w+)+)\"", self.query)
        self.phrases = list(map(lambda x: x[0], self.phrases))
        print("Phrases ", self.phrases)
        self.phrase_queries = list(map(lambda x: PhraseQuery(x), self.phrases))
        self.query = re.sub(r"\"[^\"]+\"", "", self.query)
        print("After AND ", self.query)
        # Remove NOTs
        self.not_terms = re.findall(r"\-\w+", self.query)
        self.query = re.sub(r"\-\w+", "", self.query)
        print("After NOT ", self.query)
        # Remove ORs
        self.or_terms = re.split(r"\s+", self.query.strip())
        self.query_terms = list(set(self.and_terms + self.or_terms))
        # print("Or: ", self.or_terms)
        # print("And: ", self.and_terms)
        # print("Query terms: ", self.query_terms)
        # print("Not terms: ", self.not_terms)
        # self.query = normalise_query(query)
        self.normalise_all_terms()

    def normalise_all_terms(self):
        """Normalize query and split it into AND, OR and NOT terms."""

        def n(terms):
            return [normalise_query(x) for x in terms if (x not in stop_words)]

        self.query_terms = list(filter(lambda x: len(x) != 0, n(self.query_terms)))
        self.and_terms = list(filter(lambda x: len(x) != 0, n(self.and_terms)))
        self.or_terms = list(filter(lambda x: len(x) != 0, n(self.or_terms)))
        self.not_terms = list(filter(lambda x: len(x) != 0, n(self.not_terms)))
        self.not_terms = [re.sub("-", "", x) for x in self.not_terms if x != ""]
        print("After normalising all queries")
        print("Or: ", self.or_terms)
        print("And: ", self.and_terms)
        print("Query terms: ", self.query_terms)
        print("Not terms: ", self.not_terms)

    def spell_check(self):
        return self.query
        b = TextBlob(self.query)
        corrected = b.correct()
        if corrected != self.query:
            self.was_corrected = True
            return str(corrected)
        else:
            return self.query

    def get_candidate_documents(self, inverted_index, bigram_index):
        """Gets Candidate documents from inverted index

        Args:
            inverted_index (InvertedIndex): Inverted index that was created previously.

        Returns:
            list of Document: Rerives a list of Documents from InvertedIndex.
        """
        docs = None
        if self.phrase_queries:
            # docs = bigram_index.get_documents_for_query_AND(self.phrases)
            for phrase_query in self.phrase_queries:
                docs = phrase_query.get_candidate_documents(bigram_index)
                # TODO add AND logic for multiple phrases
        elif self.and_terms:
            docs = inverted_index.get_documents_for_query_AND(self.and_terms)
        else:
            docs = inverted_index.get_documents_for_query_OR(self.or_terms)
        return inverted_index.remove_documents_for_terms(self.not_terms, docs)

    def retrieve_documents(
        self, reader, normal_corpus, bigram_corpus, inverted_index, bigram_index
    ):
        """Retrieves document from corpus.

        Args:
            reader (Reader): Instance of Reader class to fetch documents from Corpus
            corpus (Corpus): Corpus which contains all Documnets
            inverted_index (InvertedIndx): Inveted Index that was costructed on Corpus

        Returns:
            list[str]: Returns list of filenames of documents to be retrived.
        """
        # docs = inverted_index.get_documents_for_query_AND(self.and_terms)
        t0 = timeit.default_timer()
        docs = self.get_candidate_documents(inverted_index, bigram_index)

        docs_with_bm25 = {}
        for doc in docs:
            docs_with_bm25[doc] = inverted_index.BM25(
                normal_corpus.get_document(doc), self.query_terms, k1, b
            )
            if len(self.phrase_queries) > 0:
                for phrase_query in self.phrase_queries:
                    docs_with_bm25[doc] += bigram_index.BM25(
                        bigram_corpus.get_document(doc),
                        phrase_query.get_bigrams(),
                        0.2,
                        b,
                    )
        # print(docs_with_bm25)
        # sorted_docs = sorted(
        #     docs_with_bm25.items(), key=lambda item: item[1], reverse=True
        # )
        sorted_docs = docs_with_bm25.items()
        sorted_docIds = [x[0] for x in sorted_docs]

        t_0 = timeit.default_timer()
        # print("Time before mapping content", )
        global search_time
        search_time = round((t_0 - t0) * 10**3, 4)
        sorted_docs_with_filenames = list(
            map(
                lambda x: {
                    "docId": x,
                    "filename": reader.get_original_passage_filename(
                        normal_corpus.get_document(x).docId
                    ),
                    "content": reader.get_original_passage_content(
                        normal_corpus.get_document(x).docId
                    ),
                    "bm25": docs_with_bm25[x],
                },
                sorted_docIds,
            )
        )
        t_1 = timeit.default_timer()
        print("Time taken: ", round((t_1 - t_0) * 10**3, 3))
        return sorted_docs_with_filenames


def init():
    """Converts Raw Documnets(.txt files) into relevant classes.

    Returns:
        Reader: instance of Reader class for easy reading of documents
        Corpus: Corpus containing all Documents
        InvertedIndex:  Inverted Index constructed on Raw Documents
    """
    reader = Reader(path="Normal", original_files_dir="Unnormal/")
    normal_corpus = Corpus(reader)
    reader.reinit()
    bigram_corpus = BigramCorpus(reader)

    inverted_index = InvertedIndex(normal_corpus)
    bigram_index = InvertedIndex(bigram_corpus)
    return reader, normal_corpus, bigram_corpus, inverted_index, bigram_index


def search(reader, normal_corpus, bigram_corpus, inverted_index, bigram_index, query):
    t0 = timeit.default_timer()
    query = Query(query)
    t1 = timeit.default_timer()
    print("Just before returning: ", (t1 - t0) * 10**3)
    docs = query.retrieve_documents(
        reader, normal_corpus, bigram_corpus, inverted_index, bigram_index
    )
    return docs


def build_index_and_search(query):
    reader = Reader("Normal")
    corpus = Corpus(reader)
    inverted_index = InvertedIndex(corpus)
    # print(inverted_index.get_posting_list("contamin"))
    query = Query(query)
    docs = query.retrieve_documents(reader, corpus, inverted_index)
    return docs


def spell_check(query):
    b = TextBlob(query)
    corrected = b.correct()
    if corrected != query:
        return corrected
    else:
        return None


# main()


# import re

# text = 'The quick "brown fox" jumps over the "lazy dog".'

# matches = re.findall(r'"([^"]*)"', text)

# print(matches) # Output: ['brown fox', 'lazy dog']
