from inverted_index import Document, Reader, normalise_query, InvertedIndex


class Corpus:
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
        self.original_query = phrase_query
        self.phrase_query = normalise_query(phrase_query)
        terms = self.phrase_query.split()
        if len(terms) < 2:
            raise Exception("Phrase query must have at least 2 terms")
        self.bigrams = []
        for i in range(len(terms) - 1):
            self.bigrams.append((terms[i], terms[i + 1]))
        # self.phrase_queries.sort(key=lambda x: len(x), reverse=True)

    def run_phrase_query(self, bigram_index):
        return bigram_index.get_documents_for_query_AND(self.bigrams)


reader = Reader(path="Normal/", original_files_dir="Unnormal/")
corpus = Corpus(reader)
bigram_index = InvertedIndex(corpus)
query = PhraseQuery("contamination of property")
# print(query.run_phrase_query(bigram_index))
for docId in query.run_phrase_query(bigram_index):
    filename = reader.get_original_passage_content(docId)
    print(docId, filename)
