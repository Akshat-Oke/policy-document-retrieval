import argparse
import inverted_index, sys
import timeit
from flask import Flask, jsonify, request, send_file
import webbrowser

app = Flask(__name__, static_url_path="", static_folder="static/")

# global variables
# reader = 1
# normal_corpus = 1
# i_index = 1
# bigram_corpus = 1


def get_document_class(doc_title):
    auto = [
        "1215E.2.pdf",
        "Business-Auto-Policy-CA0001-03-10.pdf",
        "PP_00_01_06_98.pdf",
        "7thEditionPolicy.pdf",
        " insurance-pdf-NL-SPF-1.pdf",
        "AU127-1.pdf",
        "PL-600003-87.pdf",
    ]
    if doc_title in auto:
        return "Auto"
    else:
        return "Property"


@app.route("/", methods=["GET", "POST"])
def home():
    return send_file("static/index.html")


@app.route("/search", methods=["POST", "GET"])
def search():
    if request.method == "POST":
        json = request.get_json()
        if "query" in json:
            query = json["query"]
            t_0 = timeit.default_timer()
            docs = inverted_index.search(
                reader, normal_corpus, bigram_corpus, i_index, bigram_index, query
            )
            t_1 = timeit.default_timer()
            docs = list(
                map(
                    lambda x: {
                        "docId": x["docId"],
                        "filename": x["filename"],
                        "docClass": get_document_class(x["filename"]),
                        "bm25": x["bm25"],
                        "content": x["content"],
                    },
                    docs,
                )
            )
            res = {
                "docs": docs,
                "time": inverted_index.search_time,  # round((t_1 - t_0) * 10**3, 3),
                "corrected": str(inverted_index.spell_check(query)),
            }
            return jsonify(res)

        return jsonify({"data": "JSON received"})
    elif request.method == "GET":
        query = request.args.get("search")
        docs = inverted_index.search(query)
        print("docs", docs)
        return jsonify(docs)


# driver function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build or load the inverted index.")
    parser.add_argument(
        "folderpaths",
        metavar="FolderPath",
        help="Path to Original files' folder followed by Normalised folder",
        nargs="+",
    )
    parser.add_argument(
        "-use-saved",
        metavar="Pickle filepath",
        help="Use a saved pickle file instead of normalised corpus files",
    )
    args = parser.parse_args()
    normalised_corpus_path = None  # sys.argv[1]
    unormalised_corpus_path = None  # sys.argv[2]
    pickle_file_path = None
    if args.use_saved:
        filepath = args.use_saved
        pickle_file_path = filepath
        normalised_corpus_path = args.folderpaths[0]
        unormalised_corpus_path = args.folderpaths[0]
        print(f"Using saved pickle file at {filepath}")
        # print(args.folderpaths)
    else:
        unormalised_corpus_path = args.folderpaths[0]
        normalised_corpus_path = args.folderpaths[1]
        print(
            f"Using original: {unormalised_corpus_path} and normalised: {normalised_corpus_path}"
        )
        # print(args.filepaths)

    if sys.argv[1] == "-use-saved":
        pickle_file_path = sys.argv[2]
    (
        reader,
        normal_corpus,
        bigram_corpus,
        i_index,
        bigram_index,
    ) = inverted_index.init(
        unormalised_corpus_path, unormalised_corpus_path, "inverted_index.pickle"
    )
    # webbrowser.open("http://127.0.0.1:5000/", new=2)
    app.run()
