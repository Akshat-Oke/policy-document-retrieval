import inverted_index
import timeit
from flask import Flask, jsonify, request
import webbrowser

app = Flask(__name__, static_url_path="", static_folder="static/")

# global variables
reader = 1
corpus = 1
index = 1


# @app.route("/", methods=["GET", "POST"])
# def home():
#     if request.method == "GET":
#         data = "hello world"
#         return jsonify({"data": data})


@app.route("/search", methods=["POST", "GET"])
def search():
    if request.method == "POST":
        json = request.get_json()
        if "query" in json:
            query = json["query"]
            t_0 = timeit.default_timer()
            docs = inverted_index.search(reader, corpus, index, query)
            t_1 = timeit.default_timer()
            res = {
                "docs": docs,
                "time": round((t_1 - t_0) * 10**3, 3),
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
    reader, corpus, index = inverted_index.init()
    webbrowser.open("http://127.0.0.1:5000/index.html", new=2)
    app.run()
