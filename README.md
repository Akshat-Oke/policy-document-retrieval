# Policy Document Retrieval System

## Usage

Run `python3 server.py` to start the server.

If the browser tab is not opened automatically, open `http://127.0.0.1:5000/index.html` to access the GUI.

## Querying

You can enter your query as a normal sequence of words. Results retrieved will match atleast one of the given query terms.

You can also use the following optional qualifiers:

`"<term>"` To include a specific term. Will revert to "may or may not match" (OR) query if two or more terms are within quotes.
`-<term>` Do not include a specific term.

Any number of these qualifiers can be entered.

![GUI screenshot](static/gui.png "GUI")
