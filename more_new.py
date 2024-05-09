import flask

app = flask.Flask(__name__)

# ruleid: active-debug-code-flask
app.run(debug=True, use_debugger=False, use_reloader=False)
