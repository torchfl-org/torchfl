import flask

app = flask.Flask(__name__)

# ruleid: active-debug-code-flask
app.run(debug=True, use_debugger=False, use_reloader=False)

app.run(passthrough_errors=True)

# ok: active-debug-code-flask
app.run()

app = flask.Flask(__name__)

# ruleid: active-debug-code-flask
app.debug = True
