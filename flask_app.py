from flask import Flask

app = Flask(__name__)

app.config["DEBUG"] = True


@app.route("/")
def title():
    return "This is the title"
