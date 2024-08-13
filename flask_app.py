from flask import Flask

app = Flask(__name__)

app.config["DEBUG"] = True


@app.route("/")
def title():
    return "This is the title"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
