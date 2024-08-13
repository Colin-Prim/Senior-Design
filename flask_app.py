from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def main():
    return '''<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Pose Estimator</title>
</head>
<body>
   <form action = "/success" method = "post" enctype="multipart/form-data">
        <input type="file" name="file" />
        <input type = "submit" value="Upload">
    </form>
</body>
</html>'''


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
