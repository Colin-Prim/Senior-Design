from flask import Flask

app = Flask(__name__)


@app.route("/")
def main():
    return '''<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Pose Estimator</title>
</head>
<body>
   <form action = "/your-file" method = "post" enctype="multipart/form-data">
        <input type="file" accept=".mp4,.avi,.mov,.mkv" name="file" />
        <input type = "submit" value="Upload">
    </form>
</body>
</html>'''


@app.route("/your-file")
def start_processing():



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
