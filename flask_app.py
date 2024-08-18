import atexit
from flask import *
from werkzeug.utils import secure_filename
import os
from MotionCapture import process_video

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def clean_up():
    """Delete all files in the directory"""
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Error deleting file {file_path}: e')


atexit.register(clean_up())


@app.route("/", methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        if 'file' not in request.files:
            raise FileNotFoundError
        user_file = request.files['file']
        if user_file:
            #  make a safe file name
            filename = secure_filename(user_file.filename)
            input_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            user_file.save(input_filepath)

            #  generate output filename
            output_filename = f"{os.path.splitext(filename)[0]}_estimated_pose.bvh"
            output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

            #  pose estimation
            process_video(input_filepath, output_filepath)

            return redirect(url_for('download_page', filename=os.path.basename(output_filepath)))
    return '''<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Pose Estimator</title>
</head>
<body>
   <form method = "post" enctype="multipart/form-data">
        <input type="file" accept=".mp4,.avi,.mov,.mkv" name="file">
        <input type = "submit" value="Upload">
    </form>
</body>
</html>'''


@app.route('/download/<filename>', methods=['GET'])
def download_page(filename):
    return render_template_string('''<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Download File</title>
</head>
<body>
    <h1>Your file is ready!</h1>
    <p>Click the button below to download your file.</p>
    <form action="{{ url_for('download_file', filename=filename) }}" method="get">
        <button type="submit">Download File</button>
    </form>
</body>
</html>''', filename=filename)


@app.route('/download_file/<filename>', methods=['GET'])
def download_file(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)
    except FileNotFoundError:
        return f"File {filename} not found", 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
