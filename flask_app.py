import atexit
from flask import *
from werkzeug.utils import secure_filename
import os
from MotionCapture import process_video
import cv2
import base64
import numpy as np

app = Flask(__name__)
app.secret_key = 'BFOWN293OF202NDJ781LDP84HNA3'

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

FRAME_FOLDER = 'frames'
app.config['FRAME_FOLDER'] = FRAME_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)

cancel_processing = False


def clean_up():
    """Delete all files in the directory"""
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except FileNotFoundError as e:
            print(f'Error deleting file {file_path}: {e}')

    for filename in os.listdir(app.config['FRAME_FOLDER']):
        file_path = os.path.join(app.config['FRAME_FOLDER'], filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except FileNotFoundError as e:
            print(f'Error deleting file {file_path}: {e}')


atexit.register(clean_up)


@app.route("/", methods=['GET', 'POST'])
def main():
    try:
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

                return redirect(url_for('processing', video_filename=os.path.basename(input_filepath),
                                        output_filename=os.path.basename(output_filepath)))
        return '''<html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Pose Estimator</title>
        </head>
        <body>
            <form method = "post" enctype="multipart/form-data">
                <input type="file" accept=".mp4,.avi,.mov,.mkv" name="file">
                <input type = "submit" value="Estimate">
            </form>
            
            <script>
                function validateFile() {
                    const fileInput = document.getElementById('file-input');
                    const errorMessage = document.getElementById('error-message');
                    const allowedExtensions = ['mp4', 'avi', 'mov', 'mkv'];
                    
                    if (fileInput.files.length > 0) {
                        const fileName = fileInput.files[0].name;
                        const fileExtension = fileName.split('.').pop().toLowerCase();
        
                        if (!allowedExtensions.includes(fileExtension)) {
                            errorMessage.style.display = 'block';
                            return false;  // Prevent form submission
                        }
                    }
                    errorMessage.style.display = 'none';
                    return true;  // Allow form submission
                }
    </script>
        </body>
        </html>'''
    except FileNotFoundError as e:
        return url_for('show_error', e=e)


# Processing route to display real-time frames
@app.route('/processing/<video_filename>/<output_filename>')
def processing(video_filename, output_filename):
    # reset cancel_processing flag in event of previous cancellation
    global cancel_processing
    cancel_processing = False

    return render_template_string(''' 
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Processing Video</title>
        <script type="text/javascript">
            var eventSource = new EventSource("{{ url_for('stream', video_filename=video_filename) }}");
            eventSource.onmessage = function(event) {
                if (event.data === 'DONE') {
                    eventSource.close();
                    // Redirect to download page when processing is done
                    window.location.href = "{{ url_for('view_frames', video_filename=video_filename, output_filename=output_filename) }}";                
                } 
                else {
                    // Set the source of the image to the URL provided by the server
                    document.getElementById('video_frame').src = event.data;
                }
            };
        </script>
    </head>
    <body>
        <h1>Processing your video...</h1>
        <img id="video_frame" src="" alt="Video frame will appear here">
        <br><br>
        <form action="{{ url_for('cancel_processing_route', video_filename=video_filename) }}" method="post">
            <button type="submit">Cancel</button>
        </form>
    </body>
    </html>
    ''', video_filename=video_filename, output_filename=output_filename)


# Route to handle the cancellation of processing
@app.route('/cancel_processing/<video_filename>', methods=['POST'])
def cancel_processing_route(video_filename):
    global cancel_processing
    try:
        cancel_processing = True  # Set the cancel flag to True
        return redirect(url_for('view_frames', video_filename=video_filename, output_filename='canceled.bvh'))
    except Exception as e:
        return url_for('show_error', e=e)


# Stream frames from process_video
@app.route('/stream/<video_filename>')
def stream(video_filename):
    try:
        input_filepath = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)

        # Ensure the output file is also created in UPLOAD_FOLDER
        output_filename = f"{os.path.splitext(video_filename)[0]}_estimated_pose.bvh"
        output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

        def generate():
            frame_index = 0

            for frame_encoded in process_video(input_filepath, output_filepath, cancel_signal=lambda: cancel_processing):
                # Convert back to a NumPy array for saving
                # Use cv2.imdecode to convert the base64 encoded string back to an image
                nparr = np.frombuffer(base64.b64decode(frame_encoded), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is not None:
                    # Save frame in frames folder before displaying
                    frame_filename = f"frame_{frame_index}.jpg"
                    frame_filepath = os.path.join(app.config['FRAME_FOLDER'], frame_filename)
                    cv2.imwrite(frame_filepath, frame)
                    frame_index += 1

                    yield f"data: /frames/{frame_filename}\n\n"
                else:
                    print(f"Failed to decode frame at index {frame_index}, skipping.")

            yield 'data: DONE\n\n'

        return Response(generate(), content_type='text/event-stream')
    except Exception as e:
        return url_for('show_error', e=e)


# View the processed frames in sequence
@app.route('/view_frames/<video_filename>/<output_filename>', methods=['GET'])
def view_frames(video_filename, output_filename):
    try:
        # Get all saved frames in the FRAME_FOLDER
        frame_files = sorted(os.listdir(app.config['FRAME_FOLDER']))

        return render_template_string('''
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>View Estimated Frames</title>
            <script type="text/javascript">
                var currentIndex = 0;
                var frames = {{ frames|tojson }};
                function showNextFrame() {
                    document.getElementById('frame_image').src = "/frames/" + frames[currentIndex];
                    currentIndex = (currentIndex + 1) % frames.length;
                }
                setInterval(showNextFrame, 500);  // Change frame every 500ms
            </script>
        </head>
        <body onload="showNextFrame()">
            <h1>Process Complete</h1>
            <img id="frame_image" src="" alt="Estimated Frame">
            <br><br>
            <form action="{{ url_for('download_file', filename=output_filename) }}" method="get">
            <button type="submit">Download File</button>
            </form>
            <form action="{{ url_for('restart_processing', video_filename=video_filename) }}" method="post">
                <button type="submit">Restart Estimation</button>
            </form>
            <form action="{{ url_for('cleaning') }}" method="post">
                <button type="submit">Back to Home</button>
            </form>
        </body>
        </html>
        ''', frames=frame_files, video_filename=video_filename, output_filename=output_filename)
    except Exception as e:
        return url_for('show_error', e=e)


# Serve frame images from FRAME_FOLDER
@app.route('/frames/<filename>')
def serve_frame(filename):
    try:
        return send_from_directory(app.config['FRAME_FOLDER'], filename)
    except FileNotFoundError as e:
        return url_for('show_error', e=e)


# Route to handle the restarting of the processing with the same file
@app.route('/restart/<video_filename>', methods=['POST'])
def restart_processing(video_filename):
    try:
        # Re-generate the output file name
        output_filename = f"{os.path.splitext(video_filename)[0]}_estimated_pose.bvh"
        # output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

        # Redirect to the processing page with the same video and output file
        return redirect(url_for('processing', video_filename=video_filename, output_filename=output_filename))
    except Exception as e:
        return url_for('show_error', e=e)


"""
@app.route('/download/<video_filename>/<output_filename>', methods=['GET'])
def download_page(video_filename, output_filename):
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
    <form action="{{ url_for('restart_processing', video_filename=video_filename) }}" method="post">
        <button type="submit">Restart Estimation</button>
    </form>
    <form action="{{ url_for('cleaning') }}" method="post">
        <button type="submit">Back to Home</button>
    </form> 
</body>
</html>''', filename=output_filename, video_filename=video_filename)
"""


# Route to download the generated .bvh file
@app.route('/download_file/<filename>', methods=['GET'])
def download_file(filename):
    try:
        # Ensure the canceled .bvh file is created
        if filename == 'canceled.bvh':
            with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'w') as f:
                f.write("Processing was canceled.")
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)
    except FileNotFoundError as e:
        return url_for('show_error', e=e)


@app.route('/start_over', methods=['POST'])
def cleaning():
    clean_up()
    return redirect(url_for('main'))


@app.route('/error', methods=['POST'])
def show_error(e=None):
    error_message = str(e) if e else "An unknown error occurred. Please try again."
    return render_template_string('''<html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Error</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        background-color: #f8d7da;
                        color: #721c24;
                        text-align: center;
                        padding: 50px;
                    }
                    .error-container {
                        background-color: #f5c6cb;
                        padding: 20px;
                        border-radius: 5px;
                        display: inline-block;
                    }
                    h1 {
                        font-size: 24px;
                        margin: 0;
                    }
                    p {
                        font-size: 18px;
                        margin: 10px 0 0;
                    }
                </style>
            </head>
            <body>
                <div class="error-container">
                    <h1>Error</h1>
                    <p>{{ error_message }}</p>
                </div>
            </body>
            </html>''', error_message=error_message)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
