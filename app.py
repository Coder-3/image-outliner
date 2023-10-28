from flask import Flask, render_template, request, send_from_directory
import cv2
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
        
        file = request.files['file']
        lower_threshold = int(request.form.get('lower_threshold', 50))
        upper_threshold = int(request.form.get('upper_threshold', 150))
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)
        converted_image_path = convert_image_to_non_transparent_outline(filename, lower_threshold, upper_threshold)
        return send_from_directory(directory=os.getcwd(), filename=converted_image_path)
    return render_template('index.html')


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)

