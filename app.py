from flask import Flask, render_template, request, send_from_directory
import cv2
import os
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'

def convert_image_to_non_transparent_outline(image_path, lower_threshold, upper_threshold):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, lower_threshold, upper_threshold)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline = 255 * np.ones_like(gray)
    cv2.drawContours(outline, contours, -1, (0, 0, 0), 2)

    output_path = os.path.join(UPLOAD_FOLDER, "outlined_" + os.path.basename(image_path))
    cv2.imwrite(output_path, outline)

    return output_path

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
        return send_from_directory(os.getcwd(), converted_image_path)
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
