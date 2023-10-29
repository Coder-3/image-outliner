from flask import Flask, render_template, request, send_from_directory
import cv2
import os
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'

def flexible_image_processing(image_path, lower_threshold, upper_threshold, 
                              increase_contrast=False, use_bilateral=False, 
                              use_adaptive=False, use_dilate_erode=False):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if increase_contrast:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    
    if use_bilateral:
        gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    
    if use_adaptive:
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
    else:
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, lower_threshold, upper_threshold)
    
    if use_dilate_erode:
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline = 255 * np.ones_like(gray)
    cv2.drawContours(outline, contours, -1, (0, 0, 0), 2)

    file_extension = os.path.splitext(image_path)[1].lower()

    if file_extension == ".png":
        output_path = os.path.join(UPLOAD_FOLDER, "outlined_" + os.path.basename(image_path))
        cv2.imwrite(output_path, outline, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    else:
        output_name = "outlined_" + os.path.splitext(os.path.basename(image_path))[0] + ".jpg"
        output_path = os.path.join(UPLOAD_FOLDER, output_name)
        cv2.imwrite(output_path, outline, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    
    return output_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
        file = request.files['file']
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)

        lower_threshold = 50
        upper_threshold = 150
        increase_contrast = False
        use_bilateral = False
        use_adaptive = False
        use_dilate_erode = False
        
        preset = request.form.get('preset')
        
        if preset == 'High noise image':
            use_bilateral = True
        elif preset == 'Low contrast image':
            increase_contrast = True
            lower_threshold = 30
        else:
            lower_threshold = int(request.form.get('lower_threshold', 50))
            upper_threshold = int(request.form.get('upper_threshold', 150))
            increase_contrast = request.form.get('increase_contrast') == 'true'
            use_bilateral = request.form.get('use_bilateral') == 'true'
            use_adaptive = request.form.get('use_adaptive') == 'true'
            use_dilate_erode = request.form.get('use_dilate_erode') == 'true'
        
        converted_image_path = flexible_image_processing(filename, lower_threshold, upper_threshold, 
                                                         increase_contrast, use_bilateral, 
                                                         use_adaptive, use_dilate_erode)
        
        return send_from_directory(os.getcwd(), converted_image_path)
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
