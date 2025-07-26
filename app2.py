from flask import Flask, render_template, request, url_for
import os
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from pdf2image import convert_from_path
import cv2
import pytesseract
import re
import math
import string
from llama_cpp import Llama

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

# --- SETUP ---
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

print("Loading YOLO model...")
model = YOLO("yolov8/models/utils/bestJuthiOCR.pt")
print("YOLO model loaded.")

print("Loading Mistral model...")
llm_model = Llama(
    model_path="models/mistral/mistral-7b-v0.1.Q6_K.gguf",
    n_ctx=2048,
    n_threads=8,
)
print("Mistral model loaded.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prompt_cleaner(question_text):
    return f"You are a math expert. Solve and give only the final answer.\n\nQuestion: {question_text.strip()}\nAnswer:"

def get_llm_answer(question_text):
    prompt = prompt_cleaner(question_text)
    response = llm_model(prompt, max_tokens=32, stop=["\n", "Answer:"])
    output = response['choices'][0]['text'].strip().lower()
    match = re.search(r'answer\s*[:\-]?\s*([A-Za-z0-9]+)', output)
    return match.group(1).strip().lower() if match else output

def parse_option(text):
    match = re.match(r'^\s*\(?([A-Z])\)?\.?\s*(.*)', text)
    if match:
        return match.groups()
    return "N/A", text

def process_and_extract(image_path, output_folder):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    results = model(image)
    tick_boxes_with_conf = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            tick_boxes_with_conf.append({'box': [x1, y1, x2, y2], 'conf': confidence})

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ocr_data = pytesseract.image_to_data(thresh, config='--oem 3 --psm 4', output_type=pytesseract.Output.DICT)

    all_lines = {}
    word_boxes = []
    for i in range(len(ocr_data['level'])):
        if int(ocr_data['conf'][i]) > 30:
            x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
            word_boxes.append((x, y, x + w, y + h))
            line_key = (ocr_data['block_num'][i], ocr_data['par_num'][i], ocr_data['line_num'][i])
            if line_key not in all_lines:
                all_lines[line_key] = {'text': [], 'box': [x, y, x + w, y + h]}
            else:
                box = all_lines[line_key]['box']
                box[0] = min(box[0], x)
                box[1] = min(box[1], y)
                box[2] = max(box[2], x + w)
                box[3] = max(box[3], y + h)
            all_lines[line_key]['text'].append(ocr_data['text'][i])

    lines_with_boxes = []
    for key in sorted(all_lines.keys()):
        full_text = ' '.join(all_lines[key]['text']).strip()
        if full_text:
            lines_with_boxes.append({'text': full_text, 'box': all_lines[key]['box']})

    identified_ticked_lines = []
    for tick_info in tick_boxes_with_conf:
        t_box = tick_info['box']
        tick_center_y = t_box[1] + (t_box[3] - t_box[1]) / 2
        closest_line = None
        min_dist = float('inf')
        for line in lines_with_boxes:
            line_center_y = line['box'][1] + (line['box'][3] - line['box'][1]) / 2
            dist = abs(tick_center_y - line_center_y)
            if dist < min_dist:
                min_dist = dist
                closest_line = line
        if closest_line and not any(d['text'] == closest_line['text'] for d in identified_ticked_lines):
            identified_ticked_lines.append(closest_line)

    for line in identified_ticked_lines:
        l_box = line['box']
        cv2.rectangle(image, (l_box[0], l_box[1]), (l_box[2], l_box[3]), (0, 255, 0), 2)

    question_lines = [line for line in lines_with_boxes if re.match(r'^\d+\.\s', line['text'])]
    question_blocks = []
    for i, q_line in enumerate(question_lines):
        question_text = q_line['text']
        start_y = q_line['box'][1]
        end_y = question_lines[i + 1]['box'][1] if i + 1 < len(question_lines) else image.shape[0]
        block = {"question": question_text, "student_ticked": [], "llm_answer": "", "mark": 0, "correct": False}
        for ticked_line in identified_ticked_lines:
            ticked_center_y = ticked_line['box'][1] + (ticked_line['box'][3] - ticked_line['box'][1]) / 2
            if start_y <= ticked_center_y < end_y:
                letter, text = parse_option(ticked_line['text'])
                cleaned_text = text.strip().translate(str.maketrans('', '', string.punctuation))
                cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
                cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
                cleaned_text = re.sub(r'\bwaa\b', '', cleaned_text, flags=re.IGNORECASE).strip()
                if cleaned_text and not any(d['text'] == cleaned_text for d in block['student_ticked']):
                    block['student_ticked'].append({"option": letter, "text": cleaned_text.lower()})
        llm_ans = get_llm_answer(question_text)
        block["llm_answer"] = llm_ans
        student_ans_texts = [opt['text'] for opt in block['student_ticked']]
        correct = any(llm_ans in ans or ans in llm_ans for ans in student_ans_texts)
        block["correct"] = correct
        block["mark"] = 1 if correct else 0
        question_blocks.append(block)

    image_name = os.path.basename(image_path)
    save_path = os.path.join(output_folder, image_name)
    cv2.imwrite(save_path, image)

    total_score = sum(b['mark'] for b in question_blocks)
    return {
        'image_url': url_for('static', filename=f'outputs/{image_name}'),
        'structured_summary': question_blocks,
        'total_score': total_score,
        'out_of': len(question_blocks)
    }

def pdf_to_images(pdf_path, output_folder):
    pages = convert_from_path(pdf_path, dpi=300)
    image_paths = []
    for i, page in enumerate(pages):
        img_path = os.path.join(output_folder, f"page_{i+1}.jpg")
        page.save(img_path, 'JPEG')
        image_paths.append(img_path)
    return image_paths

@app.route('/', methods=['GET', 'POST'])
def index():
    detection_results = []
    if request.method == 'POST':
        file = request.files.get('image')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            ext = filename.rsplit('.', 1)[1].lower()
            if ext == 'pdf':
                image_paths = pdf_to_images(upload_path, app.config['OUTPUT_FOLDER'])
                for img_path in image_paths:
                    result = process_and_extract(img_path, app.config['OUTPUT_FOLDER'])
                    if result:
                        detection_results.append(result)
            else:
                result = process_and_extract(upload_path, app.config['OUTPUT_FOLDER'])
                if result:
                    detection_results.append(result)
    return render_template('index2.html', all_results=detection_results)

if __name__ == '__main__':
    app.run(debug=True)