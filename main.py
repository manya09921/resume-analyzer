import os
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template
from PyPDF2 import PdfReader
import docx2txt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
def extract_pdf(file_path):
    text=""
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

def extract_docx(file_path):
    return docx2txt.process(file_path)

def extract_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_txt(file_path)
    else:
        return ""
@app.route('/')
def matchresume():
    return render_template('matchresume.html')
@app.route('/matcher',methods=['GET','POST'])
def matcher():
    if request.method =='POST':
        job_description= request.form.get('job_description')
        resume_files = request.files.getlist('resumes')
        print("Files received:", resume_files)
        resumes = []
        filenames = []
        for resume_file in resume_files:
            print("Current directory:", os.getcwd())
            if resume_file.filename == '':
                continue

            filename = secure_filename(resume_file.filename)

            filepath = os.path.join(app.config['UPLOAD_FOLDER'],filename)

            resume_file.save(filepath)

            print("Saved:", filepath)
            print("Exists:", os.path.exists(filepath))

            resume_text = extract_text(filepath)

            print("File:", resume_file.filename)
            print("Extracted characters:", len(resume_text))
            print(resume_text[:500])

            resumes.append(resume_text)
            filenames.append(resume_file.filename)
            os.remove(filepath)
        if not resumes or not job_description:
            return render_template(
                'matchresume.html',
                message="Please upload resumes and enter job description"
            )
        vectors = TfidfVectorizer().fit_transform(
            [job_description] + resumes
        )

        vectors = vectors.toarray()
        print(vectors)

        job_vector = vectors[0]
        resume_vectors = vectors[1:]

        similarities = cosine_similarity(
            [job_vector],
            resume_vectors
        )[0]

        results = []

        for i, score in enumerate(similarities):
            results.append({
                "filename": filenames[i],
                "score": round(score * 100, 2)
            })

        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        print(results)

        top_n = min(3, len(similarities))
        top_indices = similarities.argsort()[-top_n:][::-1]
        top_resumes=[resume_files[i].filename for i in top_indices]
        similarity_score=[round(similarities[i]*100,2) for i in top_indices]

        return render_template(
            'matchresume.html',
            message= "Resume matching completed successfully. ", top_resumes=top_resumes,
            similarity_score=similarity_score
        )
    return render_template('matchresume.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)