import os
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template
from PyPDF2 import PdfReader
import docx2txt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from llm_feedback import generate_feedback
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

        resumes = []
        filenames = []
        for resume_file in resume_files:
            if resume_file.filename == '':
                continue

            filename = secure_filename(resume_file.filename)

            filepath = os.path.join(app.config['UPLOAD_FOLDER'],filename)

            resume_file.save(filepath)


            resume_text = extract_text(filepath)



            resumes.append(resume_text)
            filenames.append(resume_file.filename)
            os.remove(filepath)
        if not resumes or not job_description:
            return render_template(
                'matchresume.html',
                message="Please upload resumes and enter job description"
            )
        vectorizer = TfidfVectorizer(stop_words='english')

        vectors = vectorizer.fit_transform(
            [job_description] + resumes
        )

        vectors = vectors.toarray()

        job_vector = vectors[0]
        resume_vectors = vectors[1:]

        similarities = cosine_similarity(
            [job_vector],
            resume_vectors
        )[0]
        feedbacks = []

        results = []

        for i, score in enumerate(similarities):

            jd_words = set(job_description.lower().split())
            resume_words = set(resumes[i].lower().split())

            matched_keywords = list(jd_words & resume_words)[:10]
            missing_keywords = list(jd_words - resume_words)[:10]

            feedback = generate_feedback(
                resume_text=resumes[i],
                job_description=job_description,
                match_score=float(score),
                matched_keywords=matched_keywords,
                missing_keywords=missing_keywords
            )

            feedbacks.append(feedback)
            results.append({
                "filename": filenames[i],
                "score": round(score * 100, 2)
            })

        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )



        top_n = min(3, len(similarities))
        top_indices = similarities.argsort()[-top_n:][::-1]
        top_resumes=[filenames[i] for i in top_indices]
        similarity_score=[round(similarities[i]*100,2) for i in top_indices]

        return render_template(
            'matchresume.html',
            message= "Resume matching completed successfully. ", top_resumes=top_resumes,
            similarity_score=similarity_score,
            feedbacks=feedbacks
        )
    return render_template('matchresume.html')

if __name__ == '__main__':
    app.run(debug=True)