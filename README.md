# Resume Parser and Job Description Matcher

A web-based application built with **Flask** that allows users to upload resumes (PDF/DOCX) and compare them against multiple job descriptions using **NLP** techniques.  
It calculates similarity scores and ranks resumes based on relevance, helping recruiters quickly shortlist candidates.

---

## 🚀 Features
- **Upload Multiple Resumes** (PDF/DOCX)
- **Enter Up to 3 Job Descriptions** at once
- **NLP-based Scoring**:
  - TF-IDF Cosine Similarity  
  - BERT Semantic Similarity (using `sentence-transformers`)  
- **Relevance Labels**:  
  - Highly Relevant  
  - Moderately Relevant  
  - Not Relevant  
- **Sorted Results Table** for quick analysis
- **Simple & Responsive UI** built with Flask templates

---

## 🛠 Tech Stack
- **Frontend**: HTML, CSS (Flask Templates)
- **Backend**: Python, Flask
- **NLP**: Scikit-learn, Sentence Transformers (`all-MiniLM-L6-v2`)
- **Others**: PyPDF2, docx2txt  

---



