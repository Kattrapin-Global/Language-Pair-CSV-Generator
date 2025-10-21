🗂️ Language Pair CSV Generator

🌍 Overview

Language Pair CSV Generator is a Streamlit web app that helps users upload two text documents — one in any foreign language and one in English — then automatically align their sentences and export a bilingual CSV file. These CSVs can later be used for training translation or alignment models.

🚀 Features

- Upload parallel documents (TXT, PDF, DOCX)
- Detect and segment sentences using spaCy
- Align sentences using semantic similarity (Sentence Transformers)
- Preview aligned sentence pairs
- Export results as `<language>_to_en.csv`
- Supports multiple languages (English, Japanese, Chinese, French, German, Spanish, etc.)
- Auto-downloads or preinstalls all spaCy models
- Clean two-column layout and progress indicators

🧰 Tech Stack

- Frontend / Framework: Streamlit
- Backend / NLP: spaCy + Sentence Transformers
- File Handling: pdfminer.six, python-docx
- Data Processing: pandas

⚙️ Setup Instructions

1. Clone or download the project
   ```bash
   git clone https://github.com/Kattrapin-Global/Language-Pair-CSV-Generator.git
   cd language-pair-csv-generator
   ```

2. Create a virtual environment
   ```bash
   python -m venv .venv
   ```

3. Activate it

   Windows (PowerShell):

   ```bash
   .\.venv\Scripts\Activate.ps1
   ```

   macOS / Linux:

   ```bash
   source .venv/bin/activate
   ```

4. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

   This installs all core libraries and preloaded spaCy language models.

5. Run the app
   ```bash
   streamlit run app.py
   ```

   Open your browser at:

   http://localhost:8501

🧠 How It Works

- **Upload Files**

  Upload a text, PDF, or Word file in your target language.

  Enter the language name (e.g., Japanese or ja).

  Upload its English counterpart.

- **Sentence Splitting**

  spaCy splits both texts into sentences.

  If the language model isn’t available, it falls back to a blank pipeline with a rule-based sentencizer.

- **Alignment**

  If both files have the same number of sentences → direct pairing.

  Otherwise, sentences are semantically aligned using Sentence Transformers (paraphrase-multilingual-MiniLM-L12-v2).

- **Output**

  Preview the first 10 sentence pairs.

  Download the full dataset as a CSV file.

📦 File Structure
```
language-pair-csv-generator/
│
├── app.py                # Main Streamlit application
├── requirements.txt      # Dependencies + preinstalled spaCy models
├── README.md             # This documentation
└── .venv/                # Virtual environment (created locally)
```

🧩 Example Output
```
Japanese	English
こんにちは。	Hello.
お元気ですか？	How are you?
これはテストです。	This is a test.
```

➡️ File saved as: Japanese_to_en.csv

⚠️ Notes

- For large PDFs or scanned documents, convert them to plain text before uploading for better accuracy.

- To verify all spaCy models:

  ```bash
  python -m spacy validate
  ```

The app uses CPU-friendly models by default for speed and compatibility.

🪪 License

This project is released under the MIT License — free to use, modify, and distribute.
