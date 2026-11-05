# Mini Project: NLP Pipeline

An end-to-end Natural Language Processing (NLP) pipeline containing data loading, training, evaluation, and inference modules.

---

## 📁 Repository Structure

```text
mini_project_nlp/
├── data/
│   ├── test_data.csv          # Sample / test evaluation data
│   └── text_data.csv          # Training text data
├── src/
│   ├── dataset_loader.py      # Custom Dataset and DataLoader definitions
│   ├── model.py               # NLP model architecture
│   ├── train.py               # Training loop and checkpointing
│   └── test.py                # Evaluation and inference routines
├── .gitignore                 # Ignores __pycache__, virtual environments, and weights
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

---

## 🚀 Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd mini_project_nlp
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🛠️ Usage

### Training
To train the model:
```bash
python src/train.py
```

### Testing / Evaluation
To run inference or evaluate on the test split:
```bash
python src/test.py
```
