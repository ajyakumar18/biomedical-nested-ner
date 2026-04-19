# Biomedical Nested Named Entity Recognition (Nested NER)

A machine learning project for **Biomedical Nested Named Entity Recognition** built using **BioBERT embeddings**, **spaCy features**, and **layer-wise CRF models**.

The project is designed to detect biomedical entities from tokenized text, including overlapping and nested mentions, by training separate CRF layers for different entity depths. The implementation uses BioBERT for contextual word representations and handcrafted linguistic features to improve prediction quality. fileciteturn0file0

## Project Highlights

- **Nested entity handling** using a layered labeling strategy
- **BioBERT-based contextual embeddings** for each token
- **spaCy POS tags** and handcrafted word-shape features
- **CRF sequence labeling** for entity prediction
- **Evaluation support** for:
  - Overall F1 Score
  - Nested F1 Score
  - NDT F1 Score
  - NST F1 Score
- **Visualization** of metrics and entity distribution

## Tech Stack

- **Python**
- **PyTorch**
- **Transformers**
- **spaCy**
- **scikit-learn**
- **sklearn-crfsuite**
- **NumPy**
- **Matplotlib**

## Methodology

### 1. Data Loading
The project loads train and test JSON files that contain:
- `tokens`
- `entities` with `start`, `end`, and `type`

### 2. Layer Creation
Entities are grouped into layers so that non-overlapping entities are placed together. This helps model nested structures by training one CRF per layer.

### 3. Feature Extraction
For every token, the project extracts:
- lowercase form
- word shape
- capitalization and digit flags
- prefixes and suffixes
- word length
- hyphen and internal uppercase checks
- POS tag
- biomedical keyword flags
- BioBERT embedding values
- surrounding context tokens

### 4. Model Training
A separate **CRF model** is trained for each layer using the extracted features.

### 5. Prediction
Predictions from all layers are combined to produce the final nested entity output.

### 6. Evaluation
The project calculates:
- overall precision, recall, and F1
- nested F1
- NDT F1
- NST F1
- classification report
- per-class F1 visualization

## Dataset Format

The dataset is expected in JSON format similar to:

```json
{
  "tokens": ["The", "protein", "binds", "to", "DNA"],
  "entities": [
    {
      "start": 1,
      "end": 2,
      "type": "Protein"
    }
  ]
}
```

### Entity Fields
- `start`: starting token index
- `end`: ending token index
- `type`: entity label

## Installation

Install the required libraries:

```bash
pip install transformers torch sklearn-crfsuite spacy scikit-learn matplotlib numpy
python -m spacy download en_core_web_sm
```

## Usage

1. Place the dataset JSON files in the expected directory.
2. Run the notebook or Python script.
3. Train the CRF layers.
4. Evaluate the model on the test set.

### Example Workflow

```python
# Load data
# Extract features
# Train CRF models
# Predict entities
# Evaluate results
```

## Results

The model was evaluated on the GENIA dataset using multiple performance metrics:

| Metric        | Score |
|--------------|------|
| Overall F1   | 0.7254 |
| Nested F1    | 0.3632 |
| NDT F1       | 0.4230 |
| NST F1       | 0.2214 |

## Project Structure

```text
project/
├── code.py
├── data/
│   ├── genia_train_dev.json
│   └── genia_test_context.json
├── code_with_results_and _outputs.ipynb
└── README.md
```

## Future Improvements

- Add stronger overlap resolution for nested spans
- Try end-to-end transformer-based nested NER models
- Improve label handling for token-level classification
- Add hyperparameter tuning for CRF layers
- Export predictions in standard BIO/BILOU formats


## Acknowledgements

- **BioBERT** for biomedical language representations
- **spaCy** for linguistic preprocessing
- **sklearn-crfsuite** for CRF sequence labeling
