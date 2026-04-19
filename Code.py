import json
import torch
import numpy as np
import spacy
from transformers import AutoTokenizer, AutoModel
from sklearn_crfsuite import CRF
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

train_file_path = "data/genia_train_dev.json"

with open(train_file_path) as f:
    train_data = json.load(f)

print("Train samples:", len(train_data))
print(train_data[0])

test_file_path = "data/genia_test_context.json"

with open(test_file_path) as f:
    test_data = json.load(f)

print("Test samples:", len(test_data))
print(test_data[0])

def is_overlap(e1, e2):
    return not (e1['end'] <= e2['start'] or e2['end'] <= e1['start'])

def create_layers(entities):
    entities = sorted(entities, key=lambda x: (x['end'] - x['start']))
    layers = []
    for e in entities:
        placed = False
        for layer in layers:
            if not any(is_overlap(e, existing) for existing in layer):
                layer.append(e)
                placed = True
                break
        if not placed:
            layers.append([e])
    return layers

def create_bio_labels(tokens, layer_entities):
    labels = ["O"] * len(tokens)
    for e in layer_entities:
        start, end, etype = e["start"], e["end"], e["type"]
        labels[start] = "B-" + etype
        for i in range(start + 1, end):
            labels[i] = "I-" + etype
    return labels

def get_word_shape(word):
    shape = ""
    for ch in word:
        if ch.isupper():
            shape += "X"
        elif ch.islower():
            shape += "x"
        elif ch.isdigit():
            shape += "d"
        else:
            shape += ch
    return shape

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME)

bert_model.to(device)
bert_model.eval()

nlp = spacy.load("en_core_web_sm")
embedding_cache = {}

def get_bert_embeddings(tokens):
    key = tuple(tokens)
    if key in embedding_cache:
        return embedding_cache[key]

    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding=False
    )

    word_ids = encoding.word_ids()
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = bert_model(**encoding)

    hidden_states = outputs.last_hidden_state[0].detach().cpu()
    hidden_size = hidden_states.shape[-1]

    word_embeddings = [np.zeros(hidden_size, dtype=np.float32) for _ in tokens]
    grouped = defaultdict(list)

    for idx, word_id in enumerate(word_ids):
        if word_id is not None and word_id < len(tokens):
            grouped[word_id].append(hidden_states[idx].numpy())

    for word_id, vecs in grouped.items():
        word_embeddings[word_id] = np.mean(vecs, axis=0).astype(np.float32)

    token_embeddings = np.array(word_embeddings, dtype=np.float32)
    embedding_cache[key] = token_embeddings
    return token_embeddings

def extract_features(tokens):
    features = []
    bert_vectors = get_bert_embeddings(tokens)

    doc = nlp(" ".join(tokens))
    pos_tags = [token.pos_ for token in doc]

    if len(pos_tags) < len(tokens):
        pos_tags.extend(["X"] * (len(tokens) - len(pos_tags)))
    elif len(pos_tags) > len(tokens):
        pos_tags = pos_tags[:len(tokens)]

    biomedical_keywords = {
        "protein", "gene", "dna", "rna", "cell",
        "receptor", "factor", "kinase", "enzyme",
        "alpha", "beta", "interleukin"
    }

    for i, word in enumerate(tokens):
        lower_word = word.lower()

        feature = {
            "lower": lower_word,
            "word_shape": get_word_shape(word),
            "is_upper": word.isupper(),
            "is_title": word.istitle(),
            "is_digit": word.isdigit(),
            "pos": pos_tags[i],
            "prefix1": word[:1],
            "prefix2": word[:2],
            "prefix3": word[:3],
            "suffix1": word[-1:],
            "suffix2": word[-2:],
            "suffix3": word[-3:],
            "suffix4": word[-4:],
            "len": len(word),
            "has_hyphen": "-" in word,
            "has_digit": any(c.isdigit() for c in word),
            "has_upper_inside": any(c.isupper() for c in word[1:]),
            "contains_alpha": "alpha" in lower_word,
            "contains_beta": "beta" in lower_word,
            "is_biomedical_keyword": lower_word in biomedical_keywords,
        }

        if i < len(bert_vectors):
            vec = bert_vectors[i]
        else:
            vec = np.zeros(bert_vectors.shape[1], dtype=np.float32)

        for j, val in enumerate(vec[:128]):
            feature[f"bert_{j}"] = float(val)

        for offset in [-2, -1, 1, 2]:
            pos = i + offset
            if 0 <= pos < len(tokens):
                neighbor = tokens[pos]
                feature[f"context_{offset}_word"] = neighbor
                feature[f"context_{offset}_shape"] = get_word_shape(neighbor)
            else:
                feature[f"context_{offset}_PAD"] = True

        features.append(feature)

    return features

def prepare_layered_data(dataset, max_layers=3, limit=None):
    X_layers = [[] for _ in range(max_layers)]
    y_layers = [[] for _ in range(max_layers)]

    if limit is not None:
        dataset = dataset[:limit]

    for idx, sample in enumerate(dataset):
        if idx % 100 == 0:
            print(f"Processing sample {idx}/{len(dataset)}")

        tokens = sample["tokens"]
        entities = sample["entities"]

        layers = create_layers(entities)
        features = extract_features(tokens)

        repeat = 2 if len(layers) >= 2 else 1

        for _ in range(repeat):
            for i in range(min(len(layers), max_layers)):
                labels = create_bio_labels(tokens, layers[i])
                X_layers[i].append(features)
                y_layers[i].append(labels)

    print("Prepared dataset for", len(X_layers), "layers")
    return X_layers, y_layers

X_train_layers, y_train_layers = prepare_layered_data(train_data)
X_test_layers, y_test_layers = prepare_layered_data(test_data)

models = []

for i in range(len(X_train_layers)):
    if len(X_train_layers[i]) == 0:
        continue
    if len(set(sum(y_train_layers[i], []))) <= 1:
        continue

    print(f"Training Layer {i+1}")

    crf = CRF(
        algorithm="lbfgs",
        c1=0.2 + i * 0.08,
        c2=0.2 + i * 0.08,
        max_iterations=100,
        all_possible_transitions=True
    )

    crf.fit(X_train_layers[i], y_train_layers[i])
    models.append(crf)