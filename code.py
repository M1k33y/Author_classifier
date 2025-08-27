
# ---  Spooky Author Identification: Ensemble of RoBERTa + TF-IDF ---

# !pip install -q transformers datasets scikit-learn
# !pip install -q nlpaug nltk
# !pip install -q nlpaug transformers
#!pip install -q backtrans==0.1.2


# --- 0. Imports ---
import nltk

import pandas as pd
import os
import numpy as np
import torch
import time
import torch.nn as nn
import nlpaug.augmenter.word as naw
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelEncoder, FunctionTransformer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, log_loss
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.pipeline import make_pipeline
from torch.cuda.amp import autocast, GradScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.decomposition import TruncatedSVD
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from nlpaug.augmenter.word import SynonymAug
#from backtrans import BackTranslator
from nlpaug.augmenter.word import RandomWordAug

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')  # necesar pentru SynonymAug
nltk.download('wordnet')
nltk.download('omw-1.4')

# --- 1. Load data ---
train = pd.read_csv("labeled_sentence_corpus.csv")
test = pd.read_csv("test.csv")


context_aug = naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased',
    action="substitute",
    device='cpu',
    batch_size=16
)

syn_aug = SynonymAug(aug_src='wordnet')

if os.path.exists("augmented_texts.csv"):
    print("✅ Încarc augmentările salvate...")
    aug_df = pd.read_csv("augmented_texts.csv")
else:
  train_sample = train.sample(n=3000, random_state=42)
  texts = train_sample["text"].tolist()
  labels = train_sample["author"].tolist()

  aug_context_texts = []
  aug_synonym_texts = []

  print(f"🔄 Augmentez {len(texts)} exemple cu ContextualWordEmbsAug...")
  for i in tqdm(range(0, len(texts), 16)):
      batch = texts[i:i+16]
      aug_context_texts.extend(context_aug.augment(batch))

  print(f"🔄 Augmentez {len(texts)} exemple cu SynonymAug...")
  aug_synonym_texts = []
  for text in tqdm(texts):
      try:
          aug_synonym_texts.append(syn_aug.augment(text))
      except Exception as e:
          aug_synonym_texts.append(text)  # fallback la text original

  #bt_aug = BackTranslator(source='en', intermediate='fr')  # traduce prin franceză



  # Creez dataframe combinat
  aug_df_context = pd.DataFrame({"text": aug_context_texts, "author": labels})
  aug_df_synonym = pd.DataFrame({"text": aug_synonym_texts, "author": labels})


  # print(f"🔄 Augmentez {len(texts)} exemple cu BackTranslation...")
  # aug_bt_texts = []
  # for text in tqdm(texts):
  #     try:
  #         aug_bt_texts.append(bt_aug.translate(text))
  #     except:
  #         aug_bt_texts.append(text)

  # aug_df_bt = pd.DataFrame({"text": aug_bt_texts, "author": labels})

  insert_aug = RandomWordAug(action="insert")
  delete_aug = RandomWordAug(action="delete")

  print(f"🔄 Augmentez {len(texts)} exemple cu Random Insert/Delete...")
  aug_inserted = [insert_aug.augment(t) for t in tqdm(texts)]
  aug_deleted = [delete_aug.augment(t) for t in tqdm(texts)]

  aug_df_insert = pd.DataFrame({"text": aug_inserted, "author": labels})
  aug_df_delete = pd.DataFrame({"text": aug_deleted, "author": labels})
  aug_df = pd.concat([aug_df_context, aug_df_synonym, aug_df_insert, aug_df_delete], ignore_index=True)


  # Combin cu datele originale
  train = pd.concat([train, aug_df], ignore_index=True)
  print(f"✅ Dataset extins: {len(train)} exemple după augmentare dublă.")

  aug_df.to_csv("augmented_texts.csv", index=False)
  print("✅ Augmentările au fost salvate în 'augmented_texts.csv'")

label_encoder = LabelEncoder()
train['label'] = label_encoder.fit_transform(train['author'])



# --- 2. Feature extraction (common for both models) ---


def extract_features(texts):
    stopwords = set(ENGLISH_STOP_WORDS)

    def count_rare_words(text):
        words = text.lower().split()
        return sum(1 for word in words if word not in stopwords and len(word) > 3)

    def count_uppercase_words(text):
        return sum(1 for word in text.split() if word.isupper())

    def has_long_words(text):
        return int(any(len(word) > 12 for word in text.split()))

    def count_uppercase_chars(text):
        return sum(1 for c in text if c.isupper())

    def count_digits(text):
        return sum(1 for c in text if c.isdigit())

    def avg_sentence_length(text):
        sentences = text.split('.')  # simplificat
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        return np.mean(sentence_lengths) if sentence_lengths else 0

    def sentence_count(text):
        return text.count('.') + text.count('!') + text.count('?')

    features = pd.DataFrame()

    # Existente
    features['text_len'] = texts.str.len()
    features['avg_word_length'] = texts.apply(lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0)
    features['word_count'] = texts.str.split().apply(len)
    features['colon_count'] = texts.str.count(':')
    features['semicolon_count'] = texts.str.count(';')
    features['exclaim_count'] = texts.str.count('!')
    features['question_count'] = texts.str.count(r'\?')
    features['punctuation_ratio'] = texts.apply(lambda x: sum([1 for c in x if c in '.,;!?']) / len(x) if len(x) > 0 else 0)
    features['ellipsis_count'] = texts.str.count(r'\.\.\.')
    features['repeat_exclaim'] = texts.str.count(r'!{2,}')
    features['dash_count'] = texts.str.count('—')
    features['uppercase_word_count'] = texts.apply(count_uppercase_words)
    features['rare_word_count'] = texts.apply(count_rare_words)
    features['long_word_flag'] = texts.apply(has_long_words)
    features['hpl_words'] = texts.str.contains(r'eldritch|cosmic|nameless|gibbering', case=False).astype(int)
    features['eap_words'] = texts.str.contains(r'tomb|horror|gloom|midnight', case=False).astype(int)
    features['mws_words'] = texts.str.contains(r'thou|thy|shalt|creature', case=False).astype(int)

    # NOI
    features['capital_ratio'] = texts.apply(lambda x: count_uppercase_chars(x) / len(x) if len(x) > 0 else 0)
    features['digit_ratio'] = texts.apply(lambda x: count_digits(x) / len(x) if len(x) > 0 else 0)
    features['stopword_ratio'] = texts.apply(lambda x: len([w for w in x.lower().split() if w in stopwords]) / (len(x.split()) + 1e-5))
    features['unique_word_ratio'] = texts.apply(lambda x: len(set(x.split())) / (len(x.split()) + 1e-5))
    features['avg_sentence_length'] = texts.apply(avg_sentence_length)
    features['sentence_count'] = texts.apply(sentence_count)
    features['quotation_count'] = texts.str.count('"') + texts.str.count("'")
    features['creepy_mood'] = texts.str.contains(r'madness|abyss|insanity|dread|haunting|terror|unseen|nameless', case=False).astype(int)

    return features.astype(np.float32)
# --- 3. RoBERTaWithFeatures Model ---

#tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

class AuthorDataset(Dataset):
    def __init__(self, texts, features, labels):
        self.encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=256, return_tensors="pt")
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['features'] = self.features[idx]
        item['labels'] = self.labels[idx]
        return item

class RobertaWithFeatures(nn.Module):
    def __init__(self, n_features, num_labels):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size + n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_labels)
        )

    def forward(self, input_ids, attention_mask, features, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        combined = torch.cat((cls_output, features), dim=1)
        logits = self.classifier(combined)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(label_smoothing=0.05)(logits, labels)
        return {"loss": loss, "logits": logits}

# Split data
X_train, X_val, y_train, y_val = train_test_split(train["text"], train["label"], test_size=0.1, random_state=42, stratify=train["label"])

word_tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=100000, min_df=2, max_df=0.6, sublinear_tf=True)
char_tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,6), max_features=100000, min_df=1, max_df=0.75, sublinear_tf=True)

combined_features = FeatureUnion([
    ('word_tfidf', word_tfidf),
    ('char_tfidf', char_tfidf),
])

vectorizer_pipeline = make_pipeline(combined_features)
X_tfidf_train = vectorizer_pipeline.fit_transform(X_train.astype(str))
X_tfidf_val = vectorizer_pipeline.transform(X_val.astype(str))

Xf_train = extract_features(X_train)
Xf_val = extract_features(X_val)
Xf_test = extract_features(test["text"])

train_dataset = AuthorDataset(X_train, Xf_train, y_train)
val_dataset = AuthorDataset(X_val, Xf_val, y_val)
test_dataset = AuthorDataset(test["text"], Xf_test, pd.Series([0]*len(test)))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

roberta_val_probs = np.zeros((len(X_val), 3))  # 3 clase
roberta_test_probs_accum = np.zeros((len(test), 3))


# Get Roberta probs
def get_roberta_probs(model, dataloader):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], features=batch["features"])
            probs = torch.softmax(outputs["logits"], dim=1)
            all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)

for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\n🔁 Fold {fold+1}/{N_FOLDS}")

    X_tr = X_train.iloc[train_idx]
    y_tr = y_train.iloc[train_idx]

    Xf_tr = extract_features(X_tr)
    dataset_tr = AuthorDataset(X_tr, Xf_tr, y_tr)
    loader_tr = DataLoader(dataset_tr, batch_size=32, shuffle=True)

    model = RobertaWithFeatures(n_features=Xf_tr.shape[1], num_labels=3).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scaler = GradScaler()



    # Training

    for epoch in range(4):
        model.train()
        total_loss = 0
        for batch in tqdm(loader_tr):
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast():
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    features=batch["features"],
                    labels=batch["labels"]
                )
                loss = outputs["loss"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Fold {fold+1} Loss: {total_loss / len(loader_tr):.4f}")

    # Predict on val
    val_preds = get_roberta_probs(model, val_loader)
    roberta_val_probs += val_preds / N_FOLDS

    # Predict on test
    test_preds = get_roberta_probs(model, test_loader)
    roberta_test_probs_accum += test_preds / N_FOLDS

roberta_test_probs = roberta_test_probs_accum




# --- 4. TF-IDF +HGBC  ---
# Definește extractorii de feature-uri
stat_feature_transformer = Pipeline([
    ('extract', FunctionTransformer(lambda x: extract_features(x), validate=False)),
    ('scale', StandardScaler())
])

word_tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=100000)
char_tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 6), max_features=100000)

combined_union = FeatureUnion([
    ('word_tfidf', word_tfidf),
    ('char_tfidf', char_tfidf),
    ('stats', stat_feature_transformer)
])

pipeline_template = Pipeline([
    ('features', combined_union),
    ('svd', TruncatedSVD(n_components=300)),
    ('clf', HistGradientBoostingClassifier(loss='log_loss', max_iter=100))
])

# Pregateste structuri de stocare
tfidf_val_probs = np.zeros((len(X_val), 3))  # 3 clase
tfidf_test_probs_accum = np.zeros((len(test), 3))

# 3-fold CV
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"📦 Fold {fold+1}/3")

    X_tr, y_tr = X_train.iloc[train_idx].astype(str), y_train.iloc[train_idx]
    X_va = X_train.iloc[valid_idx].astype(str)

    # Antreneaza modelul pe fold
    model = pipeline_template
    model.fit(X_tr, y_tr)

    # Predictii pe validation split (doar pentru meta input)
    tfidf_val_fold = model.predict_proba(X_val.astype(str))
    tfidf_val_probs += tfidf_val_fold / 3  # medie peste folduri

    # Predictii pe test set
    tfidf_test_fold = model.predict_proba(test["text"].astype(str))
    tfidf_test_probs_accum += tfidf_test_fold / 3  # medie peste folduri

# Rezultatele finale
tfidf_test_probs = tfidf_test_probs_accum

#Adaug optimizare alfa + C

# print("🔎 Grid search pentru α (ensemble weight):")
# for alpha in np.arange(0.5, 0.91, 0.1):
#     for C in [1.0, 10.0, 100.0]:
#         clf = LogisticRegression(C=C, max_iter=1000)
#         clf.fit(X_tfidf_train, y_train)
#         val_probs = clf.predict_proba(X_tfidf_val)
#         blended = alpha * roberta_val_probs + (1 - alpha) * val_probs
#         loss = log_loss(y_val, blended)
#         print(f"α={alpha:.2f} | C={C:.1f} → log_loss={loss:.5f}")


# --- 5. Ensemble & Submission ---

# Combin predictiile ca input pentru meta-model
X_meta = np.hstack([roberta_val_probs, tfidf_val_probs])
X_meta_test = np.hstack([roberta_test_probs, tfidf_test_probs])

# Antrenez un meta-classifier mai expresiv
meta_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    use_label_encoder=False,
    eval_metric='mlogloss',
    verbosity=0
)
meta_model.fit(X_meta, y_val)

# Predictii finale
ensemble_val_probs = meta_model.predict_proba(X_meta)
ensemble_test_probs = meta_model.predict_proba(X_meta_test)

# Scor log loss
val_logloss = log_loss(y_val, ensemble_val_probs)
print(f"📉 Ensemble Validation Log Loss (XGBoost): {val_logloss:.5f}")

submission = pd.DataFrame(ensemble_test_probs, columns=label_encoder.inverse_transform([0, 1, 2]))
submission.insert(0, 'id', test['id'])
submission[['EAP', 'HPL', 'MWS']] = submission[['EAP', 'HPL', 'MWS']].clip(1e-15, 1 - 1e-15)
submission.to_csv('submission.csv', index=False)
print("✅ Final submission saved as 'submission.csv'")

print(f"📦 Dataset final: {len(train)} instanțe antrenare după augmentare.")
print(f"📉 Scor final VALIDATION LOG-LOSS: {val_logloss:.5f}")
