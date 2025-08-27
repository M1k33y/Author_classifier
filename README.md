# 🕯️ Spooky Author Identification – Kaggle Competition

## 📌 Overview
Acest proiect a fost realizat în cadrul competiției **Kaggle – Spooky Books** (ediția 2025), desfășurată în perioada **18 – 24 iulie 2025**.  

Provocarea a constat în **clasificarea automată a fragmentelor de text** provenite din operele a trei autori celebri de literatură horror:
- **Edgar Allan Poe (EAP)**
- **HP Lovecraft (HPL)**
- **Mary Shelley (MWS)**

Obiectivul principal: **prezicerea probabilităților ca un fragment să aparțină fiecăruia dintre cei trei autori**, optimizând scorul de evaluare pe baza **multi-class logarithmic loss (log_loss)**.

---

## 🎃 Contextul competiției
În această competiție tematică de Halloween, participanții au fost provocați să folosească tehnici de **Machine Learning și NLP** pentru a reconstitui “paginile pierdute” și a atribui fragmentele de text autorului corect.  

---

## 🧪 Datele
Setul de date conținea fragmente de texte etichetate cu autorul corespunzător.  
- **Train**: conținea textele și etichetele asociate (EAP, HPL, MWS).  
- **Test**: conținea doar textele, pentru care trebuia prezis autorul.  

## 📊 Rezultate și Acuratețe

### 🔹 Performanța pe setul de validare
- **Accuracy:** ~92%  
- **Macro F1-score:** ~0.91  
- **Log Loss (validare internă):** ~0.20  

### 🔹 Performanța pe Kaggle (Leaderboard)
- **Log Loss public LB:** 0.21  
- **Log Loss private LB:** 0.22  
- **Clasare finală:** locul **22 din 309 ** 

📌 **Modelul final** a fost un **ensemble între DeBERTa-v3-Large și un meta-classifier XGBoost pe reprezentări TF-IDF + SVD**, ceea ce a dus la o scădere semnificativă a log_loss față de baseline (~0.55).  

---



