# Predviđanje cene automobila

Projekat mašinskog učenja za istraživanje podataka o polovnim automobilima i
predviđanje njihove cene u američkim dolarima (`priceUSD`). Projekat obuhvata
eksplorativnu analizu podataka, čišćenje, inženjering karakteristika,
pretprocesiranje, treniranje i poređenje regresionih modela.

## Skup podataka

Ulazni podaci nalaze se u fajlu `data/cars.csv`. Skup sadrži ciljnu promenljivu
`priceUSD` i podatke o vozilu, uključujući proizvođača, model, godinu
proizvodnje, stanje, kilometražu, vrstu goriva, zapreminu motora, boju, menjač,
pogon i segment vozila.

## Struktura projekta

```text
car-price-prediction/
├── data/
│   └── cars.csv
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── model_comparison.py
├── models/
│   └── car_price_model.joblib  # nastaje nakon treniranja
├── README.md
└── requirements.txt
```

## Instalacija i pokretanje

Potrebni su Python 3 i `pip`. Iz korenskog direktorijuma projekta pokrenuti:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
jupyter notebook notebooks/01_eda.ipynb
```

Nakon implementacije pojedinačne faze mogu se pokretati ovim redosledom:

```powershell
python src/data_cleaning.py
python src/feature_engineering.py
python src/data_preprocessing.py
python src/model_training.py
python src/model_evaluation.py
python src/model_comparison.py
```

## Roadmap za rešavanje zadatka

### 1. Kreiranje GitHub repozitorijuma

- Postaviti projekat u javni GitHub repozitorijum.
- Dodati početni `README.md`, `.gitignore` i zavisnosti u `requirements.txt`.

### 2. Organizacija strukture projekta

- Čuvati izvorni skup u direktorijumu `data/`.
- EDA radnu svesku smestiti u `notebooks/`.
- Razdvojiti obradu podataka, treniranje i evaluaciju u module unutar `src/`.
- Sačuvane modele smestiti u `models/`.

### 3. Učitavanje i istraživanje podataka

U radnoj svesci `notebooks/01_eda.ipynb` analizirati:

- broj redova i kolona;
- tipove podataka;
- nedostajuće vrednosti;
- raspodelu ciljne promenljive `priceUSD`;
- numeričke i kategorijske kolone;
- ekstremne vrednosti i potencijalne autlajere.

### 4. Čišćenje podataka

U modulu `src/data_cleaning.py` implementirati:

- standardizaciju naziva kolona;
- obradu nedostajućih vrednosti;
- uklanjanje duplikata i nevalidnih redova;
- osnovnu standardizaciju tekstualnih i numeričkih vrednosti.

### 5. Inženjering karakteristika

U modulu `src/feature_engineering.py` dodati korisne karakteristike, na primer:

- `car_age` - starost automobila;
- `mileage_per_year` - prosečna godišnja kilometraža;
- `engine_volume_liters` - zapremina motora izražena u litrima.

Nove karakteristike treba računati bez curenja informacija iz ciljne
promenljive.

### 6. Pretprocesiranje

U modulu `src/data_preprocessing.py` definisati:

- ciljnu promenljivu `priceUSD`;
- numeričke i kategorijske kolone;
- podelu na trening i test skup;
- `ColumnTransformer` i preprocessing pipeline.

Za obradu koristiti `SimpleImputer`, `StandardScaler` i `OneHotEncoder`, a
`OrdinalEncoder` samo za kategorije koje imaju stvaran redosled.

### 7. Treniranje prvog modela

U modulu `src/model_training.py` trenirati najmanje jedan osnovni regresioni
model. Ceo pipeline, zajedno sa pretprocesiranjem, sačuvati pomoću biblioteke
`joblib` kao `models/car_price_model.joblib`.

### 8. Evaluacija modela

U modulu `src/model_evaluation.py` izračunati najmanje sledeće metrike:

- MAE (srednja apsolutna greška);
- RMSE (koren srednje kvadratne greške);
- R² (koeficijent determinacije).

Niže vrednosti MAE i RMSE znače manju grešku, dok R² bliži vrednosti 1 znači da
model objašnjava veći deo varijanse cene.

### 9. Poređenje algoritama

U modulu `src/model_comparison.py` trenirati i pod istim uslovima uporediti
najmanje tri regresiona algoritma, na primer:

- `LinearRegression` kao osnovni model;
- `RandomForestRegressor`;
- `GradientBoostingRegressor`.

Za pouzdanije poređenje koristiti isti test skup ili unakrsnu validaciju i
fiksirati `random_state` gde je primenljivo.

### 10. Izbor finalnog modela

Finalni model izabrati na osnovu najniže greške na neviđenim podacima, uz
proveru R² rezultata, stabilnosti kroz validacione podele i složenosti modela.
Izabrani pipeline sačuvati u direktorijumu `models/`.

### 11. Dokumentovanje rezultata

Nakon završetka eksperimenata u README uneti stvarne rezultate:

| Model | MAE (USD) | RMSE (USD) | R² |
|---|---:|---:|---:|
| Linear Regression | Nije još izmereno | Nije još izmereno | Nije još izmereno |
| Random Forest | Nije još izmereno | Nije još izmereno | Nije još izmereno |
| Gradient Boosting | Nije još izmereno | Nije još izmereno | Nije još izmereno |

**Izabrani model:** biće naveden nakon treniranja i poređenja modela.

**Obrazloženje:** biće zasnovano na izmerenim MAE, RMSE i R² vrednostima na
validacionom ili test skupu. Rezultati se ne navode unapred kako bi
dokumentacija ostala proverljiva i ponovljiva.
