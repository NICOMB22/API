import pandas as pd
import nltk
import unicodedata
import io
from fastapi import FastAPI, UploadFile, File
from joblib import load
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from fastapi.responses import StreamingResponse


app = FastAPI()

def limpiar_texto(texto):
    # Normaliza a la forma no acentuada
    texto = unicodedata.normalize('NFKD', texto)
    # Utiliza una expresión regular para eliminar caracteres no alfabéticos estándar
    return ''.join([c for c in texto if not unicodedata.combining(c)])

@app.post("/predict")
async def make_predictions(csv_file: UploadFile = File(...)):
    # Read the CSV file and load the data into a pandas DataFrame
    df = pd.read_csv(csv_file.file)
    
    # Ensure that the columns match the expected ones by the model
    expected_columns = ["Textos_espanol"]
    if not all(col in df.columns for col in expected_columns):
        return {"error": "El archivo CSV debe contener una columna 'Textos_espanol'."}
    
    df['Textos_espanol'] = df['Textos_espanol'].str.lower()
    df['Textos_espanol'] = df['Textos_espanol'].apply(limpiar_texto)
    df['Textos_espanol'] = df['Textos_espanol'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    df['Textos_espanol'] = df['Textos_espanol'].apply(derivar_texto)
    
    
    Prediction_tfidf = tfidf.fit_transform(df['Textos_espanol'])

    model = load("assets/pipelineAPI.joblib")
    # Make predictions using the loaded model
    df["sdg"] = model.predict(Prediction_tfidf)

    print(df["sdg"].value_counts())

     # Convertir el DataFrame a formato CSV 
    csv_data = df.to_csv(index=False)

    # Convertir el CSV a bytes
    csv_bytes = io.BytesIO(csv_data.encode())

    # Crear una respuesta en formato CSV
    response = StreamingResponse(iter([csv_bytes.read()]), media_type="text/csv")
    response.headers["Content-Disposition"] = 'attachment; filename="predictions.csv"'

    return response
    




def tokenizer(text):
    return word_tokenize(text)


    
# Descarga de las stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))
nltk.download('punkt')
    
tfidf = TfidfVectorizer(tokenizer=tokenizer, stop_words=list(stop_words), lowercase=True, max_features=9100)

stemmer = SnowballStemmer("spanish")

def derivar_texto(texto):
    return ' '.join([stemmer.stem(word) for word in texto.split()])





