from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pickle

# Cargar el modelo entrenado
with open("titanic_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

# Definir el esquema de datos de entrada para la API


class Passenger(BaseModel):
    pclass: int
    sex: int  # 0 para Male, 1 para Female
    age: float
    parent: int  # 0 o 1 para indicar si tiene hijos/parientes


# Cargar el modelo entrenado
with open("titanic_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

# Mapeo para mostrar valores legibles en los resultados
sekkusu = {0: "Masculino", 1: "Femenino"}
has_children = {0: "No", 1: "Sí"}

# Endpoint de predicción para API


@app.post("/predict")
async def predict_survival(passenger: Passenger):
    try:
        input_data = [[passenger.pclass, passenger.sex,
                       passenger.age, passenger.parent]]
        prediction = model.predict(input_data)
        return {"survived": bool(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Endpoint para la interfaz de usuario


@app.get("/form", response_class=HTMLResponse)
async def form_get():
    html_content = """
    <html>
        <head>
            <title>Interfaz Titanic - Predicción de Supervivencia</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    background-color: #f4f7f6;
                }
                .container {
                    max-width: 400px;
                    padding: 20px;
                    background-color: #ffffff;
                    border-radius: 8px;
                    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                    text-align: center;
                }
                h2 {
                    color: #333;
                }
                form {
                    display: flex;
                    flex-direction: column;
                    gap: 15px;
                }
                label {
                    font-weight: bold;
                    margin-bottom: 5px;
                    color: #555;
                }
                input[type="number"], select {
                    padding: 8px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    font-size: 14px;
                    width: 100%;
                }
                button {
                    padding: 10px;
                    border: none;
                    background-color: #28a745;
                    color: white;
                    font-size: 16px;
                    border-radius: 4px;
                    cursor: pointer;
                }
                button:hover {
                    background-color: #218838;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Predicción de Supervivencia en el Titanic</h2>
                <form action="/form" method="post">
                    <label for="pclass">Clase de Ticket (1, 2, 3):</label>
                    <select id="pclass" name="pclass" required>
                    <option value="1">1</option>
                    <option value="2">2</option>
                    <option value="3">3</option>
                </select>
                    
                    <label for="sex">Sexo:</label>
                    <select id="sex" name="sex" required>
                        <option value="0">Masculino</option>
                        <option value="1">Femenino</option>
                    </select>
                    
                    <label for="age">Edad:</label>
                    <input type="number" id="age" name="age" required>
                    
                    <label for="parent">¿Tiene hijos o parientes? (0 = No, 1 = Sí):</label>
                    <select id="parent" name="parent" required>
                        <option value="0">No</option>
                        <option value="1">Sí</option>
                    </select>
                    
                    <button type="submit">Predecir Supervivencia</button>
                </form>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Endpoint para manejar el envío del formulario


@app.post("/form", response_class=HTMLResponse)
async def form_post(pclass: int = Form(...), sex: int = Form(...), age: float = Form(...), parent: int = Form(...)):
    input_data = [[pclass, sex, age, parent]]
    prediction = model.predict(input_data)
    result = "Sobrevivió" if prediction[0] == 1 else "No sobrevivió"

    result_html = f"""
    <html>
        <head>
            <title>Resultado de Predicción</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    background-color: #f4f7f6;
                }}
                .container {{
                    max-width: 400px;
                    padding: 20px;
                    background-color: #ffffff;
                    border-radius: 8px;
                    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                    text-align: center;
                }}
                h2 {{
                    color: #333;
                }}
                p {{
                    color: #555;
                    font-size: 16px;
                }}
                .result {{
                    font-size: 18px;
                    font-weight: bold;
                    color: #28a745;
                }}
                a {{
                    display: inline-block;
                    margin-top: 20px;
                    color: #007bff;
                    text-decoration: none;
                }}
                a:hover {{
                    text-decoration: underline;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Resultado de la Predicción</h2>
                <p>La predicción indica que el pasajero con los siguientes datos:</p>
                <p>Sexo: <strong>{sekkusu[sex]}</strong><br>
                Edad: <strong>{age}</strong><br>
                Hijos: <strong>{has_children[parent]}</strong></p>
                <p class="result">Resultado: <strong>{result}</strong></p>
                <a href="/form">Volver al formulario</a>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=result_html)
