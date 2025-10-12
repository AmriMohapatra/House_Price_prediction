from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np

# Load model and encoders
model = joblib.load("house_price_model.pkl")
le_dict = joblib.load("label_encoders.pkl")

# Initialize FastAPI
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Mount static folder for CSS
app.mount("/static", StaticFiles(directory="static"), name="static")

# Categorical options
neighborhood_options = list(le_dict['Neighborhood'].classes_)
housestyle_options = list(le_dict['HouseStyle'].classes_)
exterior1st_options = list(le_dict['Exterior1st'].classes_)
functional_options = list(le_dict['Functional'].classes_)
centralair_options = list(le_dict['CentralAir'].classes_)

# Serve homepage
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "neighborhood_options": neighborhood_options,
        "housestyle_options": housestyle_options,
        "exterior1st_options": exterior1st_options,
        "functional_options": functional_options,
        "centralair_options": centralair_options
    })

# Handle prediction
@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    OverallQual: int = Form(...),
    GrLivArea: float = Form(...),
    GarageCars: int = Form(...),
    TotalBsmtSF: float = Form(...),
    FirstFlrSF: float = Form(...),
    FullBath: int = Form(...),
    YearBuilt: int = Form(...),
    YearRemodAdd: int = Form(...),
    Neighborhood: str = Form(...),
    HouseStyle: str = Form(...),
    Exterior1st: str = Form(...),
    Functional: str = Form(...),
    CentralAir: str = Form(...)
):
    input_data = {
        "OverallQual": OverallQual,
        "GrLivArea": GrLivArea,
        "GarageCars": GarageCars,
        "TotalBsmtSF": TotalBsmtSF,
        "FirstFlrSF": FirstFlrSF,
        "FullBath": FullBath,
        "YearBuilt": YearBuilt,
        "YearRemodAdd": YearRemodAdd,
        "Neighborhood": Neighborhood,
        "HouseStyle": HouseStyle,
        "Exterior1st": Exterior1st,
        "Functional": Functional,
        "CentralAir": CentralAir
    }

    cat_cols = ["Neighborhood", "HouseStyle", "Exterior1st", "Functional", "CentralAir"]
    num_cols = [c for c in input_data if c not in cat_cols]

    # Encode categorical features
    for col in cat_cols:
        le = le_dict[col]
        input_data[col] = le.transform([input_data[col]])[0]

    # Prepare feature array
    x = np.array([input_data[c] for c in num_cols + cat_cols]).reshape(1, -1)
    pred_price = model.predict(x)[0]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": round(pred_price, 2),
        "neighborhood_options": neighborhood_options,
        "housestyle_options": housestyle_options,
        "exterior1st_options": exterior1st_options,
        "functional_options": functional_options,
        "centralair_options": centralair_options
    })
