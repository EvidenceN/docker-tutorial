import logging
import os
import pickle

from fastapi import APIRouter
import pandas as pd
from pydantic import BaseModel, Field, validator

log = logging.getLogger(__name__)
router = APIRouter()


class Item(BaseModel):
    """Use this data model to parse the request body JSON."""

    room_type: str = Field(..., example = "Entire home/apt") 
    latitude: float = Field(..., example = 42.0)
    """positive value"""
    longitude: float = Field(..., example = -42.0)
    """negative value"""

    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""
        return pd.DataFrame([dict(self)])

    @validator('latitude')
    def latitude_must_be_positive(cls, value):
        """Validate that latitude is positive integer."""
        assert value > 0, f'latitude == {value}, must be > 0'
        return value

    @validator('longitude')
    def longitude_must_be_negative(cls, value):
        """Validate that longitude is negative integer."""
        assert value < 0, f'longitude == {value}, must be < 0'
        return value


@router.post('/airbnb_predict')
async def predict(item: Item):
    """
    Make AirBnB price predictions using room type, longitude, and latitude
    On the web dev backend side, longitude and latitude information is converted into city. 

    On the front-end, user selects a city and roomtype, then web dev converts that city into longitude and latitude on the back end. Then the model receives room type, longitude, and latitude information as input, this input is then used to get a model prediction. 

    ### Request Body
    - `room type`: string
    - `latitude`: positive integer or float
    - `longitude`: negative integer or float

    ### Response
    - `prediction`: airbnb price

    ### RoomType Options:
    * Entire home/apt 
    * Private room 
    * Shared room 
    * Hotel room 

    ### Longitude and Latitude
    Longitude has to be negative numbers. Can be integer or float. This type is enforced.\n 
    Latitude has to be positive numbers. Can be integer or float. This type is enforced.
    """
    data = item.to_df()

    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    my_file = os.path.join(THIS_FOLDER, 'airBnB_model_v3.pkl')

    with open(my_file, "rb") as f:
        model = pickle.load(f)

    prediction = round(model.predict(data)[0])

    return {
        'AirBnB Price Prediction': prediction
    }
