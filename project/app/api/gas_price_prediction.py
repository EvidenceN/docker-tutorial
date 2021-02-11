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

    month: int= Field(..., example=8)
    day: int = Field(..., example=30)
    year: int = Field(..., example=2021)

    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""
        return pd.DataFrame([dict(self)])

    @validator('month')
    def month_must_be_positive(cls, value):
        """Validate that month is a positive number and reasonable range."""
        assert value >= 1 and value <=12, f'month == {value}, must be >=1 and <=12'
        return value

    @validator('day')
    def day_must_be_positive(cls, value):
        """Validate that day is a positive number and reasonable range."""
        assert value >= 1 and value <=31, f'day == {value}, must be >=1 and <=31'
        return value

    @validator('year')
    def year_must_be_positive(cls, value):
        """Validate that year is a positive number and reasonable range."""
        assert value > 2019, f'month == {value}, must be > 2020'
        return value


@router.post('/gas_predict')
async def predict(item: Item):
    """
    Get gas price predictions by inputting month, day, and year

    ### Request Body
    - `month`: positive integer
    - `day`: positive integer
    - `year`: positive integer

    ### Response
    - `prediction`: Gas Price Prediction
    """

    month = item.month
    day = item.day
    year = item.year

    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    my_file = os.path.join(THIS_FOLDER, 'gas_model.pkl')

    with open(my_file, "rb") as f:
        model = pickle.load(f)

    prediction = model.predict([[month, day, year]])[0]
    prediction = round(prediction, 2)

    return {
        'prediction': prediction
    }
