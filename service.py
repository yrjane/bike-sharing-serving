import bentoml
import numpy as np
import numpy.typing as npt
import pandas as pd
from bentoml.io import JSON, NumpyNdarray
from pydantic import BaseModel


class Features(BaseModel):
    season: int
    holiday: int
    workingday: int
    weather: int
    temp: int
    atemp: int
    humidity: int
    windspeed: float


# TODO: 학습 코드에서 저장한 베스트 모델을 가져올 것 (house_rent:latest)
bento_model = bentoml.sklearn.get("house_rent:latest")
model_runner = bento_model.to_runner()
svc = bentoml.Service("bike_sharing_regressor", runners=[model_runner])
# TODO: "bike_sharing_regressor"라는 이름으로 서비스를 띄우기


@svc.api(input=JSON(pydantic_model=Features), output=NumpyNdarray())
async def predict(input_data: Features) -> npt.NDArray:
    input_df = pd.DataFrame([input_data.dict()])
    log_pred = await model_runner.predict.async_run(input_df)
    return np.expm1(log_pred)
