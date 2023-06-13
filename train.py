# TODO: 적절한 위치에 맞는 수준으로 로그 출력되도록 코드 작성

# sourcery skip: raise-specific-error
import os
import sys
import warnings
from distutils.dir_util import copy_tree

import bentoml
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

from src.common.constants import ARTIFACT_PATH, DATA_PATH, LOG_FILEPATH
from src.common.logger import (
    handle_exception,
    log_feature_importance,
    set_logger,
)
from src.common.metrics import rmse_cv_score
from src.common.utils import get_param_set
from src.preprocess import preprocess_pipeline


# 로그 들어갈 위치 
# TODO: 로그를 정해진 로그 경로에 logs.log로 저장하도록 설정

sys.excepthook = handle_exception
warnings.filterwarnings(action="ignore")



if __name__ == "__main__":
    train_df = pd.read_csv(os.path.join(DATA_PATH, "house_rent_train.csv"))

    _X = train_df.drop(["rent", "area_locality", "posted_on"], axis=1)
    y = np.log1p(train_df["rent"])
    
    # TODO: X=_X, y=y로 전처리 파이프라인을 적용해 X에 저장

    # Data storage - 피처 데이터 저장
    if not os.path.exists(os.path.join(DATA_PATH, "storage")):
        os.makedirs(os.path.join(DATA_PATH, "storage"))
    X.assign(rent=y).to_csv(
        # TODO: DATA_PATH 밑에 storage 폴더 밑에 피처 데이터를 저장
        
        index=False,
    )


    params_candidates = {
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 4, 5, 6],
        "max_features": [1.0, 0.9, 0.8, 0.7],
    }

    param_set = get_param_set(params=params_candidates)

    # Set experiment name for mlflow
    experiment_name = "new_experiment"
    mlflow.set_experiment(experiment_name=experiment_name)
    mlflow.set_tracking_uri("./mlruns")

    for i, params in enumerate(param_set):

        run_name = f"Run {i}"
        with mlflow.start_run(run_name=f"Run {i}"):
            regr = GradientBoostingRegressor(**params)
            # 전처리 이후 모델 순서로 파이프라인 작성
            pipeline = Pipeline(
                # TODO: 전처리 파이프라인와 모델을 파이프라인으로 묶을 것
            )
            pipeline.fit(_X, y)

            # get evaluations scores
            score_cv = rmse_cv_score(regr, X, y)

            name = regr.__class__.__name__
            mlflow.set_tag("estimator_name", name)

            # 로깅 정보 : 파라미터 정보
            mlflow.log_params({key: regr.get_params()[key] for key in params})

            # 로깅 정보: 평가 메트릭
            mlflow.log_metrics(
                {
                    "RMSE_CV": #TODO: RMSE_CV 라는 이름으로 score_cv.mean()을 저장
                }
            )

            # 로깅 정보 : 학습 loss
            for s in regr.train_score_:
                mlflow.log_metric("Train Loss", s)

            # 모델 아티팩트 저장
            mlflow.sklearn.log_model(
                # TODO: 최종 파이프라인을 저장
                "model",
            )

            # log charts
            mlflow.log_artifact(
                # TODO: 아티팩트 경로 설정
            )

            # generate a chart for feature importance
            log_feature_importance(train=X, model=regr)

    # Find the best regr
    best_run_df = mlflow.search_runs(
        order_by=["metrics.RMSE_CV ASC"], max_results=1
    )

    if len(best_run_df.index) == 0:
        raise Exception(f"Found no runs for experiment '{experiment_name}'")

    best_run = mlflow.get_run(best_run_df.at[0, "run_id"])
    best_params = best_run.data.params

    best_model_uri = f"{best_run.info.artifact_uri}/model"

    # TODO: 베스트 모델을 아티팩트 폴더에 복사
    copy_tree(
              # TODO: 베스트 모델 URI에서 file:// 를 지울 것, 
              ARTIFACT_PATH
    )


    # BentoML에 모델 저장
    bentoml.sklearn.save_model(
        name="house_rent",
        model=mlflow.sklearn.load_model(
            # TODO: 베스트 모델 URI
        ),
        signatures={"predict": {"batchable": True, "batch_dim": 0}},
        metadata=best_params,
    )
