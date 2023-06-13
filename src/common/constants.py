import os
import pathlib

# 경로
LOG_FILEPATH: str = os.path.join(
    pathlib.Path(__file__).parent.parent.parent.absolute(), "logs"
)
DATA_PATH: str = os.path.join(
    pathlib.Path(__file__).parent.parent.parent.absolute(), "data"
)
ARTIFACT_PATH: str = os.path.join(
    pathlib.Path(__file__).parent.parent.parent.absolute(), "artifacts"
)

PREDICTION_PATH: str = os.path.join(
    pathlib.Path(__file__).parent.parent.parent.absolute(), "predictions"
)

DRIFT_DETECTION_PATH = os.path.join(ARTIFACT_PATH, "drift_detection")

for path in [
    LOG_FILEPATH,
    ARTIFACT_PATH,
    PREDICTION_PATH,
    DRIFT_DETECTION_PATH,
]:
    if not os.path.exists(path):
        os.makedirs(path)
