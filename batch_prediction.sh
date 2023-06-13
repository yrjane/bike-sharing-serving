BASH_ENV=~/.bashrc
ROOT_PATH=/workspaces/mlops-practice
PIPENV_PIPFILE=$ROOT_PATH/Pipfile
PYENV_VERSION=3.9.16
export PATH=$PATH:/home/codespace/.pyenv/shims
export PIPENV_PIPFILE=$PIPENV_PIPFILE
export PYENV_VERSION=$PYENV_VERSION
PIPENV_PYTHON=/home/codespace/.local/share/virtualenvs/mlops-practice-2_cZx3tu/bin/python
$PIPENV_PYTHON $ROOT_PATH/batch_prediction.py >> $ROOT_PATH/cron.log 2>&1
