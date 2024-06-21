import os
import sys
import argparse
from PyQt5.QtWidgets import QDialog
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon

from app import AppController
from app import LoginDialog
from utils import read_yaml
from utils import replace_env_variables_in_config

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


if __name__ == '__main__':
    config = read_yaml('config.yaml')
    replace_env_variables_in_config(config)

    parser = argparse.ArgumentParser(
        description='Application for sketch-based medical image retrieval.')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(config['static']['icon_path']))

    login = LoginDialog(config)

    if login.exec_() == QDialog.Accepted:
        mode = list(config['modes'].keys())[0]
    else:
        mode = list(config['modes'].keys())[1]

    dataset_type = login.datasetSelection.currentText()
    dataset_name = config['dataset_types'][dataset_type]['dataset_name']

    app_controller = AppController(
        mode=mode,
        config=config,
        dataset_type=dataset_type,
        dataset_name=dataset_name,
        model_name=login.modelSelection.currentText(),
    )

    sys.exit(app.exec_())
