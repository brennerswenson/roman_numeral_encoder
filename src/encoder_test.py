import os
from datetime import datetime

import src.micchi_et_al.utils.analyse_results as micchi_analysis
from src.shared_config import DATA_FOLDER

CONFUSION_MATRICES = True  # if confusion matrices are generated
DATASET = 'valid'
MODELS_FOLDER = os.path.join('trained_models')
CHUNK_SIZE = 160
MODEL_NAME = 'encoder_spelling_bass_cut_44'

if __name__ == '__main__':
    model_file_path = os.path.join(MODELS_FOLDER, MODEL_NAME)
    ys_true, ys_pred, info = micchi_analysis.generate_results(DATA_FOLDER, model_file_path, MODEL_NAME, chunk_size=CHUNK_SIZE)
    micchi_analysis.write_tabular_annotations(ys_pred, info["timesteps"], info["file_names"], os.path.join(model_file_path, 'analyses'))
    idx = 139
    acc = micchi_analysis.analyse_results(ys_true, ys_pred, confusion_matrices=CONFUSION_MATRICES)
    model_with_accuracies = micchi_analysis.compare_results(DATA_FOLDER, MODELS_FOLDER, DATASET, export_annotations=True)
    comparison_fp = os.path.join(MODELS_FOLDER, '../../..',
                                 f'comparison_{datetime.now().strftime("%Y-%m-%d_%H-%M")}.csv')
