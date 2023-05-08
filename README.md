# LaRS Evaluator

## Setup

1. For each of the evaluation tracks (semantic segmentation, panoptic segmentation) the evaluator expects a prediction root dir, where predictions will be placed.  
Configure paths to the dataset and predictions root in config files for your version of LaRS (*e.g.* [lars_test_semantic.yaml](configs/v0.9.3/lars_test_semantic.yaml)).
2. Install requirements into your python environment
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Place the predictions of your methods into `<prediction_root_dir>/<method_name>`
    The method dir contains PNG files with predictions for all test images:
    - **Semantic segmentation**: The PNG file contains predicted segmentation masks, following the color coding of classes specified in the configuration file (*e.g.* [lars_test_semantic.yaml](configs/v0.9.3/lars_test_semantic.yaml)).
    - **Panoptic segmentation**: The PNG file contains RGB coded class and instance predictions. The format follows LaRS GT masks: class id is stored in the R component, while instance ids are stored in the G and B components. 
2. Run evaluation:
    ```bash
    $ python evaluate.py path/to/config.yaml <method_name>
    ```
