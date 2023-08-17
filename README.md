# LaRS Evaluator
[[`paper`](https://gmd.copernicus.org/articles/14/2057/2021/)] [[`dataset`](https://lojzezust.github.io/lars-dataset)] [[`evaluation webserver`](https://macvi.org)] [[`BibTeX`](#cite)]

This is the evaluator code for the paper "LaRS: A Diverse Panoptic Maritime Obstacle Detection Dataset and Benchmark". It can be used to evaluate **semantic segmentation** and **panoptic segmentation** predictions with the LaRS ground-truth annotations. Currently only the GT of the *validation* set is publicly available. For evaluation on the LaRS test set, please submit your submissions through [our evaluation server](https://macvi.org).

## Setup

1. Install requirements into your python environment
    ```bash
    pip install -r requirements.txt
    ```
2. For each of the evaluation tracks (semantic segmentation, panoptic segmentation) the evaluator expects a prediction root dir, where predictions will be placed.  
Configure paths to the dataset and predictions root in config files for your version of LaRS (*e.g.* [lars_test_semantic.yaml](configs/v1.0.0/lars_val_semantic.yaml)).

## Usage

1. Place the predictions of your methods into `<prediction_root_dir>/<method_name>`
    The method dir contains PNG files with predictions for all test images:
    - **Semantic segmentation**: The PNG file contains predicted segmentation masks, following the color coding of classes specified in the configuration file (*e.g.* [lars_test_semantic.yaml](configs/v0.9.3/lars_test_semantic.yaml)).
    - **Panoptic segmentation**: The PNG file contains RGB coded class and instance predictions. The format follows LaRS GT masks: class id is stored in the R component, while instance ids are stored in the G and B components. 
2. Run evaluation:
    ```bash
    $ python evaluate.py path/to/config.yaml <method_name>
    ```

Result files with various statistics will be placed in the configured directory (`results/v1.0.0/<track>/<method>` by default).

## <a name="cite"></a>Citation

If you use LaRS, please cite our paper.

```bibtex
@InProceedings{Zust2023LaRS,
  title={LaRS: A Diverse Panoptic Maritime Obstacle Detection Dataset and Benchmark},
  author={{\v{Z}}ust, Lojze and Per{\v{s}}, Janez and Kristan, Matej},
  booktitle={International Conference on Computer Vision (ICCV)},
  year={2023}
}
```
