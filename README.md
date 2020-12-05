# REval new - modification of REval 




## 🎓&nbsp; Introduction

REval new is a simple framework for probing sentence-level representations of Relation Extraction models.

Modified code to probe appositional and noun compound modifiers. Designed to study RE models trained on TACRED.

Part of UMass CS685 Fall 2020 project.

Annotated data is saved ./AddGR_data
Evaluation results are saved ./EvaluationResults

## ✅&nbsp; Requirements

REval new is tested with:

- Python 3.7


## 🚀&nbsp; Installation

### From source
```bash
git clone https://github.com/ykliu1892/REval_new
cd REval_new
pip install -r requirements.txt
```

## 🔬&nbsp; Probing

### Supported Datasets
/AddGR_data

### Probing Tasks

"ArgumentAddGrammarRole_Head"
"ArgumentAddGrammarRole_Tail"
"ArgumentGrammarRole_ControlHead"
"ArgumentGrammarRole_ControlTail"


## 🔧&nbsp; Usage

### **Step 1**: create the probing task datasets from the original [datasets](#supported-datasets).

#### TACRED

```bash
python reval.py generate-all-from-tacred \
    --train-file <TACRED DIR>/train.json \
    --validation-file <TACRED DIR>/dev.json \
    --test-file <TACRED DIR>/test.json \
    --output-dir ./data/tacred/
```

### **Step 2**: Train RE models using RelEx repository (recommend to do with GPU).

```bash
git clone https://github.com/DFKI-NLP/RelEx
cd RelEx
!pip install -r requirements.txt
```

save TACRED data under ../relex-data/tacred/

```
!allennlp train \
  ./configs/relation_classification/tacred/baseline_cnn_tacred_bert.jsonnet \
  -s <MODEL DIR> \
  --include-package relex
```

### **Step 3**: Run the probing tasks on a model.

Original probing tasks
    ```
    git clone https://github.com/DFKI-NLP/REval
    ```
    ```
    !python 'REval/probing_task_evaluation.py' \
      --model-dir <MODEL DIR> \
      --data-dir 'REval_new/data/tacred/' \
      --dataset tacred --cuda-device 0 --batch-size 64 --cache-representations
    ```

New probing tasks

Move "appos_nn_head.txt", "appos_nn_tail.txt", "control_head.txt", and "control_tail.txt" to REval_new/data/tacred/

Then run:
    ```
    !python 'REval_new/new_probing_task_evaluation.py' \
        --model-dir <MODEL DIR> \
        --data-dir 'REval_new/data/tacred/' \
        --dataset tacred --cuda-device 0 --batch-size 64 --cache-representations
    ```


After the run is completed, the results are stored to `new_probing_task_results.json` in the `model-dir`.

```json
{   
    [...]
    "ArgumentAddGrammarRole_Tail": {
    [...]
        "99": {
            "1) id": "61b3a179d3a9265c91e3",
            "2) sentence": "Japanese office equipment maker Konica Minolta said Tuesday it was tying up with Dutch rival Oce in a bid to focus energies on profitable business areas .",
            "3) head": "Konica Minolta at (4, 5)",
            "4) tail": "Japanese at (0, 0)",
            "5) NER": "MISC O O O ORGANIZATION ORGANIZATION O DATE O O O O O MISC O ORGANIZATION O O O O O O O O O O O",
            "6) actual label": "2: noun compound modifier",
            "7) predicted": "2: noun compound modifier"
        },
        "devacc": 86.19,
        "ndev": 449,
        "ntest": 390,
        "testF1": 72.16,
        "testacc": 82.56
    },
    "ArgumentGrammarRole_ControlHead": {
        "devacc": 37.56,
        "ndev": 450,
        "ntest": 398,
        "testF1": 29.93,
        "testacc": 30.4
    },
    "ArgumentGrammarRole_ControlTail": {
        "devacc": 36.53,
        "ndev": 449,
        "ntest": 390,
        "testF1": 29.2,
        "testacc": 33.85
    }
}
```
