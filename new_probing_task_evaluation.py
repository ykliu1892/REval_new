"""
@date: 20.11.29
@editor: Ke Liu kliu0@umass.edu
Modified code to probe appositional and noun compound modifiers. 
Part of UMass CS685 Fall 2020 project"""

from typing import Optional

import os
import logging
import argparse
import json
import numpy as np
import relex
import reval
from relex.predictors.predictor_utils import load_predictor


logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


ALL_PROBING_TASKS_TACRED = [
    # "ArgTypeHead",            # Commented to do probe evaluation on new features only
    # "ArgTypeTail",
    # "Length",
    # "EntityDistance",
    # "ArgumentOrder",
    # "EntityExistsBetweenHeadTail",
    # "PosTagHeadLeft",
    # "PosTagHeadRight",
    # "PosTagTailLeft",
    # "PosTagTailRight",
    # "TreeDepth",
    # "SDPTreeDepth",
    # "ArgumentHeadGrammaticalRole",
    # "ArgumentTailGrammaticalRole",
    # Kirk's new code
    "ArgumentAddGrammarRole_Head",
    "ArgumentAddGrammarRole_Tail",
    "ArgumentGrammarRole_ControlHead",
    "ArgumentGrammarRole_ControlTail",
]

ALL_PROBING_TASKS_SEMEVAL = [
    "ArgTypeHead",
    "ArgTypeTail",
    "Length",
    "EntityDistance",
    "EntityExistsBetweenHeadTail",
    "PosTagHeadLeft",
    "PosTagHeadRight",
    "PosTagTailLeft",
    "PosTagTailRight",
    "TreeDepth",
    "SDPTreeDepth",
    "ArgumentHeadGrammaticalRole",
    "ArgumentTailGrammaticalRole",
]


def _get_parser():
    parser = argparse.ArgumentParser(description="Run evaluation on probing tasks")

    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="directory containing the model archive file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="directory containing the probing task data files",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="dataset to be evaluated"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="directory to use for storing the probing task results",
    )
    parser.add_argument(
        "--predictor",
        type=str,
        default="relation_classifier",
        help="predictor to use for probing tasks",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="batch size to use for predictions"
    )
    parser.add_argument(
        "--cuda-device", type=int, default=0, help="a cuda device to load the model on"
    )
    parser.add_argument(
        "--result-file-name",
        type=str,
        default="new_probing_results.json",       # edited
        help="name of the file the results are written to",
    )
    parser.add_argument("--prototyping", action="store_true")

    parser.add_argument("--cache-representations", action="store_true")

    return parser


def run_evaluation(
    model_dir: str,
    data_dir: str,
    dataset: str,
    output_dir: Optional[str] = None,
    predictor: str = "relation_classifier",
    batch_size: int = 128,
    cuda_device: int = 0,
    prototyping: bool = False,
    cache_representations: bool = True,
    result_file_name: str = "new_probing_results.json",       # edited
):

    predictor = load_predictor(
        model_dir,
        predictor,
        cuda_device,
        archive_filename="model.tar.gz",
        weights_file=None,
    )

    def prepare(params, samples):
        pass

    cache = {}

    def batcher(params, batch, heads, tails, ner, pos, dep, dep_head, ids):
        if cache_representations:
            inputs = []
            inputs_ids = []

            for sent, head, tail, n, p, d, dh, id_ in zip(
                batch, heads, tails, ner, pos, dep, dep_head, ids
            ):
                if id_ not in cache:
                    inputs.append(
                        dict(
                            text=" ".join(sent),
                            head=head,
                            tail=tail,
                            ner=n,
                            pos=p,
                            dep=d,
                            dep_heads=dh,
                        )
                    )
                    inputs_ids.append(id_)

            if inputs:
                computed_sent_embeddings = {
                    id_: result["input_rep"]
                    for id_, result in zip(
                        inputs_ids, predictor.predict_batch_json(inputs)
                    )
                }
                cache.update(computed_sent_embeddings)

            sent_embeddings = np.array([cache[id_] for id_ in ids])

        else:
            inputs = []
            for sent, head, tail, n, p, d, dh in zip(
                batch, heads, tails, ner, pos, dep, dep_head
            ):
                inputs.append(
                    dict(
                        text=" ".join(sent),
                        head=head,
                        tail=tail,
                        ner=n,
                        pos=p,
                        dep=d,
                        dep_heads=dh,
                    )
                )
            results = predictor.predict_batch_json(inputs)
            sent_embeddings = np.array([result["input_rep"] for result in results])

        return sent_embeddings

    if prototyping:
        params = {
            "task_path": data_dir,
            "usepytorch": True,
            "kfold": 5,
            "batch_size": batch_size,
        }
        params["classifier"] = {
            "nhid": 0,
            "optim": "rmsprop",
            "batch_size": 128,
            "tenacity": 3,
            "epoch_size": 2,
        }
    else:
        params = {
            "task_path": data_dir,
            "usepytorch": True,
            "kfold": 10,
            "batch_size": batch_size,
        }

        ## Modify probe hyperparameters here ##
        params["classifier"] = {    # best SOTA
            "nhid": 256,        
            "optim": "adam",    
            "batch_size": 64,
            "tenacity": 5,
            "epoch_size": 12,  
            "dropout": 0,
        }

        # params["classifier"] = {   
        #     "nhid": 256,        
        #     "optim": "adam",    
        #     "batch_size": 64,
        #     "tenacity": 5,
        #     "epoch_size": 15,  
        #     "dropout": 0.2,
        # }

    if dataset == "tacred":
        tasks = ALL_PROBING_TASKS_TACRED
    elif dataset == "semeval2010":
        tasks = ALL_PROBING_TASKS_SEMEVAL
    else:
        raise ValueError(f"Unknown dataset '{dataset}'.")

    logger.info(f"Parameters: {json.dumps(params, indent=4, sort_keys=True)}")
    logger.info(f"Tasks: {tasks}")

    re = reval.engine.RE(params, batcher, prepare)
    results = re.eval(tasks)
    
    # New code: print accuracy and number of data only
    print("Probing Task Results: ")
    print_keys = ["devacc", "testacc", "testF1", "ndev", "ntest"]
    for task in results.keys():
        print(f"    {task}")
        subdict = results[task]
        for key in subdict.keys():
            if key in print_keys:
                print(f"        {key}: {subdict[key]}")

    # logger.info(
    #     f"Probing Task Results: {json.dumps(results, indent=4, sort_keys=True)}"
    # )

    output_dir = output_dir or model_dir

    with open(os.path.join(output_dir, result_file_name), "w") as prob_res_f:
        json.dump(results, prob_res_f, indent=4, sort_keys=True)


if __name__ == "__main__":
    parser = _get_parser()
    args = parser.parse_args()
    run_evaluation(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        dataset=args.dataset,
        output_dir=args.output_dir,
        predictor=args.predictor,
        batch_size=args.batch_size,
        cuda_device=args.cuda_device,
        prototyping=args.prototyping,
        cache_representations=args.cache_representations,
        result_file_name=args.result_file_name,
    )
