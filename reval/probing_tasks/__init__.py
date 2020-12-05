"""
@date: 20.11.29
@editor: Ke Liu kliu0@umass.edu
Modified code to probe appositional and noun compound modifiers. 
Part of UMass CS685 Fall 2020 project"""

import os
import io
import logging
import torch
import numpy as np
# from senteval.tools.validation import SplitClassifier
# New code
from reval.probing_tasks.probe_validation import SplitClassifier     # modified from Senteval.tools.validation
from senteval.tools.classifier import MLP
from sklearn.metrics import f1_score
# End of new code
from reval.probing_tasks import (
    sent_length,
    entity_distance,
    argument_order,
    entity_exists_between_head_tail,
    entity_type_count_between_head_tail,
    pos_tag_argument_position,
    argument_type,
    tree_depth,
    sdp_tree_depth,
    argument_grammatical_role,
    # new code
    argument_add_grammar_role,     
    argument_add_GRControl
    # End of new code
)


def get_probing_task_generator(name: str):
    task_generator = {
        "sentence_length": sent_length.generate,
        "entity_distance": entity_distance.generate,
        "argument_order": argument_order.generate,
        "entity_exists_between_head_tail": entity_exists_between_head_tail.generate,
        "entity_type_count_between_head_tail": entity_type_count_between_head_tail.generate,
        "pos_tag_argument_position": pos_tag_argument_position.generate,
        "argument_type": argument_type.generate,
        "tree_depth": tree_depth.generate,
        "sdp_tree_depth": sdp_tree_depth.generate,
        "argument_grammatical_role": argument_grammatical_role.generate,
        "argument_add_grammar_role": argument_add_grammar_role.generate,
        "argument_add_GRControl": argument_add_grammar_role.generate,
    }.get(name)

    if task_generator is None:
        raise ValueError(f"'{name}' is not a valid probing task.")

    return task_generator


class REPROBINGEval(object):
    def __init__(self, task, task_path, seed=np.random.randint(1000)):
        self.seed = seed
        self.task = task
        logging.debug(
            "***** (Probing) Transfer task : %s classification *****", self.task.upper()
        )
        self.task_data = {
            "train": {
                "X": [],
                "id": [],
                "head": [],
                "tail": [],
                "ner": [],
                "pos": [],
                "dep": [],
                "dep_head": [],
                "y": [],
            },
            "dev": {
                "X": [],
                "id": [],
                "head": [],
                "tail": [],
                "ner": [],
                "pos": [],
                "dep": [],
                "dep_head": [],
                "y": [],
            },
            "test": {
                "X": [],
                "id": [],
                "head": [],
                "tail": [],
                "ner": [],
                "pos": [],
                "dep": [],
                "dep_head": [],
                "y": [],
            },
        }
        self.loadFile(task_path)
        logging.info(
            "Loaded %s train - %s dev - %s test for %s"
            % (
                len(self.task_data["train"]["y"]),
                len(self.task_data["dev"]["y"]),
                len(self.task_data["test"]["y"]),
                self.task,
            )
        )

    def do_prepare(self, params, prepare):
        samples = (
            self.task_data["train"]["X"]
            + self.task_data["dev"]["X"]
            + self.task_data["test"]["X"]
        )
        return prepare(params, samples)

    def loadFile(self, fpath):
        self.tok2split = {"tr": "train", "va": "dev", "te": "test"}
        with io.open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip().split("\t")
                self.task_data[self.tok2split[line[0]]]["X"].append(line[-1].split())
                self.task_data[self.tok2split[line[0]]]["id"].append(line[1])
                self.task_data[self.tok2split[line[0]]]["y"].append(line[2])
                self.task_data[self.tok2split[line[0]]]["head"].append(
                    (int(line[3]), int(line[4]))
                )
                self.task_data[self.tok2split[line[0]]]["tail"].append(
                    (int(line[5]), int(line[6]))
                )
                self.task_data[self.tok2split[line[0]]]["ner"].append(line[7].split())
                self.task_data[self.tok2split[line[0]]]["pos"].append(line[8].split())
                self.task_data[self.tok2split[line[0]]]["dep"].append(line[9].split())
                self.task_data[self.tok2split[line[0]]]["dep_head"].append(
                    list(map(int, line[10].split()))
                )

        labels = sorted(np.unique(self.task_data["train"]["y"]))
        self.tok2label = dict(zip(labels, range(len(labels))))
        self.nclasses = len(self.tok2label)

        for split in self.task_data:
            for i, y in enumerate(self.task_data[split]["y"]):
                self.task_data[split]["y"][i] = self.tok2label[y]

    # Kirk's new code - modify score function from Facebook SentEval/senteval/tools/classifier.py 
    # output predictions and f1 scores from the best probe classifier

    def best_clf_pred(self, best_classifier, devX, devy):
        best_classifier.model.eval()
        correct = 0
        if not isinstance(devX, torch.cuda.FloatTensor) or best_classifier.cudaEfficient:
            devX = torch.FloatTensor(devX).cuda()
            devy = torch.LongTensor(devy).cuda()

        with torch.no_grad():
            Xbatch = devX
            ybatch = devy
            if best_classifier.cudaEfficient:
                Xbatch = Xbatch.cuda()
                ybatch = ybatch.cuda()
            output = best_classifier.model(Xbatch)
            pred = output.data.max(1)[1]
            correct += pred.long().eq(ybatch.data.long()).sum().item()
            y_true = ybatch.cpu().data.long().numpy()
            y_pred = pred.cpu().long().numpy()
            f1 = f1_score(y_true, y_pred, average='macro')
            f1 = round(100*f1, 2)
        return pred, f1

    def display_wrong_preds(self, splitClf, devX, devy):
        add_dict = {
                    0: '0, No additional grammar role',
                    1: '1, Appositional modifier',
                    2: '2, Noun compound modifier',
                    }
        if splitClf.usepytorch:
            clf = MLP(splitClf.classifier_config, inputdim=splitClf.featdim,
                      nclasses=splitClf.nclasses, l2reg=0.0001,
                      seed=splitClf.seed, cudaEfficient=splitClf.cudaEfficient)
            clf.optim = splitClf.config['optim']
            clf.epoch_size = splitClf.config['epoch_size']
            print(f"MLP has epoch size {clf.epoch_size}, optimizer {clf.optim}.")
            clf.fit(splitClf.X['train'], splitClf.y['train'],
                    validation_data=(splitClf.X['valid'], splitClf.y['valid']))
        clf.model.eval()
        correct = 0
        if not isinstance(devX, torch.cuda.FloatTensor) or clf.cudaEfficient:
            devX = torch.FloatTensor(devX).cuda()
            devy = torch.LongTensor(devy).cuda()

        with torch.no_grad():
            Xbatch = devX
            ybatch = devy
            if clf.cudaEfficient:
                Xbatch = Xbatch.cuda()
                ybatch = ybatch.cuda()
            output = clf.model(Xbatch)
            pred = output.data.max(1)[1]
            correct += pred.long().eq(ybatch.data.long()).sum().item()
            
            # Randomly display 10 samples, for error analysis
            ran_i = np.random.randint(0, len(devX)-9)
            for i in range(ran_i, ran_i+10):
                print(f"Sentence id: {self.task_data['test']['id'][i]}")
                tokens = self.task_data['test']['X'][i]
                sentence = ' '.join(tokens)
                for j in range(len(sentence)//150 + 1):                      # Wrap the text manually
                    print(sentence[150*j:150*(j+1)])
                head_index = self.task_data['test']['head'][i]
                tail_index = self.task_data['test']['tail'][i]
                head = ' '.join(tokens[head_index[0]: head_index[1]+1])
                tail = ' '.join(tokens[tail_index[0]: tail_index[1]+1])
                print(f"Head: {head} at {head_index}, tail: {tail} at {tail_index}.")
                print(f"Prediction: {pred[i]}, Actual y: {add_dict[ybatch[i].item()]}. \n")
            accuracy = 1.0 * correct / len(devX)
            print(f"Test Accuracy: {accuracy}.")

    # End of new code

    def run(self, params, batcher):
        task_embed = {"train": {}, "dev": {}, "test": {}}
        bsize = params.batch_size
        logging.info("Computing embeddings for train/dev/test")
        for key in self.task_data:
            # Sort to reduce padding
            sorted_data = sorted(
                zip(
                    self.task_data[key]["X"],
                    self.task_data[key]["id"],
                    self.task_data[key]["y"],
                    self.task_data[key]["head"],
                    self.task_data[key]["tail"],
                    self.task_data[key]["ner"],
                    self.task_data[key]["pos"],
                    self.task_data[key]["dep"],
                    self.task_data[key]["dep_head"],
                ),
                key=lambda z: (len(z[0]), z[1]),
            )
            (
                self.task_data[key]["X"],
                self.task_data[key]["id"],
                self.task_data[key]["y"],
                self.task_data[key]["head"],
                self.task_data[key]["tail"],
                self.task_data[key]["ner"],
                self.task_data[key]["pos"],
                self.task_data[key]["dep"],
                self.task_data[key]["dep_head"],
            ) = map(list, zip(*sorted_data))

            task_embed[key]["X"] = []
            for ii in range(0, len(self.task_data[key]["y"]), bsize):
                batch = self.task_data[key]["X"][ii : ii + bsize]
                id_ = self.task_data[key]["id"][ii : ii + bsize]
                id_ = id_ if id_ != "None" else None
                head = self.task_data[key]["head"][ii : ii + bsize]
                tail = self.task_data[key]["tail"][ii : ii + bsize]
                ner = self.task_data[key]["ner"][ii : ii + bsize]
                pos = self.task_data[key]["pos"][ii : ii + bsize]
                dep = self.task_data[key]["dep"][ii : ii + bsize]
                dep_head = self.task_data[key]["dep_head"][ii : ii + bsize]

                embeddings = batcher(
                    params, batch, head, tail, ner, pos, dep, dep_head, id_
                )
                task_embed[key]["X"].append(embeddings)
            task_embed[key]["X"] = np.vstack(task_embed[key]["X"])
            task_embed[key]["y"] = np.array(self.task_data[key]["y"])
        logging.info("Computed embeddings")

        config_classifier = {
            "nclasses": self.nclasses,
            "seed": np.random.randint(1000),   # modified code
            "usepytorch": params.usepytorch,
            "classifier": params.classifier,
            # new code
            # "epoch_size": 12, 
            # "optim": "sgd,lr=1e-5",           # end of new code
        }

        # if self.task == "WordContent" and params.classifier["nhid"] > 0:
        #     config_classifier = copy.deepcopy(config_classifier)
        #     config_classifier["classifier"]["nhid"] = 0
        #     print(params.classifier["nhid"])

        clf = SplitClassifier(
            X={
                "train": task_embed["train"]["X"],
                "valid": task_embed["dev"]["X"],
                "test": task_embed["test"]["X"],
            },
            y={
                "train": task_embed["train"]["y"],
                "valid": task_embed["dev"]["y"],
                "test": task_embed["test"]["y"],
            },
            config=config_classifier,
        )
        
        # print(f"clf epoch size {clf.config['epoch_size']}, optim {clf.config['optim']}")  # new code
        devacc, testacc, best_clf = clf.run()                                             # modified
          
        logging.debug(
            "\nDev acc : %.1f Test acc : %.1f for %s classification\n"
            % (devacc, testacc, self.task.upper())
        )
        
        pred, f1 = self.best_clf_pred(best_clf, clf.X['test'], clf.y['test'])   

        output = {
            "devacc": devacc,
            "testacc": testacc,
            "testF1": f1,
            "ndev": len(task_embed["dev"]["X"]),
            "ntest": len(task_embed["test"]["X"]),
        }


        
        # Saving test set and predictions into output
        if self.task not in ["ArgumentHeadAddGrammaticalRoleControl", "ArgumentTailAddGrammaticalRoleControl"]:           
            add_dict = {
                        0: '0: no additional grammatical role',
                        1: '1: appositional modifier',
                        2: '2: noun compound modifier',
                        }
            
            for j in range(len(self.task_data['test']['y'])):
                sen_id = self.task_data['test']['id'][j]
                tokens = self.task_data['test']['X'][j]
                sentence = " ".join(tokens)
                head_idx = self.task_data['test']['head'][j]
                tail_idx = self.task_data['test']['tail'][j] 
                head = " ".join(tokens[head_idx[0]: head_idx[1]+1])
                tail = " ".join(tokens[tail_idx[0]: tail_idx[1]+1])
                ner = " ".join(self.task_data['test']['ner'][j])
                y = self.task_data['test']['y'][j]
                y_hat = pred[j].item()
                output[f"{j+1}"] = {
                             "1) id": sen_id,
                             "2) sentence": sentence,
                             "3) head": f"{head} at {head_idx}",
                             "4) tail": f"{tail} at {tail_idx}",
                             '5) NER': ner,
                             "6) actual label": add_dict[y],
                             "7) predicted": add_dict[y_hat],
                            }

        return output
        # return {
        #     "devacc": devacc,
        #     "acc": testacc,
        #     "ndev": len(task_embed["dev"]["X"]),
        #     "ntest": len(task_embed["test"]["X"]),
        # }


"""
Surface Information
"""


class LengthEval(REPROBINGEval):
    def __init__(self, task_path, seed=np.random.randint(1000)):
        task_path = os.path.join(task_path, "sentence_length.txt")
        # labels: bins
        REPROBINGEval.__init__(self, "Length", task_path, seed)


class EntityDistanceEval(REPROBINGEval):
    def __init__(self, task_path, seed=np.random.randint(1000)):
        task_path = os.path.join(task_path, "entity_distance.txt")
        # labels: bins
        REPROBINGEval.__init__(self, "EntityDistance", task_path, seed)


class ArgumentOrderEval(REPROBINGEval):
    def __init__(self, task_path, seed=np.random.randint(1000)):
        task_path = os.path.join(task_path, "argument_order.txt")
        # labels: bins
        REPROBINGEval.__init__(self, "ArgumentOrder", task_path, seed)


class EntityExistsBetweenHeadTailEval(REPROBINGEval):
    def __init__(self, task_path, seed=np.random.randint(1000)):
        task_path = os.path.join(task_path, "entity_exists_between_head_tail.txt")
        # labels: bins
        REPROBINGEval.__init__(self, "EntityExistsBetweenHeadTail", task_path, seed)


class EntityTypeCountBetweenHeadTailEval(REPROBINGEval):
    def __init__(self, task_path, ner_tag="ORG", seed=np.random.randint(1000)):
        task_path = os.path.join(
            task_path, f"entity_type_count_{ner_tag}_between_head_tail.txt"
        )
        # labels: bins
        REPROBINGEval.__init__(
            self, f"EntityTypeCount{ner_tag}BetweenHeadTail", task_path, seed
        )


class PosTagArgPositionEval(REPROBINGEval):
    def __init__(self, task_path, argument, position, seed=np.random.randint(1000)):
        task_path = os.path.join(task_path, f"pos_tag_{argument}_{position}.txt")
        # labels: bins
        REPROBINGEval.__init__(
            self,
            f"PosTag{argument.capitalize()}{position.capitalize()}",
            task_path,
            seed,
        )


class ArgumentTypeEval(REPROBINGEval):
    def __init__(self, task_path, argument, seed=np.random.randint(1000)):
        task_path = os.path.join(task_path, f"argument_type_{argument}.txt")
        # labels: bins
        REPROBINGEval.__init__(self, f"ArgType{argument.capitalize()}", task_path, seed)


class TreeDepthEval(REPROBINGEval):
    def __init__(self, task_path, seed=np.random.randint(1000)):
        task_path = os.path.join(task_path, "tree_depth.txt")
        # labels: bins
        REPROBINGEval.__init__(self, "TreeDepth", task_path, seed)


class SDPTreeDepthEval(REPROBINGEval):
    def __init__(self, task_path, seed=np.random.randint(1000)):
        task_path = os.path.join(task_path, "sdp_tree_depth.txt")
        # labels: bins
        REPROBINGEval.__init__(self, "SDPTreeDepth", task_path, seed)


class ArgumentGrammaticalRoleEval(REPROBINGEval):
    def __init__(self, task_path, argument, seed=np.random.randint(1000)):
        task_path = os.path.join(task_path, f"argument_{argument}_grammatical_role.txt")
        REPROBINGEval.__init__(
            self, f"Argument{argument.capitalize()}GrammaticalRole", task_path, seed
        )

# Kirk's new code 
class ArgumentAddGrammarRoleEval(REPROBINGEval):
    def __init__(self, task_path, argument, seed=np.random.randint(1000)):
        task_path = os.path.join(task_path, f"appos_nn_{argument}.txt")
        REPROBINGEval.__init__(
            self, f"Argument{argument.capitalize()}AddGrammaticalRole", task_path, seed
        )

class ArgumentAddGrammarRoleControl(REPROBINGEval):
    def __init__(self, task_path, argument, seed=np.random.randint(1000)):
        task_path = os.path.join(task_path, f"control_{argument}.txt")
        REPROBINGEval.__init__(
            self, f"Argument{argument.capitalize()}AddGrammaticalRoleControl", task_path, seed
        )
# end of new code
