from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import PreTrainedModel
import torch
from typing import List, Dict, Tuple, Any
from abc import abstractmethod
import numpy as np
from datasets import load_metric
import evaluate
from gem_metrics.msttr import MSTTR
from gem_metrics.ngrams import NGramStats
from rl4lms.envs.text_generation.caption_metrics.cider import Cider
from rl4lms.envs.text_generation.caption_metrics.spice.spice import Spice
from gem_metrics.texts import Predictions
from rl4lms.envs.text_generation.summ_metrics.summa_c import SummaCConv, SummaCZS
from rl4lms.data_pools.task_utils.totto.eval_utils import compute_parent, compute_bleu
from rl4lms.data_pools.custom_text_generation_pools import DailyDialog
from rl4lms.envs.text_generation.gpt3_utils import avoid_keywords
from sft4lms.MultiWOZ.eval import MultiWozEvaluator

from tqdm import tqdm
import copy
import rouge
import operator
import math
import functools
from bert_score import BERTScorer

class BaseMetric:
    @abstractmethod
    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        """
        Returns a dict where key is the metric name and value is again a dict consisting of tuple of individual scores (if any) and corpus level score

        eg. {
            metric_name: (individual_scores, corpus_level_score)
            "metric_1": ([0.5, 0.5, 0.8], 0.1)
        }

        """
        raise NotImplementedError


class LearnedRewardMetric(BaseMetric):
    def __init__(
        self,
        model_name: str,
        label_ix: int,
        batch_size: int,
        include_prompt_for_eval: bool = True,
    ) -> None:
        super().__init__()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._tokenizer.truncation_side = "left"
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            self._device
        )
        self._label_ix = label_ix
        self._batch_size = batch_size
        self._include_prompt_for_eval = include_prompt_for_eval

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Dict[str, float]:
        all_scores = []
        current_ix = 0
        n_texts = len(generated_texts)
        while current_ix < n_texts:
            batch_gen_texts = generated_texts[
                current_ix : current_ix + self._batch_size
            ]
            batch_prompt_texts = prompt_texts[
                current_ix : current_ix + self._batch_size
            ]

            if self._include_prompt_for_eval:
                batch_gen_texts = [
                    (prompt + gen)
                    for gen, prompt in zip(batch_gen_texts, batch_prompt_texts)
                ]
            encoded = self._tokenizer(
                batch_gen_texts, return_tensors="pt", truncation=True, padding=True
            )
            with torch.no_grad():
                outputs = self._model(
                    input_ids=encoded.input_ids.to(self._device),
                    attention_mask=encoded.attention_mask.to(self._device),
                )
                scores = torch.softmax(outputs.logits, dim=1)
                scores = scores[:, self._label_ix].tolist()
                all_scores.extend(scores)
            current_ix += self._batch_size

        metric_dict = {
            "semantic/learned_automodel_metric": (all_scores, np.mean(all_scores))
        }
        return metric_dict


class MeteorMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()
        self._metric = load_metric("./metric/meteor.py")

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):

        score = self._metric.compute(
            predictions=generated_texts, references=reference_texts
        )["meteor"]

        metric_dict = {"lexical/meteor": (None, score)}
        return metric_dict


class RougeMetric(BaseMetric):
    def __init__(self, use_single_ref: bool = True) -> None:
        super().__init__()
        self._metric = load_metric("./metric/rouge.py")
        self._use_single_ref = use_single_ref

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        if self._use_single_ref:
            # TBD: this is required for CNN/DM dataset, without this we get low scores
            # TBD: needs investigation
            ref_texts = [ref[0] for ref in reference_texts]
        else:
            ref_texts = reference_texts

        metric_results = self._metric.compute(
            predictions=generated_texts, references=ref_texts, use_stemmer=True
        )
        score_keys = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        metric_dict = {}
        for rouge_type in score_keys:
            rouge_score = metric_results[rouge_type].mid.fmeasure
            metric_dict[f"lexical/rouge_{rouge_type}"] = (None, rouge_score)
        return metric_dict


class BERTScoreMetric(BaseMetric):
    def __init__(self, language: str = "en", metric_value: str = "f1") -> None:
        super().__init__()
        self._metric = load_metric("bertscore")
        self._language = language
        self._metric_value = metric_value # assert metric_value in ['precision', 'recall', 'f1']
        # since models are loaded heavily on cuda:0, use the last one to avoid memory
        # self._last_gpu = f"cuda:{torch.cuda.device_count() - 1}"

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        with torch.no_grad():
            metric_results = self._metric.compute(
                predictions=generated_texts,
                references=reference_texts,
                lang=self._language,
                #device=self._last_gpu,
            )
            bert_scores = metric_results[self._metric_value]
            corpus_level_score = np.mean(bert_scores)
            metric_dict = {"semantic/bert_score": (bert_scores, corpus_level_score)}
            return metric_dict


class BLEUMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()
        self._metric = load_metric("./metric/bleu.py")

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:

        tokenized_predictions = []
        tokenized_reference_texts = []
        for prediction, refs in zip(generated_texts, reference_texts):
            tokenized_prediction = prediction.split()
            tokenized_refs = [ref.split() for ref in refs]
            tokenized_predictions.append(tokenized_prediction)
            tokenized_reference_texts.append(tokenized_refs)

        try:
            metric_results = self._metric.compute(
                predictions=tokenized_predictions, references=tokenized_reference_texts
            )
            bleu_score = metric_results["bleu"]
            metric_dict = {"lexical/bleu": (None, bleu_score)}
            return metric_dict
        except Exception as e:
            return {"lexical/bleu": (None, "n/a")}


class BLEURTMetric(BaseMetric):
    def __init__(self, config_name: str = None) -> None:
        super().__init__()
        self._metric = load_metric("./metric/bleurt.py", config_name=config_name)

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        metric_results = self._metric.compute(
            predictions=generated_texts, references=reference_texts
        )
        corpus_score = np.mean(metric_results["scores"])
        metric_dict = {"semantic/bleurt": (metric_results["scores"], corpus_score)}
        return metric_dict


def get_generated_and_predictions(
    prompt_texts: List[str],
    generated_texts: List[str],
    reference_texts: List[List[str]],
    split_name: str,
):
    split_name = "" if split_name is None else split_name
    preds = {}
    refs = {}
    for ix, (prompt_text, gen_text, ref_text) in enumerate(
        zip(prompt_texts, generated_texts, reference_texts)
    ):
        preds[split_name + prompt_text] = [gen_text]
        refs[split_name + prompt_text] = ref_text
    return preds, refs


def get_individual_scores(
    prompt_texts: List[str], split_name: str, scores_dict: Dict[str, float]
):
    split_name = "" if split_name is None else split_name
    scores = []
    for prompt_text in prompt_texts:
        scores.append(scores_dict.get(split_name + prompt_text, "n/a"))
    return scores


class CIDERMetric(BaseMetric):
    def __init__(self) -> None:
        self._metric = Cider()

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        predictions, references = get_generated_and_predictions(
            prompt_texts, generated_texts, reference_texts, split_name
        )
        (
            corpus_score,
            individual_scores,
        ) = self._metric.compute_score(references, predictions)
        individual_scores = get_individual_scores(
            prompt_texts, split_name, individual_scores
        )

        metric_dict = {"lexical/cider": (individual_scores, corpus_score)}
        return metric_dict


class SpiceMetric(BaseMetric):
    def __init__(self) -> None:
        self._metric = Spice()

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        predictions, references = get_generated_and_predictions(
            prompt_texts, generated_texts, reference_texts, split_name
        )
        (
            corpus_score,
            individual_scores,
        ) = self._metric.compute_score(references, predictions)

        individual_scores = get_individual_scores(
            prompt_texts, split_name, individual_scores
        )

        metric_dict = {"lexical/spice": (individual_scores, corpus_score)}
        return metric_dict


class DiversityMetrics(BaseMetric):
    def __init__(self, window_size: int = 100) -> None:
        self._msttr_metric = MSTTR(window_size=window_size)
        self._n_gram_metric = NGramStats()

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:

        predictions = Predictions(data={"filename": "", "values": generated_texts})
        diversity_metrics = {}
        msttr_metrics = self._msttr_metric.compute(None, predictions)
        n_gram_metrics = self._n_gram_metric.compute(None, predictions)

        for key, value in msttr_metrics.items():
            diversity_metrics[f"diversity_metrics/{key}"] = (None, value)
        for key, value in n_gram_metrics.items():
            diversity_metrics[f"diversity_metrics/{key}"] = (None, value)

        return diversity_metrics


class SummaCZSMetric(BaseMetric):
    """
    Consistency metric for summarization

    https://github.com/tingofurro/summac/
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._scorer = SummaCZS(**kwargs)

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        metric_results = self._scorer.score(prompt_texts, generated_texts)
        corpus_score = np.mean(metric_results["scores"])
        metric_dict = {"consistency/summaczs": (metric_results["scores"], corpus_score)}
        return metric_dict


class SummaCConvMetric(BaseMetric):
    """
    Consistency metric for summarization

    https://github.com/tingofurro/summac/
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._scorer = SummaCConv(**kwargs)

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        metric_results = self._scorer.score(prompt_texts, generated_texts)
        corpus_score = np.mean(metric_results["scores"])
        metric_dict = {
            "consistency/summacconv": (metric_results["scores"], corpus_score)
        }
        return metric_dict


class Perplexity(BaseMetric):
    def __init__(
        self,
        stride: int,
        tokenizer_id: str,
        model_type: str = "causal",
        use_text_from_meta_data: bool = False,
    ) -> None:
        super().__init__()
        self._tokenizer_id = tokenizer_id
        self._model_type = model_type
        self._stride = stride
        self._use_text_from_meta_data = use_text_from_meta_data

    def get_device(self, model: PreTrainedModel):
        try:
            return model.transformer.first_device
        except:
            return model.device

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        if split_name == "train":
            return {}

        if self._model_type != "causal":
            raise NotImplementedError

        # we compute perplexity on reference texts
        if self._use_text_from_meta_data:
            reference_texts = [info["reference"] for info in meta_infos]
        else:
            reference_texts = [ref for refs in reference_texts for ref in refs]
        tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_id)
        encodings = tokenizer("\n\n".join(reference_texts), return_tensors="pt")

        device = self.get_device(model)

        nlls = []
        max_length = model.config.n_positions
        for i in tqdm(range(0, encodings.input_ids.size(1), self._stride)):
            begin_loc = max(i + self._stride - max_length, 0)
            end_loc = min(i + self._stride, encodings.input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop

            # run on last device
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs[0] * trg_len

            nlls.append(neg_log_likelihood)

        return {
            "fluency_metrics/perplexity": (
                None,
                torch.exp(torch.stack(nlls).sum() / end_loc).item(),
            )
        }


class ParentToTTo:
    """
    Official version
    """

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]],
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        tables = [info["raw_table"] for info in meta_infos]
        parent_overall, parent_overlap, parent_non_overlap = compute_parent(
            generated_texts, tables
        )

        metric_results = {}
        metric_names = ["parent_overall", "parent_overlap", "parent_non_overlap"]
        metric_values = [parent_overall, parent_overlap, parent_non_overlap]
        for name, value in zip(metric_names, metric_values):
            metric_results[f"table_to_text/{name}/precision"] = (
                None,
                value["precision"],
            )
            metric_results[f"table_to_text/{name}/recall"] = (None, value["recall"])

            # individual f-scores - fetch only for overall since we don't know for which samples
            if name == "parent_overall":
                f_scores = value["all_f"]
            else:
                f_scores = None

            metric_results[f"table_to_text/{name}_f_score"] = (
                f_scores,
                value["f_score"],
            )
        return metric_results


class BLEUToTTo:
    """
    Official version
    """

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]],
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        tables = [info["raw_table"] for info in meta_infos]
        bleu_overall, bleu_overlap, bleu_non_overlap = compute_bleu(
            generated_texts, tables
        )

        metric_results = {
            "table_to_text/bleu_overall": (None, bleu_overall),
            "table_to_text/bleu_overlap": (None, bleu_overlap),
            "table_to_text/bleu_non_overlap": (None, bleu_non_overlap),
        }
        return metric_results


class RougeLMax(BaseMetric):
    def __init__(self, **args) -> None:
        super().__init__()
        self._metric = rouge.Rouge(metrics=["rouge-l"], **args)

    def _rouge_max_over_ground_truths(self, prediction, ground_truths):
        """
        Computes max of Rouge-L (https://github.com/allenai/unifiedqa/blob/bad6ef339db6286f0d8bd0661a2daeeb0f800f59/evaluation/evaluate_narrativeqa.py#L25)
        """
        # load stemmer
        self._metric.load_stemmer(self._metric.ensure_compatibility)

        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = self._metric.get_scores(prediction, [ground_truth])
            scores_for_ground_truths.append(score)
        max_score = copy.deepcopy(score)
        max_score = max([score["rouge-l"]["f"] for score in scores_for_ground_truths])
        return max_score

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        all_scores = []
        for gen_text, ref_texts in zip(generated_texts, reference_texts):
            rouge_max_score = self._rouge_max_over_ground_truths(gen_text, ref_texts)
            all_scores.append(rouge_max_score)

        metric_dict = {"lexical/rouge_l_max": (all_scores, np.mean(all_scores))}
        return metric_dict


class SacreBLEUMetric(BaseMetric):
    def __init__(self, **args) -> None:
        super().__init__()
        self._args = args
        self._metric = load_metric("sacrebleu")

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:

        metric_results = self._metric.compute(
            predictions=generated_texts, references=reference_texts, **self._args
        )
        bleu_score = metric_results["score"] / 100
        metric_dict = {"lexical/sacrebleu": (None, bleu_score)}
        return metric_dict


class GoogleBLEUMetric(BaseMetric):
    def __init__(self, **args) -> None:
        super().__init__()
        self._args = args
        self._metric = evaluate.load("google_bleu")

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:

        metric_results = self._metric.compute(
            predictions=generated_texts, references=reference_texts, **self._args
        )
        bleu_score = metric_results["google_bleu"]
        metric_dict = {"lexical/google_bleu": (None, bleu_score)}
        return metric_dict


class TERMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()
        self._metric = load_metric("ter")

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:

        metric_results = self._metric.compute(
            predictions=generated_texts, references=reference_texts
        )
        score = metric_results["score"] / 100
        metric_dict = {"lexical/ter": (None, score)}
        return metric_dict


class chrFmetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()
        self._metric = load_metric("chrf")

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:

        metric_results = self._metric.compute(
            predictions=generated_texts, references=reference_texts
        )
        score = metric_results["score"] / 100
        metric_dict = {"lexical/chrf": (None, score)}
        return metric_dict


class HintHitSummarization(BaseMetric):
    def __init__(self, split: str = ';') -> None:
        super().__init__()
        self.SPLIT = split
        
    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:

        assert len(generated_texts) == len(reference_texts)
        hits, not_hits, precisions, nums = [], [], [], []
        hits_word, recall_word, nums_word = [], [], []
        for i in range(len(generated_texts)):
            #reference_texts[i][0]和generated_texts[i]是字符串
            label = reference_texts[i][0].strip().lower()
            pred = generated_texts[i].strip().lower()

            #存放另一个任务的关键字列表字符串
            phrases= meta_infos[i]['phrases'].strip().lower()

            if len(pred) > 0 and pred[-1] == ".":
                pred = pred[:-1]
            #pred = pred[:-1] if pred[-1] == "." else pred
            #pred通过pred.split(self.SPLIT)后返回 分割后的字符串列表.['a','b','c']
            pred = pred.split(self.SPLIT)
            # remove the repeated words
            pred = sorted(pred, key=lambda x: len(x), reverse=True)
            #原代码
            hit_pred = []
            for p in pred:
                p = p.strip()
                if p not in " ".join(hit_pred) and p in label and p not in avoid_keywords:
                    hit_pred.append(p)
            #修改代码
            # hit_pred0 = []  #任务1
            # hit_pred1 = []  #任务2
            # for p in pred:
            #     p = p.strip()
            #     if p not in " ".join(hit_pred0) and p in label and p not in avoid_keywords:
            #         hit_pred0.append(p)
            #     if p not in " ".join(hit_pred1) and p in phrases and p not in avoid_keywords:
            #         hit_pred1.append(p)
            # # unhit_pred没有命中的关键字，权重设置伟0.2
            # unhit_pred = 0.2 * (len(pred) - len(hit_pred0))
            # hit_pred = hit_pred0 + hit_pred1

            # hits表示预测关键词在label中，并且不在avoid_keywords的关键词数
            hits.append(len(hit_pred))
            # precisions表示关键词的精确率
            precisions.append(len(hit_pred) / len(pred) if len(pred) > 0 else 0)
            #precisions.append((len(hit_pred) + unhit_pred) / len(pred) if len(pred) > 0 else 0)
            # nums表示预测的关键词数
            nums.append(len(pred))
            # not_hits表示没有预测的关键词数
            not_hits.append(len(pred)-len(hit_pred))

            # 新增评价指标
            # hits_word表示预测单词在label中，并且不在avoid_keywords的单词数
            # hits_word = append(len(" ".join(hit_pred).split())）

            # nums_word表示label的单词数
            # nums_word = append(len(label.split()))

            # recall_word表示单词的召回率
            # recall_word = append(len(" ".join(hit_pred).split()) / len(label.split()) if len(label.split()) > 0 else 0)


        metric_dict = {
            "keyword/hint_hit": (hits, np.mean(hits)),
            "keyword/hint_not_hit": (not_hits, np.mean(not_hits)),
            "keyword/hint_hit_precision": (precisions, np.mean(precisions)),
            "keyword/hint_num": (nums, np.mean(nums)),
            # "word/hint_hit": (hits_word, np.mean(hits_word)),
            # "word/hint_hit_recall": (recall_word, np.mean(recall_word)),
            # "word/hint_num": (nums_word, np.mean(nums_word)),
        }
        return metric_dict


class HintDialogActAccuracyMultiWOZ(BaseMetric):
    def __init__(self, all_domain: list = ["[taxi]", "[police]", "[hospital]", "[hotel]",
                                           "[attraction]", "[train]", "[restaurant]"],
                       all_acts: list = ['[inform]', '[request]', '[nooffer]', '[recommend]', 
                                         '[select]', '[offerbook]', '[offerbooked]', '[nobook]', 
                                        '[bye]', '[greet]', '[reqmore]', '[welcome]']) -> None:
        super().__init__()
        self.all_domain = all_domain
        self.all_acts = all_acts

    @staticmethod
    def paser_aspn_to_dict(sent, all_domain, all_acts):
        sent = sent.split()
        dialog_act = {}
        domain_idx = [idx for idx,token in enumerate(sent) if token in all_domain+["[general]"]]
        for i,d_idx in enumerate(domain_idx):
            next_d_idx = len(sent) if i+1 == len(domain_idx) else domain_idx[i+1]
            domain = sent[d_idx]
            if domain in dialog_act:
                domain_da = dialog_act[domain]
            else:
                domain_da = {}
            sub_span = sent[d_idx+1:next_d_idx]
            sub_a_idx = [idx for idx,token in enumerate(sub_span) if token in all_acts]
            for j,a_idx in enumerate(sub_a_idx):
                next_a_idx = len(sub_span) if j+1 == len(sub_a_idx) else sub_a_idx[j+1]
                act = sub_span[a_idx]
                act_slots = sub_span[a_idx+1:next_a_idx]
                domain_da[act] = act_slots
            dialog_act[domain] = domain_da
        return dialog_act

    @staticmethod       
    def paser_dict_to_list(goal, level):
        if level == 1:
            return list(goal.keys())
        elif level == 2:
            belief_state = []
            for domain, domain_bs in goal.items():
                for slot_name, slot_value in domain_bs.items():
                    belief_state.append(" ".join([domain, slot_name]))
            return list(set(belief_state))
        elif level == 3:
            belief_state = []
            for domain, domain_bs in goal.items():
                for slot_name, slot_value in domain_bs.items():
                    if isinstance(slot_value, str):
                        belief_state.append(" ".join([domain, slot_name, slot_value]))
                    elif isinstance(slot_value, List):
                        if slot_value:
                            for slot_value_ in slot_value:
                                belief_state.append(" ".join([domain, slot_name, slot_value_]))
                        else:
                            belief_state.append(" ".join([domain, slot_name]))
            return list(set(belief_state))

    @staticmethod
    def dict_jaccard_similarity(old_dict, new_dict, levels=[3]):
        def jaccard(list1, list2):
            intersection = list(set(list1) & set(list2))
            unionset = list(set(list1).union(set(list2)))
            if unionset:
                return float(len(intersection) / len(unionset))
            else:
                return 0.0
        similarity = 0.
        for level in levels:
            old_list = HintDialogActAccuracyMultiWOZ.paser_dict_to_list(old_dict, level=level)
            new_list = HintDialogActAccuracyMultiWOZ.paser_dict_to_list(new_dict, level=level)
            similarity += jaccard(old_list, new_list)
        similarity /= len(levels)
        return similarity

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:

        assert len(generated_texts) == len(reference_texts)
        hint_accuracys = []
        for i in range(len(generated_texts)):
            label = reference_texts[i].strip().lower()
            pred = generated_texts[i].strip().lower()
            label = HintDialogActAccuracyMultiWOZ.paser_aspn_to_dict(label, self.all_domain, self.all_acts)
            pred = HintDialogActAccuracyMultiWOZ.paser_aspn_to_dict(pred, self.all_domain, self.all_acts)
            similarity = HintDialogActAccuracyMultiWOZ.dict_jaccard_similarity(pred, label, levels=[3])
            hint_accuracys.append(similarity)

        metric_dict = {
            "act/accuracy": (hint_accuracys, np.mean(hint_accuracys))
        }
        return metric_dict


class MultiWOZMetric(BaseMetric):
    """
    Metric for MultiWOZ evaluation

    https://github.com/awslabs/pptod/
    """

    def __init__(self, dataset_version="2.0") -> None:
        super().__init__()
        self.evaluator = MultiWozEvaluator(dataset_version=dataset_version)

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:

        eval_turns = []
        for i in range(len(generated_texts)):
            gen_text = generated_texts[i]
            meta_data = meta_infos[i]
            eval_turn = meta_data["eval_turn"]
            my_eval_turn = copy.deepcopy(eval_turn)
            my_eval_turn['resp_gen'] = gen_text
            eval_turns.append(my_eval_turn)

        dev_bleu, dev_success, dev_match, total_success, total_matches, dial_nums = self.evaluator.validation_metric(eval_turns)
        dev_score = 0.5 * (dev_success + dev_match) + dev_bleu
        metric_dict = {
            "multiwoz/bleu": (None, dev_bleu),
            "multiwoz/success": (None, dev_success),
            "multiwoz/inform": (None, dev_match),
            "multiwoz/combined_score": (None, dev_score),
        }
        return metric_dict



if __name__ == "__main__":
    # prompt_texts = [""]
    # gen_texts = [
    #         "Two CNN Heroes, Anuradha Koirala and Pushpa Basnet, have been struggling in the aftermath of the earthquake in Kathmandu, Nepal . Koirala rescues victims of sex trafficking and has a rehabilitation center in Kathmandu that is home to 425 young women and girls . Basnet, whose Early Childhood Development Center provides a home and education to children whose parents are incarcerated, cares for 45 children who have been forced to evacuate their residence.",
    #         "Two CNN Heroes are among the earthquake survivors in Kathmandu, Nepal . Anuradha Koirala, who rescues victims of sex trafficking, has a rehabilitation center in Kathmandu that is home to 425 young women and girls . While her primary facility seems structurally unharmed, all of the children have been sleeping outdoors because of aftershocks, followed by a second earthquake on May 12.",
    #         "CNN Heroes Anuradha Koirala and Pushpa Basnet are among the earthquake survivors in Kathmandu, Nepal . Koirala's rehabilitation center in Kathmandu is home to 425 young women and girls, while Basnet's Early Childhood Development Center provides a home and education to children whose parents are incarcerated . After the earthquake, both centers have been damaged and their inhabitants are struggling to survive ."
    #     ]
    reference_texts = [["Nicholas Salvador accused of beheading Palmira Silva in her garden .\n\
                        He pleaded not guilty on grounds of insanity at London's Old Bailey today .\nAlso pleaded not guilty to a separate assault charge, citing same reason .\
                        \nMrs Silva was found dead in a garden in Edmonton in September 2014 ."]]
    # gen_texts = [
    #         "A would-be cage fighter has pleaded not guilty by reason of insanity to charges of beheading a great-grandmother in her garden in Edmonton, London , and a separate assault charge. Nicholas Salvador, known as Fat Nick, is accused of killing Palmira Silva, 82, with a machete . He was escorted from the dock after his brief appearance at the Old Bailey ."        
    #         ]
    gen_texts = [
            "insanity"        
            ]
    prompt_texts = ["A would-be cage fighter accused of beheading a great-grandmother in her garden has pleaded not guilty by reason of insanity. Appearing briefly at London's Old Bailey, Nicholas Salvador, nicknamed 'Fat Nick', spoke only to issue his plea. Dressed in black and flanked by three security guards, he also pleaded not guilty to a separate assault charge, giving the same reason. Nicholas Salvador (right) is accused of killing Palmira Silva (left) in her garden in September last year . The widow ran a successful cafe, Silva's Cafe, in Church Street, Edmonton - where members of the public laid floral tributes after her murder on September 4 last year . Salvador, 25, is accused of killing Palmira Silva, 82, with a machete. She was found dead in a garden in Nightingale Road, Edmonton, north London, in September 2014. The suburban street became the scene of a major police operation after officers were called to apprehend the alleged killer. Happy family: Palmira and her late husband Domenico with their now grown-up children, Celestina and Rino . Mrs Silva moved to London from her native southern Italy with her husband Domenico six decades ago . Several of the victim's relatives, including her daughter Celestina, were in the courtroom to witness the plea. Mrs Silva moved to London from her native southern Italy with her husband Domenico six decades ago. They married and had a daughter, Celestina, and a son, Rino. The widow ran a successful cafe, Silva's Cafe, in Church Street, Edmonton - where members of the public laid floral tributes after her murder on September 4 last year. Mr Silva died in 2008 but the family continued the business. Mrs Silva had only recently stopped working full time at the cafe that she and her son ran. Salvador, an only child, is thought to have left home at the age of around 13 or 14 after arriving in the UK from Nigeria with his parents. Police were forced to stage a dramatic evacuation operation to save neighbours during the incident, and an armed squad Tasered the suspect. The house in the busy street where the Italian widow's body was found was the scene of a major police operation after officers were called initially to reports that a man had beheaded an animal. Scotland Yard said that its officers distracted a man after he was seen going through back gardens in Nightingale Road while they evacuated people from nearby homes. Judge Hilliard set a trial date of June 22 at the Old Bailey and Salvador was then escorted out of the dock. Sorry we are not currently accepting comments on this article."]
    
    # metric = RougeMetric()
    # print(metric.compute(prompt_texts, gen_texts, reference_texts))

    # metric = SacreBLEUMetric(tokenize="intl")
    # print(metric.compute(prompt_texts, gen_texts, reference_texts))

    # metric = TERMetric()
    # print(metric.compute(prompt_texts, gen_texts, reference_texts))

    # metric = chrFmetric()
    # print(metric.compute(prompt_texts, gen_texts, reference_texts))

    metric = BERTScoreMetric(language="en", metric_value="precision")
    print(metric.compute(prompt_texts, gen_texts, reference_texts))

    # metric = BLEUMetric()
    # print(metric.compute(prompt_texts, gen_texts, reference_texts))

    # metric = BLEURTMetric()
    # print(metric.compute(prompt_texts, gen_texts, reference_texts))

    # metric = DiversityMetrics()
    # print(metric.compute(prompt_texts, gen_texts, reference_texts))

    # document = """Jeff joined Microsoft in 1992 to lead corporate developer evangelism for Windows NT. He then served as a Group Program manager in Microsoft’s Internet Business Unit. In 1998, he led the creation of SharePoint Portal Server, which became one of Microsoft’s fastest-growing businesses, exceeding $2 billion in revenues. Jeff next served as Corporate Vice President for Program Management across Office 365 Services and Servers, which is the foundation of Microsoft’s enterprise cloud leadership. He then led Corporate Strategy supporting Satya Nadella and Amy Hood on Microsoft’s mobile-first/cloud-first transformation and acquisitions. Prior to joining Microsoft, Jeff was vice president for software development for an investment firm in New York. He leads Office shared experiences and core applications, as well as OneDrive and SharePoint consumer and business services in Office 365. Jeff holds a Master of Business Administration degree from Harvard Business School and a Bachelor of Science degree in information systems and finance from New York University."""
    # summary = "Jeff joined Microsoft in 1992 to lead the company's corporate evangelism. He then served as a Group Manager in Microsoft's Internet Business Unit. In 1998, Jeff led Sharepoint Portal Server, which became the company's fastest-growing business, surpassing $3 million in revenue. Jeff next leads corporate strategy for SharePoint and Servers which is the basis of Microsoft's cloud-first strategy. He leads corporate strategy for Satya Nadella and Amy Hood on Microsoft's mobile-first."

    # metric = SummaCZSMetric(granularity="sentence",
    #                         use_ent=True,
    #                         use_con=False)
    # print(metric.compute([document], [summary], []))

    # metric = SummaCConvMetric(granularity="sentence")
    # print(metric.compute([document], [summary], []))

    # prompt_texts = ["1", "2"]
    # gen_texts = [
    #     "The dog is the boy's cat.",
    #     "A boy is picking apples from trees and put them into bags.",
    # ]
    # reference_texts = [
    #     ["The dog is the boy's cat.", "The dog eats the cat of the boy."],
    #     ["A boy is picking apples from trees."],
    # ]
    # metric = CIDERMetric()
    # print(metric.compute(prompt_texts, gen_texts, reference_texts))

    # metric = SpiceMetric()
    # print(metric.compute(prompt_texts, gen_texts, reference_texts))
