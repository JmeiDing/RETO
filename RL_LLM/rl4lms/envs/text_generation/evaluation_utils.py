from typing import Any, Dict, List

from stable_baselines3.common.policies import BasePolicy
from tqdm import tqdm
from transformers import AutoTokenizer

from rl4lms.data_pools.custom_text_generation_pools import Sample
from rl4lms.envs.text_generation.logging_utils import Tracker
from rl4lms.envs.text_generation.metric import BaseMetric


def get_batch(samples: List[Sample], batch_size: int):
    current_ix = 0
    n_samples = len(samples)
    while current_ix < n_samples:
        current_batch = samples[current_ix : current_ix + batch_size]
        yield current_batch
        current_ix += batch_size


def evaluate_on_samples(
    policy: BasePolicy,                  #policy = MaskedSeq2SeqLMActorCriticPolicy
    tokenizer: AutoTokenizer,
    samples: List[Sample],               #samples_by_split["val", "test"]
    batch_size: int,                     #eval_batch_size: 2
    max_prompt_length: int,              #max_prompt_length: 1030
    metrics: List[BaseMetric],           #train_eval_config.get("metrics")
    epoch: int,                          #epoch=0
    split_name: str,                     #["val", "test"]
    tracker: Tracker = None,
    dt_control_token: str = "",
    gen_kwargs: Dict[str, Any] = None,   #gen_kwargs = train_eval_config.get("generation_kwargs")
):
    # generate text by batch
    all_generated_texts = []
    all_ref_texts = []
    all_prompt_texts = []
    all_meta_infos = []
    n_samples = len(samples)
    for batch in tqdm(list(get_batch(samples, batch_size)), desc="Evaluating"):
        # 生成关键词
        batch_generated_texts = generate_text(
            policy, tokenizer, batch, max_prompt_length, dt_control_token, gen_kwargs)
        # 目标摘要
        batch_ref_texts = [sample.references for sample in batch]
        # 提示词+源文本
        batch_prompt_texts = [sample.prompt_or_input_text for sample in batch]
        # meta_data={'phrases'、'target'}
        batch_meta_infos = [sample.meta_data for sample in batch]

        all_generated_texts.extend(batch_generated_texts)
        all_ref_texts.extend(batch_ref_texts)
        all_prompt_texts.extend(batch_prompt_texts)
        all_meta_infos.extend(batch_meta_infos)

    # compute metrics
    corpus_level_metrics = {}
    sample_scores_by_metric = {}
    #MetricRegistry——SummarizationWithHintMetric——compute
    if metrics is not None:
        for metric in metrics:
            metric_dict = metric.compute(
                all_prompt_texts,
                all_generated_texts,
                all_ref_texts,
                all_meta_infos,
                policy.get_language_model(),
                split_name,)
            for metric_key, (sample_scores, corpus_score) in metric_dict.items():
                if sample_scores is None:
                    sample_scores = ["n/a"] * n_samples
                corpus_level_metrics[metric_key] = corpus_score
                sample_scores_by_metric[metric_key] = sample_scores

    # aggregate sample metric scores
    sample_predictions_dict = []
    for ix, (sample, prompt_text, generated_text, ref_texts) in enumerate(
        zip(samples, all_prompt_texts, all_generated_texts, all_ref_texts)):
        sample_prediction = {
            "split_name": split_name,
            "sample_id": sample.id,
            "prompt_text": prompt_text,
            "generated_text": generated_text,
            "ref_text": "".join(
                [
                    f"<START-{ref_ix+1}>" + ref_text + f"<END-{ref_ix+1}>"
                    for ref_ix, ref_text in enumerate(ref_texts)
                ]
            ),
        }
        for metric_key, sample_scores in sample_scores_by_metric.items():
            sample_prediction[metric_key] = sample_scores[ix]
        sample_predictions_dict.append(sample_prediction)

    if tracker is not None:
        # log the entire predictions
        tracker.log_predictions(epoch, split_name, sample_predictions_dict)
        # log the corpus level scores
        tracker.log_metrics(epoch, split_name, corpus_level_metrics)


def generate_text(
    policy: BasePolicy,           #policy = MaskedSeq2SeqLMActorCriticPolicy
    tokenizer: AutoTokenizer,
    samples: List[Sample],        #samples_by_split["val", "test"]——batch
    max_prompt_length: int,       #max_prompt_length = 512
    dt_control_token: str,        #dt_control_token = ""
    gen_kwargs: Dict[str, Any],   #gen_kwargs = train_eval_config.get("generation_kwargs")
):
    #prompt_texts = [prompt_or_input_text('Extract the keywords+article) for sample in samples]
    prompt_texts = [
        dt_control_token + sample.prompt_or_input_text for sample in samples
    ]
    #max_prompt_length = 提示文本prompt_or_input_text的最大长度限制
    #gen_kwargs中的min_length = 10
    generated_texts = policy.generate(tokenizer, prompt_texts, max_prompt_length, gen_kwargs=gen_kwargs).gen_texts

    return generated_texts
