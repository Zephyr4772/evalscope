# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Optional

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.metric.semantics import MetricSelector
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags

from .utils import PAPER_URL, IndicGenBenchAdapter, TokenF1Mixin, language_name

ENGLISH = 'english'
INDIC = 'indic'

# English and the 12 Indic languages of XQuAD-IN; passage, question and answer are all in the subset's language.
INDIC_LANGUAGES = ['as', 'bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'or', 'pa', 'ta', 'te', 'ur']
SUBSET_LIST = ['en'] + INDIC_LANGUAGES

# Table 15 of the paper; the one-shot example uses the same format, followed by its answer.
PROMPT_TEMPLATE = '{context}\n\nQ: {question}\nA:'


def subset_file(subset: str, split: str) -> str:
    """Name of the released file for a subset and split, e.g. `xquad_hi_test.json`."""
    if subset not in SUBSET_LIST:
        raise ValueError(f'Unknown XQuAD-IN subset `{subset}`. Expected one of: {", ".join(SUBSET_LIST)}.')
    return f'xquad_{subset}_{split}.json'


@register_benchmark(
    BenchmarkMeta(
        name='xquad_in',
        evaluation_version='v1.0',
        pretty_name='XQuAD-IN',
        tags=[Tags.MULTI_LINGUAL, Tags.READING_COMPREHENSION],
        description="""
## Overview

XQuAD-IN is the multilingual question answering task of IndicGenBench (Google Research, ACL 2024). It extends
XQuAD, itself built from SQuAD v1.1, with professional translations of its passages, questions and answers into
12 Indic languages.

## Task Description

- **Task Type**: Extractive Question Answering (generative)
- **Input**: A passage and a question in the subset's language
- **Output**: A short answer span in the same language
- **Languages**: Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu and
  Urdu, plus English

## Key Features

- 13 subsets named by language (`en`, `hi`, `ta`, ...), each evaluated on all 1,190 questions of its released
  `test` file (the XQuAD test set), 15,470 in total
- The `train` file of the same language (103 questions) provides the in-context examples
- Results are grouped into `indic` (the 12 Indic languages) and `english`, which the paper reports separately

## Evaluation Notes

- Default configuration uses **1-shot** evaluation, the paper's main setting. The example is the first `train`
  question of the same language, so every model sees the same prompt; set `few_shot_num=0` for zero-shot
- Prompt (Table 15 of the paper): `{context}\\n\\nQ: {question}\\nA:`. The example uses the same format followed
  by ` {answer}`, then a blank line before the test item
- The first non-empty line of the output is scored; use greedy decoding (`temperature=0`) as in the paper
- Metric: SQuAD-style **Token-F1**, the best over the reference answers, averaged over questions. Answers are
  lowercased, stripped of all Unicode punctuation (MLQA's multilingual normalization, so a trailing Devanagari
  danda or Arabic comma does not count as part of a word), English articles are removed from English answers, and
  tokens are split on whitespace
- The Indic score is the macro-average of the 12 Indic languages, reported as the `macro_score` of the `indic`
  category; all subsets have the same size, so the category `score` is the same number
- License: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), from XQuAD
""",
        paper_url=PAPER_URL,
        dataset_id='google/IndicGenBench_xquad_in',
        hf_dataset_id='google/IndicGenBench_xquad_in',
        metric_list=['f1'],
        primary_metric=MetricSelector(name='f1', aggregation='mean'),
        subset_list=SUBSET_LIST,
        few_shot_num=1,
        train_split='train',
        eval_split='test',
        prompt_template=PROMPT_TEMPLATE,
    )
)
class XQuADINAdapter(TokenF1Mixin, IndicGenBenchAdapter):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category_map = {subset: ENGLISH if subset == 'en' else INDIC for subset in SUBSET_LIST}

    def subset_file(self, subset: str, split: str) -> str:
        return subset_file(subset, split)

    def check_record(self, subset: str, record: Dict[str, Any]) -> Optional[str]:
        if record['lang'] != subset:
            return f'holds a `{record["lang"]}` row, expected `{subset}`.'
        if not record['answers'] or not all(answer['text'].strip() for answer in record['answers']):
            return 'holds a row without a reference answer.'
        return None

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        answers = [answer['text'].strip() for answer in record['answers']]
        return Sample(
            input=record['question'].strip(),
            target=answers[0] if len(answers) == 1 else answers,
            metadata={
                'context': record['context'].strip(),
                'question': record['question'].strip(),
                'answers': answers,
                'answer_language': record['lang'],
                'language_name': language_name(record['lang']),
                'id': record['id'],
            },
        )

    def format_prompt_template(self, sample: Sample) -> str:
        return self.prompt_template.format(context=sample.metadata['context'], question=sample.metadata['question'])

    def sample_to_fewshot(self, sample: Sample) -> str:
        return f'{self.format_prompt_template(sample)} {sample.metadata["answers"][0]}'
