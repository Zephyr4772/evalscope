# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, List, Optional, Tuple, Union

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.metric.semantics import MetricSelector
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags

from .utils import LANGUAGES, PAPER_URL, IndicGenBenchAdapter, TokenF1Mixin, language_name

ANSWER_IN_LANGUAGE = 'xorqa_in_xx'
ANSWER_IN_ENGLISH = 'xorqa_in_en'
ENGLISH = 'english'

# The 28 Indic languages of XorQA-IN (no Nepali). Each question has an English answer and a human translation of it.
XORQA_LANGUAGES = [lang for lang in LANGUAGES if lang != 'ne']

# `<lang>`: answer in that language (XorQA-IN-Xx); `<lang>-en`: answer in English (XorQA-IN-En); `en`: the English
# questions, answered in English.
SUBSET_LIST = XORQA_LANGUAGES + [f'{lang}-en' for lang in XORQA_LANGUAGES] + ['en']

# Table 15 of the paper; the one-shot example uses the same format, followed by its answer.
PROMPT_TEMPLATE = (
    'Generate an answer in {language} for the question based on the given passage:\n\n{context}\n\nQ: {question}\nA:'
)


def subset_languages(subset: str) -> Tuple[str, str]:
    """Return the (question, answer) language codes of a subset such as `hi`, `hi-en` or `en`."""
    question, _, answer = subset.partition('-')
    if subset == 'en' or (question in XORQA_LANGUAGES and answer in ('', 'en')):
        return question, answer or question
    raise ValueError(f'Unknown XorQA-IN subset `{subset}`. Expected one of: {", ".join(SUBSET_LIST)}.')


def subset_file(subset: str, split: str) -> str:
    """Name of the released file for a subset and split, e.g. `xorqa_hi_test.json`."""
    return f'xorqa_{subset_languages(subset)[0]}_{split}.json'


def subset_group(subset: str) -> str:
    question, answer = subset_languages(subset)
    if question == 'en':
        return ENGLISH
    return ANSWER_IN_ENGLISH if answer == 'en' else ANSWER_IN_LANGUAGE


@register_benchmark(
    BenchmarkMeta(
        name='xorqa_in',
        evaluation_version='v1.0',
        pretty_name='XorQA-IN',
        tags=[Tags.MULTI_LINGUAL, Tags.READING_COMPREHENSION],
        description="""
## Overview

XorQA-IN is the cross-lingual question answering task of IndicGenBench (Google Research, ACL 2024). Built on
XOR-TyDi QA, it pairs English Wikipedia passages with questions in 28 Indic languages, and each answer is given
both in English and as a human translation into the question's language.

## Task Description

- **Task Type**: Cross-lingual Question Answering (generative)
- **Input**: A passage in English and a question in an Indic language
- **Output**: A short answer, in the question's language (XorQA-IN-Xx) or in English (XorQA-IN-En)
- **Languages**: Assamese, Awadhi, Bengali, Bhojpuri, Bodo, Chhattisgarhi, Garhwali, Gujarati, Haryanvi, Hindi,
  Kannada, Konkani, Maithili, Malvi, Malayalam, Manipuri, Marathi, Marwari, Odia, Pashto, Punjabi, Rajasthani,
  Sanskrit, Santali, Tamil, Telugu, Tibetan and Urdu, plus English questions

## Key Features

- 57 subsets: `<lang>` asks for the answer in that language and is scored against the translated answer;
  `<lang>-en` asks for the answer in English and is scored against the English answer; `en` holds the English
  questions. Each evaluates its language's released `test` file (520 to 539 questions)
- The two answer languages are kept apart: results are grouped into `xorqa_in_xx` (28 languages, target-language
  answers), `xorqa_in_en` (28 languages, English answers) and `english`, as the paper reports them separately
- The `train` file of the same language (94 to 100 questions) provides the in-context examples
- One `test` question (Malvi, row 62) is empty in the release and is skipped in `mup` and `mup-en`

## Evaluation Notes

- Default configuration uses **1-shot** evaluation, the paper's main setting. The example is the first `train`
  question of the same language, with the answer in the subset's answer language; set `few_shot_num=0` for
  zero-shot
- Prompt (Table 15 of the paper), with the English language names of its Table 7:
  `Generate an answer in {language} for the question based on the given passage:\\n\\n{context}\\n\\nQ: {question}\\nA:`
  with `English` as the language for the `-en` and `en` subsets. The example uses the same format followed by
  ` {answer}`, then a blank line before the test item
- The first non-empty line of the output is scored; use greedy decoding (`temperature=0`) as in the paper
- Metric: SQuAD-style **Token-F1**, the best over the reference answers, averaged over questions. Answers are
  lowercased, stripped of all Unicode punctuation (MLQA's multilingual normalization, so a trailing Devanagari
  danda or Tibetan shad does not count as part of a word), English articles are removed from English answers,
  and tokens are split on whitespace
- Each group's score is the macro-average of its 28 languages, reported as the `macro_score` of the
  `xorqa_in_xx` and `xorqa_in_en` categories
- Most Tibetan questions and translated answers begin with the Tibetan words for "Question" and "Answer"; the
  references are kept as released
- License: [MIT](https://opensource.org/license/mit/), from XOR-TyDi QA
""",
        paper_url=PAPER_URL,
        dataset_id='google/IndicGenBench_xorqa_in',
        hf_dataset_id='google/IndicGenBench_xorqa_in',
        metric_list=['f1'],
        primary_metric=MetricSelector(name='f1', aggregation='mean'),
        subset_list=SUBSET_LIST,
        few_shot_num=1,
        train_split='train',
        eval_split='test',
        prompt_template=PROMPT_TEMPLATE,
    )
)
class XorQAINAdapter(TokenF1Mixin, IndicGenBenchAdapter):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category_map = {subset: subset_group(subset) for subset in SUBSET_LIST}

    def subset_file(self, subset: str, split: str) -> str:
        return subset_file(subset, split)

    def check_record(self, subset: str, record: Dict[str, Any]) -> Optional[str]:
        question = subset_languages(subset)[0]
        if record['lang'] != question:
            return f'holds a `{record["lang"]}` row, expected `{question}`.'
        return None

    def record_to_sample(self, record: Dict[str, Any]) -> Union[Sample, List[Sample]]:
        _, answer_language = subset_languages(self.current_subset_name)
        field = 'answers' if answer_language == 'en' else 'translated_answers'
        answers = [answer['text'].strip() for answer in record[field] if answer['text'].strip()]
        question = record['question'].strip()
        if not question or not answers:
            # A few released rows lost their question or translated answer; they cannot be scored.
            return []
        return Sample(
            input=question,
            target=answers[0] if len(answers) == 1 else answers,
            metadata={
                'context': record['context'].strip(),
                'question': question,
                'answers': answers,
                'question_language': record['lang'],
                'answer_language': answer_language,
                'answer_language_name': language_name(answer_language),
                'title': record['title'],
            },
        )

    def format_prompt_template(self, sample: Sample) -> str:
        return self.prompt_template.format(
            language=sample.metadata['answer_language_name'],
            context=sample.metadata['context'],
            question=sample.metadata['question'],
        )

    def sample_to_fewshot(self, sample: Sample) -> str:
        return f'{self.format_prompt_template(sample)} {sample.metadata["answers"][0]}'
