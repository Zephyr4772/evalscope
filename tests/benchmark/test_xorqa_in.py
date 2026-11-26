"""XorQA-IN: the two answer languages are separate subsets with their own references, prompts and groups."""

import json
from pathlib import Path

import pytest

from evalscope.api.benchmark import DataAdapter
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.indicgenbench.xorqa_in_adapter import SUBSET_LIST, subset_file, subset_languages
from evalscope.config import TaskConfig
from evalscope.report import ReportGenerator

INSTRUCTION = 'Generate an answer in {} for the question based on the given passage:'


def _adapter(**dataset_args) -> DataAdapter:
    args = {'xorqa_in': dataset_args} if dataset_args else {}
    return get_benchmark('xorqa_in', TaskConfig(model='mock', datasets=['xorqa_in'], dataset_args=args))


def _record(question: str, english: str, translated: str, lang: str, context: str = 'Delhi is the capital.') -> dict:
    return {
        'title': 't',
        'context': context,
        'question': question,
        'answers': [{'text': english, 'answer_start': 0}],
        'translated_answers': [{'text': translated, 'answer_start': -1}],
        'lang': lang,
        'split': 'dev',
        'oracle_question': '',
    }


def _sample(adapter: DataAdapter, subset: str, record: dict):
    adapter.current_subset_name = subset
    return adapter.record_to_sample(record)


def test_subsets() -> None:
    assert len(SUBSET_LIST) == len(set(SUBSET_LIST)) == 57
    assert 'ne' not in SUBSET_LIST and 'ne-en' not in SUBSET_LIST  # XorQA-IN has no Nepali
    assert subset_languages('hi') == ('hi', 'hi')
    assert subset_languages('hi-en') == ('hi', 'en')
    assert subset_languages('en') == ('en', 'en')
    assert subset_file('hi-en', 'test') == subset_file('hi', 'test') == 'xorqa_hi_test.json'
    for bad in ('ne', 'hi-ta', 'en-en', 'xx'):
        with pytest.raises(ValueError, match='Unknown XorQA-IN subset'):
            subset_languages(bad)
    groups = _adapter().category_map
    assert groups['hi'] == 'xorqa_in_xx' and groups['hi-en'] == 'xorqa_in_en' and groups['en'] == 'english'


def test_answer_language_selects_reference_and_prompt() -> None:
    adapter = _adapter()
    record = _record('राजधानी क्या है?', 'Delhi', 'दिल्ली।', 'hi')
    in_hindi, in_english = _sample(adapter, 'hi', record), _sample(adapter, 'hi-en', record)
    assert in_hindi.target == 'दिल्ली।' and in_hindi.metadata['answer_language'] == 'hi'
    assert in_english.target == 'Delhi' and in_english.metadata['answer_language'] == 'en'
    assert adapter.format_prompt_template(in_hindi) == (
        f'{INSTRUCTION.format("Hindi")}\n\nDelhi is the capital.\n\nQ: राजधानी क्या है?\nA:'
    )
    assert adapter.format_prompt_template(in_english).startswith(INSTRUCTION.format('English'))
    assert adapter.sample_to_fewshot(in_hindi).endswith('\nA: दिल्ली।')
    assert adapter.sample_to_fewshot(in_english).endswith('\nA: Delhi')


@pytest.mark.parametrize(
    ('subset', 'completion', 'f1'),
    [('hi', 'दिल्ली', 1.0), ('hi', 'Delhi', 0.0), ('hi-en', 'the Delhi', 1.0), ('hi-en', 'दिल्ली', 0.0)],
)
def test_each_group_is_scored_against_its_own_answers(subset: str, completion: str, f1: float) -> None:
    adapter = _adapter()
    sample = _sample(adapter, subset, _record('राजधानी क्या है?', 'Delhi', 'दिल्ली।', 'hi'))
    state = TaskState(model='mock', sample=sample, output=ModelOutput.from_content('mock', completion), completed=True)
    assert adapter.calculate_metrics(state).score.value['f1'] == f1


@pytest.mark.parametrize(('question', 'translated'), [(' ', 'दिल्ली'), ('प्रश्न?', ' ')])
def test_rows_without_question_or_answer_are_skipped(question: str, translated: str) -> None:
    assert _sample(_adapter(), 'hi', _record(question, 'Delhi', translated, 'hi')) == []


def test_groups_are_macro_averaged_separately() -> None:
    from evalscope.api.metric import AggScore

    scores = {
        'hi': [AggScore(score=0.2, metric_name='f1', aggregation='mean', num=10)],
        'bo': [AggScore(score=0.4, metric_name='f1', aggregation='mean', num=30)],
        'hi-en': [AggScore(score=0.6, metric_name='f1', aggregation='mean', num=10)],
        'en': [AggScore(score=0.9, metric_name='f1', aggregation='mean', num=10)],
    }
    report = ReportGenerator.generate_report(score_dict=scores, model_name='mock', data_adapter=_adapter())
    categories = {c.name[0]: c for c in report.metrics[0].categories}
    assert categories['xorqa_in_xx'].macro_score == pytest.approx(0.3)  # languages weigh equally
    assert categories['xorqa_in_en'].macro_score == pytest.approx(0.6)
    assert categories['english'].macro_score == pytest.approx(0.9)


def _write(directory: Path, name: str, records: list) -> None:
    (directory / name).write_text(json.dumps({'examples': records, 'canary': 'x'}), encoding='utf-8')


def test_load_local_files(tmp_path: Path) -> None:
    test_rows = [_record(f'q{i}', f'en{i}', f'hi{i}', 'hi') for i in range(3)] + [_record('', 'x', 'y', 'hi')]
    _write(tmp_path, 'xorqa_hi_test.json', test_rows)
    _write(tmp_path, 'xorqa_hi_train.json', [_record(f'tq{i}', f'ten{i}', f'thi{i}', 'hi') for i in range(3)])
    test, fewshot = _adapter(dataset_id=str(tmp_path), subset_list=['hi', 'hi-en']).load()
    assert [s.target for s in test['hi']] == ['hi0', 'hi1', 'hi2']  # the empty question is skipped
    assert [s.target for s in test['hi-en']] == ['en0', 'en1', 'en2']
    assert [s.target for s in fewshot['hi']] == ['thi0'] and [s.target for s in fewshot['hi-en']] == ['ten0']


def test_load_rejects_rows_of_another_language(tmp_path: Path) -> None:
    _write(tmp_path, 'xorqa_hi_test.json', [_record('q', 'a', 'b', 'mr')])
    with pytest.raises(ValueError, match='expected `hi`'):
        _adapter(dataset_id=str(tmp_path), subset_list=['hi'], few_shot_num=0).load()
