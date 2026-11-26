"""XQuAD-IN: Token-F1 normalization for Indic scripts, multiple references, the paper's prompt and loading."""

import json
from pathlib import Path

import pytest

from evalscope.api.benchmark import DataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.indicgenbench.utils import normalize_answer, token_f1
from evalscope.benchmarks.indicgenbench.xquad_in_adapter import SUBSET_LIST, subset_file
from evalscope.config import TaskConfig


def _adapter(**dataset_args) -> DataAdapter:
    args = {'xquad_in': dataset_args} if dataset_args else {}
    return get_benchmark('xquad_in', TaskConfig(model='mock', datasets=['xquad_in'], dataset_args=args))


def _record(context: str, question: str, answers: list, lang: str) -> dict:
    return {
        'id': 'q',
        'title': '',
        'context': context,
        'question': question,
        'answers': [{'answer_start': -1, 'text': a} for a in answers],
        'lang': lang,
    }


@pytest.mark.parametrize(
    ('text', 'language', 'tokens'),
    [
        ('The Eiffel Tower!', 'en', ['eiffel', 'tower']),  # SQuAD v1.1 on ASCII text
        ('a, an and the', 'en', ['and']),
        ('पोर्टलैंड।', 'hi', ['पोर्टलैंड']),  # Devanagari danda
        ('दिल्ली, भारत॥', 'hi', ['दिल्ली', 'भारत']),  # double danda
        ('لاہور، پاکستان۔', 'ur', ['لاہور', 'پاکستان']),  # Arabic comma and full stop
        ('པོ་ལན་ལན་།', 'bo', ['པོལནལན']),  # Tibetan tsheg and shad
        ('The Hindi द', 'hi', ['the', 'hindi', 'द']),  # English articles are only removed from English answers
    ],
)
def test_normalize_answer(text: str, language: str, tokens: list) -> None:
    assert normalize_answer(text, language) == tokens


@pytest.mark.parametrize(
    ('prediction', 'references', 'language', 'f1'),
    [
        ('पोर्टलैंड', 'पोर्टलैंड।', 'hi', 1.0),  # a trailing danda no longer costs the whole answer
        ('नई दिल्ली', 'दिल्ली', 'hi', 2 * 0.5 * 1.0 / 1.5),
        ('Paris', ['London', 'paris.'], 'en', 1.0),  # the best reference counts
        ('x b', ['y', 'b c'], 'en', 0.5),
        ('', 'दिल्ली', 'hi', 0.0),
        ('।', 'दिल्ली', 'hi', 0.0),  # punctuation only normalizes to nothing
    ],
)
def test_token_f1(prediction: str, references, language: str, f1: float) -> None:
    assert token_f1(prediction, references, language) == pytest.approx(f1)


def test_subsets() -> None:
    assert len(SUBSET_LIST) == 13 and SUBSET_LIST[0] == 'en'
    assert subset_file('hi', 'test') == 'xquad_hi_test.json'
    with pytest.raises(ValueError, match='Unknown XQuAD-IN subset'):
        subset_file('sat', 'test')
    adapter = _adapter()
    assert adapter.few_shot_num == 1 and adapter.train_split == 'train'
    assert adapter.category_map['en'] == 'english'
    assert {adapter.category_map[s] for s in SUBSET_LIST[1:]} == {'indic'}


def test_prompts() -> None:
    adapter = _adapter()
    example = adapter.record_to_sample(_record('संदर्भ एक।', 'प्रश्न एक?', [' उत्तर '], 'hi'))
    test = adapter.record_to_sample(_record('संदर्भ दो।', 'प्रश्न दो?', ['जवाब'], 'hi'))
    prompt = 'संदर्भ दो।\n\nQ: प्रश्न दो?\nA:'
    assert adapter.format_prompt_template(test) == prompt
    fewshot = adapter.sample_to_fewshot(example)
    assert fewshot == 'संदर्भ एक।\n\nQ: प्रश्न एक?\nA: उत्तर'
    assert adapter.format_fewshot_template(fewshot, test) == f'{fewshot}\n\n{prompt}'


def test_multiple_references_are_scored_separately() -> None:
    adapter = _adapter()
    sample = adapter.record_to_sample(_record('c', 'q?', ['London', 'Paris'], 'en'))
    state = TaskState(model='mock', sample=sample, output=ModelOutput.from_content('mock', 'paris\nmore'), completed=True)
    assert adapter.calculate_metrics(state).score.value['f1'] == 1.0


def _write(directory: Path, name: str, records: list) -> None:
    (directory / name).write_text(json.dumps({'examples': records, 'canary': 'x'}), encoding='utf-8')


def test_load_local_files(tmp_path: Path) -> None:
    _write(tmp_path, 'xquad_hi_test.json', [_record(f'c{i}', f'q{i}', [f'a{i}'], 'hi') for i in range(3)])
    _write(tmp_path, 'xquad_hi_train.json', [_record(f't{i}', f'tq{i}', [f'ta{i}'], 'hi') for i in range(3)])
    test, fewshot = _adapter(dataset_id=str(tmp_path), subset_list=['hi']).load()
    assert [s.input for s in test['hi']] == ['q0', 'q1', 'q2']
    assert [s.input for s in fewshot['hi']] == ['tq0']  # the first `train` row


@pytest.mark.parametrize(
    ('record', 'message'),
    [(_record('c', 'q', ['a'], 'ta'), 'expected `hi`'), (_record('c', 'q', [' '], 'hi'), 'without a reference')],
)
def test_load_rejects_bad_rows(tmp_path: Path, record: dict, message: str) -> None:
    _write(tmp_path, 'xquad_hi_test.json', [record])
    with pytest.raises(ValueError, match=message):
        _adapter(dataset_id=str(tmp_path), subset_list=['hi'], few_shot_num=0).load()
