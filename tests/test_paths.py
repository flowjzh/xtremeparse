"""xtremeparse.paths.resolve: dotted lookup with list-index hops."""

from xtremeparse.paths import resolve

DATA = {
    'name': '张三',
    'jobs': [{'company': 'Nike', 'sub': [{'title': 'A'}, {'title': 'B'}]},
             {'company': 'VSPN'}],
}


def test_dotted_path_reads_nested_dicts():
    assert resolve(DATA, 'name') == '张三'
    assert resolve(DATA, 'jobs[0].company') == 'Nike'


def test_index_hops_into_lists():
    assert resolve(DATA, 'jobs[0].sub[1].title') == 'B'
    assert resolve(DATA, 'jobs[1].company') == 'VSPN'


def test_absent_or_crossing_hops_return_none():
    assert resolve(DATA, 'nowhere.field') is None
    assert resolve(DATA, 'name.title') is None  # crosses a string leaf
    assert resolve(DATA, 'jobs[5].company') is None  # index out of range
    assert resolve(DATA, 'jobs[0].sub[2]') is None
    assert resolve(DATA, 'jobs[-1].company') is None  # bare key, not an index
