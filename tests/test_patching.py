"""RFC 6902 apply semantics: the correction round's patch language."""

from xtremeparse.patching import apply_patch, is_patch


def test_add_appends_and_inserts_by_pointer():
    prev = [{'name': 'a'}, {'name': 'b'}]
    out, err = apply_patch(prev, [
        {'op': 'add', 'path': '/project_experiences/2', 'value': {'name': 'c'}}],
        root='project_experiences')
    assert err is None and [e['name'] for e in out] == ['a', 'b', 'c']
    out, err = apply_patch(prev, [{'op': 'add', 'path': '/0', 'value': 9}])
    assert err is None and out[0] == 9
    out, err = apply_patch(prev, [{'op': 'add', 'path': '/-', 'value': 9}])
    assert err is None and out[-1] == 9


def test_remove_replace_nested_and_root():
    out, err = apply_patch({'x': {'y': [1, 2, 3]}}, [
        {'op': 'remove', 'path': '/x/y/1'},
        {'op': 'replace', 'path': '/x/y/1', 'value': 9}])
    assert err is None and out == {'x': {'y': [1, 9]}}
    out, err = apply_patch([1], [{'op': 'replace', 'path': '/items', 'value': [2]}],
                           root='items')
    assert err is None and out == [2]


def test_move_copies_then_drops_source():
    out, err = apply_patch([{'n': 'a'}, {'n': 'b'}], [
        {'op': 'move', 'from': '/0', 'path': '/-'}])
    assert err is None and [e['n'] for e in out] == ['b', 'a']
    out, err = apply_patch([{'n': 'a'}], [
        {'op': 'copy', 'from': '/0', 'path': '/-'}])
    assert err is None and [e['n'] for e in out] == ['a', 'a']


def test_errors_keep_the_previous_result():
    prev = [{'name': 'a'}]
    for ops in (
            [{'op': 'frobnicate', 'path': '/0'}],
            [{'op': 'remove', 'path': '/9'}],
            [{'op': 'replace', 'path': '/x/y', 'value': 1}],
            [{'op': 'add', 'path': 'no-slash', 'value': 1}]):
        out, err = apply_patch(prev, ops)
        assert out is None and err


def test_empty_patch_is_a_noop_not_a_wipe():
    prev = [{'name': 'a'}]
    out, err = apply_patch(prev, [])
    assert err is None and out == prev


def test_previous_result_is_never_mutated():
    prev = [{'name': 'a'}]
    apply_patch(prev, [{'op': 'add', 'path': '/-', 'value': {'name': 'z'}}])
    assert prev == [{'name': 'a'}]


def test_is_patch_needs_op_and_path_on_every_element():
    assert is_patch([])
    assert is_patch([{'op': 'add', 'path': '/1'}])
    assert not is_patch([{'name': 'a'}])
    assert not is_patch([{'op': 1, 'path': '/1'}])
    assert not is_patch('x') and not is_patch({'op': 'add', 'path': '/'})
