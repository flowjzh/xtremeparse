"""The shared schema reader: type shapes, options, walks, heading labels."""

from xtremeparse.schema import (format_of, heading_label, options_of,
                                type_set, value_branches, walk_fields)

SCHEMA = {
    'type': 'object',
    'properties': {
        'name': {'type': 'string', 'x-verbatim': 0.9},
        'education': {
            'type': 'array',
            'items': {'type': 'object', 'properties': {
                'school': {'type': 'string'},
                'details': {'type': 'array', 'items': {'type': 'object',
                            'properties': {'note': {'type': 'string'}}}}}}},
        'choice': {'anyOf': [
            {'type': 'object', 'properties': {'a': {'type': 'string'}}},
            {'type': 'object', 'properties': {'b': {'type': 'string'}}}]},
    },
}


def test_type_set_reads_one_name_or_a_list():
    assert type_set({'type': 'string'}) == {'string'}
    assert type_set({'type': ['object', 'null']}) == {'object', 'null'}


def test_value_branches_unwrap_anyof_only():
    plain = {'type': 'string'}
    assert value_branches(plain) == [plain]
    branches = [{'type': 'string'}, {'enum': ['']}]
    assert value_branches({'anyOf': branches}) == branches


def test_options_of_collects_enums_formats_and_skips_the_sentinel():
    prop = {'anyOf': [
        {'type': 'string', 'format': 'date-month'},
        {'enum': ['至今']},
        {'anyOf': [{'enum': ['', '至今', '在职']}]},
    ]}
    assert options_of(prop) == ['date-month', '至今', '在职']


def test_options_of_prefers_first_seen_order():
    prop = {'enum': ['A', 'B'], 'anyOf': [{'enum': ['B', 'C']}]}
    assert options_of(prop) == ['A', 'B', 'C']


def test_format_of_reads_only_the_declared_shape():
    assert format_of({'type': 'string', 'format': 'date-year'}) == 'date-year'
    assert format_of({'type': 'string'}) is None


def test_walk_fields_is_preorder_through_properties_items_and_branches():
    assert [(path, prop.get('type')) for path, prop in walk_fields(SCHEMA)] == [
        ('name', 'string'),
        ('education', 'array'),
        ('education.school', 'string'),
        ('education.details', 'array'),
        ('education.details.note', 'string'),
        ('choice', None),
        ('choice.a', 'string'),
        ('choice.b', 'string'),
    ]


def test_walk_fields_preserves_extension_keywords():
    _, prop = next(walk_fields(SCHEMA))
    assert prop['x-verbatim'] == 0.9
    assert options_of(prop) == []


def test_heading_label_splits_at_either_colon():
    assert heading_label({'type': 'string',
                          'description': '汇报对象：汇报线'}) == '汇报对象'
    assert heading_label({'type': 'string',
                          'description': 'Report to: manager'}) == 'Report to'


def test_heading_label_falls_back_to_the_whole_description():
    assert heading_label({'type': 'string',
                          'description': ' 公司名照抄原文 '}) == '公司名照抄原文'
    assert heading_label({'type': 'string'}) == ''


def test_heading_label_is_string_fields_only():
    assert heading_label({'type': 'object', 'description': 'x：y'}) is None
    assert heading_label({'type': ['string', 'null'],
                          'description': 'x：y'}) is None
