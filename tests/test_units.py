"""Boundary probes for schema decomposition and semantic cards."""

from xtremeparse.units import MISC, decompose

GOLDEN = {
    'type': 'object',
    'properties': {
        'basic_info': {
            'type': 'object',
            'description': '候选人基本信息',
            'properties': {
                'name': {'type': 'string', 'description': '候选人的姓名'},
                'employment_status': {'type': 'string',
                                      'description': '当前在职状态 (在职=employed; 离职=unemployed)'},
                'start_work_date': {'type': 'string', 'format': 'date', 'description': '开始工作日期'},
                'is_married_with_children': {'type': 'boolean', 'description': '是否已婚已育'},
            },
        },
        'career': {
            'type': 'object',
            'description': '职业经历',
            'properties': {
                'work_experiences': {
                    'type': 'array',
                    'description': '工作经历（多段重复）',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'start_date': {'type': 'string', 'description': '工作开始时间'},
                            'company': {'type': 'string', 'description': '公司'},
                        },
                    },
                },
            },
        },
        'creation_time': {'type': 'string', 'format': 'date-time', 'description': '简历生成的时间'},
    },
}


def test_golden_shape_decomposes_to_three_units():
    units = decompose(GOLDEN)
    assert [(u.path, u.kind) for u in units] == [
        ('basic_info', 'object'),
        ('career.work_experiences', 'array'),
        (MISC, 'scalar'),
    ]


def test_array_unit_sub_schema_is_the_item_schema():
    unit = next(u for u in decompose(GOLDEN) if u.kind == 'array')
    assert unit.sub_schema['properties']['company']['description'] == '公司'


def test_cards_carry_the_description_chain():
    cards = {u.path: u.card for u in decompose(GOLDEN)}
    assert '[basic_info | object] 候选人基本信息' in cards['basic_info']
    assert '候选人的姓名' in cards['basic_info']
    assert '(在职=employed; 离职=unemployed)' in cards['basic_info']
    assert '[career.work_experiences | array] 工作经历（多段重复）' in cards['career.work_experiences']
    assert 'per item:' in cards['career.work_experiences']
    assert '(date)' in cards['basic_info'] and '(date-time)' in cards[MISC]
    assert '(boolean)' in cards['basic_info']


def test_bare_schema_degrades_to_name_only_cards():
    units = decompose({'type': 'object', 'properties': {
        'a': {'type': 'string'},
        'b': {'type': 'object', 'properties': {'x': {'type': 'string'}}},
    }})
    assert [u.path for u in units] == ['b', MISC]
    assert 'x' in units[0].card and '[b | object]' in units[0].card
    assert units[1].card == f'[{MISC} | scalar] a'


def test_empty_and_propertyless_schemas_yield_nothing():
    assert decompose({}) == []
    assert decompose({'type': 'object'}) == []
    assert decompose({'type': 'object', 'properties': {}}) == []


def test_required_naming_an_array_child_is_filtered_from_object_unit():
    schema = {'type': 'object', 'properties': {
        'career': {'type': 'object', 'required': ['work', 'summary'],
                   'properties': {
                       'summary': {'type': 'string'},
                       'work': {'type': 'array', 'items': {'type': 'object'}},
                   }},
    }}
    unit = next(u for u in decompose(schema) if u.kind == 'object')
    assert unit.sub_schema['required'] == ['summary']


def test_toplevel_array_is_an_array_unit():
    units = decompose({'type': 'object', 'properties': {
        'tags': {'type': 'array', 'items': {'type': 'string'}},
    }})
    assert [(u.path, u.kind) for u in units] == [('tags', 'array')]
    assert units[0].sub_schema == {'type': 'string'}  # scalar items: no per-item line
    assert 'per item' not in units[0].card


NESTED = {'type': 'object', 'properties': {
    'work': {'type': 'array', 'items': {'type': 'object', 'properties': {
        'company': {'type': 'string'},
        'roles': {'type': 'array', 'items': {'type': 'object', 'properties': {
            'title': {'type': 'string'},
        }}},
    }}},
}}


def test_array_under_array_lifts_to_its_own_unit():
    units = decompose(NESTED)
    assert [(u.path, u.parent) for u in units] == [
        ('work', None),
        ('work.roles', 'work'),
    ]


def test_lifted_field_is_stripped_from_the_parent_item_schema():
    parent = decompose(NESTED)[0]
    assert 'roles' not in parent.sub_schema['properties']


def test_nested_unit_card_names_the_parent_path():
    child = decompose(NESTED)[1]
    assert '[work.roles | array]' in child.card
    assert child.field == 'roles'


def test_required_naming_a_lifted_child_is_filtered_from_parent():
    schema = {'type': 'object', 'properties': {
        'work': {'type': 'array',
                 'items': {'type': 'object', 'required': ['company', 'roles'],
                           'properties': {
                               'company': {'type': 'string'},
                               'roles': {'type': 'array',
                                         'items': {'type': 'object'}},
                           }}},
    }}
    assert decompose(schema)[0].sub_schema['required'] == ['company']


def test_deeper_nesting_stays_inside_parent_sub_schema():
    schema = {'type': 'object', 'properties': {
        'work': {'type': 'array', 'items': {'type': 'object', 'properties': {
            'company': {'type': 'string'},
            'roles': {'type': 'array', 'items': {'type': 'object', 'properties': {
                'projects': {'type': 'array', 'items': {'type': 'object'}},
            }}},
        }}},
    }}
    units = decompose(schema)
    assert [u.path for u in units] == ['work', 'work.roles']  # one lift only
    assert 'projects' in units[1].sub_schema['properties']


def test_object_with_arrays_and_scalars_produces_both_units():
    units = decompose({'type': 'object', 'properties': {
        'profile': {'type': 'object', 'properties': {
            'summary': {'type': 'string'},
            'skills': {'type': 'array', 'items': {'type': 'string'}},
        }},
    }})
    assert [(u.path, u.kind) for u in units] == [('profile', 'object'), ('profile.skills', 'array')]


def test_misc_sub_schema_wraps_its_fields():
    unit = next(u for u in decompose(GOLDEN) if u.kind == 'scalar')
    assert unit.sub_schema == {'type': 'object', 'properties': {
        'creation_time': GOLDEN['properties']['creation_time'],
    }}


def test_deterministic():
    assert decompose(GOLDEN) == decompose(GOLDEN)


def test_nullable_section_decomposes_like_a_plain_object():
    # docxcast encodes an optional section as a type list — the unit
    # walker must see through it, not file it under $misc
    schema = {'type': 'object', 'properties': {
        'info': {'type': ['object', 'null'], 'description': '基本信息', 'properties': {
            'name': {'type': 'string', 'description': '姓名'},
        }},
    }}
    assert [u.path for u in decompose(schema)] == ['info']
