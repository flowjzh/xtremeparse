"""Boundary probes for the deterministic chunker."""

from xtremeparse.chunking import chunk_text


def test_empty_and_whitespace_only():
    assert chunk_text('') == []
    assert chunk_text('   \n\t \n\n') == []


def test_tiny_input_is_one_chunk():
    assert chunk_text('姓名') == ['姓名']


def test_plain_hard_wrapped_lines_form_one_paragraph_chunk():
    text = '姓名：张三\n年龄：30\n电话：13800000000'
    assert chunk_text(text) == [text]


def test_blank_lines_separate_paragraphs():
    assert chunk_text('第一段\n\n第二段\n \n第三段') == ['第一段', '第二段', '第三段']


def test_heading_opens_block_absorbing_following_plain_lines():
    text = '## 工作经历\n2020-2023 腾讯 后端'
    assert chunk_text(text) == [text]


def test_list_items_and_numbered_items_are_individual_chunks():
    text = '经历：\n- 腾讯 后端\n• 阿里 高级专家\n1. 早期创业\n2、外企'
    assert chunk_text(text) == ['经历：', '- 腾讯 后端', '• 阿里 高级专家', '1. 早期创业', '2、外企']


def test_negative_numbers_are_not_list_items():
    assert chunk_text('-10分\n正常行') == ['-10分\n正常行']


def test_line_leading_dates_are_not_list_items():
    text = '2015.3 joined 腾讯\n2018.5 left'
    assert chunk_text(text) == [text]


def test_table_rows_are_individual_chunks():
    text = '|公司|职位|\n|---|---|\n|腾讯|后端|'
    assert chunk_text(text) == ['|公司|职位|', '|---|---|', '|腾讯|后端|']


def test_sentences_split_and_pack_to_ceiling():
    text = '第一句。' * 300  # 300 four-char sentences, 1200 chars
    chunks = chunk_text(text, max_chars=200)
    assert all(len(c) <= 200 for c in chunks)
    assert len(chunks) == -(-len(text) // 200)  # minimal packing, no slack
    assert ''.join(chunks) == text


def test_fullwidth_full_stop_is_a_sentence_ender():
    chunks = chunk_text('第一段．第二段．', max_chars=6)
    assert chunks == ['第一段．', '第二段．']


def test_digit_dots_never_split_but_sentence_dots_do():
    chunks = chunk_text('Joined 2015.3. Left 2018.5.', max_chars=15)
    assert chunks == ['Joined 2015.3.', ' Left 2018.5.']


def test_unpunctuated_line_falls_to_character_windows():
    assert chunk_text('密' * 1300, max_chars=600) == ['密' * 600, '密' * 600, '密' * 100]


def test_dense_block_inside_markdown_uses_windows_but_keeps_heading():
    text = f'## 经历\n{"密" * 700}\n\n## 其他\n简短'
    chunks = chunk_text(text, max_chars=600)
    assert ''.join(chunks[:2]) == f'## 经历\n{"密" * 700}'  # dense block windowed whole
    assert chunks[0].startswith('## 经历\n')
    assert chunks[2] == '## 其他\n简短'


def test_crlf_is_normalized():
    assert chunk_text('第一段\r\n\r\n第二段') == ['第一段', '第二段']


def test_deterministic_and_ordered_substrings():
    text = '# 简历\n张三，30岁\n\n## 经历\n- 腾讯 后端\n\n自我介绍。很长。'
    first, second = chunk_text(text), chunk_text(text)
    assert first == second
    pos = 0
    for chunk in first:
        i = text.find(chunk, pos)
        assert i >= 0, 'chunks must be ordered substrings'
        pos = i + len(chunk)
