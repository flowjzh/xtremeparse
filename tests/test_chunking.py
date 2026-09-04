"""Boundary probes for the deterministic chunker."""

from xtremeparse.chunking import chunk_text


def test_empty_and_whitespace_only():
    assert chunk_text('') == []
    assert chunk_text('   \n\t \n\n') == []


def test_tiny_input_is_one_chunk():
    assert chunk_text('姓名') == ['姓名']


def test_varied_lines_keep_their_breaks():
    # OCR/converted text: a line is a layout boundary — the packer may
    # not bury one unit's band inside another unit's chunk
    text = '姓名：张三\n年龄：30\n电话：13800000000'
    assert chunk_text(text) == text.split('\n')


def test_layout_band_lines_stay_addressable():
    text = ('business negotiations\nPersonal Skills\n'
            'Language: Fluent in oral English, Mandarin Class1 Grade A\n'
            "Professional: Driver's license, driving experience for10 years\n"
            'Office: Mastered the office software such as Word,Excel and PPT')
    assert chunk_text(text) == text.split('\n')


def test_box_wrapped_lines_join_into_sentence_chunks():
    # every line a full measure — box-width wrapping: the cuts are
    # mid-sentence, so the sentence tier takes over
    line = '姓名张三，某公司后端，负责平台组。'
    text = '\n'.join([line] * 5)
    chunks = chunk_text(text, max_chars=30)
    assert ''.join(chunks) == text
    assert all(len(c) <= 30 for c in chunks)
    assert len(chunks) == 5  # five sentences, one per chunk


def test_blank_lines_separate_paragraphs():
    assert chunk_text('第一段\n\n第二段\n \n第三段') == ['第一段', '第二段', '第三段']


def test_heading_is_its_own_chunk():
    assert chunk_text('## 工作经历\n2020-2023 腾讯 后端') == [
        '## 工作经历', '2020-2023 腾讯 后端']


def test_list_items_and_numbered_items_are_individual_chunks():
    text = '经历：\n- 腾讯 后端\n• 阿里 高级专家\n1. 早期创业\n2、外企'
    assert chunk_text(text) == ['经历：', '- 腾讯 后端', '• 阿里 高级专家', '1. 早期创业', '2、外企']


def test_negative_numbers_are_not_list_items():
    # near-uniform short lines take the wrap path, where a list split
    # would wrongly fire if '-' led a list — it must not
    assert chunk_text('-10分\n-5分') == ['-10分\n-5分']


def test_line_leading_dates_are_not_list_items():
    text = '2015.3 joined 腾讯\n2018.5 went away'
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


def test_ticker_dot_never_splits_mid_token():
    text = '投资霸王茶姬(CHA. US)；投资宁德时代(03750.HK)；'
    chunks = chunk_text(text, max_chars=40)
    assert any('03750.HK' in c for c in chunks)
    assert ''.join(chunks) == text


def test_semicolon_clause_runs_split_per_entry():
    text = '职责一，做了a；职责二，做了b；职责三，做了c。总结。'
    assert chunk_text(text, max_chars=40) == [
        '职责一，做了a；', '职责二，做了b；', '职责三，做了c。总结。']


def test_lone_semicolon_clause_still_packs():
    assert chunk_text('前句；后句继续。', max_chars=40) == ['前句；后句继续。']


def test_mid_line_symbol_bullets_split_per_entry():
    bullet = chr(0x9f)
    text = f'{bullet} 条目一提供法律服务； {bullet} 条目二提供法律服务； {bullet} 条目三；'
    assert chunk_text(text, max_chars=40) == [
        f'{bullet} 条目一提供法律服务；', f' {bullet} 条目二提供法律服务；',
        f' {bullet} 条目三；']


def test_entry_over_ceiling_still_windowed():
    bullet = chr(0x9f)
    text = f'{bullet} {"长" * 300}；{bullet} 短条目；'
    chunks = chunk_text(text, max_chars=250)
    assert chunks[0] == f'{bullet} ' + '长' * 248  # windowed to the ceiling
    assert chunks[-1] == f'{bullet} 短条目；'


def test_unpunctuated_line_falls_to_character_windows():
    assert chunk_text('密' * 1300, max_chars=600) == ['密' * 600, '密' * 600, '密' * 100]


def test_dense_line_windows_within_itself_and_heading_stays_alone():
    text = f'## 经历\n{"密" * 700}\n\n## 其他\n简短'
    chunks = chunk_text(text, max_chars=600)
    assert chunks[0] == '## 经历'
    assert chunks[1] == '密' * 600 and chunks[2] == '密' * 100
    assert chunks[3] == '## 其他' and chunks[4] == '简短'


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
