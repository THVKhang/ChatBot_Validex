import pytest
from bs4 import BeautifulSoup
from app.collect_au_sources import (
    _table_to_markdown,
    _simhash,
    _hamming_distance,
    _extract_text_from_html,
)

def test_table_to_markdown_basic():
    html = """
    <table>
        <tr>
            <th>Name</th>
            <th>Age</th>
        </tr>
        <tr>
            <td>Alice</td>
            <td>30</td>
        </tr>
        <tr>
            <td>Bob</td>
            <td>25</td>
        </tr>
    </table>
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    md = _table_to_markdown(table)
    
    assert "| Name | Age |" in md
    assert "| --- | --- |" in md
    assert "| Alice | 30 |" in md
    assert "| Bob | 25 |" in md

def test_simhash_near_duplicates():
    text1 = "Australian National Police Check process and requirements for NDIS worker screening."
    text2 = "Australian National Police Check process and requirements for NDIS worker screening. (Copyright 2026)"
    text3 = "Completely different text about immigration visa check rules in state government offices."

    h1 = _simhash(text1)
    h2 = _simhash(text2)
    h3 = _simhash(text3)

    # Near-duplicates should have small Hamming distance
    dist12 = _hamming_distance(h1, h2)
    assert dist12 <= 6  # Allow up to 6 bits difference for slightly modified templates

    # Completely different texts should have large Hamming distance
    dist13 = _hamming_distance(h1, h3)
    assert dist13 > 6

def test_extract_text_from_html_preserves_tables():
    html = """
    <main>
        <h2>Fees Section. Check out the fees below.</h2>
        <p>Please review the pricing below. It contains the updated fees for standard options.</p>
        <table>
            <tr><th>Service</th><th>Fee</th></tr>
            <tr><td>Police Check</td><td>$45</td></tr>
        </table>
        <p>This is a second paragraph. It explains the payment methods.</p>
    </main>
    """
    text, rejected = _extract_text_from_html(html)
    assert "Fees Section" in text
    assert "| Service | Fee |" in text
    assert "| Police Check | $45 |" in text
    assert "This is a second paragraph" in text
