import pytest
from app.agents.scraper import scrape_url

def test_scrape_url_blocks_localhost():
    # Localhost should be blocked by SSRF checks
    res = scrape_url("http://localhost:8000/api/admin")
    assert res is None

def test_scrape_url_blocks_loopback_ip():
    # Loopback IP should be blocked by SSRF checks
    res = scrape_url("http://127.0.0.1:8000/api/admin")
    assert res is None

def test_scrape_url_blocks_private_ip():
    # Private network IP should be blocked by SSRF checks
    res = scrape_url("http://192.168.1.1/index.html")
    assert res is None

def test_scrape_url_blocks_link_local_ip():
    # Link local IP (cloud metadata endpoints) should be blocked
    res = scrape_url("http://169.254.169.254/latest/meta-data/")
    assert res is None

def test_scrape_url_blocks_blocked_domains():
    # Social media domains listed in BLOCKED_DOMAINS should be blocked
    res = scrape_url("https://facebook.com/someprofile")
    assert res is None

def test_scrape_url_blocks_ipv6_loopback():
    # IPv6 loopback should be blocked
    res = scrape_url("http://[::1]/")
    assert res is None
