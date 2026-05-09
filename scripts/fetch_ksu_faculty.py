#!/usr/bin/env python3
"""Fetch all KSU faculty data and store in PostgreSQL."""
import requests
import json
import re
from bs4 import BeautifulSoup

API_URL = "https://faculty.ksu.edu.sa/en/views/ajax"
HEADERS = {
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://faculty.ksu.edu.sa/en/faculty",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
}

def fetch_page(page_num):
    params = {
        "_wrapper_format": "drupal_ajax",
        "view_name": "faculty_websites_list",
        "view_display_id": "professor_list",
        "view_args": "",
        "view_path": "/faculty",
        "view_base_path": "faculty",
        "view_dom_id": "844bdb9659dfaf69104ceab0f9accc85f1b3f90f8fbeb3fc302ed6d0d0bd0bf6",
        "pager_element": "0",
        "page": str(page_num),
        "_drupal_ajax": "1",
        "ajax_page_state[theme]": "fac",
        "ajax_page_state[theme_token]": "",
        "ajax_page_state[libraries]": "eJxVikkOgCAMAD9E5OR7mgJVMcUaWrffG_FgvM1MJm5qUmDA6EeWgPyg-7ThZIV7oFpFm--ZDoWUdWW8QFbLsigMmY2q00uNig-o5Nr47h3OeP5CkbQx3aV7MiQ",
    }
    resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()

def parse_faculty_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    results = []
    for card in soup.select(".views-row"):
        link = card.select_one("a[href]")
        name_el = card.select_one(".views-field-title")
        dept_el = card.select_one(".views-field-field-department")
        col_el = card.select_one(".views-field-field-college")
        img_el = card.select_one("img")

        name = name_el.get_text(strip=True) if name_el else None
        department = dept_el.get_text(strip=True) if dept_el else None
        college = col_el.get_text(strip=True) if col_el else None
        profile_url = link["href"] if link else None
        if profile_url and profile_url.startswith("/"):
            profile_url = f"https://faculty.ksu.edu.sa{profile_url}"
        image_url = img_el["src"] if img_el and img_el.has_attr("src") else None
        if image_url and image_url.startswith("/"):
            image_url = f"https://faculty.ksu.edu.sa{image_url}"

        results.append({
            "name": name,
            "department": department,
            "college": college,
            "profile_url": profile_url,
            "image_url": image_url,
        })
    return results

if __name__ == "__main__":
    # Test page 0
    data = fetch_page(0)
    print(f"Got {len(data)} command entries for page 0")
    for i, entry in enumerate(data):
        print(f"  [{i}] command={entry.get('command')} selector={entry.get('selector')}")
        if entry.get("command") == "insert":
            html = entry.get("data", "")
            faculty = parse_faculty_from_html(html)
            print(f"    Parsed {len(faculty)} faculty members")
            for f in faculty[:3]:
                print(f"      {f}")
            break
