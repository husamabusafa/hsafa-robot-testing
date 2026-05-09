#!/usr/bin/env python3
"""Fetch all KSU faculty data and store in PostgreSQL."""
import requests
import json
import time
import sys
from bs4 import BeautifulSoup
import psycopg2
import os

API_URL = "https://faculty.ksu.edu.sa/en/views/ajax"
HEADERS = {
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://faculty.ksu.edu.sa/en/faculty",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
}

DB_NAME = "ksu_faculty"
DB_USER = os.environ.get("USER", "postgres")


def get_connection():
    return psycopg2.connect(dbname=DB_NAME, user=DB_USER)


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS faculty (
            id SERIAL PRIMARY KEY,
            name TEXT,
            academic_degree TEXT,
            job_title TEXT,
            email TEXT,
            phone TEXT,
            profile_url TEXT,
            image_url TEXT,
            raw_html TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS fetch_progress (
            key TEXT PRIMARY KEY,
            value INT DEFAULT 0
        )
    """)
    conn.commit()
    cur.close()
    conn.close()


def get_last_page():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT value FROM fetch_progress WHERE key = 'last_page'")
    row = cur.fetchone()
    if row:
        return row[0]
    # Determine total pages from page 0
    params = {
        "_wrapper_format": "drupal_ajax",
        "view_name": "faculty_websites_list",
        "view_display_id": "professor_list",
        "view_args": "",
        "view_path": "/faculty",
        "view_base_path": "faculty",
        "view_dom_id": "844bdb9659dfaf69104ceab0f9accc85f1b3f90f8fbeb3fc302ed6d0d0bd0bf6",
        "pager_element": "0",
        "page": "0",
        "_drupal_ajax": "1",
        "ajax_page_state[theme]": "fac",
        "ajax_page_state[theme_token]": "",
        "ajax_page_state[libraries]": "eJxVikkOgCAMAD9E5OR7mgJVMcUaWrffG_FgvM1MJm5qUmDA6EeWgPyg-7ThZIV7oFpFm--ZDoWUdWW8QFbLsigMmY2q00uNig-o5Nr47h3OeP5CkbQx3aV7MiQ",
    }
    resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    for entry in data:
        if entry.get("command") == "insert":
            soup = BeautifulSoup(entry.get("data", ""), "html.parser")
            last_link = soup.select_one('a[title="Go to last page"]')
            if last_link and "page=" in (last_link.get("href") or ""):
                total_pages = int(last_link["href"].split("page=")[1].split("&")[0]) + 1
                cur.execute("INSERT INTO fetch_progress (key, value) VALUES ('last_page', %s) ON CONFLICT (key) DO NOTHING", (total_pages,))
                conn.commit()
                cur.close()
                conn.close()
                return total_pages
    cur.close()
    conn.close()
    return 440  # fallback


def parse_page(html):
    soup = BeautifulSoup(html, "html.parser")
    results = []
    for row in soup.select(".professor.views-row"):
        name_el = row.select_one(".professor_name a")
        degree_el = row.select_one(".professor_academic_degree")
        job_el = row.select_one(".professor_job_title")
        mail_el = row.select_one(".professor_mail")
        phone_el = row.select_one(".professor_phone")
        img_el = row.select_one(".professor_image img")

        name = name_el.get_text(strip=True) if name_el else None
        profile_path = name_el["href"] if name_el else None
        profile_url = f"https://faculty.ksu.edu.sa{profile_path}" if profile_path and profile_path.startswith("/") else profile_path

        academic_degree = degree_el.get_text(strip=True) if degree_el else None
        job_title = job_el.get_text(strip=True) if job_el else None
        email = mail_el.get_text(strip=True) if mail_el else None
        phone = phone_el.get_text(strip=True) if phone_el else None
        image_path = img_el["src"] if img_el and img_el.has_attr("src") else None
        image_url = f"https://faculty.ksu.edu.sa{image_path}" if image_path and image_path.startswith("/") else image_path

        results.append({
            "name": name,
            "academic_degree": academic_degree,
            "job_title": job_title,
            "email": email,
            "phone": phone,
            "profile_url": profile_url,
            "image_url": image_url,
            "raw_html": str(row),
        })
    return results


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
    data = resp.json()
    for entry in data:
        if entry.get("command") == "insert":
            return entry.get("data", "")
    return ""


def store_batch(conn, items):
    cur = conn.cursor()
    cur.executemany("""
        INSERT INTO faculty (name, academic_degree, job_title, email, phone, profile_url, image_url, raw_html)
        VALUES (%(name)s, %(academic_degree)s, %(job_title)s, %(email)s, %(phone)s, %(profile_url)s, %(image_url)s, %(raw_html)s)
    """, items)
    conn.commit()
    cur.close()


def get_progress():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT value FROM fetch_progress WHERE key = 'current_page'")
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row[0] if row else 0


def set_progress(page_num):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO fetch_progress (key, value) VALUES ('current_page', %s)
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
    """, (page_num,))
    conn.commit()
    cur.close()
    conn.close()


def main():
    init_db()
    total_pages = get_last_page()
    start_page = get_progress()
    print(f"Total pages: {total_pages}, starting from page {start_page}")

    conn = get_connection()
    for page in range(start_page, total_pages):
        try:
            html = fetch_page(page)
            items = parse_page(html)
            if items:
                store_batch(conn, items)
            set_progress(page + 1)
            if (page + 1) % 10 == 0 or page == 0:
                print(f"Fetched page {page + 1}/{total_pages} ({len(items)} items)")
            time.sleep(0.3)
        except Exception as e:
            print(f"Error on page {page}: {e}")
            time.sleep(1)
            # Retry once
            try:
                html = fetch_page(page)
                items = parse_page(html)
                if items:
                    store_batch(conn, items)
                set_progress(page + 1)
            except Exception as e2:
                print(f"Retry failed on page {page}: {e2}")
                break
    conn.close()
    print("Done fetching all faculty data.")


if __name__ == "__main__":
    main()
