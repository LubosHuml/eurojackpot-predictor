import requests
from bs4 import BeautifulSoup
import re
import time
from datetime import datetime
import database

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def scrape_year(year):
    """
    Scrapes Eurojackpot draws for a specific year from euro-jackpot.net.
    Returns a list of tuples: (date, main_nums, euro_nums)
    """
    url = f"https://www.euro-jackpot.net/en/results-archive-{year}"
    print(f"Scraping archive for year {year} from {url}...")
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        if response.status_code != 200:
            print(f"Warning: Received status code {response.status_code} for year {year}")
            return []
            
        soup = BeautifulSoup(response.text, "html.parser")
        draws = []
        for tr in soup.find_all("tr"):
            link = tr.find("a", href=re.compile(r"/results/\d{2}-\d{2}-\d{4}"))
            if not link:
                continue
                
            href = link.get("href")
            date_match = re.search(r"(\d{2})-(\d{2})-(\d{4})", href)
            if not date_match:
                continue
                
            day, month, year_str = date_match.groups()
            db_date = f"{year_str}-{month}-{day}"
            
            balls_ul = tr.find("ul", class_="balls small")
            if not balls_ul:
                continue
                
            main_balls = balls_ul.find_all("li", class_="ball")
            euro_balls = balls_ul.find_all("li", class_="euro")
            
            if len(main_balls) != 5 or len(euro_balls) != 2:
                continue
                
            try:
                main_nums = sorted([int(b.text.strip()) for b in main_balls])
                euro_nums = sorted([int(e.text.strip()) for e in euro_balls])
                draws.append((db_date, main_nums, euro_nums))
            except ValueError:
                continue
                
        # Sort draws ascending by date
        draws.sort(key=lambda x: x[0])
        print(f"Scraped {len(draws)} draws for year {year}.")
        return draws
    except Exception as e:
        print(f"Error scraping year {year}: {e}")
        return []

def fetch_latest_draw():
    """
    Fetches the latest draw from Lottoland's JSON API.
    Returns: (date, main_nums, euro_nums) or None
    """
    url = "https://media.lottoland.com/api/drawings/euroJackpot"
    print(f"Fetching latest draw from Lottoland API: {url}...")
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        if response.status_code != 200:
            print(f"Warning: Received status code {response.status_code} from Lottoland API")
            return None
            
        data = response.json()
        last = data.get("last")
        if not last:
            return None
            
        date_info = last.get("date", {})
        year = date_info.get("year")
        month = date_info.get("month")
        day = date_info.get("day")
        
        if not (year and month and day):
            return None
            
        db_date = f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
        
        main_nums = sorted([int(n) for n in last.get("numbers", [])])
        euro_nums = sorted([int(n) for n in last.get("euroNumbers", [])])
        
        if len(main_nums) == 5 and len(euro_nums) == 2:
            print(f"Fetched latest draw: Date={db_date}, Main={main_nums}, Euro={euro_nums}")
            return db_date, main_nums, euro_nums
        else:
            print("Warning: Fetched latest draw numbers do not match Eurojackpot format.")
            return None
    except Exception as e:
        print(f"Error fetching latest draw: {e}")
        return None

def update_database(force_all=False):
    """
    Scrapes and updates the database with new draws.
    If force_all is True, scrapes all years from 2012 to current year.
    Otherwise, checks the latest draw date and decides which years to scrape.
    """
    database.init_db()
    
    current_year = datetime.now().year
    latest_db_date = database.get_latest_draw_date()
    
    start_year = 2012
    if latest_db_date and not force_all:
        # We can start scraping from the year of the latest draw in the database
        db_year = int(latest_db_date.split("-")[0])
        start_year = max(2012, db_year)
        print(f"Latest draw in DB is from {latest_db_date}. Scraping starting from year {start_year}.")
    else:
        print(f"No existing data or force_all set. Scraping all history starting from {start_year}.")
        
    inserted_count = 0
    # Scrape historical years
    for y in range(start_year, current_year + 1):
        draws = scrape_year(y)
        for date, main_nums, euro_nums in draws:
            if database.insert_draw(date, main_nums, euro_nums):
                inserted_count += 1
        # Polite scraping delay
        time.sleep(1.0)
        
    # Also check the Lottoland API for incremental update
    latest = fetch_latest_draw()
    if latest:
        date, main_nums, euro_nums = latest
        if database.insert_draw(date, main_nums, euro_nums):
            inserted_count += 1
            
    print(f"Database update complete. Inserted {inserted_count} new draws.")
    return inserted_count

if __name__ == "__main__":
    # Test execution
    update_database(force_all=False)
