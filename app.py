import streamlit as st
from bs4 import BeautifulSoup
import json
import requests 
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException, TimeoutException
import time
from datetime import datetime
import traceback
import io
from PIL import Image
import asyncio
import aiohttp
import urllib.parse
import os
import random # Added import
from google import genai
from urllib.parse import urlparse, parse_qs, unquote # Import necessary functions
from langdetect import detect
import numpy as np
from tokencost import count_string_tokens
import imagehash


st.set_page_config(layout="wide",page_title= "FB Scrape", page_icon="🚀")

# --- Gemini Import and Configuration ---
try:
    # Get API keys from secrets - assumes it's a comma-separated string or a single key
    api_keys_str = st.secrets.get("GEMINI_API_KEY", "")
    if api_keys_str:
        GEMINI_API_KEYS = api_keys_str
      

    else:
        st.warning("GEMINI_API_KEY not found in Streamlit secrets. Gemini functionality will be disabled.", icon="⚠️")
        GEMINI_API_KEYS = None


except Exception as e:
    st.error(f"Error initializing Gemini configuration: {e}")
    GEMINI_API_KEYS = None


# --- Gemini Function ---
def gemini_text_lib(prompt, model='gemini-2.5-pro-exp-03-25',max_retries=5): # Using a stable model  
    tries = 0
    while tries < max_retries:
        
        st.text(f"Gemini working.. {model} trial {tries+1}")
        """ Calls Gemini API, handling potential list of keys """
        if not GEMINI_API_KEYS:
            st.error("Gemini API keys not available.")
            return None
    
        # If multiple keys, choose one randomly; otherwise use the configured one (if single) or the first.
        selected_key = random.choice(GEMINI_API_KEYS)
    
        client = genai.Client(api_key=selected_key)
    
    
        try:
            response = client.models.generate_content(
                model=model, contents=  prompt
            )
            st.text(str(response))
    
            return response.text
        except Exception as e:
            st.text('gemini_text_lib error ' + str(e)) 
            time.sleep(15)
            tries += 1
    
    return None

def get_top_3_images_hash(img_list):


    hashes_map = {}
    for image_url in img_list:
        # try:
            if '60x60' not in image_url:
                # st.text(f"Processing image URL: {image_url}")
                image_res = requests.get(image_url)
                image_res.raise_for_status()  

                img = Image.open(io.BytesIO(image_res.content))
                hash = imagehash.phash(img)
                if str(hash) not in hashes_map.keys():
                    hashes_map[str(hash)] = None
                    hashes_map[str(hash)] = {'count':1 , 'data' : [image_url]}
                elif str(hash) in hashes_map.keys():
                    hashes_map[str(hash)]['data'] = hashes_map[str(hash)]['data'] + [image_url]
                    hashes_map[str(hash)]['count'] += 1

        # except Exception as e:
        #     print(f'get_top_3_images_hash failed  : {e}')
    # most_common_hash =max(hashes_map, key= lambda k: k[1]['count'], reversed=True)[:3]
    # st.text(f"hashes_map : {hashes_map}")
    top3_most_common_hash = sorted(hashes_map.items(), key = lambda k :k[1]['count'] , reverse= True)[:3]
    # st.text(top3_most_common_hash)
    return top3_most_common_hash




# --- Initialize Session State ---
if 'combined_df' not in st.session_state:
    st.session_state.combined_df = None # Initialize as None

# --- Scraping Function (scrape_facebook_ads) ---
# [NO CHANGES NEEDED TO THE scrape_facebook_ads function itself from the previous version]
# ... (Keep the entire function as it was) ...
def scrape_facebook_ads(url, search_term, scroll_pause_time=5, max_scrolls=50):
    """
    Scrapes ads from a given Facebook Ads Library URL using Selenium (Cloud-ready).

    Args:
        url (str): The specific Facebook Ads Library URL to scrape (with q= term).
        search_term (str): The search term used for this specific scrape run.
        scroll_pause_time (int): Base pause time between scrolls.
        max_scrolls (int): Maximum number of scroll attempts.

    Returns:
        pandas.DataFrame: A DataFrame containing the scraped ad data, or None if error.
        list: Status messages.
    """
    status_messages = []
    ads_data = []
    driver = None

    status_messages.append(f"Attempting to initialize WebDriver for term: '{search_term}' in cloud environment...")
    try:
        options = Options()
        options.add_argument("--headless")  # Run headless REQUIRED for Streamlit Cloud
        options.add_argument("--no-sandbox")  # REQUIRED
        options.add_argument("--disable-dev-shm-usage")  # REQUIRED
        options.add_argument("--disable-gpu") # Also often recommended
        options.add_argument("--window-size=1920,1080") # Can be helpful
        options.add_argument('--log-level=3') # Suppress logs

        # In Streamlit Cloud, Selenium should automatically find chromedriver
        # if it's installed via packages.txt and in the PATH.
        # We try initializing without specifying executable_path.
        try:
             # Let Selenium handle the driver path if installed via packages.txt
             # No explicit 'service' needed if chromedriver is in PATH
             driver = webdriver.Chrome(options=options)
             status_messages.append("WebDriver initialized successfully using system PATH.")
        except WebDriverException as e:
             status_messages.append(f"WebDriver auto-init failed: {e}. Trying with default Service()...")
             # Fallback: Sometimes explicitly using Service() helps Selenium find it
             try:
                 service = Service() # Initialize without path
                 driver = webdriver.Chrome(service=service, options=options)
                 status_messages.append("WebDriver initialized successfully using default Service().")
             except WebDriverException as e2:
                 status_messages.append(f"WebDriver explicit Service() failed: {e2}")
                 # Updated message based on packages.txt correction
                 status_messages.append("Ensure 'chromium' and 'chromium-driver' are in packages.txt")
                 st.error("Fatal: Could not initialize WebDriver in the cloud environment. Check packages.txt.")
                 return None, status_messages # Critical failure

        status_messages.append(f"Loading URL for '{search_term}': {url}")
        driver.get(url)
        driver.implicitly_wait(10) # Give page elements time to appear
        time.sleep(5) # Initial wait after load

        # --- Scrolling Logic ---
        # (Keep scrolling logic as before, but add more robust waits/error handling)
        screen_height = driver.execute_script("return window.screen.height;")
        last_height = driver.execute_script("return document.body.scrollHeight")
        scroll_count = 0
        status_messages.append(f"Starting scroll process for '{search_term}'...")
        scroll_status_placeholder = st.empty()

        while True:
            try:
                # Scroll down
                driver.execute_script(f"window.scrollTo(0, {last_height + screen_height});")
                time.sleep(0.5) # Short pause between scrolls
                driver.execute_script(f"window.scrollTo(0, document.body.scrollHeight);") # Try to ensure bottom
                scroll_count += 1
                wait_time = scroll_pause_time + (scroll_count * 0.1) # Dynamic wait
                scroll_status_placeholder.info(f"Term '{search_term}': Scroll attempt {scroll_count}, waiting {wait_time:.1f}s...")
                time.sleep(wait_time)

                # Check new height
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    # Sometimes content loads just after height stabilizes, try one more time
                    scroll_status_placeholder.info(f"Term '{search_term}': Height stable, checking one last time...")
                    time.sleep(scroll_pause_time * 1.5) # Longer wait
                    new_height = driver.execute_script("return document.body.scrollHeight")
                    if new_height == last_height:
                        status_messages.append(f"Reached end of page for '{search_term}'.")
                        break

                last_height = new_height

                if scroll_count >= max_scrolls:
                    status_messages.append(f"Reached max scroll attempts ({max_scrolls}) for '{search_term}'.")
                    break

            except TimeoutException:
                 status_messages.append(f"Warning: Timeout during scroll execution for '{search_term}'. Page might be slow or stuck.")
                 time.sleep(scroll_pause_time * 2) # Longer wait after timeout
            except WebDriverException as scroll_err:
                 status_messages.append(f"Warning: WebDriver error during scroll for '{search_term}': {scroll_err}. Trying to continue...")
                 time.sleep(scroll_pause_time)


        scroll_status_placeholder.empty()
        status_messages.append(f"Scrolling finished for '{search_term}'.")

        # Get page source
        status_messages.append(f"Getting page source for '{search_term}'...")
        html = driver.page_source
        if not html or len(html) < 500: # Basic check for empty or minimal HTML
             status_messages.append(f"Warning: Page source seems empty or too small for '{search_term}'. Check if the page loaded correctly.")
             # Decide whether to continue or return early based on severity

        status_messages.append(f"Parsing HTML for '{search_term}'...")
        soup = BeautifulSoup(html, "lxml")

        # --- Data Extraction Logic ---
        # (Keep extraction logic as before - selectors remain the fragile part)
        ad_block_selector = 'div.xh8yej3' # VERIFY THIS SELECTOR REGULARLY
        ad_blocks = soup.select(ad_block_selector)
        status_messages.append(f"Found {len(ad_blocks)} potential ad blocks for '{search_term}'.")

        # ... (rest of the extraction loop is identical to previous version) ...
        extraction_count = 0
        for i, ad_block in enumerate(ad_blocks):
            # (Same extraction logic for status, text, media_url)
            # ...
            status = "Not Found" # Placeholder
            ad_text = "Not Found" # Placeholder
            media_url = "Not Found" # Placeholder
            count = 1
            # --- [INSERT EXACT EXTRACTION CODE FROM PREVIOUS VERSION HERE] ---
             # --- Extract Status ---
            try:
                status_selectors = [
                    'span.x1fp01tm', 'div[role="button"] > span', 'div > span[dir="auto"] > span[dir="auto"]'
                ]
                # Simplified status extraction (take first non-empty match)
                for selector in status_selectors:
                    elem = ad_block.select_one(selector)
                    if elem and elem.text.strip():
                        status = elem.text.strip()
                        break
            except Exception: pass # Ignore errors in finding elements

            # --- Extract Ad Text ---
            try:
                text_selectors = [
                    'div[data-ad-preview="message"]', 'div._7jyr',
                    'div > div > span[dir="auto"]', 'div[style*="text-align"]'
                ]
                for selector in text_selectors:
                    elem = ad_block.select_one(selector)
                    if elem:
                        all_texts = elem.find_all(string=True, recursive=True)
                        full_text = ' '.join(filter(None, (t.strip() for t in all_texts)))
                        cleaned_text = ' '.join(full_text.split())
                        if cleaned_text and cleaned_text.lower() not in ["sponsored", "suggested for you", ""]:
                            ad_text = cleaned_text
                            break # Found good text
                if ad_text in ["Not Found", ""]: ad_text = "Not Found"
            except Exception: pass

            # --- Extract Page ID ---
            try:
                text_selectors = [
                        'a.xt0psk2.x1hl2dhg.xt0b8zv.x8t9es0.x1fvot60.xxio538.xjnfcd9.xq9mrsl.x1yc453h.x1h4wwuj.x1fcty0u']
                for selector in text_selectors:
                    elem = ad_block.select_one(selector)
                    if elem:
                        page_name = elem.find_all(string=True, recursive=True)
                        page_name =list(page_name)[0]
                        page_id = elem.get("href", "Not Found")
                        page_id = page_id.split("/")[3]

                if page_id in ["Not Found", ""]: page_id = "Not Found"
                if page_name in ["Not Found", ""]: page_id = "Not Found"

                # page_id = str(page_id)
            except Exception:
                page_id ='fail'
                page_name = 'fail'

# --- Extract count Text ---
            try:
                count_selectors = [
                   'div.x6s0dn4.x78zum5.xsag5q8 span[dir="auto"]', # Specific parent + specific span type
                 'div.x6s0dn4.x78zum5.xsag5q8 span', # General span  
                ]
                for selector in count_selectors:
                    elem = ad_block.select_one(selector)
                    if elem:
                        all_texts = elem.find_all(string=True, recursive=True)
                        full_text = ' '.join(filter(None, (t.strip() for t in all_texts)))
                        cleaned_text = ' '.join(full_text.split())
                        if cleaned_text and cleaned_text.lower() not in ["sponsored", "suggested for you", ""]:
                            count = int("".join(filter(str.isdigit, cleaned_text)))
                            break # Found good text
                if count in ["Not Found", ""]: ad_text = "Not Found"
            except Exception: pass

            # --- Extract Image or Video Poster URL ---
            try:
                media_url = "Not Found"
                img_selectors = [
                    'img.x168nmei', 'img.xt7dq6l', 'img[referrerpolicy="origin-when-cross-origin"]',
                    'div[role="img"] > img', 'img:not([width="16"]):not([height="16"])'
                ]
                # Look for images
                for selector in img_selectors:
                     img_elem = ad_block.select_one(selector)
                     if img_elem and img_elem.has_attr('src'):
                         src = img_elem['src']
                         if 'data:image' not in src and '/emoji.php/' not in src and 'static.xx.fbcdn.net/rsrc.php' not in src:
                             media_url = src
                             break
                # Look for video posters if no image found
                if media_url == "Not Found":
                    video_selectors = [
                        'video.x1lliihq', 'video.xvbhtw8', 'video[poster]'
                    ]
                    for selector in video_selectors:
                         vid_elem = ad_block.select_one(selector)
                         if vid_elem and vid_elem.has_attr('poster'):
                            media_url = vid_elem['poster']
                            break



                


            except Exception: pass
            # --- Extract Ad Link using CSS Selectors ---
            ad_link = "Not Found"
            found_link_tag = None
            # Define potential CSS selectors for the link, ordered from most specific/reliable to more general
            link_selectors = [
                'a[href^="https://l.facebook.com/l.php?u="]', # Starts with FB redirect + contains domain
                'div._7jyr + a[target="_blank"]', # Positional selector + target attribute
                'a[data-lynx-mode="hover"]',       # Just the data attribute
                'a[href^="https://l.facebook.com/l.php?u="]' # Just the FB redirect start
            ]

            try:
                for selector in link_selectors:
                    # print(f"Trying selector: {selector}") # Optional debug print
                    elem = ad_block.select_one(selector)
                    # Check if an element was found and if it has an 'href' attribute
                    if elem and elem.has_attr('href'):
                        # print(f"Selector matched: {selector}") # Optional debug print
                        found_link_tag = elem
                        ad_link = found_link_tag['href'] # Extract the href value


                        if ad_link and ad_link != "Not Found" and "l.facebook.com/l.php" in ad_link:
                            try:
                                parsed_url = urlparse(ad_link)
                                query_params = parse_qs(parsed_url.query)
                                
                                # Check if the 'u' parameter exists
                                if 'u' in query_params:
                                    # parse_qs returns a list for each param, get the first value
                                    encoded_url = query_params['u'][0]
                                    # Decode the URL
                                    ad_link = unquote(encoded_url)
                                else:
                                    ad_link = "Redirect link found, but 'u' parameter missing."
                                    
                            except Exception as e:
                                print(f"Error parsing or decoding redirect URL: {e}")
                                actual_destination_url = "Error processing redirect link"
                        elif ad_link and ad_link != "Not Found":
                            # If it wasn't a facebook redirect link, the extracted link is the actual one
                            ad_link = ad_link
                        break # Stop searching once a link is found
            except Exception as e:
                print(f"An error occurred while selecting the link: {e}")
                ad_link = "Error finding link"
            # --- [END OF EXTRACTION CODE] ---


            # Append data - include the search_term
            # ** NOTE: Filtering based on Text/Media presence is now done AFTER collecting all rows **
            ads_data.append({ 
                 'Search_Term': search_term,
                 'Status': status,
                 'Text': ad_text,
                 'Count': count,
                 'Media_URL': media_url,
                 'Landing_Page': ad_link,
                 'Page ID' :page_id,
                 'Page Name' : page_name
             })
            extraction_count += 1 # Count raw extracted rows

        # --- End of Extraction Loop ---

        final_message = f"Extracted {extraction_count} raw ad data entries for term '{search_term}' (before filtering)."
        status_messages.append(final_message)

        if ads_data:
            df = pd.DataFrame(ads_data)
            # Note: Filtering is now applied AFTER concatenation in the main app logic
            return df, status_messages # Return the unfiltered data for this term
        else:
            status_messages.append(f"No raw data extracted into DataFrame for '{search_term}'.")
            return pd.DataFrame(), status_messages # Return empty DF

    except WebDriverException as e:
        error_msg = f"WebDriver Error during operation for term '{search_term}': {e}\n{traceback.format_exc()}"
        status_messages.append(error_msg)
        st.error(f"WebDriver Error occurred for '{search_term}'. Check logs. The cloud environment might be unstable or the page blocked.")
        return None, status_messages # Indicate failure
    except Exception as e:
        error_msg = f"Unexpected Error for term '{search_term}': {e}\n{traceback.format_exc()}"
        status_messages.append(error_msg)
        st.error(f"Unexpected Error occurred for '{search_term}'. Check logs.")
        return None, status_messages # Indicate failure

    finally:
        if driver:
            status_messages.append(f"Closing WebDriver for '{search_term}'...")
            try:
                driver.quit()
                status_messages.append(f"WebDriver closed for '{search_term}'.")
            except Exception as quit_err:
                 status_messages.append(f"Error closing WebDriver for '{search_term}': {quit_err}")


# Fetch a single title with a semaphore (concurrency limiter)
async def fetch_title(session, url, semaphore, timeout=10):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    }

    async with semaphore:
        try:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                text = await resp.text()
                soup = BeautifulSoup(text, 'html.parser')
                title_tag = soup.find('title')
                return url, title_tag.get_text(strip=True) if title_tag else "[No title found]"
        except Exception as e:
            return url, f"[Error: {e}]"

# Limit concurrency using a semaphore
async def fetch_all_titles(urls, max_concurrent=20):
    semaphore = asyncio.Semaphore(max_concurrent)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_title(session, url, semaphore) for url in urls]
        results = await asyncio.gather(*tasks)
        return results



def get_html_content(url):
    # Set up headless browser
    options = Options()
    options.add_argument("--headless")  # Run headless REQUIRED for Streamlit Cloud
    options.add_argument("--no-sandbox")  # REQUIRED
    options.add_argument("--disable-dev-shm-usage")  # REQUIRED
    options.add_argument("--disable-gpu") # Also often recommended
    options.add_argument("--window-size=1920,1080") # Can be helpful
    options.add_argument('--log-level=3') # Suppress logs
    # options.headless = True
    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)
        time.sleep(2)  # Let JS load

        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        # # Keep only allowed tags
        # allowed_tags = ['a', 'p', 'h1', 'h2', 'h3', 'h4', 'li', 'ul', 'img']
        # for tag in soup.find_all(True):
        #     if tag.name not in allowed_tags:
        #         tag.decompose()

        # # Clean unwanted attributes
        # for tag in soup.find_all(allowed_tags):
        #     tag.attrs = {k: v for k, v in tag.attrs.items() if k in ['href', 'src', 'alt']}
        for tag in ["style", "link", "meta", "header", "footer", "input", "svg", "script"]:


            for element in soup.find_all(tag):
                element.decompose()
        
        # Optional: also remove iframes, noscript, or style if needed
        for tag in soup(['iframe', 'noscript']):
            tag.decompose()

        return str(soup)

    except Exception as e:
        st.text("error in get_html_content" + e)
        return None
    finally:
        driver.quit()

# --- Streamlit App UI ---
st.title("Facebook Ads Library Multi-Term Scraper + Gemini Analysis")
st.markdown("""
Provide Base URL & Search Terms. Scrapes ads in the cloud, combines results, **filters for ads with Text & Media**, displays them, and optionally analyzes trends with Gemini.
""")

# --- Inputs ---
st.subheader("Configuration")
mode = st.radio("Select Search Mode",["General Search","Page Search"] , index=0)

if mode == 'General Search':
    default_base_url = "https://www.facebook.com/ads/library/?active_status=active&ad_type=all&country=ALL&is_targeted_country=false&media_type=all&search_type=keyword_unordered&q="
    base_url_template = st.text_input(
        "Enter Base URL Template (ending with 'q=' or ready for term):",
        default_base_url,
        help="Example: https://www.facebook.com/ads/library/?active_status=all&ad_type=all&country=ALL&q="
    )
if mode == 'Page Search':
    default_base_url = "https://www.facebook.com/ads/library/?active_status=active&ad_type=all&country=ALL&is_targeted_country=false&media_type=all&search_type=page&view_all_page_id="
    base_url_template = st.text_input(
        "Enter Base URL Template (ending with 'view_all_page_id=' or ready for term):",
        default_base_url,
    )


search_terms_input = st.text_area(
    "Enter Search Terms\Page IDs (one per line):",
    height=150,
    help="Each line is a separate search query."
)
auto_gemini = st.checkbox("Auto Gemini Analyze?", value=False)


st.info("ℹ️ WebDriver configured for Streamlit Cloud.", icon="☁️")

col1, col2 = st.columns(2)
with col1:
    scroll_pause = st.slider("Scroll Pause Time (seconds):", min_value=1, max_value=20, value=2, help="Base time between scrolls.")
with col2:
     max_scrolls = st.slider("Max Scroll Attempts:", min_value=0, max_value=75, value=40, help="Max scrolls per term.")


# --- Scrape Button and Logic ---
if st.button("🚀 Scrape All Terms in Cloud", type="primary"):
    # --- [Identical validation logic as before] ---
    if not base_url_template or not base_url_template.startswith("http"):
        st.error("Please enter a valid Base URL Template.")
    elif not search_terms_input:
        st.error("Please enter at least one search term.")
    else:
        search_terms = [term.strip() for term in search_terms_input.splitlines() if term.strip()]
        if not search_terms:
             st.error("No valid search terms found.")
        else:
            st.info(f"Preparing to scrape for {len(search_terms)} terms...")
            all_results_dfs = []
            all_log_messages = []
            overall_start_time = time.time()
            overall_status_placeholder = st.empty()

            # --- [Identical scraping loop as before] ---
            for i, term in enumerate(search_terms):
                term_start_time = time.time()
                overall_status_placeholder.info(f"Processing term {i+1}/{len(search_terms)}: '{term}'... total scraped : {len(all_results_dfs)}")
                encoded_term = urllib.parse.quote_plus(term)
            


                if mode == 'General Search':
                    if "?" in base_url_template:
                        scrape_url = f"{base_url_template.split('?')[0]}?{base_url_template.split('?')[1]}&q={encoded_term}"
                    else:
                        scrape_url = f"{base_url_template}?q={encoded_term}"
                    scrape_url = scrape_url.replace("?&", "?").replace("&&", "&").replace("= ", "=")
                elif mode == "Page Search":
                    if "?" in base_url_template:
                        scrape_url = f"{base_url_template.split('?')[0]}?{base_url_template.split('?')[1]}&view_all_page_id={encoded_term}"
                    else:
                        scrape_url = f"{base_url_template}?view_all_page_id={encoded_term}"
                    scrape_url = scrape_url.replace("?&", "?").replace("&&", "&").replace("= ", "=")




                with st.spinner(f"Scraping '{term}'..." ):
                    scraped_df, log_messages = scrape_facebook_ads(
                        scrape_url, term, scroll_pause, max_scrolls
                    )
                    all_log_messages.extend(log_messages)

                term_duration = time.time() - term_start_time
                if scraped_df is not None: # Check for None (fatal error)
                    if not scraped_df.empty:
                         # Append even if empty, filtering happens after concat
                        all_results_dfs.append(scraped_df)
                    # Don't display success message per term here, do it after combining/filtering
                else:
                    st.error(f"Scraping failed for term '{term}' after {term_duration:.2f}s.")

            overall_status_placeholder.empty()
            overall_duration = time.time() - overall_start_time
            st.info(f"Finished scraping all {len(search_terms)} terms in {overall_duration:.2f} seconds. Now combining and filtering...")

            # --- Combine, Filter, and Store in Session State ---
            if all_results_dfs:
                # Combine all collected data (including potentially empty DFs from terms with no results)
                combined_raw_df = pd.concat(all_results_dfs, ignore_index=True)

                # Apply filtering HERE, after combining
                combined_filtered_df = combined_raw_df[
                    (combined_raw_df['Text'] != "Not Found") &
                    (combined_raw_df['Media_URL'] != "Not Found")
                ].copy() # Apply the filter condition from your code

                combined_filtered_df.reset_index(drop=True, inplace=True)

                if not combined_filtered_df.empty:
                    st.success(f"Combined and filtered data: {len(combined_filtered_df)} ads found with Text & Media.")
                    # Store the *filtered* DataFrame in session state
                    st.session_state.combined_df = combined_filtered_df
                else:
                    st.warning("No ads with both Text and Media found across all terms after filtering.")
                    st.session_state.combined_df = pd.DataFrame() # Store empty DF
            else:
                 st.warning("No data was scraped from any term.")
                 st.session_state.combined_df = None # Ensure state is None if scraping failed

            # --- Display Logs ---
            st.subheader("Combined Scraping Log")
            # Always show logs, even if no data was found
            with st.expander("Show detailed log", expanded=False):
                 log_text = "\n".join(all_log_messages)
                 st.text_area("Log Output:", log_text, height=300)


# --- Display Results Area (uses session state) ---
st.subheader("Scraped & Filtered Data")
if st.session_state.combined_df is not None and not st.session_state.combined_df.empty:
    st.dataframe(
        st.session_state.combined_df,
        use_container_width=True,
        column_config={"Media_URL": st.column_config.ImageColumn("Preview Image", width="medium")},
        # row_height = 100 # Increase row height for images if needed
    )
    # Download Button - Placed logically after data display
    @st.cache_data
    def convert_combined_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    csv_data = convert_combined_df_to_csv(st.session_state.combined_df)
    now_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
         label="💾 Download Filtered Data as CSV",
         data=csv_data,
         file_name=f"fb_ads_filtered_scrape_{now_ts}.csv",
         mime='text/csv',
         key='download_csv_button' # Add key for widget uniqueness
    )
elif st.session_state.combined_df is not None and st.session_state.combined_df.empty:
    st.info("Scraping complete, but no ads matched the filtering criteria (Text + Media found).")
else:
    st.info("Click 'Scrape All Terms in Cloud' to fetch data.")


# --- Gemini Processing Button (uses session state) ---
st.subheader(" Analyze Trends (Optional)")
if st.button("Process trends with Gemini?", key='gemini_button', disabled=(GEMINI_API_KEYS is None) ) or ( auto_gemini and st.session_state.combined_df is not None and not st.session_state.combined_df.empty and st.session_state['final_merged_df'] is None) :
    if st.session_state.combined_df is not None and not st.session_state.combined_df.empty:
        df_to_process = st.session_state.combined_df

        # Check if 'Text' column exists
        if "Text" in df_to_process.columns:
            df_to_process  = df_to_process[df_to_process["Text"].str.len() <= 500]

            tokens =count_string_tokens(prompt = "\n".join(list(df_to_process["Text"])),model="gemini-2.0-flash-001	")
            chunks_num = tokens//200000 + 1  
            df_appends = []
            max_rows = 3500
            dfs_splits = np.array_split(df_to_process,chunks_num)
            st.text(f"Tokens :{tokens} Num of chucks {len(dfs_splits)}")

          

            for df_idx, df_chunk  in enumerate(dfs_splits):
                st.text(f"{df_idx} {len(df_chunk)}")
                # st.text("\n".join(list(df_chunk["Text"])))
                
                df_chunk = df_chunk.reset_index(drop=True)
                df_to_process_text  = pd.DataFrame(df_chunk[["Text","Count"]], columns = ["Text","Count"])
                df_to_process_text  = df_to_process_text[df_to_process_text["Text"].str.len() <= 500]
                df_to_process_text['Count'] = pd.to_numeric(df_to_process_text['Count'], errors='coerce')
                df_to_process_text['Count'] = df_to_process_text['Count'].fillna(0)

                # st.text(df_to_process_text)
                df_counts = (
                                df_to_process_text.reset_index()
                                .groupby("Text")
                                .agg(Count=("Count", "sum"), Indices=("index", list))
                                .reset_index()
                            )
                # st.text(df_counts.to_string())
                st.markdown(f"Proccessing {df_idx+1} df...")
                st.dataframe(df_counts)
                # st.text(df_counts.to_string())
                # st.text("\n".join(list(df_counts)))
                
                
                # --- Prepare prompt (consider limits) ---
                # Example: Use unique texts, limit number of texts sent
             
            
                # # Construct the final prompt
                # gemini_prompt = f"""Please go over the following search arbitrage ideas, deeply think about patterns and reoccurring. I want to get the ideas that would show the most potential. This data is scraped from competitors, so whatever reoccurs is probably successful.\nReturn a list of ideas txt new line delimited!      (no Analysis at all! )of the ideas (just the ideas consicly, no explaning, and not as given), descending order by potential like i described. \nanalyze EACH entry!  BE VERY thorough. be  specific in the topic. don't mix beteern languages and simillar ideas, show them in differnet rows (but still just the ideas consicly , not original input) , return in original language
    
                #         Ad Text:
                #         {'\n'.join(df_to_process["Text"])}
                        
                        
                #         """
                
    
                gemini_prompt = """Please go over the following search arbitrage ideas table, deeply think about patterns and reoccurring. I want to get the ideas that would show the most potential. This data is scraped from competitors, so whatever reoccurs is probably successful.\nReturn a list of ideas txt new line delimited!      (no Analysis at all! )of the ideas (just the ideas consicly, no explaning, and not as given), descending order by potential like i described. \nanalyze EACH entry!  BE VERY thorough. be  specific in the topic. don't mix beteern languages, show them in differnet rows (but still just the ideas consicly , not original input) , return in original language. use the text in 'Text' col to understand the topic and merge simillar text about the similar ideas. then return the indices of the rows from input table per row of output table. return in json example : [{"idea" : "idea text..." , "indices" : [1,50]} , ....]""" + f"""
                I will provide the how many times the text occurred for you and the indices
                Each "idea" value should be 2-5 words specific
                the idea column texts needs to be a simple concise terms\keyword, no special characters like ( ) & / , etc 
                RETURN ONLY THE JSON NO INTROS OR ANYTHING ELSE!
                table:
                {df_counts.to_string()}"""
            
                st.info(f"Sending  unique text samples to Gemini for analysis...")
                with st.spinner("🧠 Processing with Gemini... This might take a moment."):
                    with st.expander("Prompt:"):
                        st.text(gemini_prompt)
                    gemini_res = gemini_text_lib(gemini_prompt,model ="gemini-2.5-pro-preview-03-25") # Use the dedicated function gemini-2.5-pro-exp-03-25
            
                if gemini_res:
                    # st.text(gemini_res) 

                    final_df = pd.DataFrame()
                    st.subheader(" Gemini Analysis Results")
                    gemini_res =gemini_res.replace("```json", '').replace("```", '') # Clean up the response
                    #st.text(gemini_res) 
                    # with st.expander("Gemini Results:"):
                    st.text(gemini_res)
                    gemini_df = pd.read_json(gemini_res) # Convert to DataFrame
    
    
    
                    for index, row in gemini_df.iterrows():
                        idea = row['idea']
        
                        indices = row['indices']
                        inx_len = len(list(indices))
                        hash_urls={}

                        urls = [df_chunk.iloc[idx]["Landing_Page"] for idx in indices]
                        url_title_map = asyncio.run(fetch_all_titles(urls))
                        # st.text(url_title_map)
                        
                        count_map ={}

                        for elem in url_title_map:
                            title = elem[1]
                            if title not in count_map.keys():
                                count_map[title] = 1
                            else:
                                count_map[title] += 1
                        max_seen_url_title = max(count_map, key=count_map.get)
                        max_seen_url =  max_seen_url = next((url for url, title in url_title_map if title == max_seen_url_title),
                                                             None)



                        # for idx in list(indices): #url:times
                        #     landing_page = df_chunk.iloc[idx]["Landing_Page"]
                        #     if landing_page in hash_urls:
                        #         hash_urls[landing_page] += 1
                        #     else:
                        #         hash_urls[landing_page] = 1
                        # max_seen_url = max(hash_urls, key=hash_urls.get)
                        
                        text_urls = {}
                        for idx in list(indices): #text:times
                            text = df_chunk.iloc[idx]["Text"]
                            if text in text_urls:
                                text_urls[text] += 1
                            else:
                                text_urls[text] = 1
                        max_seen_text = max(text_urls, key=text_urls.get)
    
    
                        matching_rows = df_chunk.iloc[indices]
                        try:
                            most_common_hash = get_top_3_images_hash(matching_rows['Media_URL'].tolist())
                            most_common_img_urls= [elem[1]['data'][0] for elem in most_common_hash]
                            images = "|".join(most_common_img_urls)

                        except Exception as e:
                            print(f"Error processing most_common_img_urls: {e}")


                        padded_urls = (list(most_common_img_urls or []) + [None] * 3)[:3]

                        
                        img1, img2, img3 = padded_urls


                        try:
                            lang= detect(max_seen_text)
                        except: 
                            lang=''
                        row_df = pd.DataFrame([{
                            "selected" : False,
                            "idea": idea,
                            "lang": lang,

                            "len": inx_len,
                            "images": images,
                            "max_url": max_seen_url,
                            "max_text": max_seen_text,
                            "max_seen_url_title" :max_seen_url_title,
                            "img1": img1,
                            "img2": img2,
                            "img3": img3,
                            "indices": indices,                            
                        }])

                        df_appends.append(row_df)

                else:
                    # Error message already displayed within gemini_text_lib
                    st.error("Gemini processing failed or returned no result.")
        else:
            st.error("Could not find 'Text' column in the scraped data. Cannot analyze.")
    else:
        st.error("No filtered data available to process. Please scrape data first.")

    
    final_merged_df = pd.concat(df_appends)
                    
    st.session_state['final_merged_df'] = final_merged_df

elif GEMINI_API_KEYS is None:
    st.warning("Gemini analysis disabled because GEMINI_API_KEY is not configured in secrets.", icon="🚫")



if 'final_merged_df' in st.session_state :
    # df = pd.concat(df_appends)
    # df["selected"] = False  # Ensure column exists
    # st.session_state['final_merged_df'] = df

    # Use a local variable to hold current version
    current_df = st.session_state['final_merged_df'].copy()
 
    # Display editor - do NOT connect to session_state via key
    edited_df = st.data_editor(
        current_df,
        column_config={
            "idea": st.column_config.TextColumn(pinned = True),
            'img1': st.column_config.ImageColumn("Image 1", width="medium"),
            'img2': st.column_config.ImageColumn("Image 2", width="medium"),
            'img3': st.column_config.ImageColumn("Image 3", width="medium"),
            "selected": st.column_config.CheckboxColumn("Selected",pinned = True)
        },
        use_container_width=True,
        hide_index=True,
        
    )

    # Let user manually confirm selection changes to sync
    is_gen_html = st.checkbox("Gen HTML content")
    if st.button("Process Selected Rows"):
        st.session_state['final_merged_df_selected'] = edited_df.copy()

        # Work with updated session state
        selected_df = st.session_state['final_merged_df_selected'][st.session_state['final_merged_df_selected']["selected"] == True]

        if is_gen_html:
            title_res = []
            html_res=[]
            for index, row in selected_df.iterrows():
                tries = 0
                done = False
                while tries < 5 and done is False:
                        
                    try:

                        content = get_html_content(row['max_url'])
                        # st.text(content)
                        prompt = """write as html using only  <a>, <p>, <h2>–<h4>, <li>, <ul>, <img>.\nNEVER use <br> or <br\> or <ol> or <ol\> NEVER!
                        only the article content no footers no images!! no images! no writer name!, no <div>!!! first element is ALWAYS <p>. NEVER write\return the domain name ( like xxx.com) in the title or html , omit that!! return in language same as input . return json dict, 2 keys : 'title', 'html'  . \n example :{"title" : "Learn more about how veterans ...", 'html' :"full article w/o title with html tags..'}  no <div>\n\n""" + content
                        gemini_res =gemini_text_lib(prompt=prompt, model='gemini-2.0-flash-exp' ) # gemini-2.0-flash-exp
                        # st.text(gemini_res)
                        pure_html = gemini_res.replace("```html","").replace("```","").replace("```json","").replace("json","")
                        pure_html = json.loads(pure_html)
                        done = True
    
                    except Exception as e:
                        pure_html = f"error {e} "
                        tries += 1
                    # st.text(str(pure_html)) 
                    title_res.append(pure_html['title'].replace("```json",""))
                    html_res.append(pure_html['html'].replace("```html","").replace("```","").replace("```json",""))

            selected_df['html'] = html_res
            selected_df['html_title'] = title_res

                
    # if st.button("👁 Show Selected Rows"):
        st.dataframe(selected_df, hide_index=True, use_container_width=True,column_config={
            'img1': st.column_config.ImageColumn("Image 1", width="medium"),
            'img2': st.column_config.ImageColumn("Image 2", width="medium"),
            'img3': st.column_config.ImageColumn("Image 3", width="medium")})

# --- Footer ---
st.markdown("---")
