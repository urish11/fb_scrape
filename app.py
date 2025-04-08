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
import urllib.parse
import os
import random # Added import
from google import genai
 
st.set_page_config(layout="wide")

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
def gemini_text_lib(prompt, model='gemini-2.5-pro-exp-03-25'): # Using a stable model
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

        return response.text
    except Exception as e:
        st.text('gemini_text_lib error ' + str(e))
        time.sleep(4)
        return None



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
            # --- [END OF EXTRACTION CODE] ---


            # Append data - include the search_term
            # ** NOTE: Filtering based on Text/Media presence is now done AFTER collecting all rows **
            ads_data.append({
                 'Search_Term': search_term,
                 'Status': status,
                 'Text': ad_text,
                 'Media_URL': media_url
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


# --- Streamlit App UI ---
st.title("Facebook Ads Library Multi-Term Scraper + Gemini Analysis")
st.markdown("""
Provide Base URL & Search Terms. Scrapes ads in the cloud, combines results, **filters for ads with Text & Media**, displays them, and optionally analyzes trends with Gemini.
""")

# --- Inputs ---
st.subheader("Configuration")
default_base_url = "https://www.facebook.com/ads/library/?active_status=active&ad_type=all&country=ALL&is_targeted_country=false&media_type=all&search_type=keyword_unordered&q="
base_url_template = st.text_input(
    "Enter Base URL Template (ending with 'q=' or ready for term):",
    default_base_url,
    help="Example: https://www.facebook.com/ads/library/?active_status=all&ad_type=all&country=ALL&q="
)
search_terms_input = st.text_area(
    "Enter Search Terms (one per line):",
    height=150,
    help="Each line is a separate search query."
)
st.info("ℹ️ WebDriver configured for Streamlit Cloud.", icon="☁️")

col1, col2 = st.columns(2)
with col1:
    scroll_pause = st.slider("Scroll Pause Time (seconds):", min_value=1, max_value=20, value=7, help="Base time between scrolls.")
with col2:
     max_scrolls = st.slider("Max Scroll Attempts:", min_value=1, max_value=75, value=40, help="Max scrolls per term.")


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
                if "?" in base_url_template:
                     scrape_url = f"{base_url_template.split('?')[0]}?{base_url_template.split('?')[1]}&q={encoded_term}"
                else:
                     scrape_url = f"{base_url_template}?q={encoded_term}"
                scrape_url = scrape_url.replace("?&", "?").replace("&&", "&").replace("= ", "=")

                with st.spinner(f"Scraping '{term}'..."):
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
if st.button("Process trends with Gemini?", key='gemini_button', disabled=(GEMINI_API_KEYS is None)):
    if st.session_state.combined_df is not None and not st.session_state.combined_df.empty:
        df_to_process = st.session_state.combined_df

        # Check if 'Text' column exists
        if "Text" in df_to_process.columns:
            # --- Prepare prompt (consider limits) ---
            # Example: Use unique texts, limit number of texts sent
         
        
            # Construct the final prompt
            gemini_prompt = f"""Please go over the following search arbitrage ideas, deeply think about patterns and reoccurring. I want to get the ideas that would show the most potential. This data is scraped from competitors, so whatever reoccurs is probably successful.\nReturn a list of ideas txt new line delimited!      (no Analysis at all! )of the ideas (just the ideas consicly, no explaning, and not as given), descending order by potential like i described. \nanalyze EACH entry!  BE VERY thorough. be  specific in the topic. don't mix beteern languages, show them in differnet rows (but still just the ideas consicly , not original input) , return in original language

                    Ad Text:
                    {'\n'.join(df_to_process["Text"])}
                    
                    
                    """
            st.info(f"Sending  unique text samples to Gemini for analysis...")
            with st.spinner("🧠 Processing with Gemini... This might take a moment."):
                gemini_res = gemini_text_lib(gemini_prompt) # Use the dedicated function

            if gemini_res:
                st.subheader(" Gemini Analysis Results")
                st.text(gemini_res) # Use markdown for better formatting
            else:
                # Error message already displayed within gemini_text_lib
                st.error("Gemini processing failed or returned no result.")
        else:
            st.error("Could not find 'Text' column in the scraped data. Cannot analyze.")
    else:
        st.error("No filtered data available to process. Please scrape data first.")
elif GEMINI_API_KEYS is None:
     st.warning("Gemini analysis disabled because GEMINI_API_KEY is not configured in secrets.", icon="🚫")


# --- Footer ---
st.markdown("---")
