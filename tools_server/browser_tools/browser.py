from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.action_chains import ActionChains
from typing import List, Tuple, Type, Dict
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from fake_useragent import UserAgent
from selenium_stealth import stealth
import undetected_chromedriver as uc
import chromedriver_autoinstaller
import certifi
import ssl
import time
import random
import os
import shutil
import tempfile
import markdownify
import sys
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil

def get_chrome_path() -> str:
    """Get the path to the Chrome executable."""
    paths = {
        "win": [
            "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
            "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google\\Chrome\\Application\\chrome.exe"),
        ],
        "darwin": [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Google Chrome Beta.app/Contents/MacOS/Google Chrome Beta",
        ],
        "linux": [
            "/usr/bin/google-chrome",
            "/usr/bin/chromium-browser",
            "/usr/bin/chromium",
            "/opt/chrome/chrome",
            "/usr/local/bin/chrome",
        ],
    }

    platform = sys.platform
    candidates = []

    if platform.startswith("win"):
        candidates = paths["win"]
    elif platform.startswith("darwin"):
        candidates = paths["darwin"]
    else:
        candidates = paths["linux"]

    for path in candidates:
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path

    chrome_path_env = os.environ.get("CHROME_EXECUTABLE_PATH")
    if chrome_path_env and os.path.exists(chrome_path_env) and os.access(chrome_path_env, os.X_OK):
        return chrome_path_env

    chrome_path = shutil.which("google-chrome") or shutil.which("chrome") or shutil.which("chromium")
    if chrome_path:
        return chrome_path

    print("Looking for Google Chrome in these locations failed:")
    print('\n'.join(candidates))
    path = input("Google Chrome not found. Please enter the path to the Chrome executable: ")
    if os.path.exists(path) and os.access(path, os.X_OK):
        os.environ["CHROME_EXECUTABLE_PATH"] = path
        print("Chrome path saved to environment variable CHROME_EXECUTABLE_PATH")
        return path
    return None

def get_random_user_agent() -> str:
    """Get a random user agent string with associated vendor."""
    user_agents = [
        {"ua": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36", "vendor": "Google Inc."},
        {"ua": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36", "vendor": "Apple Inc."},
        {"ua": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36", "vendor": "Google Inc."},
    ]
    return random.choice(user_agents)

def install_chromedriver() -> str:
    """
    Install the ChromeDriver if not already installed. Return the path.
    """
    chromedriver_path = shutil.which("chromedriver")
    print("Chromedriver:", chromedriver_path)
    if not chromedriver_path:
        print("trying to auto-install chromedriver...")
        try:
            chromedriver_path = chromedriver_autoinstaller.install()
        except Exception as e:
            raise FileNotFoundError(
                "ChromeDriver not found and could not be installed automatically. "
                "Please install it manually from https://chromedriver.chromium.org/downloads."
                "and ensure it's in your PATH or specify the path directly."
                "See know issues in readme if your chrome version is above 115."
            ) from e
    if not chromedriver_path:
        raise FileNotFoundError("ChromeDriver not found. Please install it or add it to your PATH.")
    return chromedriver_path

def bypass_ssl() -> str:
    """
    This is a fallback for stealth mode to bypass SSL verification. Which can fail on some setup.
    """
    print("Bypassing SSL verification issues, we strongly advice you update your certifi SSL certificate.")
    ssl._create_default_https_context = ssl._create_unverified_context

def create_undetected_chromedriver(service, chrome_options) -> webdriver.Chrome:
    """Create an undetected ChromeDriver instance."""
    try:
        driver = uc.Chrome(service=service, options=chrome_options)
    except Exception as e:
        print(f"Failed to create Chrome driver: {str(e)}. Trying to bypass SSL...")
        try:
            bypass_ssl()
            driver = uc.Chrome(service=service, options=chrome_options)
        except Exception as e:
            print(f"Failed to create Chrome driver, fallback failed:\n{str(e)}.")
            raise e
        raise e
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})") 
    return driver

def create_driver(headless=False, stealth_mode=True, crx_path="./crx/nopecha.crx", lang="en") -> webdriver.Chrome:
    """Create a Chrome WebDriver with specified options."""
    chrome_options = Options()
    chrome_path = get_chrome_path()
    
    if not chrome_path:
        raise FileNotFoundError("Google Chrome not found. Please install it.")
    chrome_options.binary_location = chrome_path
    
    if headless:
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-webgl")
    user_data_dir = tempfile.mkdtemp()
    user_agent = get_random_user_agent()
    width, height = (1920, 1080)
    chrome_options.add_argument(f"--user-data-dir={user_data_dir}")
    chrome_options.add_argument(f"--accept-lang={lang}-{lang.upper()},{lang};q=0.9")
    chrome_options.add_argument("--timezone=Europe/Paris")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--mute-audio")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--autoplay-policy=user-gesture-required")
    chrome_options.add_argument("--disable-features=SitePerProcess,IsolateOrigins")
    chrome_options.add_argument("--enable-features=NetworkService,NetworkServiceInProcess")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument(f'user-agent={user_agent["ua"]}')
    chrome_options.add_argument(f'--window-size={width},{height}')
    if not stealth_mode:
        if not os.path.exists(crx_path):
            print(f"Anti-captcha CRX not found at {crx_path}.")
        else:
            chrome_options.add_extension(crx_path)

    chromedriver_path = install_chromedriver()

    service = Service(chromedriver_path)
    if stealth_mode:
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        driver = create_undetected_chromedriver(service, chrome_options)
        chrome_version = driver.capabilities['browserVersion']
        stealth(driver,
            languages=["en-US", "en"],
            vendor=user_agent["vendor"],
            platform="Win64" if "windows" in user_agent["ua"].lower() else "MacIntel" if "mac" in user_agent["ua"].lower() else "Linux x86_64",
            webgl_vendor="Intel Inc.",
            renderer="Intel Iris OpenGL Engine",
            fix_hairline=True,
        )
        return driver
    security_prefs = {
        "profile.default_content_setting_values.geolocation": 0,
        "profile.default_content_setting_values.notifications": 0,
        "profile.default_content_setting_values.camera": 0,
        "profile.default_content_setting_values.microphone": 0,
        "profile.default_content_setting_values.midi_sysex": 0,
        "profile.default_content_setting_values.clipboard": 0,
        "profile.default_content_setting_values.media_stream": 0,
        "profile.default_content_setting_values.background_sync": 0,
        "profile.default_content_setting_values.sensors": 0,
        "profile.default_content_setting_values.accessibility_events": 0,
        "safebrowsing.enabled": True,
        "credentials_enable_service": False,
        "profile.password_manager_enabled": False,
        "webkit.webprefs.accelerated_2d_canvas_enabled": True,
        "webkit.webprefs.force_dark_mode_enabled": False,
        "webkit.webprefs.accelerated_2d_canvas_msaa_sample_count": 4,
        "enable_webgl": True,
        "enable_webgl2_compute_context": True
    }
    chrome_options.add_experimental_option("prefs", security_prefs)
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    return webdriver.Chrome(service=service, options=chrome_options)

class Browser:
    def __init__(self, driver, anticaptcha_manual_install=False):
        """Initialize the browser with optional AntiCaptcha installation."""
        self.js_scripts_folder = "./sources/web_scripts/" if not __name__ == "__main__" else "./web_scripts/"
        self.anticaptcha = "https://chrome.google.com/webstore/detail/nopecha-captcha-solver/dknlfmjaanfblgfdfebhijalfmhmjjjo/related"
        self.screenshot_folder = os.path.join(os.getcwd(), ".screenshots")
        self.tabs = []
        try:
            self.driver = driver
            self.wait = WebDriverWait(self.driver, 10)
        except Exception as e:
            raise Exception(f"Failed to initialize browser: {str(e)}")
        self.setup_tabs()
        if anticaptcha_manual_install:
            self.load_anticatpcha_manually()
    
    def setup_tabs(self):
        self.tabs = self.driver.window_handles
        try:
            self.driver.get("https://www.google.com")
        except Exception as e:
            pass
        self.screenshot()
    
    def switch_control_tab(self):
        self.driver.switch_to.window(self.tabs[0])
            
    def load_anticatpcha_manually(self):
        print("You might want to install the AntiCaptcha extension for captchas.")
        try:
            self.driver.get(self.anticaptcha)
        except Exception as e:
            pass

    def human_move(self, element):
        """Simulate human-like mouse movement to click an element.
        
        Args:
            element: The web element to click on
            
        The method creates random small mouse movements before clicking
        to appear more human-like and avoid detection as automation.
        """
        actions = ActionChains(self.driver)
        x_offset = random.randint(-5,5)
        for _ in range(random.randint(2,5)):
            actions.move_by_offset(x_offset, random.randint(-2,2))
            actions.pause(random.uniform(0.1,0.3))
        actions.click().perform()

    def human_scroll(self):
        for _ in range(random.randint(1, 3)):
            scroll_pixels = random.randint(150, 1200)
            self.driver.execute_script(f"window.scrollBy(0, {scroll_pixels});")
            time.sleep(random.uniform(0.5, 2.0))
            if random.random() < 0.4:
                self.driver.execute_script(f"window.scrollBy(0, -{random.randint(50, 300)});")
                time.sleep(random.uniform(0.3, 1.0))

    def patch_browser_fingerprint(self) -> None:
        script = self.load_js("spoofing.js")
        self.driver.execute_script(script)
    
    def go_to(self, url:str) -> bool:
        """Navigate to a specified URL."""
        time.sleep(random.uniform(0.4, 2.5))
        try:
            initial_handles = self.driver.window_handles
            self.driver.get(url)
            time.sleep(random.uniform(0.01, 0.3))
            try:
                wait = WebDriverWait(self.driver, timeout=10)
                wait.until(
                    lambda driver: (
                        not any(keyword in driver.page_source.lower() for keyword in ["checking your browser", "captcha"])
                    ),
                    message="stuck on 'checking browser' or verification screen"
                )
            except TimeoutException:
                time.sleep(random.uniform(0.01, 0.2))
            self.human_scroll()
            return True
        except TimeoutException as e:
            return False
        except WebDriverException as e:
            return False
        except Exception as e:
            raise e

    def is_sentence(self, text:str) -> bool:
        """Check if the text qualifies as a meaningful sentence or contains important error codes."""
        text = text.strip()

        if any(c.isdigit() for c in text):
            return True
        words = re.findall(r'\w+', text, re.UNICODE)
        word_count = len(words)
        has_punctuation = any(text.endswith(p) for p in ['.', '，', ',', '!', '?', '。', '！', '？', '।', '۔'])
        is_long_enough = word_count > 4
        return (word_count >= 5 and (has_punctuation or is_long_enough))

    def get_text(self) -> str | None:
        """Get page text as formatted Markdown"""
        try:
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            for element in soup(['script', 'style', 'noscript', 'meta', 'link']):
                element.decompose()
            markdown_converter = markdownify.MarkdownConverter(
                heading_style="ATX",
                strip=['a'],
                autolinks=False,
                bullets='•',
                strong_em_symbol='*',
                default_title=False,
            )
            markdown_text = markdown_converter.convert(str(soup.body))
            lines = []
            for line in markdown_text.splitlines():
                stripped = line.strip()
                if stripped and self.is_sentence(stripped):
                    cleaned = ' '.join(stripped.split())
                    lines.append(cleaned)
            result = "[Start of page]\n\n" + "\n\n".join(lines) + "\n\n[End of page]"
            result = re.sub(r'!\[(.*?)\]\(.*?\)', r'[IMAGE: \1]', result)
            return result[:4096]
        except Exception as e:
            return None
    
    def clean_url(self, url:str) -> str:
        """Clean URL to keep only the part needed for navigation to the page"""
        clean = url.split('#')[0]
        parts = clean.split('?', 1)
        base_url = parts[0]
        if len(parts) > 1:
            query = parts[1]
            essential_params = []
            for param in query.split('&'):
                if param.startswith('_skw=') or param.startswith('q=') or param.startswith('s='):
                    essential_params.append(param)
                elif param.startswith('_') or param.startswith('hash=') or param.startswith('itmmeta='):
                    break
            if essential_params:
                return f"{base_url}?{'&'.join(essential_params)}"
        return base_url
    
    def is_link_valid(self, url:str) -> bool:
        """Check if a URL is a valid link (page, not related to icon or metadata)."""
        if len(url) > 72:
            return False
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            return False
        if re.search(r'/\d+$', parsed_url.path):
            return False
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
        metadata_extensions = ['.ico', '.xml', '.json', '.rss', '.atom']
        for ext in image_extensions + metadata_extensions:
            if url.lower().endswith(ext):
                return False
        return True

    def get_navigable(self) -> List[str]:
        """Get all navigable links on the current page."""
        try:
            links = []
            elements = self.driver.find_elements(By.TAG_NAME, "a")
            
            for element in elements:
                href = element.get_attribute("href")
                if href and href.startswith(("http", "https")):
                    links.append({
                        "url": href,
                        "text": element.text.strip(),
                        "is_displayed": element.is_displayed()
                    })
            
            return [self.clean_url(link['url']) for link in links if (link['is_displayed'] == True and self.is_link_valid(link['url']))]
        except Exception as e:
            return []

    def click_element(self, xpath: str) -> bool:
        """Click an element specified by XPath."""
        try:
            element = self.wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))
            if not element.is_displayed():
                return False
            if not element.is_enabled():
                return False
            try:
                self.driver.execute_script("arguments[0].scrollIntoView({block: 'center', behavior: 'smooth'});", element)
                time.sleep(0.1)
                element.click()
                return True
            except Exception as e:
                return False
        except TimeoutException:
            return False
        except Exception as e:
            return False
        
    def load_js(self, file_name: str) -> str:
        """Load javascript from script folder to inject to page."""
        path = os.path.join(self.js_scripts_folder, file_name)
        try:
            with open(path, 'r') as f:
                return f.read()
        except FileNotFoundError as e:
            raise Exception(f"Could not find: {path}") from e
        except Exception as e:
            raise e

    def find_all_inputs(self, timeout=3):
        """Find all inputs elements on the page."""
        try:
            WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
        except Exception as e:
            return []
        time.sleep(0.5)
        return None

    def get_form_inputs(self) -> List[str]:
        """Extract all input from the page and return them."""
        try:
            input_elements = self.find_all_inputs()
            if not input_elements:
                return ["No input forms found on the page."]

            form_strings = []
            for element in input_elements:
                input_type = element.get("type") or "text"
                if input_type in ["hidden", "submit", "button", "image"] or not element["displayed"]:
                    continue
                input_name = element.get("text") or element.get("id") or input_type
                if input_type == "checkbox" or input_type == "radio":
                    try:
                        checked_status = "checked" if element.is_selected() else "unchecked"
                    except Exception as e:
                        continue
                    form_strings.append(f"[{input_name}]({checked_status})")
                else:
                    form_strings.append(f"[{input_name}]("")")
            return form_strings

        except Exception as e:
            raise e

    def get_buttons_xpath(self) -> List[str]:
        """
        Find buttons and return their type and xpath.
        """
        buttons = self.driver.find_elements(By.TAG_NAME, "button") + \
                  self.driver.find_elements(By.XPATH, "//input[@type='submit']")
        result = []
        for i, button in enumerate(buttons):
            if not button.is_displayed() or not button.is_enabled():
                continue
            text = (button.text or button.get_attribute("value") or "").lower().replace(' ', '')
            xpath = f"(//button | //input[@type='submit'])[{i + 1}]"
            result.append((text, xpath))
        result.sort(key=lambda x: len(x[0]))
        return result

    def wait_for_submission_outcome(self, timeout: int = 10) -> bool:
        """
        Wait for a submission outcome (e.g., URL change or new element).
        """
        try:
            wait = WebDriverWait(self.driver, timeout)
            wait.until(
                lambda driver: driver.current_url != self.driver.current_url or
                               driver.find_elements(By.XPATH, "//*[contains(text(), 'success')]")
            )
            return True
        except TimeoutException:
            return False

    def find_and_click_btn(self, btn_type: str = 'login', timeout: int = 5) -> bool:
        """Find and click a submit button matching the specified type."""
        buttons = self.get_buttons_xpath()
        if not buttons:
            return False

        for button_text, xpath in buttons:
            if btn_type.lower() in button_text.lower() or btn_type.lower() in xpath.lower():
                try:
                    wait = WebDriverWait(self.driver, timeout)
                    element = wait.until(
                        EC.element_to_be_clickable((By.XPATH, xpath)),
                        message=f"Button with XPath '{xpath}' not clickable within {timeout} seconds"
                    )
                    if self.click_element(xpath):
                        return True
                    else:
                        return False
                except TimeoutException:
                    return False
                except Exception as e:
                    return False
        return False

    def tick_all_checkboxes(self) -> bool:
        """
        Find and tick all checkboxes on the page.
        Returns True if successful, False if any issues occur.
        """
        try:
            checkboxes = self.driver.find_elements(By.XPATH, "//input[@type='checkbox']")
            if not checkboxes:
                return True

            for index, checkbox in enumerate(checkboxes, 1):
                try:
                    WebDriverWait(self.driver, 10).until(
                        EC.element_to_be_clickable(checkbox)
                    )
                    self.driver.execute_script(
                        "arguments[0].scrollIntoView({block: 'center', inline: 'center'});", checkbox
                    )
                    if not checkbox.is_selected():
                        try:
                            checkbox.click()
                        except Exception as e:
                            self.driver.execute_script("arguments[0].click();", checkbox)
                except TimeoutException:
                    continue
                except Exception as e:
                    continue
            return True
        except Exception as e:
            return False

    def find_and_click_submission(self, timeout: int = 10) -> bool:
        possible_submissions = ["login", "submit", "register", "continue", "apply",
                                "ok", "confirm", "proceed", "accept", 
                                "done", "finish", "start", "calculate"]
        for submission in possible_submissions:
            if self.find_and_click_btn(submission, timeout):
                return True
        return False
    
    def find_input_xpath_by_name(self, inputs, name: str) -> str | None:
        for field in inputs:
            if name in field["text"]:
                return field["xpath"]
        return None

    def fill_form_inputs(self, input_list: List[str]) -> bool:
        """Fill inputs based on a list of [name](value) strings."""
        if not isinstance(input_list, list):
            return False
        inputs = self.find_all_inputs()
        try:
            for input_str in input_list:
                match = re.match(r'\[(.*?)\]\((.*?)\)', input_str)
                if not match:
                    continue

                name, value = match.groups()
                name = name.strip()
                value = value.strip()
                xpath = self.find_input_xpath_by_name(inputs, name)
                if not xpath:
                    continue
                try:
                    element = WebDriverWait(self.driver, 10).until(
                        EC.element_to_be_clickable((By.XPATH, xpath))
                    )
                except TimeoutException:
                    continue
                self.driver.execute_script("arguments[0].scrollIntoView(true);", element)
                if not element.is_displayed() or not element.is_enabled():
                    continue
                input_type = (element.get_attribute("type") or "text").lower()
                if input_type in ["checkbox", "radio"]:
                    is_checked = element.is_selected()
                    should_be_checked = value.lower() == "checked"

                    if is_checked != should_be_checked:
                        element.click()
                else:
                    element.clear()
                    element.send_keys(value)
            return True
        except Exception as e:
            return False
    
    def fill_form(self, input_list: List[str]) -> bool:
        """Fill form inputs based on a list of [name](value) and submit."""
        if not isinstance(input_list, list):
            return False
        if self.fill_form_inputs(input_list):
            self.tick_all_checkboxes()
            if self.find_and_click_submission():
                if self.wait_for_submission_outcome():
                    return True
        return False

    def get_current_url(self) -> str:
        """Get the current URL of the page."""
        return self.driver.current_url

    def get_page_title(self) -> str:
        """Get the title of the current page."""
        return self.driver.title

    def scroll_bottom(self) -> bool:
        """Scroll to the bottom of the page."""
        try:
            self.driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
            )
            time.sleep(0.5)
            return True
        except Exception as e:
            return False
    
    def get_screenshot(self) -> str:
        return self.screenshot_folder + "/updated_screen.png"

    def screenshot(self, filename:str = 'updated_screen.png') -> bool:
        """Take a screenshot of the current page, attempt to capture the full page by zooming out."""
        time.sleep(0.1)
        try:
            original_zoom = self.driver.execute_script("return document.body.style.zoom || 1;")
            self.driver.execute_script("document.body.style.zoom='75%'")
            time.sleep(0.1)
            path = os.path.join(self.screenshot_folder, filename)
            if not os.path.exists(self.screenshot_folder):
                os.makedirs(self.screenshot_folder)
            self.driver.save_screenshot(path)
        except Exception as e:
            return False
        finally:
            self.driver.execute_script(f"document.body.style.zoom='1'")
        return True

    def apply_web_safety(self):
        """
        Apply security measures to block any website malicious/annoying execution, privacy violation etc..
        """
        script = self.load_js("inject_safety_script.js")
        self.driver.execute_script(script)