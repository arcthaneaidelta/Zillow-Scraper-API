"""
Enhanced Zillow Property Scraper API
===================================
A robust FastAPI for scraping Zillow property listings with advanced anti-detection
and flexible search parameters.

Requirements:
- Python 3.7+
- fastapi
- uvicorn
- httpx
- lxml  
- pandas

Installation:
pip install fastapi uvicorn httpx lxml pandas

Usage:
uvicorn zillow_api:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Literal
import httpx
from lxml import html
import asyncio
import logging
import time
from urllib.parse import urljoin, quote
import random
from datetime import datetime
import json
import base64
import hashlib

# Configure logging to capture HTTP requests and responses
class LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.logs = []
        self.max_logs = 1000  # Keep last 1000 log entries
    
    def emit(self, record):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "message": self.format(record),
            "module": record.module
        }
        self.logs.append(log_entry)
        
        # Keep only recent logs
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs:]

# Initialize log capture
log_capture = LogCapture()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        log_capture,
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add HTTP client logging
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.INFO)
httpx_logger.addHandler(log_capture)

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Zillow Property Scraper API",
    description="A robust REST API for scraping property listings from Zillow.com with advanced search parameters",
    version="2.0.0",
    docs_url="/docs"
)


class PropertyData(BaseModel):
    """Property data model"""
    address: Optional[str] = None
    price: Optional[str] = None
    bedrooms: Optional[str] = None
    bathrooms: Optional[str] = None
    square_feet: Optional[str] = None
    property_type: Optional[str] = None
    listing_url: Optional[str] = None
    zestimate: Optional[str] = None
    days_on_market: Optional[str] = None
    listing_status: Optional[str] = None


class SearchRequest(BaseModel):
    """Request model for POST /search"""
    city: str = Field(..., description="City to scrape (e.g., 'new-york-ny', 'miami-fl')")
    max_properties: int = Field(
        default=30, 
        ge=1, 
        le=200, 
        description="Maximum number of properties to scrape (1-200)"
    )
    min_price: Optional[int] = Field(None, description="Minimum price filter")
    max_price: Optional[int] = Field(None, description="Maximum price filter")
    property_type: Optional[Literal["for_sale", "sold", "for_rent"]] = Field(
        default="for_sale", 
        description="Property listing type"
    )
    map_bounds: Optional[str] = Field(
        None, 
        description="Map bounds in format 'lat1,lng1,lat2,lng2' for geographic filtering"
    )
    
    @validator('city')
    def validate_city(cls, v):
        """Validate and format city"""
        if not v or len(v.strip()) == 0:
            raise ValueError("City cannot be empty")
        return v.lower().replace(' ', '-').replace(',', '').strip()

    @validator('map_bounds')
    def validate_map_bounds(cls, v):
        """Validate map bounds format"""
        if v is None:
            return v
        try:
            coords = v.split(',')
            if len(coords) != 4:
                raise ValueError("Map bounds must have 4 coordinates")
            [float(coord) for coord in coords]  # Validate all are numbers
            return v
        except (ValueError, AttributeError):
            raise ValueError("Map bounds must be in format 'lat1,lng1,lat2,lng2'")


class SearchResponse(BaseModel):
    """Response model for search endpoints"""
    success: bool
    city: str
    properties_requested: int
    properties_found: int
    scraping_time_seconds: float
    properties: List[PropertyData]
    message: str
    search_url: Optional[str] = None


class ZillowURLGenerator:
    """Generate Zillow search URLs based on parameters"""
    
    @staticmethod
    def generate_search_url(
        city: str,
        property_type: str = "for_sale",
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
        map_bounds: Optional[str] = None,
        page: int = 1
    ) -> str:
        """Generate Zillow search URL with all parameters"""
        
        # Base URL structure
        base_url = f"https://www.zillow.com/{city}/"
        
        # Build search query state
        search_state = {
            "pagination": {"currentPage": page},
            "usersSearchTerm": city.replace('-', ' ').title(),
            "mapBounds": {},
            "filterState": {}
        }
        
        # Add property type filter
        if property_type == "sold":
            search_state["filterState"]["isRecentlySold"] = {"value": True}
            search_state["filterState"]["isForSaleByAgent"] = {"value": False}
            search_state["filterState"]["isForSaleByOwner"] = {"value": False}
            search_state["filterState"]["isNewConstruction"] = {"value": False}
            search_state["filterState"]["isComingSoon"] = {"value": False}
            search_state["filterState"]["isAuction"] = {"value": False}
            search_state["filterState"]["isForSaleForeclosure"] = {"value": False}
        elif property_type == "for_rent":
            search_state["filterState"]["isForRent"] = {"value": True}
            search_state["filterState"]["isForSaleByAgent"] = {"value": False}
            search_state["filterState"]["isForSaleByOwner"] = {"value": False}
        else:  # for_sale (default)
            search_state["filterState"]["isForSaleByAgent"] = {"value": True}
            search_state["filterState"]["isForSaleByOwner"] = {"value": True}
            search_state["filterState"]["isNewConstruction"] = {"value": False}
            search_state["filterState"]["isComingSoon"] = {"value": False}
            search_state["filterState"]["isAuction"] = {"value": False}
            search_state["filterState"]["isForSaleForeclosure"] = {"value": False}
        
        # Add price filters
        if min_price or max_price:
            price_filter = {}
            if min_price:
                price_filter["min"] = min_price
            if max_price:
                price_filter["max"] = max_price
            search_state["filterState"]["price"] = price_filter
        
        # Add map bounds if provided
        if map_bounds:
            coords = map_bounds.split(',')
            search_state["mapBounds"] = {
                "west": float(coords[1]),
                "east": float(coords[3]),
                "south": float(coords[0]),
                "north": float(coords[2])
            }
        
        # Convert to JSON and URL encode
        search_query = json.dumps(search_state, separators=(',', ':'))
        encoded_query = quote(search_query)
        
        # Build final URL
        final_url = f"{base_url}?searchQueryState={encoded_query}"
        
        logger.info(f"Generated URL: {final_url}")
        return final_url


class ZillowScraper:
    """Enhanced Zillow scraper with anti-detection measures"""
    
    def __init__(self):
        # Rotate user agents to avoid detection
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
        ]
        
        self.url_generator = ZillowURLGenerator()
        self.min_delay = 2
        self.max_delay = 5
        self.request_timeout = 30

    def get_headers(self) -> Dict[str, str]:
        """Get randomized headers for requests"""
        return {
            "User-Agent": random.choice(self.user_agents),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
            "sec-ch-ua": '"Not A(Brand";v="99", "Google Chrome";v="121", "Chromium";v="121"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"'
        }

    async def create_session(self) -> httpx.AsyncClient:
        """Create HTTP session with enhanced configuration"""
        return httpx.AsyncClient(
            timeout=httpx.Timeout(self.request_timeout),
            follow_redirects=True,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            headers=self.get_headers()
        )

    async def fetch_page_content(self, url: str, session: httpx.AsyncClient) -> Optional[str]:
        """Fetch HTML content from URL with enhanced error handling"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}: Fetching URL: {url}")
                
                # Update headers for each request
                session.headers.update(self.get_headers())
                
                response = await session.get(url)
                
                logger.info(f"Response status: {response.status_code} for {url}")
                
                if response.status_code == 200:
                    content_length = len(response.text)
                    logger.info(f"Successfully fetched {content_length} characters")
                    return response.text
                elif response.status_code == 403:
                    logger.warning(f"403 Forbidden - Anti-bot detection triggered")
                    wait_time = (attempt + 1) * 10
                    await asyncio.sleep(wait_time)
                elif response.status_code == 429:
                    wait_time = (attempt + 1) * 15
                    logger.warning(f"Rate limited. Waiting {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                elif response.status_code == 404:
                    logger.error(f"404 Not Found - Invalid URL or city: {url}")
                    return None
                else:
                    logger.warning(f"HTTP {response.status_code} for {url}")
                    
            except httpx.TimeoutException:
                logger.error(f"Timeout fetching {url}")
            except httpx.ConnectError:
                logger.error(f"Connection error fetching {url}")
            except Exception as e:
                logger.error(f"Error fetching {url}: {str(e)}")
                
            if attempt < max_retries - 1:
                wait_time = random.uniform(3, 8)
                logger.info(f"Waiting {wait_time:.2f} seconds before retry...")
                await asyncio.sleep(wait_time)
                
        return None

    def parse_property_details(self, card_element) -> Dict[str, Optional[str]]:
        """Parse property card details with enhanced extraction"""
        property_data = {
            "address": None,
            "price": None,
            "bedrooms": None,
            "bathrooms": None,
            "square_feet": None,
            "property_type": None,
            "listing_url": None,
            "zestimate": None,
            "days_on_market": None,
            "listing_status": None
        }
        
        try:
            # Extract address - multiple selectors
            address_selectors = [
                './/address/text()',
                './/a/address/text()',
                './/*[@data-test="property-card-addr"]/text()',
                './/*[contains(@class, "address")]/text()'
            ]
            
            for selector in address_selectors:
                address_elements = card_element.xpath(selector)
                if address_elements:
                    property_data["address"] = address_elements[0].strip()
                    break
            
            # Extract price - multiple selectors
            price_selectors = [
                './/span[@data-test="property-card-price"]/text()',
                './/*[contains(@class, "price")]/text()',
                './/*[contains(@data-test, "price")]/text()'
            ]
            
            for selector in price_selectors:
                price_elements = card_element.xpath(selector)
                if price_elements:
                    property_data["price"] = price_elements[0].strip()
                    break
            
            # Extract Zestimate
            zestimate_elements = card_element.xpath('.//*[contains(text(), "Zestimate")]/text()')
            if zestimate_elements:
                zestimate_text = zestimate_elements[0]
                # Extract price from Zestimate text
                import re
                price_match = re.search(r'\$[\d,]+', zestimate_text)
                if price_match:
                    property_data["zestimate"] = price_match.group()
            
            # Extract property details
            details_selectors = [
                './/ul[contains(@class, "StyledPropertyCardHomeDetailsList")]',
                './/ul[contains(@class, "property-card-details")]',
                './/*[contains(@class, "details")]'
            ]
            
            details_container = None
            for selector in details_selectors:
                containers = card_element.xpath(selector)
                if containers:
                    details_container = containers[0]
                    break
            
            if details_container:
                detail_items = details_container.xpath('.//li | .//span')
                
                for item in detail_items:
                    text_content = ''.join(item.xpath('.//text()')).strip().lower()
                    
                    if 'bd' in text_content or 'bed' in text_content:
                        beds = ''.join(filter(str.isdigit, text_content))
                        property_data["bedrooms"] = beds if beds else None
                    elif 'ba' in text_content or 'bath' in text_content:
                        baths = text_content.split()[0] if text_content.split() else None
                        property_data["bathrooms"] = baths
                    elif 'sqft' in text_content or 'sq ft' in text_content:
                        import re
                        sqft_match = re.search(r'[\d,]+', text_content)
                        if sqft_match:
                            property_data["square_feet"] = sqft_match.group().replace(',', '')
                    elif 'days on market' in text_content or 'dom' in text_content:
                        days_match = re.search(r'\d+', text_content)
                        if days_match:
                            property_data["days_on_market"] = days_match.group()
            
            # Extract listing URL
            link_selectors = [
                './/a[@data-test="property-card-link"]/@href',
                './/a[contains(@href, "/homedetails/")]/@href',
                './/a[contains(@class, "property-card-link")]/@href'
            ]
            
            for selector in link_selectors:
                link_elements = card_element.xpath(selector)
                if link_elements:
                    relative_url = link_elements[0]
                    property_data["listing_url"] = urljoin("https://www.zillow.com", relative_url)
                    break
            
            # Determine property type from address
            if property_data["address"]:
                address_lower = property_data["address"].lower()
                if any(keyword in address_lower for keyword in ['apt', 'unit', '#']):
                    property_data["property_type"] = "Apartment"
                elif any(keyword in address_lower for keyword in ['condo', 'condominium']):
                    property_data["property_type"] = "Condo"
                elif any(keyword in address_lower for keyword in ['townhouse', 'townhome']):
                    property_data["property_type"] = "Townhouse"
                else:
                    property_data["property_type"] = "House"
            
            # Extract listing status
            status_elements = card_element.xpath('.//*[contains(@class, "status") or contains(@class, "label")]/text()')
            if status_elements:
                property_data["listing_status"] = status_elements[0].strip()
                    
        except Exception as e:
            logger.warning(f"Error parsing property card: {str(e)}")
            
        return property_data

    async def scrape_properties_from_page(self, html_content: str) -> List[Dict]:
        """Extract properties from HTML content with enhanced parsing"""
        try:
            tree = html.fromstring(html_content)
            
            # Multiple selectors to find property cards
            property_selectors = [
                '//li[contains(@class, "ListItem-c11n")]',
                '//article[contains(@class, "property-card")]',
                '//div[contains(@class, "property-card")]',
                '//li[contains(@class, "result-list-item")]',
                '//div[contains(@data-test, "property-card")]'
            ]
            
            property_cards = []
            for selector in property_selectors:
                cards = tree.xpath(selector)
                if cards:
                    property_cards = cards
                    logger.info(f"Found {len(cards)} property cards using selector: {selector}")
                    break
            
            if not property_cards:
                logger.warning("No property cards found on page")
                # Log page content for debugging
                if len(html_content) < 1000:
                    logger.debug(f"Page content: {html_content[:500]}...")
                return []
            
            properties = []
            for i, card in enumerate(property_cards):
                try:
                    property_data = self.parse_property_details(card)
                    
                    # Only include properties with essential data
                    if property_data["address"] and property_data["price"]:
                        properties.append(property_data)
                        logger.debug(f"Property {i+1}: {property_data['address']} - {property_data['price']}")
                    else:
                        logger.debug(f"Skipping property {i+1}: missing essential data")
                        
                except Exception as e:
                    logger.warning(f"Error parsing property card {i+1}: {str(e)}")
                    continue
            
            logger.info(f"Successfully parsed {len(properties)} valid properties")
            return properties
            
        except Exception as e:
            logger.error(f"Error parsing HTML content: {str(e)}")
            return []

    async def scrape_properties(
        self,
        city: str,
        max_properties: int,
        property_type: str = "for_sale",
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
        map_bounds: Optional[str] = None
    ) -> tuple[List[Dict], str]:
        """Main scraping function with enhanced parameters"""
        start_time = time.time()
        logger.info(f"Starting scrape for {city} (max {max_properties} properties)")
        logger.info(f"Filters - Type: {property_type}, Price: {min_price}-{max_price}, Bounds: {map_bounds}")
        
        scraped_properties = []
        search_url = None
        
        try:
            async with await self.create_session() as session:
                page_number = 1
                consecutive_empty_pages = 0
                max_empty_pages = 3
                max_pages = 15  # Increased for better coverage
                
                while len(scraped_properties) < max_properties and page_number <= max_pages:
                    
                    # Generate URL for current page
                    page_url = self.url_generator.generate_search_url(
                        city=city,
                        property_type=property_type,
                        min_price=min_price,
                        max_price=max_price,
                        map_bounds=map_bounds,
                        page=page_number
                    )
                    
                    if page_number == 1:
                        search_url = page_url  # Store first page URL for response
                    
                    html_content = await self.fetch_page_content(page_url, session)
                    
                    if not html_content:
                        consecutive_empty_pages += 1
                        logger.warning(f"Failed to fetch page {page_number}")
                        if consecutive_empty_pages >= max_empty_pages:
                            logger.info("Too many failed pages, stopping scrape")
                            break
                        page_number += 1
                        continue
                    
                    page_properties = await self.scrape_properties_from_page(html_content)
                    
                    if not page_properties:
                        consecutive_empty_pages += 1
                        logger.warning(f"No properties found on page {page_number}")
                        if consecutive_empty_pages >= max_empty_pages:
                            logger.info("No more properties found, ending scrape")
                            break
                    else:
                        consecutive_empty_pages = 0
                        scraped_properties.extend(page_properties)
                        logger.info(f"Page {page_number}: Added {len(page_properties)} properties "
                                  f"(Total: {len(scraped_properties)})")
                    
                    if len(scraped_properties) >= max_properties:
                        logger.info(f"Reached target of {max_properties} properties")
                        break
                    
                    page_number += 1
                    
                    # Enhanced rate limiting with randomization
                    delay = random.uniform(self.min_delay, self.max_delay)
                    logger.info(f"Waiting {delay:.2f} seconds before next request...")
                    await asyncio.sleep(delay)
            
            # Trim to max_properties
            if len(scraped_properties) > max_properties:
                scraped_properties = scraped_properties[:max_properties]
            
            end_time = time.time()
            scraping_time = end_time - start_time
            logger.info(f"Scraping completed in {scraping_time:.2f} seconds. Found {len(scraped_properties)} properties")
            
            return scraped_properties, search_url
            
        except Exception as e:
            logger.error(f"Scraping failed: {str(e)}")
            raise


# Initialize scraper
scraper = ZillowScraper()


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Enhanced Zillow Property Scraper API",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "Anti-detection measures for production deployment",
            "Advanced search filters (price, type, map bounds)",
            "Enhanced property data extraction",
            "Robust error handling and retry logic"
        ],
        "endpoints": {
            "GET /": "API information",
            "GET /search": "Search properties with query parameters",
            "POST /search": "Search properties with JSON body",
            "GET /logs": "View API logs",
            "GET /health": "Health check",
            "GET /docs": "API documentation"
        },
        "search_parameters": {
            "city": "City slug (e.g., 'miami-fl')",
            "max_properties": "Max properties to return (1-200)",
            "min_price": "Minimum price filter",
            "max_price": "Maximum price filter", 
            "property_type": "Type: 'for_sale', 'sold', 'for_rent'",
            "map_bounds": "Geographic bounds: 'lat1,lng1,lat2,lng2'"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for deployment platforms"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Enhanced Zillow Property Scraper API",
        "version": "2.0.0"
    }


@app.get("/search", response_model=SearchResponse)
async def search_properties_get(
    city: str = Query(..., description="City to scrape (e.g., 'miami-fl', 'new-york-ny')"),
    max_properties: int = Query(30, ge=1, le=200, description="Maximum properties to scrape (1-200)"),
    min_price: Optional[int] = Query(None, description="Minimum price filter"),
    max_price: Optional[int] = Query(None, description="Maximum price filter"),
    property_type: Literal["for_sale", "sold", "for_rent"] = Query("for_sale", description="Property listing type"),
    map_bounds: Optional[str] = Query(None, description="Map bounds: 'lat1,lng1,lat2,lng2'")
):
    """Search for properties using query parameters"""
    start_time = time.time()
    
    try:
        # Format city name
        formatted_city = city.lower().replace(' ', '-').replace(',', '').strip()
        logger.info(f"GET /search - City: {formatted_city}, Type: {property_type}, Max: {max_properties}")
        
        # Scrape properties
        properties, search_url = await scraper.scrape_properties(
            city=formatted_city,
            max_properties=max_properties,
            property_type=property_type,
            min_price=min_price,
            max_price=max_price,
            map_bounds=map_bounds
        )
        
        scraping_time = time.time() - start_time
        
        # Convert to PropertyData models
        property_objects = [PropertyData(**prop) for prop in properties]
        
        return SearchResponse(
            success=True,
            city=formatted_city,
            properties_requested=max_properties,
            properties_found=len(properties),
            scraping_time_seconds=round(scraping_time, 2),
            properties=property_objects,
            message=f"Successfully scraped {len(properties)} properties from {formatted_city}",
            search_url=search_url
        )
        
    except Exception as e:
        logger.error(f"GET /search failed: {str(e)}")
        scraping_time = time.time() - start_time
        
        return SearchResponse(
            success=False,
            city=city,
            properties_requested=max_properties,
            properties_found=0,
            scraping_time_seconds=round(scraping_time, 2),
            properties=[],
            message=f"Scraping failed: {str(e)}"
        )


@app.post("/search", response_model=SearchResponse)
async def search_properties_post(request: SearchRequest):
    """Search for properties using JSON request body"""
    start_time = time.time()
    
    try:
        logger.info(f"POST /search - City: {request.city}, Type: {request.property_type}, Max: {request.max_properties}")
        
        # Scrape properties
        properties, search_url = await scraper.scrape_properties(
            city=request.city,
            max_properties=request.max_properties,
            property_type=request.property_type,
            min_price=request.min_price,
            max_price=request.max_price,
            map_bounds=request.map_bounds
        )
        
        scraping_time = time.time() - start_time
        
        # Convert to PropertyData models
        property_objects = [PropertyData(**prop) for prop in properties]
        
        return SearchResponse(
            success=True,
            city=request.city,
            properties_requested=request.max_properties,
            properties_found=len(properties),
            scraping_time_seconds=round(scraping_time, 2),
            properties=property_objects,
            message=f"Successfully scraped {len(properties)} properties from {request.city}",
            search_url=search_url
        )
        
    except Exception as e:
        logger.error(f"POST /search failed: {str(e)}")
        scraping_time = time.time() - start_time
        
        return SearchResponse(
            success=False,
            city=request.city,
            properties_requested=request.max_properties,
            properties_found=0,
            scraping_time_seconds=round(scraping_time, 2),
            properties=[],
            message=f"Scraping failed: {str(e)}"
        )


@app.get("/logs")
async def get_logs(limit: int = Query(100, ge=1, le=1000, description="Number of recent logs to return")):
    """Get recent API logs"""
    recent_logs = log_capture.logs[-limit:] if log_capture.logs else []
    
    return {
        "total_logs": len(log_capture.logs),
        "returned_logs": len(recent_logs),
        "logs": recent_logs
    }


@app.get("/test-url")
async def test_url_generation(
    city: str = Query(..., description="City to test"),
    property_type: Literal["for_sale", "sold", "for_rent"] = Query("for_sale"),
    min_price: Optional[int] = Query(None),
    max_price: Optional[int] = Query(None),
    map_bounds: Optional[str] = Query(None)
):
    """Test URL generation without scraping"""
    try:
        url_generator = ZillowURLGenerator()
        generated_url = url_generator.generate_search_url(
            city=city,
            property_type=property_type,
            min_price=min_price,
            max_price=max_price,
            map_bounds=map_bounds
        )
        
        return {
            "success": True,
            "city": city,
            "parameters": {
                "property_type": property_type,
                "min_price": min_price,
                "max_price": max_price,
                "map_bounds": map_bounds
            },
            "generated_url": generated_url
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
