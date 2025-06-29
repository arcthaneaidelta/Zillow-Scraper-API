
"""
Zillow Property Scraper Simple API
=================================
A streamlined FastAPI for scraping Zillow property listings with JSON responses only.

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
from typing import List, Dict, Optional
import httpx
from lxml import html
import asyncio
import logging
import time
from urllib.parse import urljoin
import random
from datetime import datetime
import io
import sys

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
    title="Zillow Property Scraper API",
    description="A simple REST API for scraping property listings from Zillow.com",
    version="1.0.0",
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


class SearchRequest(BaseModel):
    """Request model for POST /search"""
    city: str = Field(..., description="City to scrape (e.g., 'new-york-ny', 'miami-fl')")
    max_properties: int = Field(
        default=30, 
        ge=1, 
        le=200, 
        description="Maximum number of properties to scrape (1-200)"
    )
    
    @validator('city')
    def validate_city(cls, v):
        """Validate and format city"""
        if not v or len(v.strip()) == 0:
            raise ValueError("City cannot be empty")
        return v.lower().replace(' ', '-').replace(',', '').strip()


class SearchResponse(BaseModel):
    """Response model for search endpoints"""
    success: bool
    city: str
    properties_requested: int
    properties_found: int
    scraping_time_seconds: float
    properties: List[PropertyData]
    message: str


class ZillowScraper:
    """Simplified Zillow scraper for API usage"""
    
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Cache-Control": "max-age=0"
        }
        
        self.min_delay = 1
        self.max_delay = 3
        self.request_timeout = 20

    async def fetch_page_content(self, url: str, session: httpx.AsyncClient) -> Optional[str]:
        """Fetch HTML content from URL"""
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Fetching URL: {url}")
                
                response = await session.get(
                    url, 
                    headers=self.headers,
                    timeout=self.request_timeout,
                    follow_redirects=True
                )
                
                logger.info(f"Response status: {response.status_code} for {url}")
                
                if response.status_code == 200:
                    return response.text
                elif response.status_code == 429:
                    wait_time = (attempt + 1) * 5
                    logger.warning(f"Rate limited. Waiting {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.warning(f"HTTP {response.status_code} for {url}")
                    
            except Exception as e:
                logger.error(f"Error fetching {url}: {str(e)}")
                
            if attempt < max_retries - 1:
                await asyncio.sleep(random.uniform(1, 3))
                
        return None

    def parse_property_details(self, card_element) -> Dict[str, Optional[str]]:
        """Parse property card details"""
        property_data = {
            "address": None,
            "price": None,
            "bedrooms": None,
            "bathrooms": None,
            "square_feet": None,
            "property_type": None,
            "listing_url": None
        }
        
        try:
            # Extract address
            address_elements = card_element.xpath('.//address/text() | .//a/address/text()')
            if address_elements:
                property_data["address"] = address_elements[0].strip()
            
            # Extract price
            price_elements = card_element.xpath('.//span[@data-test="property-card-price"]/text()')
            if price_elements:
                property_data["price"] = price_elements[0].strip()
            
            # Extract property details
            details_container = card_element.xpath('.//ul[contains(@class, "StyledPropertyCardHomeDetailsList")]')
            if details_container:
                detail_items = details_container[0].xpath('.//li')
                
                for item in detail_items:
                    text_content = ''.join(item.xpath('.//text()')).strip()
                    
                    if 'bd' in text_content.lower() or 'bed' in text_content.lower():
                        beds = ''.join(filter(str.isdigit, text_content))
                        property_data["bedrooms"] = beds if beds else None
                    elif 'ba' in text_content.lower() or 'bath' in text_content.lower():
                        baths = text_content.split()[0] if text_content.split() else None
                        property_data["bathrooms"] = baths
                    elif 'sqft' in text_content.lower() or 'sq ft' in text_content.lower():
                        sqft = ''.join(filter(lambda x: x.isdigit() or x == ',', text_content))
                        property_data["square_feet"] = sqft.replace(',', '') if sqft else None
            
            # Extract listing URL
            link_elements = card_element.xpath('.//a[@data-test="property-card-link"]/@href')
            if link_elements:
                relative_url = link_elements[0]
                property_data["listing_url"] = urljoin("https://www.zillow.com", relative_url)
            
            # Determine property type
            if property_data["address"]:
                address_lower = property_data["address"].lower()
                if any(keyword in address_lower for keyword in ['apt', 'unit', '#']):
                    property_data["property_type"] = "Apartment"
                elif any(keyword in address_lower for keyword in ['condo', 'condominium']):
                    property_data["property_type"] = "Condo"
                else:
                    property_data["property_type"] = "House"
                    
        except Exception as e:
            logger.warning(f"Error parsing property card: {str(e)}")
            
        return property_data

    async def scrape_properties_from_page(self, html_content: str) -> List[Dict]:
        """Extract properties from HTML content"""
        try:
            tree = html.fromstring(html_content)
            
            property_selectors = [
                '//li[contains(@class, "ListItem-c11n")]',
                '//article[contains(@class, "property-card")]',
                '//div[contains(@class, "property-card")]'
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
                return []
            
            properties = []
            for card in property_cards:
                property_data = self.parse_property_details(card)
                
                if property_data["address"] and property_data["price"]:
                    properties.append(property_data)
            
            logger.info(f"Successfully parsed {len(properties)} valid properties")
            return properties
            
        except Exception as e:
            logger.error(f"Error parsing HTML content: {str(e)}")
            return []

    async def scrape_properties(self, city: str, max_properties: int) -> List[Dict]:
        """Main scraping function"""
        start_time = time.time()
        logger.info(f"Starting scrape for {city} (max {max_properties} properties)")
        
        base_url = f"https://www.zillow.com/{city}/"
        scraped_properties = []
        
        try:
            async with httpx.AsyncClient() as session:
                page_number = 1
                consecutive_empty_pages = 0
                max_empty_pages = 3
                max_pages = 10  # Limit to prevent infinite loops
                
                while len(scraped_properties) < max_properties and page_number <= max_pages:
                    
                    page_url = f"{base_url}?searchQueryState=%7B%22pagination%22%3A%7B%22currentPage%22%3A{page_number}%7D%7D"
                    
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
                    
                    # Rate limiting
                    delay = random.uniform(self.min_delay, self.max_delay)
                    logger.info(f"Waiting {delay:.2f} seconds before next request...")
                    await asyncio.sleep(delay)
            
            # Trim to max_properties
            if len(scraped_properties) > max_properties:
                scraped_properties = scraped_properties[:max_properties]
            
            end_time = time.time()
            scraping_time = end_time - start_time
            logger.info(f"Scraping completed in {scraping_time:.2f} seconds. Found {len(scraped_properties)} properties")
            
            return scraped_properties
            
        except Exception as e:
            logger.error(f"Scraping failed: {str(e)}")
            raise


# Initialize scraper
scraper = ZillowScraper()


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Zillow Property Scraper API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "GET /": "API information",
            "GET /search": "Search properties with query parameters",
            "POST /search": "Search properties with JSON body",
            "GET /logs": "View API logs",
            "GET /docs": "API documentation"
        },
        "usage": {
            "GET": "/search?city=miami-fl&max_properties=20",
            "POST": "Body: {\"city\": \"miami-fl\", \"max_properties\": 20}"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for deployment platforms"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Zillow Property Scraper API"
    }

@app.get("/search", response_model=SearchResponse)
async def search_properties_get(
    city: str = Query(..., description="City to scrape (e.g., 'miami-fl', 'new-york-ny')"),
    max_properties: int = Query(30, ge=1, le=200, description="Maximum properties to scrape (1-200)")
):
    """Search for properties using query parameters"""
    start_time = time.time()
    
    try:
        # Format city name
        formatted_city = city.lower().replace(' ', '-').replace(',', '').strip()
        logger.info(f"GET /search - City: {formatted_city}, Max: {max_properties}")
        
        # Scrape properties
        properties = await scraper.scrape_properties(formatted_city, max_properties)
        
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
            message=f"Successfully scraped {len(properties)} properties from {formatted_city}"
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
        logger.info(f"POST /search - City: {request.city}, Max: {request.max_properties}")
        
        # Scrape properties
        properties = await scraper.scrape_properties(request.city, request.max_properties)
        
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
            message=f"Successfully scraped {len(properties)} properties from {request.city}"
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

# The /docs endpoint is automatically created by FastAPI

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
