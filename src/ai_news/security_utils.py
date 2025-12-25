"""Security utilities for safe input handling and validation."""

import re
import urllib.parse
import ipaddress
from typing import Optional, List, Set
import urllib.request
import ssl
from urllib.request import Request

# HTML sanitization with bleach
try:
    import bleach
    BLEACH_AVAILABLE = True
except ImportError as e:
    BLEACH_AVAILABLE = False
    print(f"Warning: bleach not available: {e}. HTML sanitization will be limited.")

# Secure XML parsing with defusedxml
try:
    from defusedxml.ElementTree import fromstring as safe_fromstring, parse as safe_parse
    from defusedxml import ElementTree as SafeET
    DEFUSEDXML_AVAILABLE = True
except ImportError as e:
    DEFUSEDXML_AVAILABLE = False
    print(f"Warning: defusedxml not available: {e}. XML parsing may be vulnerable.")
    import xml.etree.ElementTree as ET


class URLValidator:
    """Validates URLs to prevent SSRF attacks."""
    
    # Private IP ranges to block
    PRIVATE_IP_RANGES = [
        '127.0.0.0/8',     # localhost
        '10.0.0.0/8',      # Private network
        '172.16.0.0/12',   # Private network
        '192.168.0.0/16',  # Private network
        '169.254.0.0/16',  # Link-local
        '224.0.0.0/4',     # Multicast
        '240.0.0.0/4',     # Reserved
        '0.0.0.0/8',       # This network
    ]
    
    # Allowed schemes
    ALLOWED_SCHEMES = {'http', 'https'}
    
    # Blocked domains
    BLOCKED_DOMAINS = {
        'localhost',
        '127.0.0.1',
        '0.0.0.0',
        'metadata.google.internal',
        '169.254.169.254',  # AWS metadata
        'metadata',
        'instance-data'
    }
    
    # Suspicious TLDs and patterns
    SUSPICIOUS_PATTERNS = [
        r'\.local$',
        r'\.localhost$',
        r'\.internal$',
        r'\.corp$',
        r'\.private$',
    ]
    
    def __init__(self):
        """Initialize URL validator."""
        self._private_networks = []
        for cidr in self.PRIVATE_IP_RANGES:
            try:
                self._private_networks.append(ipaddress.ip_network(cidr))
            except ValueError:
                continue
    
    def _is_private_ip(self, ip_str: str) -> bool:
        """Check if IP address is in private range."""
        try:
            ip = ipaddress.ip_address(ip_str)
            for network in self._private_networks:
                if ip in network:
                    return True
            return False
        except ValueError:
            return False
    
    def _resolve_hostname(self, hostname: str) -> Optional[str]:
        """Resolve hostname to IP address."""
        try:
            import socket
            # Use timeout to prevent hanging
            socket.setdefaulttimeout(5)
            ip = socket.gethostbyname(hostname)
            return ip
        except Exception:
            return None
    
    def validate_url(self, url: str) -> tuple[bool, Optional[str]]:
        """Validate URL for security.
        
        Args:
            url: URL to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not url or not isinstance(url, str):
            return False, "URL must be a non-empty string"
        
        try:
            # Parse URL
            parsed = urllib.parse.urlparse(url)
            
            # Check scheme
            if parsed.scheme not in self.ALLOWED_SCHEMES:
                return False, f"Scheme '{parsed.scheme}' not allowed. Only {self.ALLOWED_SCHEMES} are permitted"
            
            # Check hostname
            hostname = parsed.hostname
            if not hostname:
                return False, "URL must have a valid hostname"
            
            # Check for blocked domains
            hostname_lower = hostname.lower()
            for blocked in self.BLOCKED_DOMAINS:
                if blocked in hostname_lower or hostname_lower == blocked:
                    return False, f"Domain '{hostname}' is blocked"
            
            # Check for suspicious patterns
            for pattern in self.SUSPICIOUS_PATTERNS:
                if re.search(pattern, hostname_lower):
                    return False, f"Domain '{hostname}' matches suspicious pattern"
            
            # Check if hostname is an IP address
            try:
                ip = ipaddress.ip_address(hostname)
                # Direct IP access
                if self._is_private_ip(str(ip)):
                    return False, f"Direct access to private IP '{hostname}' is not allowed"
            except ValueError:
                # Hostname is not an IP, resolve it
                resolved_ip = self._resolve_hostname(hostname)
                if resolved_ip and self._is_private_ip(resolved_ip):
                    return False, f"Hostname '{hostname}' resolves to private IP '{resolved_ip}'"
            
            # Check for suspicious URL patterns
            if '@' in url:  # Potential credential injection
                return False, "URL contains potentially dangerous characters"
            
            # Check port ranges (block common internal service ports)
            if parsed.port:
                if parsed.port in [22, 23, 25, 53, 135, 137, 138, 139, 445, 993, 995]:
                    return False, f"Port {parsed.port} is not allowed"
            
            return True, None
            
        except Exception as e:
            return False, f"URL validation failed: {str(e)}"
    
    def safe_urlopen(self, url: str, headers: Optional[dict] = None, timeout: int = 30) -> Optional[object]:
        """Safely open URL with validation.
        
        Args:
            url: URL to open
            headers: HTTP headers
            timeout: Request timeout
            
        Returns:
            Response object or None if validation fails
        """
        # Validate URL first
        is_valid, error = self.validate_url(url)
        if not is_valid:
            print(f"URL validation failed: {error}")
            return None
        
        try:
            # Create secure request
            req = Request(url, headers=headers or {})
            
            # Use SSL context for HTTPS
            context = ssl.create_default_context()
            context.check_hostname = True
            context.verify_mode = ssl.CERT_REQUIRED
            
            # Make request
            response = urllib.request.urlopen(req, timeout=timeout, context=context)
            return response
            
        except Exception as e:
            print(f"Failed to open URL {url}: {e}")
            return None


class HTMLSanitizer:
    """HTML sanitization utilities."""
    
    # Allowed HTML tags (safe subset)
    ALLOWED_TAGS = {
        'p', 'br', 'strong', 'em', 'u', 'i', 'b',
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'ul', 'ol', 'li', 'blockquote',
        'code', 'pre',
        'div', 'span'
    }
    
    # Allowed attributes
    ALLOWED_ATTRIBUTES = {
        '*': ['class'],
        'a': ['href', 'title'],
        'img': ['src', 'alt', 'title', 'width', 'height'],
    }
    
    # Schemes allowed in URLs
    ALLOWED_SCHEMES = {'http', 'https', 'mailto'}
    
    def __init__(self):
        """Initialize HTML sanitizer."""
        if not BLEACH_AVAILABLE:
            print("Warning: Using fallback HTML sanitization. Install bleach for better security.")
    
    def sanitize_html(self, html_content: str) -> str:
        """Sanitize HTML content to prevent XSS.
        
        Args:
            html_content: HTML content to sanitize
            
        Returns:
            Sanitized HTML content
        """
        if not html_content:
            return ""
        
        # Preprocess to remove dangerous content completely
        cleaned_content = self._remove_dangerous_content(html_content)
        
        if BLEACH_AVAILABLE:
            try:
                # Use bleach for proper sanitization
                return bleach.clean(
                    cleaned_content,
                    tags=self.ALLOWED_TAGS,
                    attributes=self.ALLOWED_ATTRIBUTES,
                    strip=True,
                    strip_comments=True
                )
            except Exception as e:
                print(f"Bleach sanitization failed: {e}")
                # Fall back to basic sanitization
        
        # Fallback sanitization (basic)
        return self._basic_html_sanitization(cleaned_content)
    
    def _remove_dangerous_content(self, html_content: str) -> str:
        """Remove dangerous content completely before sanitization."""
        if not html_content:
            return ""
        
        content = html_content
        
        # Remove script tags and their content completely
        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove style tags and their content
        content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove iframe tags and their content
        content = re.sub(r'<iframe[^>]*>.*?</iframe>', '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove object, embed, and applet tags and their content
        content = re.sub(r'<(object|embed|applet)[^>]*>.*?</\1>', '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove meta tags that might be dangerous
        content = re.sub(r'<meta[^>]*>', '', content, flags=re.IGNORECASE)
        
        # Remove link tags (potential for dangerous CSS)
        content = re.sub(r'<link[^>]*>', '', content, flags=re.IGNORECASE)
        
        # Remove dangerous event handlers from all remaining tags
        content = re.sub(r'\son\w+\s*=\s*["\'][^"\']*["\']', '', content, flags=re.IGNORECASE)
        content = re.sub(r'\son\w+\s*=\s*[^\s>]*', '', content, flags=re.IGNORECASE)
        
        # Remove dangerous protocols from attributes
        content = re.sub(r'(javascript|vbscript|data|file|ftp):[^\s"\'>]*', '', content, flags=re.IGNORECASE)
        
        return content
    
    def _basic_html_sanitization(self, html_content: str) -> str:
        """Basic HTML sanitization fallback."""
        import html as html_module
        
        # Remove dangerous elements and attributes first
        content = html_content
        
        # Use the dangerous content removal method
        content = self._remove_dangerous_content(content)
        
        # Remove all remaining HTML tags (conservative approach)
        content = re.sub(r'<[^>]+>', ' ', content)
        
        # Handle common problematic patterns
        content = re.sub(r'javascript:', '', content, flags=re.IGNORECASE)
        content = re.sub(r'vbscript:', '', content, flags=re.IGNORECASE)
        content = re.sub(r'on\w+\s*=', '', content, flags=re.IGNORECASE)
        
        # Decode HTML entities
        content = html_module.unescape(content)
        
        # Clean up whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        
        return content
    
    def clean_text_content(self, text: str) -> str:
        """Clean text content by removing HTML and normalizing."""
        if not text:
            return ""
        
        # First sanitize if it contains HTML
        if '<' in text and '>' in text:
            text = self.sanitize_html(text)
        
        # Remove any remaining HTML-like content
        text = re.sub(r'<[^>]*>', '', text)
        
        # Decode entities
        import html as html_module
        text = html_module.unescape(text)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


class SecureXMLParser:
    """Secure XML parsing utilities."""
    
    def __init__(self):
        """Initialize secure XML parser."""
        if not DEFUSEDXML_AVAILABLE:
            print("Warning: Using fallback XML parsing. Install defusedxml for better security.")
    
    def safe_fromstring(self, xml_content: str, source_url: str = "") -> Optional[object]:
        """Safely parse XML from string.

        Args:
            xml_content: XML content to parse
            source_url: Source URL for error reporting

        Returns:
            Parsed XML element or None if parsing fails
        """
        if not xml_content:
            return None

        # Check if content is HTML (blocking page) before attempting XML parsing
        content_lower = xml_content.lower().strip()
        
        # More lenient check: reject ONLY if it's clearly an HTML document
        # Valid RSS/Atom feeds start with <?xml or <rss or <feed or <atom
        if content_lower.startswith('<!doctype html>') or (content_lower.startswith('<html>') and not content_lower.startswith('<?xml')):
            url_hint = f" from {source_url}" if source_url else ""
            print(f"⚠ Received HTML instead of XML{url_hint} (likely blocked/403)")
            return None

        if DEFUSEDXML_AVAILABLE:
            try:
                return safe_fromstring(xml_content)
            except Exception as e:
                error_msg = str(e)
                url_hint = f" from {source_url}" if source_url else ""
                if 'not well-formed' in error_msg:
                    print(f"⚠ XML parsing failed{url_hint} (received non-XML response - site may be blocking requests)")
                else:
                    print(f"⚠ XML parsing failed{url_hint}: {e}")
                return None
        else:
            # Fallback with basic protection
            return self._fallback_xml_parse(xml_content)
    
    def safe_parse(self, xml_content: str) -> Optional[object]:
        """Safely parse XML from string with full features."""
        if not xml_content:
            return None
        
        if DEFUSEDXML_AVAILABLE:
            try:
                from io import BytesIO
                return safe_parse(BytesIO(xml_content.encode('utf-8')))
            except Exception as e:
                print(f"Secure XML parsing failed: {e}")
                return None
        else:
            # Fallback
            return self._fallback_xml_parse(xml_content)
    
    def _fallback_xml_parse(self, xml_content: str) -> Optional[object]:
        """Fallback XML parsing with basic protection."""
        try:
            # Basic entity filtering
            xml_content = re.sub(r'<!ENTITY[^>]*>', '', xml_content, flags=re.IGNORECASE)
            xml_content = re.sub(r'&[^;\s]+;', '', xml_content)
            
            # Remove potentially dangerous DTD declarations
            xml_content = re.sub(r'<!DOCTYPE[^>]*>', '', xml_content, flags=re.IGNORECASE)
            
            # Parse with standard library
            root = ET.fromstring(xml_content)
            return root
            
        except Exception as e:
            print(f"Fallback XML parsing failed: {e}")
            return None


# Global instances for convenience
url_validator = URLValidator()
html_sanitizer = HTMLSanitizer()
xml_parser = SecureXMLParser()


def validate_url(url: str) -> tuple[bool, Optional[str]]:
    """Validate a URL for security."""
    return url_validator.validate_url(url)


def safe_urlopen(url: str, headers: Optional[dict] = None, timeout: int = 30) -> Optional[object]:
    """Safely open a URL."""
    return url_validator.safe_urlopen(url, headers, timeout)


def sanitize_html(html_content: str) -> str:
    """Sanitize HTML content."""
    return html_sanitizer.sanitize_html(html_content)


def clean_text_content(text: str) -> str:
    """Clean text content by removing HTML."""
    return html_sanitizer.clean_text_content(text)


def parse_xml_safe(xml_content: str, source_url: str = "") -> Optional[object]:
    """Safely parse XML content."""
    return xml_parser.safe_fromstring(xml_content, source_url)
