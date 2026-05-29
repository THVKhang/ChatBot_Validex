"""SEO Optimizer (0 Token Python Native).

Extracts meta keywords, generates URL slugs, and checks for keyword stuffing
in the generated blog post using native Python math.
"""

import re
from collections import Counter
import string

# Stop words to ignore during TF calculation
_STOP_WORDS = {
    "the", "and", "a", "to", "of", "in", "i", "is", "that", "it", "on", "you", 
    "this", "for", "but", "with", "are", "have", "be", "at", "or", "as", "was", 
    "so", "if", "out", "not", "an", "what", "can", "about", "we", "your", "by",
    "will", "how", "which", "there", "when", "their", "more", "they", "from", "has"
}

def _clean_word(word: str) -> str:
    """Strip punctuation and lowercase."""
    return word.strip(string.punctuation).lower()

def extract_meta_keywords(text: str, top_n: int = 5) -> list[str]:
    """Extract top N keywords using simple term frequency (TF)."""
    words = [_clean_word(w) for w in text.split()]
    words = [w for w in words if w and w not in _STOP_WORDS and len(w) > 3]
    
    # Also extract bigrams (2-word phrases) since they are often better keywords
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    
    # Combine and count
    all_terms = words + bigrams
    counter = Counter(all_terms)
    
    # Filter out bigrams that occur only once
    filtered_terms = {term: count for term, count in counter.items() if len(term.split()) == 1 or count > 1}
    
    # Return top N
    return [term for term, count in sorted(filtered_terms.items(), key=lambda x: x[1], reverse=True)[:top_n]]

def generate_slug(title: str) -> str:
    """Generate URL-friendly slug from title."""
    # Convert to lowercase and replace non-alphanumeric chars with hyphen
    slug = re.sub(r'[^a-z0-9]+', '-', title.lower())
    return slug.strip('-')

def check_keyword_stuffing(text: str) -> list[str]:
    """Check if any keyword constitutes more than 5% of the text.
    
    Returns a list of warnings.
    """
    words = [_clean_word(w) for w in text.split()]
    total_words = len(words)
    if total_words < 50:
        return []
        
    meaningful_words = [w for w in words if w and w not in _STOP_WORDS and len(w) > 3]
    counter = Counter(meaningful_words)
    
    warnings = []
    for word, count in counter.items():
        density = count / total_words
        # Keyword stuffing typically considered > 5% for a single word
        if density > 0.05:
            pct = round(density * 100, 1)
            warnings.append(f"Keyword stuffing risk: '{word}' appears {count} times ({pct}% density).")
            
    return warnings

def run_seo_analysis(title: str, draft: str) -> dict:
    """Run full SEO analysis on the generated blog."""
    if not draft:
        return {}
        
    return {
        "url_slug": generate_slug(title),
        "meta_keywords": extract_meta_keywords(title + " " + draft),
        "seo_warnings": check_keyword_stuffing(draft)
    }
