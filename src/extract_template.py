import os
import re
import json
from glob import glob
import pdfplumber
import pandas as pd
from datetime import datetime

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PDF_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Valid transaction categories
CATEGORIES = {
    'Restaurants', 'Retail and Grocery', 'Health and Education',
    'Professional and Financial Services', 'Personal and Household Expenses',
    'Foreign Currency Transactions', 'Transportation', 
    'Home and Office Improvement', 'Cash Advances and Balance Transfers',
    'Entertainment', 'Payment'
}


def extract_text_from_pdf(pdf_path):
    """Read all text from PDF file"""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            if page_text := page.extract_text():
                text += page_text + "\n"
    return text


def get_year(text):
    """Find the statement year"""
    patterns = [
        r'statement period\s+[A-Za-z]+\s+\d{1,2}\s+to\s+[A-Za-z]+\s+\d{1,2},?\s+(\d{4})',
        r'Statement Date\s+[A-Za-z]+\s+\d{1,2},?\s+(\d{4})',
        r'Prepared for:.*?(\d{4})',
    ]
    
    for pattern in patterns:
        if match := re.search(pattern, text, re.IGNORECASE | re.DOTALL):
            return match.group(1)
    
    return str(datetime.now().year)


def parse_date(month, day, year):
    """Convert month/day/year to YYYY-MM-DD format"""
    try:
        date_str = f"{month} {day} {year}"
        dt = datetime.strptime(date_str, "%b %d %Y")
        return dt.strftime("%Y-%m-%d")
    except:
        return f"{year}-??-??"


def extract_payments(text, year, card_number, source_pdf):
    """Extract payment transactions"""
    payments = []
    patterns = [
        r'^([A-Z][a-z]{2})\s+(\d{1,2})\s+([A-Z][a-z]{2})\s+(\d{1,2})\s+PAYMENT\s+THANK\s+YOU\s*/?\s*PAIEMENT\s+MERCI\s+(\d{1,3}(?:,\d{3})*\.\d{2})\s*$',
        r'^([A-Z][a-z]{2})\s+(\d{1,2})\s+([A-Z][a-z]{2})\s+(\d{1,2})\s+PAYMENT.*?MERCI\s+(\d{1,3}(?:,\d{3})*\.\d{2})\s*$',
    ]
    
    for line in text.split('\n'):
        line = line.strip()
        if 'PAYMENT' not in line:
            continue
        
        for pattern in patterns:
            if match := re.match(pattern, line, re.IGNORECASE):
                payments.append({
                    'transaction_date': parse_date(match.group(1), match.group(2), year),
                    'transaction_postdate': parse_date(match.group(3), match.group(4), year),
                    'transaction_description': 'PAYMENT THANK YOU/PAIEMENT MERCI',
                    'spend_category': 'Payment',
                    'amount': float(match.group(5).replace(',', '')),
                    'card_number': card_number,
                    'source_pdf': source_pdf,
                    'transaction_type': 'payment'
                })
                break
    
    return payments


def extract_charges(text, year, card_number, source_pdf):
    """Extract charge transactions"""
    charges = []
    pattern = r'([A-Z][a-z]{2})\s+(\d{1,2})\s+([A-Z][a-z]{2})\s+(\d{1,2})\s+(?:Ã\s+)?(.+?)\s+([A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+)*(?:\s+[A-Z][a-z]+)*)\s+(\d{1,3}(?:,\d{3})*\.\d{2})\s*$'
    
    for line in text.split('\n'):
        if match := re.match(pattern, line.strip()):
            description = match.group(5).strip()
            category = match.group(6).strip()
            
            # Skip payments and invalid categories
            if 'PAYMENT' in description.upper() or category not in CATEGORIES:
                continue
            
            charges.append({
                'transaction_date': parse_date(match.group(1), match.group(2), year),
                'transaction_postdate': parse_date(match.group(3), match.group(4), year),
                'transaction_description': description,
                'spend_category': category,
                'amount': float(match.group(7).replace(',', '')),
                'card_number': card_number,
                'source_pdf': source_pdf,
                'transaction_type': 'charge'
            })
    
    return charges


def get_card_sections(text):
    """Split text by card numbers"""
    pattern = r'Card number\s+(\d{4}\s+\d{4}\s+\d{4}\s+\d{4})'
    matches = list(re.finditer(pattern, text))
    
    if not matches:
        return [(text, "Unknown")]
    
    sections = []
    for i, match in enumerate(matches):
        card_num = match.group(1)
        start = match.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        sections.append((text[start:end], card_num))
    
    return sections


def process_pdf(pdf_path):
    """Extract all transactions from a PDF statement"""
    text = extract_text_from_pdf(pdf_path)
    if not text:
        return []
    
    year = get_year(text)
    source_pdf = os.path.basename(pdf_path)
    transactions = []
    
    # Get main account number for payments
    if account_match := re.search(r'Account number\s+(\d{4}\s+\d{4}\s+\d{4}\s+\d{4})', text):
        main_card = account_match.group(1)
    else:
        main_card = "Unknown"
    
    # Extract payments
    transactions.extend(extract_payments(text, year, main_card, source_pdf))
    
    # Extract charges by card
    for section_text, card_number in get_card_sections(text):
        transactions.extend(extract_charges(section_text, year, card_number, source_pdf))
    
    return transactions


def main():
    print("\n" + "="*60)
    print("Credit Card Statement Extractor")
    print("="*60 + "\n")
    
    pdf_files = sorted(glob(os.path.join(PDF_DIR, "*.pdf")))
    
    if not pdf_files:
        print(f"No PDF files found in {PDF_DIR}")
        return
    
    print(f"Found {len(pdf_files)} PDF files\n")
    
    # Process all PDFs
    all_transactions = []
    for pdf_path in pdf_files:
        print(f"Processing: {os.path.basename(pdf_path)}")
        transactions = process_pdf(pdf_path)
        all_transactions.extend(transactions)
        print(f"  Extracted {len(transactions)} transactions\n")
    
    # Sort by date
    all_transactions.sort(key=lambda x: x['transaction_date'])
    
    # Calculate totals
    charges = [t for t in all_transactions if t['transaction_type'] == 'charge']
    payments = [t for t in all_transactions if t['transaction_type'] == 'payment']
    
    print("="*60)
    print("Summary")
    print("="*60)
    print(f"Total Transactions: {len(all_transactions)}")
    print(f"  Charges:  {len(charges)}")
    print(f"  Payments: {len(payments)}")
    print(f"\nTotal Charges:  ${sum(t['amount'] for t in charges):,.2f}")
    print(f"Total Payments: ${sum(t['amount'] for t in payments):,.2f}")
    
    # Save output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(os.path.join(OUTPUT_DIR, "transactions_array.json"), 'w', encoding='utf-8') as f:
        json.dump(all_transactions, f, indent=2, ensure_ascii=False)
    
    df = pd.DataFrame(all_transactions)
    df.to_csv(os.path.join(OUTPUT_DIR, "transactions.csv"), index=False, encoding='utf-8')
    
    print(f"\n✓ Saved to: {OUTPUT_DIR}/transactions_array.json")
    print(f"✓ Saved to: {OUTPUT_DIR}/transactions.csv")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()