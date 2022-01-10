# 1. Install and Import Baseline Dependencies
get_ipython().run_line_magic('pip', 'install transformers')
get_ipython().run_line_magic('pip', 'install sentencepiece')
get_ipython().system('conda install pytorch torchvision torchaudio cpuonly -c pytorch')
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from bs4 import BeautifulSoup
import requests
import re
from transformers import pipeline
import csv

# 2. Setup Summarization Model
model_name = 'human-centered-summarization/financial-summarization-pegasus'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

# 3. Setup Pipeline
monitored_tickers = ['GME', 'TSLA', 'BTC']

# 4.1 Search for Stock News Using Google and Yahoo Finance
print('Searhcing for stock news for', monitored_tickers)
def search_for_stock_news_urls(ticker):
        search_url = 'https://www.google.com/search?q=yahoo+finance+{}&source=lnms&tbm=nws'.format(ticker)
        r = requests.get(search_url)
        soup = BeautifulSoup(r.text, 'html.parser')
        tags = soup.find_all(lambda tag: tag.name == 'a' and tag.get('href') and tag.text)
        hrefs = []
        for a in tags:
            hrefs.append(a['href'])
        return hrefs

raw_urls = {ticker:search_for_stock_news_urls(ticker) for ticker in monitered_tickers}

# 4.2 Strip Out Unwanted URLs
print('Cleaning URLs.')
exclude_list = ['maps', 'policies', 'preferences', 'accounts', 'support']
def strip_unwanted_urls(urls, exclude_list):
    val = []
    for url in urls:
        if 'https://' in url and not any(exlude_word in url for exlude_word in exclude_list):
            res = re.findall(r'(https?://\S+)', url)[0].split('&')[0]
            val.append(res)
    return list(set(val)) #converting val to a set gets rid of duplicates

cleaned_urls = {ticker:strip_unwanted_urls(raw_urls[ticker], exclude_list) for ticker in monitered_tickers}

# 4.3 Search and Scrape Cleaned URLs
print('Scraping news links.')
def scrape_and_process(URLs):
    ARTICLES = []
    for url in URLs:
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = [paragraph.text for paragraph in paragraphs]
        words = ' '.join(text).split(' ')[:350]
        ARTICLE = ' '.join(words)
        ARTICLES.append(ARTICLE)
    return ARTICLES

articles = {ticker:scrape_and_process(cleaned_urls[ticker]) for ticker in monitered_tickers}

# 4.4 Summarize all Articles
print('Summarizing articles.')
def summarize(articles):
    summaries = []
    for article in articles:
        input_ids = tokenizer.encode(article, return_tensors='pt', max_length=512, truncation=True)
        output = model.generate(input_ids, max_length=55, num_beams=5, early_stopping=True)
        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        summaries.append(summary)
    return summaries

summaries = {ticker:summarize(articles[ticker]) for ticker in monitered_tickers}
        
# 5. Adding Sentiment Analysis
print('Calculating sentiment.')
sentiment = pipeline('sentiment-analysis')
scores = {ticker: sentiment(summaries[ticker]) for ticker in monitered_tickers}

# 6. Exporting Results to CSV
print('Exporting results.')
def create_output_array(summaries, scores, urls):
    output = []
    for ticker in monitered_tickers:
        for counter in range(len(summaries[ticker])):
            output_this = [
                ticker, 
                summaries[ticker][counter],
                scores[ticker][counter]['label'],
                scores[ticker][counter]['score'],
                urls[ticker][counter]
            ]
            output.append(output_this)
    return output

final_output = create_output_array(summaries, scores, cleaned_urls)
final_output.insert(0, ['Ticker', 'Summary', 'Label', 'Score', 'Confidence', 'URL'])

with open('assetsummaries.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerows(final_output)