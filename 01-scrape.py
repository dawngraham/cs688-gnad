#!/usr/bin/env python3

import math
import time

import pandas as pd
import regex as re
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

ROOT_URL = 'https://nvdatabase.swarthmore.edu'
COUNTRY = 'United%20States'
TGT_FILE = './data/campaigns.parquet.gzip'


def get_country_slugs(country):
    """
    Get slugs to get urls for individual campaigns
    :param country: string
    :return: list
    """
    url = f'{ROOT_URL}/browse/{country}/all/all/all/all'
    res = requests.get(url)
    soup = BeautifulSoup(res.content, 'lxml')

    # Get number of pages to go through from "Showing 1-{showing} of {total_results}"
    results = soup.find('div', {'class': 'view-browse-cases'}
                        ).find('div', {'class': 'view-header'}).text.strip()
    total_results = int(re.search(r'(?<= of )\d*?(?= results)', results)[0])
    showing = int(re.search(r'(?<=-)\d*?(?= of)', results)[0])
    pages = math.ceil(total_results / showing)

    articles = []

    print('Getting slugs... Directory page # ')

    # Cycle through all pages in directory
    for page in tqdm(range(pages)):
        url = f'{ROOT_URL}/browse/{country}/all/all/all/all?page={page}'
        res = requests.get(url)
        soup = BeautifulSoup(res.content, 'lxml')

        for article in soup.find_all('article'):
            articles.append(article['about'])

        time.sleep(1)

    return articles


def get_campaign_details(country_slugs):
    """
    Get details for individual campaigns
    :param country_slugs: list created by get_country_slugs()
    :return: dataframe with campaign title, details, and url
    """
    campaigns = []
    total_campaigns = len(country_slugs)

    print(f'\nGetting {total_campaigns} campaign details... ')
    for article in tqdm(range(total_campaigns)):
        url = f'{ROOT_URL}{country_slugs[article]}'
        res = requests.get(url)
        soup = BeautifulSoup(res.content, 'lxml')

        campaign = {}

        campaign['title'] = soup.find('h1', {'class': 'page-header'}
                                      ).find('span').text.strip()

        try:
            campaign['details'] = soup.find('div', {'id': 'case-study-detail--content'}
                                            ).find('div', {'class': 'field--item'}).text.strip()
        except:
            pass

        campaign['url'] = url

        campaigns.append(campaign)

        time.sleep(1)

    # Save to dataframe
    campaigns = pd.DataFrame(campaigns)

    return campaigns


if __name__ == '__main__':
    country_slugs = get_country_slugs(COUNTRY)
    country_details = get_campaign_details(country_slugs)

    # Save file
    country_details.to_parquet(TGT_FILE, compression='gzip')

    print('Done.')
