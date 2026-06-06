import sys
import os
sys.path.append(os.getcwd())

import logging

import ccxt

from data import update_historical_db

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')

START_DATE = '2021-01-01T00:00:00Z'
TIMEFRAME = '1m'
EXCHANGE_ID = 'binance'
MARKET_TYPE = 'swap'

exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
exchange.load_markets()

symbols = sorted(
    m['symbol'] for m in exchange.markets.values()
    if m.get('swap') and m.get('linear') and m.get('active')
    and m.get('quote') == 'USDT'
)

print(f"Discovered {len(symbols)} active Binance USDT-M perpetual futures.")
print(f"Fetching {TIMEFRAME} data for {len(symbols)} symbols since {START_DATE}...")

update_historical_db(
    symbols, START_DATE,
    exchange_id=EXCHANGE_ID, timeframe=TIMEFRAME,
    overwrite=False, market_type=MARKET_TYPE,
)
print("Done! Check the 'arctic_data' folder.")
