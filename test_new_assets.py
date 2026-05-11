"""Verify BNB and HYPE asset support across data sources."""
import requests

CHECKS = {
    "BNB": {
        "kalshi": ("KXBNB15M", "https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker=KXBNB15M&limit=1"),
        "kraken": ("BNBUSD", "https://api.kraken.com/0/public/Ticker?pair=BNBUSD"),
        "coinbase": ("BNB-USD", "https://api.exchange.coinbase.com/products/BNB-USD/ticker"),
        "binance_us": ("BNBUSDT", "https://api.binance.us/api/v3/ticker/price?symbol=BNBUSDT"),
        "kraken_futures": ("PF_BNBUSD", "https://futures.kraken.com/derivatives/api/v3/tickers"),
    },
    "HYPE": {
        "kalshi": ("KXHYPE15M", "https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker=KXHYPE15M&limit=1"),
        "kraken": ("HYPEUSD", "https://api.kraken.com/0/public/Ticker?pair=HYPEUSD"),
        "coinbase": ("HYPE-USD", "https://api.exchange.coinbase.com/products/HYPE-USD/ticker"),
        "binance_us": ("HYPEUSDT", "https://api.binance.us/api/v3/ticker/price?symbol=HYPEUSDT"),
        "kraken_futures": ("PF_HYPEUSD", "https://futures.kraken.com/derivatives/api/v3/tickers"),
    },
}

print("Testing data sources for new assets...\n")

for asset, sources in CHECKS.items():
    print(f"=== {asset} ===")
    for source, (symbol, url) in sources.items():
        try:
            r = requests.get(url, timeout=10)
            ok = r.status_code == 200
            data = r.json() if ok else {}
            if source == "kalshi":
                n = len(data.get("markets", []))
                has = n > 0
                detail = f"got {n} markets" if has else "no markets returned"
            elif source == "kraken":
                has = bool(data.get("result")) and not data.get("error")
                detail = "ok" if has else str(data.get("error", "no data"))[:60]
            elif source == "coinbase":
                has = "price" in data
                detail = f"price={data.get('price', 'n/a')}" if has else str(data.get("message", "no price"))[:60]
            elif source == "binance_us":
                has = "price" in data
                detail = f"price={data.get('price', 'n/a')}" if has else str(data.get("msg", "no price"))[:60]
            elif source == "kraken_futures":
                if "tickers" in data:
                    syms = {t.get("symbol", "").upper() for t in data["tickers"]}
                    has = symbol.upper() in syms
                    detail = "in tickers" if has else f"not in {len(syms)} symbols"
                else:
                    has = False
                    detail = "tickers field missing"
            status = "OK" if has else "--"
            print(f"  [{status}] {source:<15s} {symbol:<15s} {detail}")
        except Exception as e:
            print(f"  [--] {source:<15s} {symbol:<15s} ERROR: {str(e)[:60]}")
    print()
