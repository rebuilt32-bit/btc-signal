"""
One-shot diagnostic: tests candidate funding-rate APIs from this runner's IP.
Run once, examine output, then choose source for real implementation.
"""
import requests
import json

CANDIDATES = [
    {
        "name": "Kraken Futures bulk tickers",
        "url": "https://futures.kraken.com/derivatives/api/v3/tickers",
        "params": None,
        "extract": "Look for tickers[].symbol like PF_XBTUSD with fundingRate",
    },
    {
        "name": "OKX public funding rate (BTC-USDT-SWAP)",
        "url": "https://www.okx.com/api/v5/public/funding-rate",
        "params": {"instId": "BTC-USDT-SWAP"},
        "extract": "Look for data[0].fundingRate",
    },
    {
        "name": "Deribit public ticker (BTC-PERPETUAL)",
        "url": "https://www.deribit.com/api/v2/public/ticker",
        "params": {"instrument_name": "BTC-PERPETUAL"},
        "extract": "Look for result.funding_8h or result.current_funding",
    },
    {
        "name": "dYdX v4 indexer markets",
        "url": "https://indexer.dydx.trade/v4/perpetualMarkets",
        "params": {"ticker": "BTC-USD"},
        "extract": "Look for markets.BTC-USD.nextFundingRate",
    },
    {
        "name": "Coinglass public funding (free tier, no auth)",
        "url": "https://open-api-v3.coinglass.com/api/futures/fundingRate/v2",
        "params": {"symbol": "BTC"},
        "extract": "May require API key on free tier",
    },
]


def test_one(c):
    print(f"\n{'='*70}")
    print(f"TEST: {c['name']}")
    print(f"URL:  {c['url']}")
    if c['params']:
        print(f"Params: {c['params']}")
    print(f"What to look for: {c['extract']}")
    print('-'*70)

    try:
        r = requests.get(c["url"], params=c["params"], timeout=15)
        print(f"Status: {r.status_code}")
        print(f"Response length: {len(r.text)} chars")
        if r.status_code == 200:
            try:
                j = r.json()
                snippet = json.dumps(j, indent=2)
                if len(snippet) > 1500:
                    snippet = snippet[:1500] + "\n... [truncated]"
                print(f"Body:\n{snippet}")
            except Exception as e:
                print(f"Could not parse JSON: {e}")
                print(f"Raw body (first 500 chars): {r.text[:500]}")
        else:
            print(f"Non-200 body (first 500 chars): {r.text[:500]}")
    except requests.exceptions.Timeout:
        print("FAILED: timeout")
    except requests.exceptions.ConnectionError as e:
        print(f"FAILED: connection error - {str(e)[:200]}")
    except Exception as e:
        print(f"FAILED: {type(e).__name__} - {str(e)[:200]}")


for c in CANDIDATES:
    test_one(c)

print(f"\n{'='*70}")
print("Diagnostic complete. Review above to identify working sources.")
