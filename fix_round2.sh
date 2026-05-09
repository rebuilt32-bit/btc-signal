#!/bin/bash
# Round 2 fix: VPS does only live closing_gap. GitHub does predictions + backtest.
set -e
cd /root/btc-signal

echo "=== Round 2: Split VPS/GitHub responsibilities ==="

# 1. Trim closing_gap_analysis.py: live calls only, output to closing_gap_live.json
echo "[1/3] Trimming closing_gap_analysis.py..."
python3 << 'FIX1'
p = '/root/btc-signal/closing_gap_analysis.py'
c = open(p).read()
c = c.replace('OUT_PATH = "data/closing_gap_analysis.json"', 'OUT_PATH = "data/closing_gap_live.json"')
c = c.replace('backtest = compute_backtest(history)', 'backtest = None  # disabled on VPS')
open(p, 'w').write(c)
print("  Patched: writes to closing_gap_live.json, skips backtest")
FIX1

# 2. Trim collect_continuous.py: only run closing_gap
echo "[2/3] Trimming collect_continuous.py..."
python3 << 'FIX2'
import re
p = '/root/btc-signal/collect_continuous.py'
c = open(p).read()
c = re.sub(
    r'for s in \[[^\]]+\]:',
    'for s in ["closing_gap_analysis.py"]:',
    c
)
open(p, 'w').write(c)
print("  Patched: only runs closing_gap_analysis")
FIX2

# 3. Update index.html: GitHub for predictions+backtest, VPS for live calls
echo "[3/3] Updating index.html..."
python3 << 'FIX3'
import re
p = '/root/btc-signal/index.html'
c = open(p).read()

# Reset URL constants and add LIVE_GAP_URL
c = re.sub(r"const REPO = [^;]+;", "const REPO = 'rebuilt32-bit/btc-signal';", c)
c = re.sub(r'const DATA_URL = [^;]+;', "const DATA_URL = `https://api.github.com/repos/${REPO}/contents/data/prediction.json?ref=main`;", c)
c = re.sub(r'const GAP_URL = [^;]+;', "const GAP_URL = `https://api.github.com/repos/${REPO}/contents/data/closing_gap_analysis.json?ref=main`;", c)
if 'LIVE_GAP_URL' not in c:
    c = c.replace(
        "const GAP_URL = `https://api.github.com/repos/${REPO}/contents/data/closing_gap_analysis.json?ref=main`;",
        "const GAP_URL = `https://api.github.com/repos/${REPO}/contents/data/closing_gap_analysis.json?ref=main`;\nconst LIVE_GAP_URL = '/data/closing_gap_live.json';"
    )
    print("  Added LIVE_GAP_URL")

# Patch fetchJson to handle both GitHub and local URLs
old_fetch = '''async function fetchJson(url) {
  const r = await fetch(url + '?t=' + Date.now(), {
    headers: { 'Accept': 'application/vnd.github.v3.raw' }
  });
  if (!r.ok) throw new Error('HTTP ' + r.status);
  return r.json();
}'''
new_fetch = '''async function fetchJson(url) {
  const isGithub = url.includes('api.github.com');
  const sep = url.includes('?') ? '&' : '?';
  const headers = isGithub ? { 'Accept': 'application/vnd.github.v3.raw' } : {};
  const r = await fetch(url + sep + 't=' + Date.now(), { headers });
  if (!r.ok) throw new Error('HTTP ' + r.status);
  return r.json();
}'''
if old_fetch in c:
    c = c.replace(old_fetch, new_fetch)
    print("  fetchJson updated")
else:
    # Try alternate form (with &t=)
    alt_fetch = '''async function fetchJson(url) {
  const r = await fetch(url + '&t=' + Date.now(), {
    headers: { 'Accept': 'application/vnd.github.v3.raw' }
  });
  if (!r.ok) throw new Error('HTTP ' + r.status);
  return r.json();
}'''
    if alt_fetch in c:
        c = c.replace(alt_fetch, new_fetch)
        print("  fetchJson updated (alt form)")
    else:
        print("  WARNING: fetchJson not found")

# Patch the loadData gap section to use LIVE_GAP_URL for live and GAP_URL for backtest
old_block = '''try {
    const gap = await fetchJson(GAP_URL);
    gapDataTime = gap.data_snapshot_time || gap.generated_at || null;
    const calls = (gap && gap.live_calls) || [];
    if (calls.length > 0) {
      gapSection.innerHTML =
        `<h2>Closing-time calls (final 4 min) <span id="gap-freshness" class="gap-freshness gap-fresh-0">0s old</span></h2>` +
        calls.map(renderClosingGap).join('');
      updateGapFreshness();
    }
    if (gap && gap.backtest) {
      trackSection.innerHTML = renderTrackRecord(gap.backtest);
    }
  } catch (e) {
    // Silent fail
  }'''
new_block = '''try {
    const live = await fetchJson(LIVE_GAP_URL);
    gapDataTime = live.data_snapshot_time || live.generated_at || null;
    const calls = (live && live.live_calls) || [];
    if (calls.length > 0) {
      gapSection.innerHTML =
        `<h2>Closing-time calls (final 4 min) <span id="gap-freshness" class="gap-freshness gap-fresh-0">0s old</span></h2>` +
        calls.map(renderClosingGap).join('');
      updateGapFreshness();
    }
  } catch (e) { /* VPS unreachable or no calls */ }
  try {
    const bt = await fetchJson(GAP_URL);
    if (bt && bt.backtest) {
      trackSection.innerHTML = renderTrackRecord(bt.backtest);
    }
  } catch (e) { /* GitHub fetch failed */ }'''
if old_block in c:
    c = c.replace(old_block, new_block)
    print("  loadData split into live/backtest fetches")
else:
    print("  WARNING: loadData gap block not matched (may need manual update)")

open(p, 'w').write(c)
FIX3

# Clean up old VPS-side closing_gap output (now using closing_gap_live.json)
rm -f /root/btc-signal/data/closing_gap_analysis.json

echo ""
echo "Restarting service..."
systemctl restart btc-signal.service
sleep 15
echo ""
echo "=== Recent logs ==="
tail -20 /var/log/btc-signal.log
echo ""
echo "=== closing_gap_live.json check ==="
if [ -f /root/btc-signal/data/closing_gap_live.json ]; then
    ls -la /root/btc-signal/data/closing_gap_live.json
    head -20 /root/btc-signal/data/closing_gap_live.json
else
    echo "File not yet created — wait 30s and check again with: head -20 /root/btc-signal/data/closing_gap_live.json"
fi
