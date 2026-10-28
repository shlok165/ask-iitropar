"""Async headless-browser crawler for JavaScript-rendered sites."""
import asyncio
from typing import Callable

from playwright.async_api import async_playwright

from common import UA, clean_text, is_excluded, normalize_url

EXTRACT_JS = r"""
async () => {
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const root = document.querySelector('main') || document.body;
  // Open collapsed accordions / dropdown sections inside the content area.
  for (const el of [...root.querySelectorAll('[aria-expanded="false"], details:not([open]) > summary')].slice(0, 60)) {
    try { el.click(); } catch (e) {}
  }
  await sleep(400);
  const links = [...document.querySelectorAll('a[href]')].map(a => a.href);
  // Collect text from every tab panel, not just the active one.
  const parts = [];
  const tabs = [...root.querySelectorAll('[role="tab"]')].slice(0, 25);
  if (tabs.length > 1) {
    for (const t of tabs) {
      try { t.click(); await sleep(350); } catch (e) {}
      const panel = root.querySelector('[role="tabpanel"]:not([hidden])');
      if (panel) parts.push('## ' + t.innerText.trim() + '\n' + panel.innerText);
      links.push(...[...root.querySelectorAll('a[href]')].map(a => a.href));
    }
  }
  document.querySelectorAll('header, nav, footer, script, style, noscript, svg, [role="navigation"]')
    .forEach(e => e.remove());
  const h = document.querySelector('main h1, h1, main h2, h2');
  return {
    title: (h && h.innerText.trim()) || document.title,
    doc_title: document.title,
    text: (document.querySelector('main') || document.body).innerText + '\n\n' + parts.join('\n\n'),
    links,
  };
}
"""


async def _render_one(ctx, url, wait_ms):
    page = await ctx.new_page()
    try:
        await page.goto(url, wait_until="networkidle", timeout=45000)
        await page.wait_for_timeout(wait_ms)
        return await page.evaluate(EXTRACT_JS)
    except Exception as e:
        print(f"  ! render {url}: {e.__class__.__name__}")
        return None
    finally:
        await page.close()


async def render_crawl(seeds: list[str], allow: Callable[[str], bool], max_pages: int,
                       concurrency: int = 4, wait_ms: int = 1200, on_link: Callable[[str, str], None] | None = None):
    """BFS crawl starting at `seeds`, following links for which allow(url) is True.

    Returns a list of {url, title, text, links}. `on_link(page_url, link)` sees every link found.
    """
    seen = {normalize_url(s) for s in seeds}
    queue: asyncio.Queue = asyncio.Queue()
    for s in seen:
        queue.put_nowait(s)
    results = []

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        ctx = await browser.new_context(user_agent=UA, viewport={"width": 1366, "height": 900})
        await ctx.route("**/*.{png,jpg,jpeg,gif,webp,svg,mp4,woff,woff2,ttf}", lambda r: r.abort())

        async def worker():
            while True:
                url = await queue.get()
                try:
                    if len(results) >= max_pages:
                        continue
                    res = await _render_one(ctx, url, wait_ms)
                    if not res:
                        continue
                    res["url"] = url
                    res["text"] = clean_text(res["text"])
                    results.append(res)
                    print(f"  [{len(results)}] {url} ({len(res['text'])} chars)")
                    for link in res["links"]:
                        if on_link:
                            on_link(url, link)
                        n = normalize_url(link)
                        if n not in seen and not is_excluded(n) and allow(n):
                            seen.add(n)
                            queue.put_nowait(n)
                finally:
                    queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
        await queue.join()
        for w in workers:
            w.cancel()
        await browser.close()
    return results
