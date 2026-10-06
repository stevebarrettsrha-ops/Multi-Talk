"""The interface itself, in a browser: the page boots without errors, a
picture and a recording go in, Generate renders a clip through the fake
engine, and the other pages draw. Needs Playwright; steps aside without it."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from harness import (SAMPLE_PNG, SAMPLE_WAV, Suite, Workspace, fake_weights,
                     studio)


def available() -> str:
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except ImportError:
        return "Playwright is not installed (pip install playwright)"
    return ""


def chromium() -> str:
    if os.environ.get("MT_CHROMIUM"):
        return os.environ["MT_CHROMIUM"]
    for cand in ("/opt/pw-browsers/chromium", shutil.which("chromium"),
                 shutil.which("google-chrome")):
        if cand and Path(cand).exists() and Path(cand).is_file():
            return cand
    return ""


def run(slow: bool = False) -> Suite:
    from playwright.sync_api import sync_playwright

    s = Suite("ui")
    with Workspace() as ws:
        weights = ws / "weights"
        fake_weights(weights)
        with studio(ws / "data", weights) as app, sync_playwright() as p:
            exe = chromium()
            browser = p.chromium.launch(**({"executable_path": exe} if exe else {}))
            page = browser.new_page(viewport={"width": 1400, "height": 900})
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            # a 400 is the server refusing a request on purpose (the test
            # sends one); the browser logs it as an error, it is not one
            page.on("console", lambda m: m.type == "error" and
                    "favicon" not in m.text and "status of 400" not in m.text
                    and errors.append(m.text))
            shots = os.environ.get("MT_SCREENSHOTS")
            page.goto(app.url)
            page.wait_for_function(
                "document.querySelector('#enginePill span').textContent"
                ".startsWith('Ready')", timeout=15000)
            s.check("the page boots and finds the engine ready", True)
            s.check("no setup sheet when the machine is set up",
                    page.locator("#veilSetup").is_hidden())
            s.equal("8 GB defaults are applied: 480 px",
                    page.locator("#segSize button.on").get_attribute("data-v"),
                    "multitalk-360")
            s.equal("FusionX's steps are the default",
                    page.locator("#stepsVal").text_content(), "8")

            page.locator("#imageFile").set_input_files(str(SAMPLE_PNG))
            page.wait_for_selector("#refsRow img")
            page.locator("#audio1File").set_input_files(str(SAMPLE_WAV))
            page.wait_for_selector("#refsRow audio")
            s.check("the picture and the recording show under the bar",
                    page.locator("#refsRow .ref").count() == 2)
            s.check("the picture pill stops asking once it has one",
                    "need" not in (page.locator("#btnImage").get_attribute("class")))
            page.fill("#prompt", "A woman sings into a microphone.")
            page.click("#btnGenerate")
            page.wait_for_selector("#feed .tile video", timeout=30000)
            s.check("Generate renders a clip into the feed", True)
            if shots:
                page.wait_for_timeout(600)
                page.screenshot(path=f"{shots}/generate.png")
            s.check("the toast says the clip is ready",
                    page.wait_for_function(
                        "document.querySelector('#toast').textContent"
                        ".includes('Clip ready')", timeout=10000) is not None)

            page.locator("#feed .tile .over").first.click(force=True)
            s.check("a clip opens in the lightbox with its recipe",
                    page.locator("#lightbox").is_visible()
                    and "480 px" in page.locator("#lbMeta").text_content())
            page.click("#lbClose")

            page.click('#segPeople [data-v="2"]')
            s.check("two people shows the right-voice button",
                    page.locator("#btnAudio2").is_visible())
            page.click('#segSource [data-v="tts"]')
            s.check("typed speech shows the text box and both voices",
                    page.locator("#ttsText").is_visible()
                    and page.locator("#voice2").is_visible()
                    and page.locator("#voice1 option").count() == 2)
            page.fill("#ttsText", "hello there")
            page.click("#btnGenerate")
            page.wait_for_function(
                "document.querySelector('#toast').textContent.includes('(s1)')",
                timeout=10000)
            s.check("two people without (s1)/(s2) is explained, not sent", True)

            page.click("#btnSettings")
            s.check("the settings popover opens", page.locator("#settingsPop").is_visible())
            page.click("#btnCloseSettings")

            page.click('.nav[data-view="engine"]')
            page.wait_for_selector("#depList .fitem", timeout=60000)
            s.check("the Engine page lists the dependencies",
                    page.locator("#depList .fitem").count() >= 6)
            page.wait_for_selector("#preflight .fitem", timeout=60000)
            if shots:
                page.screenshot(path=f"{shots}/engine.png", full_page=True)
            s.check("the preflight gives a verdict",
                    page.locator("#preflight .state").count() == 1)
            page.click('.nav[data-view="models"]')
            page.wait_for_selector("#setsList .setrow", timeout=15000)
            s.check("the Models page lists both model sets",
                    page.locator("#setsList .setrow").count() == 2)
            page.wait_for_selector("#localList .fitem", timeout=15000)
            page.click('.nav[data-view="library"]')
            s.check("the Library shows the clip",
                    page.locator("#libGrid .tile").count() == 1)
            page.reload()
            page.wait_for_selector("#refsRow img")
            s.check("the draft survives a reload",
                    page.locator("#ttsText").input_value() == "hello there")
            s.check("no script errors along the way", not errors, str(errors[:3]))
            browser.close()
    return s
