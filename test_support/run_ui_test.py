"""Drive the real browser UI: create panel, upload both lamp photos, run the
replacement-match tool, retain it and download the selection sheet."""
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


BASE = "http://127.0.0.1:8000"
FIXTURES = Path("/workspace/test_artifacts/fixtures")


def dialog_form(page, form_name):
    return page.locator(f'form[data-form="{form_name}"]')


def main():
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        page.goto(BASE, wait_until="networkidle")

        # Create a panel through the dialog.
        page.get_by_role("button", name="New panel").click()
        form = dialog_form(page, "new-panel")
        form.locator('[name=title]').fill("Browser cyan replacement")
        form.locator('[name=width_mm]').fill("220")
        form.locator('[name=height_mm]').fill("165")
        form.locator('[name=accession]').fill("SYN-UI-1")
        form.get_by_role("button", name="Create panel").click()
        expect(page.locator(".rail h1")).to_have_text("Browser cyan replacement")

        # Upload both captures.
        for filename, title, role, illuminant in (
            ("white-lamp.png", "white lamp layout", "transmitted", "Daylight white lamp 6500K"),
            ("warm-lamp.png", "warm lamp layout", "comparison", "Evening warm lamp 2700K"),
        ):
            page.get_by_role("button", name="Add capture").click()
            upload = dialog_form(page, "upload")
            upload.locator('input[type=file]').set_input_files(str(FIXTURES / filename))
            upload.locator('[name=title]').fill(title)
            upload.locator('[name=role]').select_option(role)
            upload.locator('[name=illuminant]').fill(illuminant)
            upload.get_by_role("button", name="Import capture").click()
            expect(page.locator("dialog[open]")).to_have_count(0)
        expect(page.locator(".photo-row")).to_have_count(2)

        # Select the new tool.
        page.locator("#tool").select_option("replacement-match")
        expect(page.locator(".settings .quiet")).to_contain_text(
            "Match three replacement samples")
        # Both captures must be ticked for the two-input tool.
        checkboxes = page.locator('[data-select-photo]')
        assert checkboxes.count() == 2
        for i in range(2):
            box = checkboxes.nth(i)
            if not box.is_checked():
                box.check()

        # Run and wait for the verdict banner.
        page.locator('form[data-form=analysis] button[type=submit]').click()
        banner = page.locator(".notice.verdict-swap")
        expect(banner).to_be_visible(timeout=15000)
        expect(banner.locator("strong").first).to_have_text("Rankings differ between the two lights")
        blocks = banner.locator(".scoped")
        expect(blocks).to_have_count(2)
        white_block = blocks.filter(has=page.get_by_text("white lamp layout", exact=True))
        warm_block = blocks.filter(has=page.get_by_text("warm lamp layout", exact=True))
        expect(white_block.locator(".rank-list li").first).to_contain_text("#1 Sample 1")
        expect(warm_block.locator(".rank-list li").first).to_contain_text("#1 Sample 2")
        # Brightness flag for the dark sample is surfaced in the UI.
        expect(banner.locator(".warn").first).to_contain_text("brightness flag")
        # Five result assets rendered (3 PNG tabs + 2 CSV downloads).
        expect(page.locator(".asset-tabs button")).to_have_count(3)
        expect(page.locator(".asset-tabs a")).to_have_count(2)

        # Retain the inspection.
        page.get_by_role("button", name="Retain inspection").click()
        retain = dialog_form(page, "retain")
        retain.locator('[name=name]').fill("UI two-lamp selection")
        retain.get_by_role("button", name="Keep observed result").click()
        expect(page.locator("dialog[open]")).to_have_count(0)
        expect(page.locator("h3").first).to_have_text("Retained inspection")
        sheet_link = page.locator('a[href*="selection-sheet"]')
        expect(sheet_link).to_have_count(1)

        # Download the selection sheet through the UI link.
        with page.expect_download() as info:
            sheet_link.click()
        download = info.value
        target = Path("/workspace/test_artifacts/ui-selection-sheet.pdf")
        download.save_as(str(target))
        assert target.read_bytes()[:5] == b"%PDF-"
        print("downloaded:", download.suggested_filename, target.stat().st_size, "bytes")

        # Open the retained record from the Records tab and confirm banner persists.
        page.locator('[data-act=tab][data-key=record]').click()
        page.locator(".report-row").first.click()
        expect(page.locator(".notice.verdict-swap")).to_be_visible()

        page.screenshot(path="/workspace/test_artifacts/ui-records.png", full_page=True)
        assert not errors, f"browser errors: {errors}"
        print("UI FLOW OK, no console errors")
        browser.close()


if __name__ == "__main__":
    main()
