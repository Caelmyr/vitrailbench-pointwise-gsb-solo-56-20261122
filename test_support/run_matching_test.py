"""End-to-end exercise of the replacement-match tool through the Flask API."""
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace")

from make_fixtures import BOXES, make  # noqa: E402

from vitrail.http import create_app  # noqa: E402


def photo_payload(image, title, role, illuminant):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    stream.seek(0)
    capture = json.dumps({"title": title, "note": "synthetic layout", "role": role,
                          "exposure_ev": 0, "black_code": 0, "illuminant": illuminant})
    return {"photo": (stream, f"{title}.png"), "capture": capture}


def main(data_dir, image_dir):
    image_dir = Path(image_dir)
    app = create_app(data_dir)
    client = app.test_client()

    panel = client.post("/api/panels", json={
        "title": "Cyan fragment replacement trial", "accession": "SYN-MATCH-1",
        "width_mm": 220.0, "height_mm": 165.0, "note": "two-lamp sample layout",
        "synthetic": True,
    })
    assert panel.status_code == 201, panel.data
    pid = panel.get_json()["panel"]["id"]

    photo_ids = []
    for lamp, role, illuminant, seed in (
        ("white", "transmitted", "Daylight white lamp 6500K", 11),
        ("warm", "comparison", "Evening warm lamp 2700K", 23),
    ):
        payload = photo_payload(make(lamp, seed), f"{lamp} lamp layout", role, illuminant)
        response = client.post(f"/api/panels/{pid}/photos", data=payload,
                               content_type="multipart/form-data")
        assert response.status_code == 201, response.data
        photo_ids.append(response.get_json()["photo"]["id"])

    order = {
        "captures": photo_ids,
        "region": [0, 0, 1, 1],
        "options": {**{k: list(v) for k, v in BOXES.items()},
                    "brightness_limit_dl": 6.0, "hue_tolerance_deg": 4.0,
                    "minimum_pixels": 64},
        "registrations": {},
        "resolution": 700,
    }
    preview = client.post(f"/api/panels/{pid}/tools/replacement-match", json=order)
    assert preview.status_code == 200, preview.data
    body = preview.get_json()
    result = body["result"]
    print("verdict:", result["verdict"])
    print("recommendation:", result["recommendation"])
    for condition in result["conditions"]:
        print("\n==", condition["condition"])
        for item in condition["ranking"]:
            print(f"  #{item['hue_rank']} {item['name']:10s} hue={item['hue_distance_deg']:6.2f} "
                  f"dL={item['lightness_delta_l']:6.2f} dE={item['total_delta_e']:6.2f} flags={item['flags']}")
        print("  original hue:", condition["patches"]["original"]["hue_deg"],
              "L:", condition["patches"]["original"]["lab"][0])
        print("  blank coverage:", condition["patches"]["blank"]["coverage"],
              "clipped:", condition["patches"]["blank"]["clipped_pixels"])

    assert body["kind"] == "replacement-match"
    white, warm = result["conditions"]
    white_order = [r["sample"] for r in white["ranking"]]
    warm_order = [r["sample"] for r in warm["ranking"]]
    print("\nwhite order:", white_order)
    print("warm  order:", warm_order)
    assert white_order != warm_order, "fixture should swap rankings"
    assert result["ranking_consistent"] is False
    assert result["verdict"] == "ranking_changes"
    assert any(any(r["brightness_flag"] for r in c["ranking"]) for c in result["conditions"]), \
        "sample 3 should trip the brightness flag"
    asset_keys = [a["key"] for a in body["assets"]]
    assert len(asset_keys) == 5, asset_keys
    assert all(k.startswith(("patch-map-", "sample-chip-strip", "sample-readings", "sample-ranking"))
               for k in asset_keys)

    # Retain the inspection.
    retain = client.post(f"/api/panels/{pid}/tools/replacement-match",
                         json={**order, "save": True, "preview": body["fingerprint"],
                               "name": "Two-lamp replacement selection"})
    assert retain.status_code == 201, retain.data
    record = retain.get_json()["record"]
    rid = record["id"]

    sheet = client.get(f"/api/inspections/{rid}/selection-sheet")
    assert sheet.status_code == 200, sheet.data
    assert sheet.data[:5] == b"%PDF-"
    (image_dir.parent / "selection-sheet.pdf").write_bytes(sheet.data)
    print("\nselection sheet bytes:", len(sheet.data))

    # Evidence packet must still verify.
    packet = client.get(f"/api/inspections/{rid}/packet").data
    verify = client.post("/api/packets/verify", data={"archive": (io.BytesIO(packet), "packet.zip")},
                         content_type="multipart/form-data")
    assert verify.status_code == 200, verify.data
    assert verify.get_json()["valid"] is True

    # Save the rendered maps for visual inspection.
    for asset in record["assets"]:
        if asset["mime"] == "image/png":
            blob = client.get(f"/api/inspections/{rid}/files/{asset['key']}").data
            (image_dir.parent / asset["filename"]).write_bytes(blob)

    # Validation path: too-small patch region.
    bad = dict(order)
    bad["options"] = {**order["options"], "original": [0.49, 0.49, 0.50, 0.50],
                      "minimum_pixels": 64}
    rejected = client.post(f"/api/panels/{pid}/tools/replacement-match", json=bad)
    assert rejected.status_code == 422, rejected.data
    print("small-patch rejection:", rejected.get_json()["error"])
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
