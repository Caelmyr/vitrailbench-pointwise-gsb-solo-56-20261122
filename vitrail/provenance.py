from .contracts import Problem, rounded


def _difference(recorded,current):
    unchanged=recorded["revision"]==current["revision"]
    ledger_exposure=recorded["metadata"]["exposure_ev"]
    active_exposure=current["data"]["exposure_ev"]
    diff=rounded(active_exposure-ledger_exposure)
    original_matches=recorded["original"]==current["original"]
    chosen_alignment=recorded["alignment_id"]
    current_alignment=current["active_alignment"]
    return dict(
        capture=recorded["id"],
        title=recorded["title"],
        current_revision=current["revision"],
        archived_revision=recorded["revision"],
        active_alignment=current_alignment,
        inspection_alignment=chosen_alignment,
        revised=not unchanged,
        original_matches=original_matches,
        same_alignment=chosen_alignment==current_alignment,
        exposure_change_ev=diff,
        uses_archived_settings=not unchanged or chosen_alignment!=current_alignment,
    )


def inspection_sources(library,record):
    report=library.report(record)
    info=report["facts"]
    archived=info["panel"]
    current=library.panel_summary(archived["id"])["panel"]
    changed_size=[archived["width_mm"],archived["height_mm"]]!=[current["width_mm"],current["height_mm"]]
    audit=[]
    for photo in info["sources"]:
        try:
            active=library.photo(photo["id"])
        except Problem as error:
            if error.status!=404:
                raise
            audit.append(dict(capture=photo["id"],title=photo["title"],missing=True))
            continue
        if active["panel_id"]!=archived["id"]:
            raise Problem("Stored source no longer belongs to its panel.",409)
        changed=_difference(photo,active)
        changed["missing"]=False
        changed["preview_url"]=f"/api/photos/{photo['id']}/view/preview"
        changed["original_url"]=f"/api/photos/{photo['id']}/view/original"
        audit.append(changed)
    return dict(inspection=report["id"],panel_size_changed=changed_size,sources=audit,
                current_panel_revision=current["revision"],archived_panel_revision=archived["revision"],
                notes="The retained result keeps the recorded settings and measurements. Active capture settings are reported for comparison.")
