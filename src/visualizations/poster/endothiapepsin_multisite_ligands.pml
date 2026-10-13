# Multi-site ligand transplant example for the enhanced scPDB dataset
# ======================================================================
# Illustrates how the AHoJ-DB enhancement (see
# CryptoBench/src/A-filter-ahojdb/extract-scPDB.ipynb and
# src/data-extraction/extract-scPDB.ipynb) pulls ligands from *other*
# conformations of the same protein onto the single representative
# structure that scPDB keeps.
#
# Representative structure : 4Y45  (endothiapepsin, chain A, 1.06 A)
#   - part of the Diamond XChem / PanDDA fragment-screening series
#   - already carries its own native fragment, F91 ("fragment 291")
#
# Alternative-conformation ligands transplanted from 5 other PDB entries
# of the same crystal series (same auth numbering in every source
# structure, verified by superposition: CA RMSD ~0.13-0.16 A onto 4Y45).
#
# NOTE on site assignment: an earlier version of this script grouped
# sites by *residue-ID overlap* of the annotated pocket, which is
# wrong for this protein - endothiapepsin is a bilobal aspartic
# protease whose single active-site cleft is lined by residues from
# BOTH lobes, so two ligands sitting almost on top of each other can
# be reported against completely non-overlapping residue lists. Site
# membership below was instead determined from actual post-superposition
# atom-to-atom distances between the ligands. The native ligand F91 is
# itself modeled as two copies (resi 405 and 406); these sit 5.0 A apart
# and are treated as two separate sites, not merged:
#
#   Site  Ligand(s)        Source(s)          Notes
#   ----  ---------------  -----------------  ---------------------------------
#   1     F91 (native,     4Y45 itself        overlaps R9J: 0.2 A min distance
#         resi 405)
#         R9J              5RBX
#   2     F91 (native,     4Y45 itself        5.0 A from resi 405 / 9.0 A
#         resi 406)                           from R9J - its own site
#   3     RA7              5RBT               13-24 A from sites 1-2
#   4     RDJ              5RCA               10.6-11.6 A from RDM/RDY -
#                                              close, but kept as its own site
#   5     RDM              5RCB               RDM-RDY: 0.3 A (same pocket)
#         RDY              5RCD
#   6     R8V              5RBS               isolated: 25-53 A from everything
#
# For each source structure we: fetch it, strip solvent/crystallization
# additives, superpose it onto 4Y45's chain A, cut out only the ligand
# copy sitting in the annotated pocket, then discard the rest of the
# fetched structure - leaving one representative protein plus one
# colored ligand per binding site. Atom counts are printed for every
# created ligand object so you can confirm none of them came back empty.
#
# Requires network access (uses `fetch` to pull structures from the PDB).
# Run with: pymol endothiapepsin_multisite_ligands.pml

reinitialize
bg_color white
set assembly, ""
set fetch_type_default, pdb

# ----------------------------------------------------------------------
# 1) Representative structure: 4Y45, chain A, keep its native ligand F91.
#    F91 is modeled as two copies (resi 405, 406), 5.0 A apart - real
#    separate sites, so each copy gets its own color: 405 -> site 1
#    (shared with the transplanted R9J), 406 -> site 2 (alone).
# ----------------------------------------------------------------------
fetch 4y45, type=pdb, async=0
remove solvent
remove not (alt ""+A)
alter 4y45, alt=""
sort
remove 4y45 and not (polymer or resn F91)
remove 4y45 and not chain A

hide everything, 4y45
show cartoon, 4y45
color grey70, 4y45

show sticks, 4y45 and resn F91 and resi 405
color orange, 4y45 and resn F91 and resi 405
util.cnc 4y45 and resn F91 and resi 405 and not elem C

show sticks, 4y45 and resn F91 and resi 406
color yellow, 4y45 and resn F91 and resi 406
util.cnc 4y45 and resn F91 and resi 406 and not elem C

# leading "/" runs the rest of the line as a python expression at top level
/print("native F91 405 atoms: %d, 406 atoms: %d" % (cmd.count_atoms("4y45 and resn F91 and resi 405"), cmd.count_atoms("4y45 and resn F91 and resi 406")))

# ----------------------------------------------------------------------
# 2) Helper macro: fetch a source structure, superpose it onto 4Y45,
#    extract only the ligand sitting in the given pocket, then discard
#    everything else from that source structure.
# ----------------------------------------------------------------------
python
from pymol import util

def transplant_ligand(site, source, ligand, pocket_resi, color):
    # object names are keyed by ligand (not just site), since site 5
    # is fed by two ligands (RDM, RDY) that must stay as separate objects
    src_obj = f"src_{source}"
    lig_obj = f"lig_site{site}_{ligand}"
    lbl_obj = f"lbl_site{site}_{ligand}"

    # use cmd.do (not cmd.fetch) so the `async=0` keyword doesn't collide
    # with the reserved python keyword `async`/`async_` across PyMOL versions
    cmd.do(f"fetch {source}, {src_obj}, type=pdb, async=0")
    cmd.remove(f"{src_obj} and solvent")
    # NOTE: do NOT strip altlocs down to ""+A here. These are PanDDA
    # "changed state model" depositions where the ligand itself is only
    # ever modeled as altloc C/D (partial-occupancy conformers) - never ""
    # or A - so an early ""+A filter deletes every ligand atom before we
    # ever get to select it. Altloc cleanup happens below, after
    # extraction, once we know which letters this specific ligand uses.
    cmd.remove(f"{src_obj} and not (polymer or resn {ligand})")
    cmd.remove(f"{src_obj} and not chain A")

    # superpose the source structure onto the representative structure
    cmd.align(f"{src_obj} and polymer", "4y45 and polymer")

    # select the specific ligand copy that sits in the annotated pocket
    # (guards against unrelated crystallization-artifact copies elsewhere)
    cmd.select("_pocket", f"{src_obj} and chain A and resi {pocket_resi}")
    cmd.create(lig_obj, f"({src_obj} and resn {ligand}) within 5 of _pocket")
    cmd.delete("_pocket")
    cmd.delete(src_obj)

    n = cmd.count_atoms(lig_obj)
    print(f"{lig_obj}: {n} atoms" + ("  <-- EMPTY, check ligand code / pocket residues!" if n == 0 else ""))

    # keep only one altloc conformer of the ligand, for a clean, single
    # pose (PanDDA ligands are often modeled as 2 partial-occupancy
    # conformers, e.g. altloc C and D, which would otherwise draw as
    # overlapping double sticks)
    alt_values = sorted({a.alt for a in cmd.get_model(lig_obj).atom if a.alt})
    if len(alt_values) > 1:
        keep, discard = alt_values[0], "+".join(alt_values[1:])
        cmd.remove(f"{lig_obj} and alt {discard}")
        cmd.alter(lig_obj, 'alt=""')
        cmd.sort(lig_obj)
        print(f"  {lig_obj}: kept altloc {keep}, dropped {discard} ({cmd.count_atoms(lig_obj)} atoms remain)")

    cmd.show("sticks", lig_obj)
    cmd.color(color, lig_obj)
    util.cnc(f"{lig_obj} and not elem C")

    cmd.pseudoatom(lbl_obj, lig_obj)
    cmd.hide("everything", lbl_obj)
    cmd.set("label_size", 20, lbl_obj)
    cmd.set("label_color", color, lbl_obj)
    cmd.label(lbl_obj, f'"{ligand}  (site {site})"')
    cmd.show("labels", lbl_obj)

    cmd.group(f"site_{site}", f"{lig_obj} {lbl_obj}")

# site 1: R9J transplant joins native F91/405, already colored orange above
transplant_ligand("1", "5rbx", "R9J", "202+210+212+213+214+228+230+231",                "orange")
# site 3: isolated (site 2 is native F91/406, handled above - no transplant)
transplant_ligand("3", "5rbt", "RA7", "233+236+237+242+243+244+251",                    "purple")
# site 4: isolated (close to site 5, but kept separate per manual review)
transplant_ligand("4", "5rca", "RDJ", "129+130+131+132+143+145+146+149+190+319+320+321+323", "teal")
# site 5: two ligands sharing the same pocket region
transplant_ligand("5", "5rcb", "RDM", "184+185+186+187+267+322+324",                    "hotpink")
transplant_ligand("5", "5rcd", "RDY", "185+186+187+322+324",                            "hotpink")
# site 6: isolated, farthest from everything else
transplant_ligand("6", "5rbs", "R8V", "61+62+67+68+69+90",                              "marine")
python end

# label both native F91 copies the same way as the transplanted ligands
# (they can't be added to a PyMOL group since they live inside the 4y45
# object, not as separate objects - but each shares its site's color)
pseudoatom lbl_native_F91_405, 4y45 and resn F91 and resi 405
hide everything, lbl_native_F91_405
set label_size, 20, lbl_native_F91_405
set label_color, orange, lbl_native_F91_405
label lbl_native_F91_405, "F91  (site 1, native)"
show labels, lbl_native_F91_405

pseudoatom lbl_native_F91_406, 4y45 and resn F91 and resi 406
hide everything, lbl_native_F91_406
set label_size, 20, lbl_native_F91_406
set label_color, yellow, lbl_native_F91_406
label lbl_native_F91_406, "F91  (site 2, native)"
show labels, lbl_native_F91_406

# ----------------------------------------------------------------------
# 3) Final view
# ----------------------------------------------------------------------
set label_font_id, 7
set label_outline_color, white
set stick_radius, 0.22, lig_*
set cartoon_transparency, 0
set ray_opaque_background, 0
orient 4y45
zoom all, buffer=8

/print("objects in scene: %s" % cmd.get_object_list())
deselect
