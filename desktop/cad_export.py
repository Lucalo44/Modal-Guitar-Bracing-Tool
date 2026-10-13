"""
STEP export for solved designs -- desktop build only (needs OpenCascade via
the cadquery-ocp package; the browser build has no CAD kernel).

Builds true B-rep solids, not a faceted mesh, so a simulation tool can mesh
them properly:

  * top_plate -- the traced outline (the same Catmull-Rom spline the UI
    draws, sent as dense points and re-interpolated here as a periodic
    B-spline), extruded to the plate thickness, sound hole cut out.
  * braces_<material> -- every brace's tapered rectangular cross-section
    lofted along its centreline (straight, quadratic or cubic Bezier, same
    sampling as engine._sample_brace), sitting on the plate's +z face,
    trimmed to the plate footprint and hole, and fused per material so
    crossing braces (an X-brace) don't export as interfering solids.

Coordinates are mm: x along the body length (Lx, the grain direction the
solver assumes), y across the width (Ly), z normal to the plate. The braced
face is +z. Same frame as the STL export and the simulation-info JSON.
"""
import math

from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace,
                                BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeWire)
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakePrism
from OCP.GeomAPI import GeomAPI_Interpolate
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.Message import Message, Message_Gravity
from OCP.Interface import Interface_Static
from OCP.STEPCAFControl import STEPCAFControl_Writer
from OCP.STEPControl import STEPControl_AsIs
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.collections import HArray1_gp_Pnt
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDocStd import TDocStd_Document
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Vec

import engine

# OpenCascade's translators print transfer statistics to stdout by default.
for _printer in Message.DefaultMessenger_s().Printers():
    _printer.SetTraceLevel(Message_Gravity.Message_Fail)

BRACE_STATIONS = 17   # cross-sections per brace loft -- plenty for a parabolic taper on a quadratic/cubic curve


def _volume(shape):
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return props.Mass()


def _clean(shape):
    u = ShapeUpgrade_UnifySameDomain(shape, True, True, True)
    u.Build()
    return u.Shape()


def _outline_face(outline_mm):
    """Closed outline points (mm, no repeated closing point) -> planar face
    bounded by a periodic B-spline through them."""
    pts = [p for i, p in enumerate(outline_mm) if i == 0 or math.dist(p, outline_mm[i-1]) > 1e-6]
    if math.dist(pts[0], pts[-1]) < 1e-6:
        pts = pts[:-1]
    arr = HArray1_gp_Pnt(1, len(pts))
    for i, (x, y) in enumerate(pts, start=1):
        arr.SetValue(i, gp_Pnt(x, y, 0.0))
    interp = GeomAPI_Interpolate(arr, True, 1e-6)
    interp.Perform()
    if not interp.IsDone():
        raise RuntimeError('could not fit a smooth curve through the outline')
    edge = BRepBuilderAPI_MakeEdge(interp.Curve()).Edge()
    wire = BRepBuilderAPI_MakeWire(edge).Wire()
    return BRepBuilderAPI_MakeFace(wire, True).Face()


def _brace_solid(b, LxMm, LyMm, z0):
    """Loft of the brace's tapered cross-section along its centreline.
    Rectangle corners are offset perpendicular to the local tangent (the same
    construction the UI's footprint drawing uses)."""
    pts = engine._sample_brace(b, LxMm, LyMm, BRACE_STATIONS - 1)
    wC, wT = b['width'], b.get('widthTip')
    hC, hT = b['height'], b.get('heightTip')
    loft = BRepOffsetAPI_ThruSections(True, False, 1e-6)
    for k, (x, y, _, t) in enumerate(pts):
        xa, ya = pts[max(0, k-1)][:2]; xb, yb = pts[min(len(pts)-1, k+1)][:2]
        L = math.hypot(xb-xa, yb-ya) or 1.0
        nx, ny = -(yb-ya)/L, (xb-xa)/L
        hw = engine._dim_at(wC, wT, t) / 2.0
        hh = engine._dim_at(hC, hT, t)
        poly = BRepBuilderAPI_MakePolygon()
        for sx, z in ((1, z0), (-1, z0), (-1, z0+hh), (1, z0+hh)):
            poly.Add(gp_Pnt(x + sx*nx*hw, y + sx*ny*hw, z))
        poly.Close()
        loft.AddWire(poly.Wire())
    loft.CheckCompatibility(False)
    loft.Build()
    if not loft.IsDone():
        raise RuntimeError('could not build a brace solid')
    return loft.Shape()


def build_design_shapes(payload):
    """payload (from the page): LxMm, LyMm, hMm, outline [[x,y],...] mm,
    hole {x,y,r} mm or None, braces [...] (engine brace dicts).
    Returns [(name, shape), ...]."""
    LxMm, LyMm, h = payload['LxMm'], payload['LyMm'], payload['hMm']
    face = _outline_face(payload['outline'])
    plate = BRepPrimAPI_MakePrism(face, gp_Vec(0, 0, h)).Shape()
    # Everything braces may occupy: the plate footprint, extruded well above
    # the tallest brace, minus the sound hole.
    keep = BRepPrimAPI_MakePrism(face, gp_Vec(0, 0, h + 500.0)).Shape()
    hole = payload.get('hole')
    if hole and hole['r'] > 0.01:
        cyl = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(hole['x'], hole['y'], -1.0), gp_Dir(0, 0, 1)),
                                       hole['r'], h + 502.0).Shape()
        plate = BRepAlgoAPI_Cut(plate, cyl).Shape()
        keep = BRepAlgoAPI_Cut(keep, cyl).Shape()
    shapes = [('top_plate', _clean(plate))]

    groups = {}   # (E, rho) -> fused solid
    for b in payload['braces']:
        key = (round(float(b.get('modulus', 10e9)) / 1e9, 2), round(float(b.get('density', 420.0))))
        solid = BRepAlgoAPI_Common(_brace_solid(b, LxMm, LyMm, h), keep).Shape()
        if _volume(solid) <= 1e-6:
            continue
        groups[key] = solid if key not in groups else BRepAlgoAPI_Fuse(groups[key], solid).Shape()
    placed = None
    for (E, rho), solid in groups.items():
        # Braces of different materials can still cross each other; give the
        # crossing to whichever material group came first, so no two exported
        # bodies ever overlap.
        if placed is not None:
            solid = BRepAlgoAPI_Cut(solid, placed).Shape()
        placed = solid if placed is None else BRepAlgoAPI_Fuse(placed, solid).Shape()
        name = f'braces_E{E:g}GPa_rho{rho:g}' if len(groups) > 1 else 'braces'
        shapes.append((name, _clean(solid)))
    for name, s in shapes:
        if not BRepCheck_Analyzer(s).IsValid():
            raise RuntimeError(f'{name}: generated solid failed validity check')
    return shapes


def write_step(shapes, path):
    """Write named solids to an AP214 STEP file in millimetres."""
    Interface_Static.SetCVal_s('write.step.unit', 'MM')
    Interface_Static.SetCVal_s('write.step.schema', 'AP214IS')
    doc = TDocStd_Document(TCollection_ExtendedString('XmlOcaf'))
    tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    for name, shape in shapes:
        label = tool.AddShape(shape, False)
        TDataStd_Name.Set_s(label, TCollection_ExtendedString(name))
    writer = STEPCAFControl_Writer()
    writer.SetNameMode(True)
    if not writer.Transfer(doc, STEPControl_AsIs):
        raise RuntimeError('STEP transfer failed')
    if writer.Write(path) != IFSelect_RetDone:
        raise RuntimeError(f'could not write {path}')


def export_step(payload, path):
    shapes = build_design_shapes(payload)
    write_step(shapes, path)
    return [{'name': n, 'volume_mm3': round(_volume(s), 1)} for n, s in shapes]
