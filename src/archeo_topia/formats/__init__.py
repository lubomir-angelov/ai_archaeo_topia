"""Format classes shared by inference, training and the GIS deliverable.

Four types, and one rule that shapes all of them.

``LabelSchema``
    The annotation schema of record: classes, attributes, defaults, and the
    projections CVAT and OGR each need.
``CocoDocument``
    COCO instance annotations, in **image pixel** coordinates.
``FeatureCollection``
    Vector features, in **world** coordinates, written as GeoJSON or
    GeoPackage.
``SheetReference``
    The georeferencing that connects the two.

**The rule: COCO and GeoJSON do not convert into one another directly.** One
is in pixels and the other is on the ground, so every conversion takes a
``SheetReference`` as a required argument. ``AGENTS.md`` puts it as *"avoid
hidden coordinate assumptions"*; making the reference impossible to omit is
how that is enforced rather than merely intended.

**GDAL appears only as a subprocess.** Its Python bindings must match the
system ``libgdal`` exactly and must be built against an already-installed
numpy, or they import and then fail with ``no module named _gdal_array`` --
see the GDAL section of ``README.md``. Nothing here imports ``osgeo``;
GeoJSON is written with the standard library and ``ogr2ogr`` converts it.
"""

from archeo_topia.formats.coco import CocoDocument
from archeo_topia.formats.features import Feature, FeatureCollection
from archeo_topia.formats.georeference import SheetReference
from archeo_topia.formats.labels import LabelSchema, LabelValidationError

__all__ = [
    "CocoDocument",
    "Feature",
    "FeatureCollection",
    "LabelSchema",
    "LabelValidationError",
    "SheetReference",
]
