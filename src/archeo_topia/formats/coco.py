#!/usr/bin/env python3
"""COCO instance annotations, in image pixel coordinates.

One class replacing logic that had spread across five modules: two independent
uncompressed-RLE decoders with different fallback behaviour, the category
triple written literally in three places, ``{id: name}`` index building in
four, and the sheet-id rule in three.

**The decoder here is the union of the two it replaces.**
``prepare_mapsam_coco.decode_uncompressed_coco_rle`` validated
``sum(counts) == h*w`` and rejected compressed string counts but lived behind a
PIL-image API; ``build_detection_windows.decode_mask`` returned a numpy array
but silently produced an **empty mask** for multi-polygon segmentation and for
box-only annotations. Silently empty is the dangerous one: it costs an
annotation with no error at all. Measured on the v0.0.3 export, the 714
annotations are 161 uncompressed RLE, 20 polygons, and **533 carrying a box and
no segmentation** -- the last group being exactly what the old decoder returned
empty for.

**Coordinates here are pixels.** Converting to and from ground coordinates
needs a :class:`~archeo_topia.formats.georeference.SheetReference`, which is
why :meth:`CocoDocument.to_features` takes one as a required argument.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from archeo_topia.formats.features import FeatureCollection, point, polygon
from archeo_topia.formats.georeference import SheetReference
from archeo_topia.formats.labels import LabelSchema

logger = logging.getLogger(__name__)

#: Class names, in the id order every export of this project has used.
MOUND = "mound"
HARD_NEGATIVE = "hard_negative_symbol"
UNCERTAIN = "uncertain_ignore"

#: The single home for the category triple. ``mapsam_end_to_end`` notes that
#: ids are not stable across re-exports, so resolve by name and treat these ids
#: as the convention for documents this project *writes*.
CATEGORIES: tuple[dict[str, Any], ...] = (
    {"id": 1, "name": MOUND, "supercategory": ""},
    {"id": 2, "name": HARD_NEGATIVE, "supercategory": ""},
    {"id": 3, "name": UNCERTAIN, "supercategory": ""},
)

#: Smallest ring, in positions, that CVAT will accept as a polygon.
MIN_POLYGON_POINTS = 3

Image.MAX_IMAGE_PIXELS = None


def sheet_of(file_name: str) -> str:
    """Return the sheet id a clip file name belongs to.

    Clips are ``<sheet>_<index>.png`` and sheet ids contain hyphens, so the
    split is on the **last** underscore. Written three times across the
    codebase before this.

    Args:
        file_name: Clip file name or stem.

    Returns:
        The sheet id.
    """
    return Path(file_name).stem.rsplit("_", 1)[0]


def decode_rle(segmentation: dict[str, Any], height: int, width: int) -> np.ndarray:
    """Decode uncompressed, column-major COCO RLE.

    Args:
        segmentation: RLE dict with ``size`` and ``counts``.
        height: Image height.
        width: Image width.

    Returns:
        Boolean mask of shape ``(height, width)``.

    Raises:
        ValueError: If counts are compressed or negative, if the declared size
            disagrees with the image, or if the counts do not sum to the mask
            area.
    """
    counts = segmentation.get("counts")
    if not isinstance(counts, list):
        raise ValueError(
            "compressed COCO RLE is not supported; expected counts to be a list of integers"
        )
    if any(count < 0 for count in counts):
        raise ValueError("invalid negative RLE count")

    # The two decoders this replaces disagreed here. One validated the counts
    # against the *image* dimensions; the other trusted the RLE's own ``size``
    # and pasted a differently-sized mask into the top-left corner. The strict
    # reading is taken, because a size mismatch is a data bug and pasting it
    # into a corner hides it. Every RLE annotation in v0.0.1 to v0.0.3 declares
    # a size equal to its image, so this rejects nothing that exists today.
    declared = tuple(segmentation.get("size", (height, width)))
    if declared != (height, width):
        raise ValueError(f"RLE size {declared} does not match the image ({height}, {width})")

    expected = height * width
    if sum(counts) != expected:
        raise ValueError(f"invalid RLE counts: sum={sum(counts)} != h*w={expected}")

    flat = np.zeros(expected, dtype=bool)
    index = 0
    filled = False
    for count in counts:
        if filled:
            flat[index : index + count] = True
        index += count
        filled = not filled

    return flat.reshape((height, width), order="F")


def decode_polygon(ring: list[float], height: int, width: int) -> np.ndarray:
    """Rasterize one flat COCO polygon ring.

    Args:
        ring: Flat ``[x0, y0, x1, y1, ...]``.
        height: Image height.
        width: Image width.

    Returns:
        Boolean mask.
    """
    canvas = Image.new("L", (width, height), 0)
    if len(ring) >= MIN_POLYGON_POINTS * 2:
        ImageDraw.Draw(canvas).polygon(
            [(ring[i], ring[i + 1]) for i in range(0, len(ring), 2)], fill=1
        )
    return np.array(canvas) > 0


def decode_bbox(bbox: list[float], height: int, width: int) -> np.ndarray:
    """Rasterize a COCO ``[x, y, w, h]`` box as a filled rectangle.

    The fallback for a box-only annotation, which the hard negatives all are.

    Args:
        bbox: COCO box.
        height: Image height.
        width: Image width.

    Returns:
        Boolean mask.
    """
    x, y, w, h = bbox
    canvas = Image.new("L", (width, height), 0)
    ImageDraw.Draw(canvas).rectangle([x, y, x + w, y + h], fill=1)
    return np.array(canvas) > 0


def decode_mask(
    annotation: dict[str, Any],
    width: int,
    height: int,
    bbox_fallback: bool = True,
) -> np.ndarray:
    """Rasterize any COCO annotation this project produces or consumes.

    Handles, in order: uncompressed RLE, a single flat polygon, a list of
    polygons unioned together, and a bbox fallback when the segmentation is
    empty. The last two are the cases ``build_detection_windows.decode_mask``
    returned an empty mask for.

    The argument order is ``(width, height)`` to match the function it
    replaces, so existing call sites move across unchanged.

    **``bbox_fallback`` exists to protect a published result.** The v0.0.1
    export has one mound carrying a box and no mask -- a symbol truncated at a
    tile edge -- and v0.1 to v0.5 were all computed treating it as having no
    geometry. Filling its box instead would silently change the component
    grouping and the ``mounds_without_geometry`` count those runs reported. So
    the dataset builders pass ``False`` and keep their historical behaviour,
    while anything new gets the more useful default. v0.0.2 and v0.0.3 have no
    such annotation, so the flag changes nothing for current work.

    Args:
        annotation: A COCO annotation.
        width: Image width.
        height: Image height.
        bbox_fallback: Fill the bounding box when there is no segmentation.

    Returns:
        Boolean mask of shape ``(height, width)``.
    """
    segmentation = annotation.get("segmentation")
    bbox = annotation.get("bbox")

    if isinstance(segmentation, dict):
        return decode_rle(segmentation, height, width)

    if isinstance(segmentation, list) and segmentation:
        first = segmentation[0]
        if isinstance(first, (int, float)):
            return decode_polygon(list(segmentation), height, width)
        if isinstance(first, list):
            mask = np.zeros((height, width), dtype=bool)
            for ring in segmentation:
                if len(ring) >= MIN_POLYGON_POINTS * 2:
                    mask |= decode_polygon(ring, height, width)
            return mask

    if bbox and bbox_fallback:
        return decode_bbox(list(bbox), height, width)

    return np.zeros((height, width), dtype=bool)


@dataclass
class CocoDocument:
    """A COCO instance-segmentation document.

    Attributes:
        document: The parsed COCO dict.
    """

    document: dict[str, Any] = field(default_factory=dict)

    # ---- construction -------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> CocoDocument:
        """Read a COCO JSON file.

        JSON only. YAML is offered for the label schema, which a person edits,
        and deliberately not here: these documents are hundreds of kilobytes of
        machine-generated geometry, and CVAT reads JSON.

        Args:
            path: The file.

        Returns:
            The document.
        """
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def empty(cls, description: str = "") -> CocoDocument:
        """Build an empty document shaped like a CVAT export.

        Args:
            description: Text for the ``info`` block.

        Returns:
            The document.
        """
        return cls(
            {
                "licenses": [{"name": "", "id": 0, "url": ""}],
                "info": {
                    "contributor": "",
                    "date_created": "",
                    "description": description,
                    "url": "",
                    "version": "",
                    "year": "",
                },
                "categories": [dict(category) for category in CATEGORIES],
                "images": [],
                "annotations": [],
            }
        )

    def save(self, path: str | Path, indent: int | None = None) -> Path:
        """Write the document as JSON.

        Args:
            path: Destination.
            indent: JSON indent, or None for the compact form CVAT exports use.

        Returns:
            The path written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.document, indent=indent), encoding="utf-8")
        return path

    # ---- indexes ------------------------------------------------------------

    @property
    def images(self) -> list[dict[str, Any]]:
        """The image records."""
        return self.document.setdefault("images", [])

    @property
    def annotations(self) -> list[dict[str, Any]]:
        """The annotation records."""
        return self.document.setdefault("annotations", [])

    @property
    def categories(self) -> list[dict[str, Any]]:
        """The category records."""
        return self.document.setdefault("categories", [])

    @property
    def images_by_id(self) -> dict[int, dict[str, Any]]:
        """Image records keyed by id."""
        return {image["id"]: image for image in self.images}

    @property
    def images_by_name(self) -> dict[str, dict[str, Any]]:
        """Image records keyed by file name."""
        return {image["file_name"]: image for image in self.images}

    @property
    def category_names(self) -> dict[int, str]:
        """Category names keyed by id."""
        return {category["id"]: category["name"] for category in self.categories}

    @property
    def category_ids(self) -> dict[str, int]:
        """Category ids keyed by name.

        Resolve by name rather than by id: ``mapsam_end_to_end`` records that
        ids are not stable across CVAT re-exports.
        """
        return {category["name"]: category["id"] for category in self.categories}

    def annotations_for(self, image_id: int) -> list[dict[str, Any]]:
        """Return one image's annotations.

        Args:
            image_id: The image id.

        Returns:
            Its annotations.
        """
        return [a for a in self.annotations if a["image_id"] == image_id]

    def annotations_by_image(self) -> dict[int, list[dict[str, Any]]]:
        """Group annotations by image id in one pass.

        Returns:
            Image id to annotations.
        """
        grouped: dict[int, list[dict[str, Any]]] = {image["id"]: [] for image in self.images}
        for annotation in self.annotations:
            grouped.setdefault(annotation["image_id"], []).append(annotation)
        return grouped

    def counts_by_category(self) -> dict[str, int]:
        """Count annotations per category name.

        Returns:
            Category name to count.
        """
        names = self.category_names
        counts: dict[str, int] = {category["name"]: 0 for category in self.categories}
        for annotation in self.annotations:
            counts[names[annotation["category_id"]]] += 1
        return counts

    # ---- mutation -----------------------------------------------------------

    def add_image(self, file_name: str, width: int, height: int) -> dict[str, Any]:
        """Append an image record shaped like CVAT's own exports.

        Args:
            file_name: Clip file name.
            width: Image width.
            height: Image height.

        Returns:
            The record added.
        """
        record = {
            "id": len(self.images) + 1,
            "file_name": file_name,
            "width": width,
            "height": height,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0,
        }
        self.images.append(record)
        return record

    def add_annotation(
        self,
        image_id: int,
        category: str,
        bbox: list[float],
        segmentation: list[list[float]] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append an annotation record.

        Args:
            image_id: Owning image id.
            category: Category name.
            bbox: COCO ``[x, y, w, h]``.
            segmentation: Polygon rings, or None for a box-only shape.
            attributes: Attribute bag.

        Returns:
            The record added.
        """
        record = {
            "id": len(self.annotations) + 1,
            "image_id": image_id,
            "category_id": self.category_ids[category],
            "segmentation": segmentation or [],
            "area": float(bbox[2] * bbox[3]),
            "bbox": [float(v) for v in bbox],
            "iscrowd": 0,
            "attributes": dict(attributes or {}),
        }
        self.annotations.append(record)
        return record

    def decode(self, annotation: dict[str, Any]) -> np.ndarray:
        """Rasterize an annotation using its own image's dimensions.

        Args:
            annotation: A COCO annotation belonging to this document.

        Returns:
            Boolean mask.
        """
        image = self.images_by_id[annotation["image_id"]]
        return decode_mask(annotation, image["width"], image["height"])

    # ---- conversion to and from world coordinates ---------------------------

    def to_features(
        self,
        reference: SheetReference,
        kind: str = "point",
        schema: LabelSchema | None = None,
        category: str = MOUND,
        layer_name: str | None = None,
        id_prefix: str | None = None,
    ) -> FeatureCollection:
        """Project annotations into world coordinates.

        ``reference`` is required, not optional. COCO is in pixels and a
        feature collection is on the ground, so without a geotransform this
        conversion has no defined answer -- and a silently-assumed one is the
        *"hidden coordinate assumption"* ``AGENTS.md`` forbids.

        Args:
            reference: The sheet's georeferencing.
            kind: ``point`` for the archaeological location, ``polygon`` for
                the printed symbol geometry.
            schema: Label schema, used to normalise the attribute bag.
            category: Which category to project.
            layer_name: Layer name; defaults by ``kind``.
            id_prefix: Prefix for ``mound_id``; defaults to the sheet id.

        Returns:
            The collection, in the reference's CRS.

        Raises:
            ValueError: If the sheet is not georeferenced, or ``kind`` is not
                recognised.
        """
        if kind not in ("point", "polygon"):
            raise ValueError(f"kind must be 'point' or 'polygon', not {kind!r}")
        if not reference.georeferenced:
            raise ValueError(
                f"{reference.sheet_id} carries no geotransform, so it cannot be projected. "
                "Read the parent GeoTIFF or a clips.json that records one."
            )

        prefix = id_prefix or reference.sheet_id
        collection = FeatureCollection(
            crs=reference.crs,
            name=layer_name or ("mound_points" if kind == "point" else "mound_symbols"),
        )
        names = self.category_names
        images = self.images_by_id

        for index, annotation in enumerate(self.annotations, start=1):
            if names[annotation["category_id"]] != category:
                continue
            image = images[annotation["image_id"]]
            clip = image["file_name"]
            mound_id = f"{prefix}-{index:05d}"

            if kind == "point":
                x, y = _bbox_centre(annotation["bbox"])
                ground = reference.clip_to_ground(clip, x, y)
                if ground is None:
                    continue
                properties = {"mound_id": mound_id, "sheet_id": reference.sheet_id}
                attributes = annotation.get("attributes", {})
                properties.update(
                    schema.coerce(category, attributes) if schema else dict(attributes)
                )
                collection.add(point(*ground), **properties)
            else:
                rings = annotation.get("segmentation")
                if not isinstance(rings, list) or not rings:
                    continue
                ring = rings[0] if isinstance(rings[0], list) else rings
                ground_ring = []
                for i in range(0, len(ring), 2):
                    ground = reference.clip_to_ground(clip, ring[i], ring[i + 1])
                    if ground is None:
                        break
                    ground_ring.append(ground)
                if len(ground_ring) < MIN_POLYGON_POINTS:
                    continue
                collection.add(
                    polygon(ground_ring),
                    mound_id=mound_id,
                    sheet_id=reference.sheet_id,
                    symbol_area_px=float(annotation.get("area", 0.0)),
                )

        return collection

    @classmethod
    def from_features(
        cls,
        collection: FeatureCollection,
        reference: SheetReference,
        schema: LabelSchema | None = None,
        category: str = MOUND,
        description: str = "",
    ) -> CocoDocument:
        """Project world-coordinate features back into a COCO document.

        The return path: a reviewer edits in QGIS or ArcGIS, and their verdicts
        have to land in CVAT beside the existing annotations.

        Args:
            collection: Features in the reference's CRS.
            reference: The sheet's georeferencing.
            schema: Label schema, used to normalise the attribute bag.
            category: Category to assign.
            description: Text for the ``info`` block.

        Returns:
            A COCO document in clip pixel coordinates.

        Raises:
            ValueError: If the sheet is not georeferenced or has no clips.
        """
        if not reference.georeferenced:
            raise ValueError(f"{reference.sheet_id} carries no geotransform")
        if not reference.clips:
            raise ValueError(
                f"{reference.sheet_id} has no clips recorded, so features cannot be placed "
                "in a clip's pixel space. Build the reference from a clips.json."
            )

        document = cls.empty(description or f"Reviewed features for {reference.sheet_id}")
        for clip in reference.clips:
            document.add_image(clip.file, clip.size[0], clip.size[1])
        image_ids = {image["file_name"]: image["id"] for image in document.images}

        for feature in collection.features:
            geometry = feature.geometry
            if geometry.get("type") != "Point":
                continue
            easting, northing = geometry["coordinates"][:2]
            sheet_xy = reference.from_ground(easting, northing)
            if sheet_xy is None:
                continue
            placed = reference.sheet_to_clip(*sheet_xy)
            if placed is None:
                logger.warning(
                    "feature %s at (%.1f, %.1f) falls outside every clip of %s; dropped",
                    feature.properties.get("mound_id", "?"),
                    easting,
                    northing,
                    reference.sheet_id,
                )
                continue
            clip_name, x, y = placed
            properties = dict(feature.properties)
            width = float(properties.pop("symbol_width_px", 25.0))
            height = float(properties.pop("symbol_height_px", 23.0))
            attributes = schema.coerce(category, properties) if schema else properties
            document.add_annotation(
                image_id=image_ids[clip_name],
                category=category,
                bbox=[x - width / 2, y - height / 2, width, height],
                attributes=attributes,
            )
        return document


def _bbox_centre(bbox: list[float]) -> tuple[float, float]:
    """Return a COCO box's centre.

    Args:
        bbox: COCO ``[x, y, w, h]``.

    Returns:
        ``(x, y)``.
    """
    x, y, w, h = bbox
    return (x + w / 2, y + h / 2)
