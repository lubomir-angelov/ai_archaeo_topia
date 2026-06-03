## 🗺️ The Architecture: How the Models Work Together
Instead of forcing a single model to do everything, you are creating a specialized pipeline where each model has a clear operational boundary:

* **The Brain / Geospatial Agent (Qwen_3.6_27B):** This model acts as your primary Agent Orchestrator. Because it has multimodal understanding, it owns the geospatial reasoning loop: visual interpretation of map tiles, archaeological domain reasoning, coding strategy, tool selection, MCP routing, and final decision-making. When a map image, tile, PDF preview, or extracted artifact is fed into the system, Qwen_3.6_27B maps out the technical blueprint, decides whether OCR, segmentation, geometry extraction, or document understanding is required, and structures the overall logic.
* **The OCR Engine (DeepSeek OCR MCP):** This service provides high-resolution OCR and PDF-to-text extraction. It is not the reasoning layer. Its job is to extract faithful text from scanned PDFs, map legends, marginalia, elevation labels, place names, symbol tables, dense annotations, and other text-heavy artifacts. It returns structured Markdown/text, page references, bounding boxes where possible, and OCR confidence metadata.
* **The Geometry Engine (SAM 2 MCP):** This service provides object detection support, segmentation, mask generation, polygon strings, and geometry artifacts. It operates only on images with maps. It should not be used for ordinary documents, screenshots, UI images, or non-map visual material. Qwen_3.6_27B decides when a map tile contains a target worth segmenting, then sends SAM 2 a point prompt, bounding box, or cropped map tile. SAM 2 returns masks, polygons, bounding boxes, and mask artifacts.
* **The Document Understanding / Support Model (Qwen_3.6_35B_A3B):** This model handles document understanding, lightweight reasoning over extracted PDF text, schema cleanup, smaller coding tasks, validation helpers, metadata extraction, and annotation-format transformations. It supports the Brain, but it does not replace the geospatial agent. Its role is to keep document-heavy and smaller implementation tasks away from the main geospatial reasoning loop.

------------------------------
## 🛠️ The MCP Stack: Should You Deploy DeepSeek OCR & SAM 2?
Absolutely yes. Integrating these services via the Model Context Protocol (MCP) addresses the inherent architectural weaknesses of even the best multimodal LLMs while keeping the system modular, inspectable, and easy to improve.

## 1. DeepSeek OCR MCP Service (Deploy It)

* **Why you need it:** Multimodal LLMs can still hallucinate text inside images, especially when the text is small, rotated, degraded, compressed, handwritten, or embedded in scanned PDFs. Archaeological maps frequently contain legends, site labels, contour values, grid references, marginal notes, publication metadata, and symbol tables that must be extracted exactly.
* **How the Agent uses it:** If Qwen_3.6_27B receives a PDF, scanned map sheet, legend crop, title block, or map tile with dense text, it executes a tool call to the DeepSeek OCR MCP service instead of guessing. The service returns structured Markdown/text plus OCR metadata. Qwen_3.6_27B then uses that exact text injection for reasoning, geospatial interpretation, and routing.
* **Where Qwen_3.6_35B_A3B helps:** For document-heavy inputs, Qwen_3.6_35B_A3B can summarize extracted text, normalize tables, classify document sections, extract map metadata, validate schema fields, and prepare clean intermediate JSON for the geospatial agent.
* **The Result:** You get reliable text extraction without letting the visual reasoning model invent labels, elevations, place names, or legend entries.

## 2. Segment Anything 2 (SAM 2) MCP Service (Deploy It, But Restrict It)

* **Why you need it:** Qwen_3.6_27B can understand what is visible in a map tile, but it should not be expected to produce exact pixel masks for complex archaeological symbols, irregular polygons, settlement outlines, roads, rivers, contours, field boundaries, hachures, or decorative map artifacts. SAM 2 provides the high-fidelity segmentation layer.
* **How the Agent uses it:** Qwen_3.6_27B identifies a candidate map feature and passes a point prompt, rough bounding box, or cropped tile to the SAM 2 MCP service. SAM 2 then returns a mask, polygon string, crop reference, and optional confidence/quality metadata.
* **Hard boundary:** SAM 2 MCP operates only on images with maps. It should not process ordinary PDFs as documents, raw text pages, UI screenshots, photographs unrelated to map annotation, or generic images. If the input is not a map image or map tile, Qwen_3.6_27B must route it elsewhere.
* **The Result:** You get automated, enterprise-grade data labeling and high-fidelity map annotations that an LLM alone could never reliably produce.

## 🗺️ Architecture Loop for Auto Annotation

The annotation loop should be rendered as a fenced `text` block so Markdown preview keeps the spacing, arrows, and box layout stable.

```text
[PDF / GeoTIFF / Map Image / Map Tile]
                  │
                  ▼
┌────────────────────────────────────────────┐
│ Qwen_3.6_27B                               │
│ Brain / Geospatial Agent                   │
│ Reasoning + Knowledge + Coding             │
│ Visual Understanding + MCP Routing         │
└─────────────────────┬──────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
┌────────────────┐ ┌────────────────┐ ┌────────────────────┐
│ DeepSeek OCR   │ │ SAM 2 MCP      │ │ Qwen_3.6_35B_A3B   │
│ MCP            │ │                │ │                    │
│                │ │                │ │                    │
│ OCR / PDF text │ │ Map-only       │ │ Document           │
│ High-res OCR   │ │ segmentation   │ │ understanding      │
│ Legends/labels │ │ Masks/polygons │ │ Small code/schema  │
└───────┬────────┘ └───────┬────────┘ └─────────┬──────────┘
        │                  │                    │
        └──────────────────┼────────────────────┘
                           ▼
┌────────────────────────────────────────────┐
│ Qwen_3.6_27B                               │
│ Merges OCR + geometry + metadata           │
│ Validates labels and spatial logic         │
│ Produces annotation instructions           │
└─────────────────────┬──────────────────────┘
                      ▼
┌────────────────────────────────────────────┐
│ Annotation / GIS Outputs                   │
│ COCO JSON / YOLO / GeoJSON / masks         │
│ Pixel coordinates + geo-coordinates        │
│ QA reports + review queues                 │
└────────────────────────────────────────────┘
```

## 🛠️ Strategic Shifts for Your Specific Use Case

## 1. Qwen_3.6_27B Becomes the Geospatial Brain

* **The Role:** Qwen_3.6_27B is not just a coding model in this architecture. It is the main geospatial agent: reasoning layer, visual understanding layer, archaeological knowledge layer, coding planner, and MCP orchestrator.
* **The Logic:** It decides whether an input should go to OCR, SAM 2, Qwen_3.6_35B_A3B, or direct annotation formatting. It understands the context of archaeological maps, distinguishes candidate symbols from hard negatives, and decides when a segmentation proposal is plausible enough to keep.
* **The Payoff:** You keep one strong multimodal model responsible for the global state of the annotation task, instead of scattering reasoning across disconnected tools.

## 2. SAM 2 MCP Becomes a Map-Only Bootstrapper

* **The Role:** Instead of SAM 2 being the final output, it acts as a pre-labeler inside a weak-supervision loop.
* **The Logic:** Qwen_3.6_27B identifies a landmass, archaeological symbol, settlement mark, water body, road, field boundary, contour zone, or suspicious map feature. It sends a point prompt, bounding box, or crop to the local SAM 2 MCP service. SAM 2 returns a raw binary mask, polygon string, and mask artifact.
* **The Fine-Tuning Payoff:** Qwen_3.6_27B and Qwen_3.6_35B_A3B can format the resulting polygon data into standard training schemas such as COCO JSON, YOLO segmentation, GeoJSON, or mask PNG references. These auto-generated labels can feed directly into the training pipeline for a bespoke map-optimized SAM 2 model.

## 3. DeepSeek OCR Owns High-Resolution OCR and PDF-to-Text

Archaeological maps are not just images. They are also documents with legends, scales, coordinates, publication notes, sheet names, grid references, contour labels, settlement names, and symbol explanations.

* **Text Extraction:** DeepSeek OCR MCP extracts text from full PDFs, page crops, legends, rotated labels, dense map annotations, and small elevation labels. This avoids hallucinated labels and gives the geospatial agent a factual text layer.
* **PDF Handling:** If the input is a PDF, DeepSeek OCR MCP should be the first document extraction step. It should produce text, Markdown, page-level metadata, and references to any image crops that need visual follow-up.
* **Agent Use:** Qwen_3.6_27B consumes the OCR output and decides whether the text should be linked to map geometry, used for legend interpretation, or passed to Qwen_3.6_35B_A3B for document cleanup and schema extraction.

## 4. Qwen_3.6_35B_A3B Handles Document Understanding and Smaller Coding Tasks

* **The Role:** Qwen_3.6_35B_A3B is the support model for document-heavy reasoning, smaller coding tasks, validation scripts, schema normalization, promptable transformations, and extracted-text interpretation.
* **The Logic:** It can take the DeepSeek OCR output and convert it into structured JSON, normalize labels, clean tables, infer document sections, prepare metadata, and generate small helper scripts.
* **The Boundary:** It should not own the global geospatial loop. The final spatial decision-making, map interpretation, and tool orchestration stay with Qwen_3.6_27B.

## 5. Isolines: The Crucial Combo of OCR + Geometry + Geospatial Agent

Isolines present a notorious challenge because text labels such as “2500 ft” or “450 m” break up the line geometry, and the lines themselves require dense geometric mapping.

* **Text Extraction (DeepSeek OCR):** Offloading contour-label crops to DeepSeek OCR MCP ensures that tiny, rotated, curved, or degraded elevation numbers are extracted from the map instead of guessed by the agent.
* **Geometry Processing (SAM 2 MCP + local scripts):** SAM 2 can propose masks or segments for contour-like structures when prompted on map tiles. For continuous tracing, Qwen_3.6_27B can generate or request local Python routines using OpenCV, GDAL, Rasterio, or Shapely-style geometry processing to trace linework and associate it with OCR-derived elevation values.
* **Final Association (Qwen_3.6_27B):** The Brain links the OCR label, the traced geometry, the tile offset, and the georeferencing transform into a coherent annotation record.

## 6. Handling Georeferencing (Spatial Transforms)

Maps contain real-world coordinates such as Latitude/Longitude, UTM, national grids, or sheet-specific projected coordinates. AI models and segmentation tools operate primarily in pixel coordinates.

* **The Fix:** Qwen_3.6_27B must treat georeferencing as a first-class part of the annotation workflow. Pixel coordinates, tile offsets, masks, polygons, and bounding boxes must be mapped back into the source raster coordinate system.
* **The Workflow:** When SAM 2 returns a mask or polygon in tile-local pixel coordinates, Qwen_3.6_27B converts it into full-image pixel coordinates, then into geospatial coordinates using GeoTIFF metadata, world files, GCPs, or external georeferencing metadata.
* **The Output:** Each annotation should preserve both pixel-space geometry and geo-space geometry where possible. This allows the same label to be used for ML training, GIS review, and downstream archaeological analysis.

------------------------------
## ⚠️ Critical Adjustments for Map Pipelines

To make this architecture successful for map annotation, you must account for these hardware, software, and data-quality constraints:

1. **Implement Tiling (Image Slicing):** Georeferenced maps are often massive, for example 10000 × 10000 pixels or larger. Passing a massive map straight into the agent loop will downsample fine linework and destroy thin isolines, small symbols, grid marks, and labels. Your orchestration code must slice maps into smaller tiles, such as 1024 × 1024 or 2048 × 2048, preserve tile offsets, and stitch resulting annotations back into full-image and geo-coordinate space.
2. **Keep SAM 2 Map-Only:** SAM 2 MCP must only process image tiles that are confirmed to be maps. This prevents the segmentation service from becoming a generic visual tool and keeps its outputs meaningful for archaeological annotation.
3. **Separate OCR From Segmentation:** Text extraction and geometry extraction should remain separate MCP calls. DeepSeek OCR produces reliable text; SAM 2 produces geometry. Qwen_3.6_27B merges them.
4. **Preserve Provenance:** Every annotation should record the source file, page number if applicable, tile ID, tile offset, OCR source crop if used, SAM 2 prompt type, mask path, polygon string, model/tool version, and confidence metadata.
5. **Add Review Queues:** Auto-annotation should produce reviewable outputs, not silently accepted ground truth. Low-confidence masks, ambiguous symbols, OCR conflicts, and geometry/text mismatches should be routed to manual review in CVAT or the chosen annotation UI.
6. **Optimize VRAM and Model Residency:** Keep the active models resident where possible. Thousands of map tiles will be processed sequentially, and model reloads or CPU offloading will bottleneck the pipeline. The practical goal is stable throughput, not just peak tokens/sec.
7. **Use Hard Negatives Explicitly:** Roads, decorative marks, grid artifacts, text, borders, water hachures, print noise, and contour fragments should be modeled as hard negatives. Qwen_3.6_27B should explicitly tag confusing regions so they can improve the training dataset.

------------------------------
## ✅ Summary of the Architectural Validity
Your layout is highly effective for this task. It cleanly decouples geospatial reasoning, high-resolution OCR, map-only segmentation, document understanding, and annotation formatting.

The key improvement is that **Qwen_3.6_27B becomes the Brain**: the multimodal geospatial agent responsible for reasoning, archaeological knowledge, visual interpretation, coding strategy, and MCP orchestration.

DeepSeek OCR MCP becomes the reliable high-resolution text and PDF extraction service. SAM 2 MCP becomes the map-only segmentation and polygon-generation service. Qwen_3.6_35B_A3B becomes the document-understanding and smaller-coding support model.

This gives you a stronger architecture than a single-model setup because each component has a narrow responsibility:

* Qwen_3.6_27B decides what the map means and what tools to call.
* DeepSeek OCR MCP extracts the text faithfully.
* SAM 2 MCP extracts masks and polygons from map imagery only.
* Qwen_3.6_35B_A3B cleans documents, formats schemas, and handles smaller support tasks.
* The output layer produces COCO JSON, YOLO, GeoJSON, mask PNGs, QA reports, and review queues.

If you are ready to implement the data pipeline, the next files to refine are the MCP tool contracts and the agent routing prompts:

* `deepseek_ocr_mcp`: PDF/text extraction contract.
* `sam2_mcp`: map-only segmentation contract.
* `qwen_27b_geospatial_agent`: orchestration and routing prompt.
* `qwen_35b_a3b_document_support`: document cleanup and smaller coding prompt.
* `annotation_output_writer`: COCO / YOLO / GeoJSON / mask artifact writer.
