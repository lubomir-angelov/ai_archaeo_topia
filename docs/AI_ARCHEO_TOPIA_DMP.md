# Data Management Plan: AI ArchaeoTopia

**Project Title:** AI ArchaeoTopia: Archaeology of Topographic Maps Using Artificial Intelligence

**Institution:** National Archaeological Institute with Museum, Bulgarian Academy of Sciences (NAIM-BAS)

**Project reference number:** Contract with Bulgarian National Science Fund, Project No. КП-06-Н100/5 from 10.12.2025

**Project duration:** 2025-2028

**Principal Investigator:** Assist. Prof. Dr. Nadezhda Kecheva, NAIM-BAS

**Team members:**
* **Assist. Prof. Dr. Angel Grigorov**, NAIM-BAS
* **Assoc. Prof. Dr. Ivo D. Cholakov**, NAIM-BAS
* **Assoc. Prof. Dr. Julia Tzvetkova**, Sofia University
* **Assist. Prof. Dr. Kalin Dimitrov**, NAIM-BAS
* **Veronika Gencheva**, PhD student, NAIM-BAS
* **Ivet Kirilova**, PhD student, NAIM-BAS
* **Lyubomir Angelov**, master student, IT and AI specialist, Burgas Free University
* **Lyubomir Nedyalkov**, IT developer
* **Assoc. Prof. Dr. Adela Sobotkova**, Aarhus University, Denmark, foreign expert in digital archaeology
* **Dr. Iban Berganzo-Besga**, Barcelona Supercomputing Center, Spain, foreign expert in AI applications in archaeology


## Document History (Version Control)
*As a dynamic DMP, this table tracks major changes to the project's data architecture during execution.*

| Version | Date | Description of Changes | Author |
| :--- | :--- | :--- | :--- |
| 1.0 | 11.01.2026 | Initial DMP drafted. Project commenced. Automated georeferencing of 1:25,000 maps initiated. | N. Kecheva |
| | | | |
| 1.1 | 28.04.2026 | DMP updated to align with project progress. Clarification of data categories, refined estimates of data volume, and updated workflow descriptions for georeferencing, model training, and validation processes. | N. Kecheva |

# **Table of Contents**

1. **Data Summary**
2. **FAIR Data**
3. **Allocation of Resources**
4. **Data Protection**
5. **Archiving and Preservation**
6. **Roles and Responsibilities**


---

## 1. Data Summary

### 1.1 Purpose of Data Collection and Generation
The project generates and processes geospatial and archaeological data to automatically identify and digitize burial and settlement mounds from mid-20th-century topographic maps using Artificial Intelligence. The ultimate goals are to:
*   Facilitate the rapid preservation of endangered archaeological sites - burial and settlement mounds. 
*   Provide a pilot digital terrain model (DTM) free from modern anthropogenic disturbances for landscape archaeology.
*   Produce reproducible AI (computer vision) workflows for archaeological cartography.

### 1.2 Types, Formats, and Volume of Data
The data workflow is divided into distinct categories and data management. All outputs will have corresponding **metadata**, which will be stored in files with `.json` extension, in the same directory as the corresponding data file. This metadata will include information such as data provenance, date of creation, data format, data type, etc. 
The total estimated volume of data generated throughout the project is **over 500 GB**. This is primarily due to the large size of the raster maps and the comprehensive DTM, which is substantial for a specific region. The sensitive vector data, while smaller in volume, is the most critical for heritage protection. All data will be managed through a structured folder system on the NAIM-BAS file servers and backed up regularly to institutional storage.

#### 1.2.1 Archival topographic maps at **1:25,000 scale** from "Balance of earth, 1969"
*   These maps are in raster format and are provided by the Agency for Geodesy, Cartography and Cadastre (AGKK).
*   **Total volume:** 50GB (approx. 1600 scanned maps with duplicates).
*   **Total coverage:** entire territory of Bulgaria in 1:25,000 scale maps
*   **Coordinate system:** 1950 System, vector grid in `EPSG:25835 - ETRS89 / UTM zone 35N`
*   **Storage location:** NAIM-BAS file server

#### 1.2.1.1. Goals

The goal of this data category is to provide the AI models with the necessary raw data to perform their tasks.

*   **1. Raw Data (Inputs):** 
    *   Archival topographic maps at **1:25,000 scale, 1969** provided by the Agency for Geodesy, Cartography and Cadastre (AGKK) (Raster formats, e.g., TIFF/JPEG).
    *   *Estimated Total Volume:* `over 30GB`
    *   Ground-truth verification data collected via GNSS and unmanned aerial vehicles (drones) with RTK corrections (Note: Conducted only on a targeted sample within specific test regions, not nationwide).
*   **2. Processed Data:** 
    *   Georeferenced topographic maps aligned to **EPSG:25835 - ETRS89 / UTM zone 35N** coordinate systems (GeoTIFF format).
    *   *Estimated Total Volume:* `over 30GB`
*   **3. Analytical Data & Code:** 
    *   Deep learning models for object detection and semantic segmentation (e.g., Foundation models like SAM, or specific models like CNN, U-Net, YOLO weights or fine-tuned models).
    *   Source code and data processing scripts (Python).
*   **4. Outputs & Results:** 
    *   **Sensitive:** Vector datasets representing the specific points and coordinates of archaeological symbols (burial and settlement mounds) (GeoJSON/GeoPackage).
    *   **Non-Sensitive:** Extracted isolines (contour lines) and the digital terrain model (DTM) (GeoTIFF).

**Sensitive data** refers to the vector data (point locations of the mounds). While the mounds themselves are not sensitive in a personal data sense, their precise coordinates are treated as sensitive to prevent looting. This data will only be uploaded to the restricted access GIS version of the Archaeological Map of Bulgaria (AIS AKB). The AIS AKB is a closed national system under the Ministry of Culture (Ordinance No. 2 of 6 April 2011) and is accessible only to authorized heritage professionals. This restriction ensures that the locations of the newly identified and vulnerable sites remain protected from unauthorized access.

**Non-sensitive data** includes the georeferenced topographic maps, the extracted contour lines, and the derived digital terrain model (DTM). These outputs are free from any personal data and do not pose a risk to heritage security. As such, they will be made fully accessible to the public through international data repositories, encouraging broader scientific reuse and collaboration.

#### 1.2.2 Archival topographic maps at **1:25,000 scale**, "Balance of earth, 1956"
*   These maps are in raster format and are provided by the Agency for Geodesy, Cartography and Cadastre (AGKK).
*   **Total volume:** 63GB (approx. `[to be added]` scanned maps with duplicates).
*   **Total coverage:** part of the territory of Bulgaria in 1:25,000 scale maps (specific regions to be documented as data is processed)
*   **Coordinate system:** 1930 System, vector grid in `EPSG:25835 - ETRS89 / UTM zone 35N`
*   **Storage location:** NAIM-BAS file server

#### 1.2.2.1. Goals
*  Georeferencing of the 1:25,000 scale maps from "Balance of earth, 1956" in coordinate system 1930 System, grid in `EPSG:25835 - ETRS89 / UTM zone 35N`. Uploading the digital archive of georeferenced maps to AIS AKB for future use in archaeological research. Giving them to AGKK for their archive.

#### 1.2.2.2. Processed Data
*  Georeferenced topographic maps in 1:25,000 scale, 1956, in coordinate system 1930 System, grid in `EPSG:25835 - ETRS89 / UTM zone 35N`.

#### 1.2.3 Archival topographic maps at **1:5,000 scale**
*   These maps are in raster format and are provided by the Agency for Geodesy, Cartography and Cadastre (AGKK).
*   **Total volume:** `220GB` (approx. `[to be added]` scanned maps).
*   **Total coverage:** `part of the territory of Bulgaria in 1:5,000 scale maps` (specific regions to be documented as data is processed)
*   **Coordinate system:** 1950 System, grid in `EPSG:25835 - ETRS89 / UTM zone 35N`
*   **Storage location:** NAIM-BAS file server

#### 1.2.3.1. Goals
* Georeferencing of the 1:5,000 scale maps in coordinate system 1950 System, grid in `EPSG:25835 - ETRS89 / UTM zone 35N`. Uploading the digital archive of georeferenced maps to AIS AKB for future use in archaeological research. Giving them to AGKK for their archive.

#### 1.2.3.2. Processed Data
*  Georeferenced topographic maps in coordinate system 1950 System, grid in `EPSG:25835 - ETRS89 / UTM zone 35N`.


#### 1.2.4 Archival topographic maps at **1:10,000 scale**
*   These maps are in raster format and are provided by the Agency for Geodesy, Cartography and Cadastre (AGKK).
*   **Total volume:** `32GB` (approx. `[to be added]` scanned maps).
*   **Total coverage:** `part of the territory of Bulgaria in 1:10,000 scale maps` (specific regions to be documented as data is processed)
*   **Coordinate system:** 1950 System, vector grid in `EPSG:25835 - ETRS89 / UTM zone 35N`
*   **Storage location:** NAIM-BAS file server

#### 1.2.4.1. Goals
* Georeferencing of the 1:10,000 scale maps in coordinate system 1950 System, grid in `EPSG:25835 - ETRS89 / UTM zone 35N`. Uploading the digital archive of georeferenced maps to AIS AKB for future use in archaeological research. Giving them to AGKK for their archive.

#### 1.2.4.2. Processed Data
*  Georeferenced topographic maps in 1:10,000 scale in coordinate system 1950 System, grid in `EPSG:25835 - ETRS89 / UTM zone 35N`.


### 1.3 Data Generation & Quality Control Workflow
*   **Georeferencing Protocol:** The 1:25,000 maps, 1969, and 1:5,000 maps undergo an initial automated georeferencing process. Because automated processes can introduce spatial shifts on historical maps, every map is subsequently subjected to **manual expert verification** in GIS software to correct polynomial distortions. The quality control metrics of this process (e.g., automated RMSE, manual RMSE, script diagnostics, and operator initials) are meticulously logged in a standardized Operational Metadata tracking table to ensure end-to-end traceability before final archival.
*   **Coordinate Reference System (CRS):** All spatial data is standardized to **EPSG:25835 - ETRS89 / UTM zone 35N**.
*   **AI Validation & Confidence Levels:** All automated mound detections are subjected to an **Expert Visual Check** in GIS. Due to the national scale of the project, **Field Validation (GNSS/Drone)** will be conducted only on a targeted sample within selected test regions to establish statistical reliability. This splits the output into two confidence levels: "Visually Verified" (potential sites) and "Field Verified" (confirmed sites). Validation data from AIS AKB and other projects (e.g. "TRAP project in Yambol region") will be used for validation of the AI models and for checking their accuracy.

### 1.4 File Naming Conventions
To ensure data remains findable and machine-readable across the team, strict naming conventions are applied:
*   **Rasters:** `[MapSheet_Nomenclature]_[Scale]_[Status].tif` (e.g., `K-35-043-1_25K_georef.tif`)
*   **Vectors:** `Mounds_[Region]_[Date].geojson`
*   **Code:** Scripts should be versioned via Git, with self-explanatory module names (e.g., `01_georeference_auto.py`).

### 1.5 Scientific Outputs and Deliverables
Beyond the raw and processed digital data, the project will generate several high-level academic and public-facing deliverables:
*   **Scientific Publications:** Articles in peer-reviewed open-access journals detailing the AI methodology, georeferencing accuracy, and archaeological findings. These publications and their associated metadata will be deposited in the **Bulgarian Portal for Open Science** (https://bpos.bg/).
*   **Edited Volume / Monograph:** A comprehensive edited volume synthesizing the project's results and the theoretical implications of applying AI to national archaeological cartography.
*   **Master Thesis:** A Master thesis by one of the team members based on one aspect of the project's findings.
*   **Presentations:** Presentations and proceedings at international conferences (e.g., CAA, EAA, etc.).
*   **Training & Workshops:** A dedicated workshop/seminar organized to train heritage professionals, students, and stakeholders in the AI and GIS methodologies developed during the project.
*   **Project Reports:** Interim and final reports submitted to the BNSF.

---

## 2. FAIR Data

### 2.1 Making Data Findable
*   **Metadata Standards:** All data will be described using established national and international metadata standards, including:
    *   **ISO 19115 / INSPIRE:** For all geographic and spatial datasets (Rasters, DTM).
    *   **CIDOC CRM (ISO 21127) / ARIADNEplus:** For all cultural heritage and archaeological datasets (Mound vectors).
    *   **STAC (SpatioTemporal Asset Catalog):** To catalog imagery specifically for deep learning pipelines.
*   **Persistent Identifiers (PIDs):** Non-sensitive datasets, project documentation, and software code will be deposited in trusted international repositories (e.g., Zenodo) and the **Bulgarian Portal for Open Science** (https://bpos.bg/), where they will be assigned persistent DOIs.

### 2.2 Making Data Accessible & Handling Sensitive Data
The project strictly follows the principle "as open as possible, as closed as necessary."

*   **RESTRICTED ACCESS (SENSITIVE DATA):** 
    *   The vector outputs representing the exact points, coordinates, and spatial locations of burial and settlement mounds are classified as **highly sensitive data**. 
    *   To protect these newly discovered and vulnerable sites from looting, this data will **ONLY** be uploaded to the GIS version of the Archaeological Map of Bulgaria (AIS AKB).
    *   The AIS AKB is a closed, legally regulated national system under Ordinance No. 2 of 6 April 2011 of the Ministry of Culture. Access to this specific data is strictly restricted to authorized heritage professionals and institutions.
*   **Open Access (Non-Sensitive Data):** 
    *   Georeferenced maps, the pilot DTM, AI models, and processing code will be freely accessible via Zenodo and code repositories (e.g., GitHub).

### 2.3 Making Data Interoperable
*   **Standardized Formats:** Data will be stored and exchanged in widely accepted, machine-readable, and open GIS formats:
    *   **Rasters:** GeoTIFF.
    *   **Vectors:** GeoJSON, GeoPackage (GPKG)
    *   **Text/Documentation:** Markdown, TXT.
*   **System Integration:** By adhering to the AIS AKB infrastructure standards, the newly generated data achieves immediate semantic interoperability with existing national archaeological records.

### 2.4 Increase Data Re-use
*   **Licensing for Data:** Open datasets and metadata will be released under standard open licenses (e.g., Creative Commons CC-BY or CC-BY-NC) to maximize scholarly reuse.
*   **Licensing for Code:** Open-source software and AI scripts developed by the team will be licensed under standard open-source licenses (e.g., MIT or GPL).
*   **Embargo Periods:** Data will be made available as soon as possible after the mandatory field verification and validation phase (WP3) is completed.

---

## 3. Allocation of Resources

*   **Computational Infrastructure:** 
    *   A professional high-performance workstation equipped with dedicated GPUs will be utilized specifically for training the deep learning AI models.
*   **Storage Infrastructure:** 
    *   Centralized server and disk arrays located at NAIM-BAS will provide the primary storage for heavy geospatial datasets.
*   **Management Responsibilities:** 
    *   **Overall DMP Adherence:** Principal Investigator (Assist. Prof. Dr. Nadezhda Kecheva).
    *   **Data Quality Control:** Managed individually by the Work Package Leaders (Angel Grigorov, Ivo Cholakov, Julia Tzvetkova) during their respective project phases.

---

## 4. Data Security

*   **Active Storage Protection:** 
    *   All active project data resides on centralized NAIM-BAS servers.
    *   Hardware is protected by UPS (Uninterruptible Power Supply) modules to prevent data corruption during sudden power outages.
*   **Backup Strategy:** 
    *   Regular, systematic backups of the server data will be maintained on external storage drives to ensure redundancy.
*   **Long-term Preservation Archive:** 
    *   **AIS AKB:** Serves as the primary, secure, and legally mandated long-term archive for all sensitive archaeological site data.
    *   **Zenodo:** Serves as the long-term archive for public datasets, models, and code.

---

## 5. Ethical Aspects and Legal Compliance

*   **Regulatory Frameworks:** 
    *   The project fully complies with the **EU Artificial Intelligence Act (AI Act)**.
    *   Adheres to the **Ethics Guidelines for Trustworthy AI** and the ALTAI self-assessment tool.
    *   Follows the **General Data Protection Regulation (GDPR)** for all personal data.
*   **Cultural Sensitivity & Heritage Protection:** 
    *   The project acknowledges the severe threat of looting. Consequently, AI-assisted interpretations regarding site locations are treated with extreme cultural sensitivity.
    *   All automated results must be expert-reviewed prior to any integration. A targeted sample will also be field-verified (WP3) to statistically validate the AI's accuracy, acknowledging that field-verifying every detected site nationally is unfeasible.
    *   Public dissemination is strictly limited to non-sensitive aggregated data to ensure the research does not inadvertently endanger cultural heritage.
