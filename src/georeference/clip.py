import os
import glob
import json
import re
from pathlib import Path
from osgeo import gdal
from tqdm import tqdm

# Enable GDAL Exceptions
gdal.UseExceptions()

# ==========================================
# CONFIGURATION
# ==========================================

SCRIPT_DIR = Path(__file__).resolve().parent          

# PATHS
INPUT_FOLDER  = SCRIPT_DIR / "georeferenced"
OUTPUT_FOLDER = SCRIPT_DIR / "geo_clipped"
GEOJSON_PATH  = SCRIPT_DIR / "grid_25k.geojson"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
GEOJSON_NAME_FIELD = "mapsheet"

# ==========================================
# PART 1: DB & UTILS
# ==========================================

def transliterate_bg_to_en(text):
    translit_map = {
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ж': 'zh',
        'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n',
        'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u', 'ф': 'f',
        'х': 'h', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'sht', 'ъ': 'a',
        'ь': 'y', 'ю': 'yu', 'я': 'ya',
        'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E', 'Ж': 'Zh',
        'З': 'Z', 'И': 'I', 'Й': 'Y', 'К': 'K', 'Л': 'L', 'М': 'M', 'Н': 'N',
        'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U', 'Ф': 'F',
        'Х': 'H', 'Ц': 'Ts', 'Ч': 'Ch', 'Ш': 'Sh', 'Щ': 'Sht', 'Ъ': 'A',
        'Ь': 'Y', 'Ю': 'Yu', 'Я': 'Ya'
    }
    result = ""
    for char in text:
        result += translit_map.get(char, char)
    return result

def generate_match_keys(s):
    if not s: return []
    s_en = transliterate_bg_to_en(str(s))
    s_upper = s_en.upper()
    v1 = re.sub(r'[^A-Z0-9]', '', s_upper)
    parts = re.split(r'[^A-Z0-9]+', s_upper)
    norm = [str(int(p)) if p.isdigit() else p for p in parts if p]
    v2 = "".join(norm)
    return list(set([v1, v2]))

def load_db_lookup(json_path, name_field):
    """
    Creates a dictionary mapping:
    NORMALIZED_KEY (Latin) -> ORIGINAL_NAME (Cyrillic/GeoJSON property)
    """
    print(f"Loading Lookup DB: {json_path}")
    if not os.path.exists(json_path):
        print(f"ERROR: GeoJSON not found at {json_path}")
        return {}
        
    with open(json_path, 'r', encoding='utf-8') as f: data = json.load(f)

    lookup = {}
    for ft in data.get('features', []):
        props = ft.get('properties', {})
        name_bg = props.get(name_field, "")
        if not name_bg: continue
        
        # Generate keys from the GeoJSON name
        # e.g. "К-35-5-А" -> ["K355A"]
        keys = generate_match_keys(name_bg)
        
        for k in keys:
            lookup[k] = name_bg  # Map Key -> Original Name
            
    return lookup

# ==========================================
# CLIPPER FUNCTION
# ==========================================

def clip_raster(src_path, output_dir, base_name, geojson_path, layer_name_val):
    try:
        # SQL WHERE clause: Must match the exact string in the GeoJSON
        sql_where = f"{GEOJSON_NAME_FIELD} = '{layer_name_val}'"

        # 1. GENERATE ML-READY COG (Clipped)
        ml_out = str(output_dir / f"{base_name}_clipped.tif")
        
        gdal.Warp(
            ml_out,
            str(src_path),
            format='COG',
            cutlineDSName=str(geojson_path),
            cutlineWhere=sql_where,
            cropToCutline=True,
            dstAlpha=True,
            
            creationOptions=[
                'COMPRESS=LZW',
                'PREDICTOR=2',
                'BIGTIFF=IF_NEEDED',
                'OVERVIEWS=IGNORE_EXISTING',
                'RESAMPLING=CUBIC',
                'BLOCKSIZE=512'
            ]
        )

        # 2. GENERATE WEB-READY IIIF (Clipped)
        iiif_out = str(output_dir / f"{base_name}_clipped.ptif")
        
        gdal.Translate(
            iiif_out,
            ml_out,
            format='GTiff',
            bandList=[1, 2, 3], # Drop Alpha for JPEG
            creationOptions=[
                'COMPRESS=JPEG', 
                'JPEG_QUALITY=90',
                'TILED=YES',
                'BLOCKXSIZE=256',
                'BLOCKYSIZE=256',
                'PHOTOMETRIC=YCBCR',
                'COPY_SRC_OVERVIEWS=NO'
            ]
        )
        
        # Build Overviews
        ds = gdal.Open(iiif_out, 1)
        if ds:
            ds.BuildOverviews("CUBIC", [2, 4, 8, 16, 32, 64])
            ds = None
            
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False

# ==========================================
# MAIN
# ==========================================

def main():
    if not os.path.exists(GEOJSON_PATH):
        print("GeoJSON not found.")
        return
    
    if not os.path.exists(INPUT_FOLDER):
        print(f"Input folder not found: {INPUT_FOLDER}")
        return

    # Load the lookup dictionary
    db_lookup = load_db_lookup(GEOJSON_PATH, GEOJSON_NAME_FIELD)
    
    # CHANGED: Look for "_georef.tif" (matching your previous script output)
    files = glob.glob(os.path.join(INPUT_FOLDER, "*_georef.tif"))
    print(f"Found {len(files)} maps to clip.")
    
    success_count = 0
    
    for img_path in tqdm(files, desc="Clipping Maps", unit="map"):
        filename = os.path.basename(img_path)
        
        # CHANGED: Replace "_georef.tif" to get clean name
        clean_name = filename.replace("_georef.tif", "")
        
        # Generate keys from filename (e.g. "K-35-5-A-g" -> ["K355AG"])
        file_keys = generate_match_keys(clean_name)
        
        exact_geojson_name = None
        for key in file_keys:
            if key in db_lookup:
                exact_geojson_name = db_lookup[key]
                break
        
        if not exact_geojson_name:
            # print(f"SKIPPING {filename} (No Match)")
            continue
            
        # Perform Clip
        if clip_raster(img_path, OUTPUT_FOLDER, clean_name, GEOJSON_PATH, exact_geojson_name):
            success_count += 1

    print(f"\nDone. {success_count} maps clipped.")

if __name__ == "__main__":
    main()