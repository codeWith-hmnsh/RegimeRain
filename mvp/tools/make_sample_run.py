"""Write web/public/geo/land.json: Natural Earth 1:50m land (public domain), clipped to the map box.

Coastline only. The web draws it as plain ground when the street map is off or offline, and the mock
producer uses it to tell a coast from a land border.

The synthetic sample RUN is no longer written here: it comes from the regimerain mock producer, the
same exporter every real run uses (docs/BACKEND_BUILD_PLAN.md):

    python3 -m regimerain.cli run --source mock --publish sample     # from the repo root

    python3 mvp/tools/make_sample_run.py                              # land.json only
"""
import json
import pathlib
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
WEB = HERE.parent / "web" / "public"
GEO = WEB / "geo"
CACHE = HERE / "cache"
NE_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_land.geojson"

for d in (GEO, CACHE):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- land polygons
MAP_BOX = (10.0, -12.0, 125.0, 50.0)  # lon0, lat0, lon1, lat1 (map extent, wider than the data box)


def clip_ring(ring, box):
    """Sutherland-Hodgman clip of one ring against an axis-aligned box."""
    x0, y0, x1, y1 = box
    edges = [
        (lambda p: p[0] >= x0, lambda a, b: (x0, a[1] + (b[1] - a[1]) * (x0 - a[0]) / (b[0] - a[0]))),
        (lambda p: p[0] <= x1, lambda a, b: (x1, a[1] + (b[1] - a[1]) * (x1 - a[0]) / (b[0] - a[0]))),
        (lambda p: p[1] >= y0, lambda a, b: (a[0] + (b[0] - a[0]) * (y0 - a[1]) / (b[1] - a[1]), y0)),
        (lambda p: p[1] <= y1, lambda a, b: (a[0] + (b[0] - a[0]) * (y1 - a[1]) / (b[1] - a[1]), y1)),
    ]
    pts = [tuple(p) for p in ring]
    for inside, cut in edges:
        if not pts:
            break
        out = []
        for i, cur in enumerate(pts):
            prev = pts[i - 1]
            if inside(cur):
                if not inside(prev):
                    out.append(cut(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(cut(prev, cur))
        pts = out
    return pts


src = CACHE / "ne_50m_land.geojson"
if not src.exists():
    urllib.request.urlretrieve(NE_URL, src)
land_rings = []
for f in json.loads(src.read_text())["features"]:
    g = f["geometry"]
    polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
    for poly in polys:
        outer = poly[0]
        xs = [p[0] for p in outer]
        ys = [p[1] for p in outer]
        if max(xs) < MAP_BOX[0] or min(xs) > MAP_BOX[2] or max(ys) < MAP_BOX[1] or min(ys) > MAP_BOX[3]:
            continue
        r = clip_ring(outer, MAP_BOX)
        if len(r) >= 4:
            r = [[round(x, 3), round(y, 3)] for x, y in r]
            if r[0] != r[-1]:
                r.append(r[0])
            land_rings.append(r)
(GEO / "land.json").write_text(json.dumps({
    "type": "FeatureCollection",
    "attribution": "Made with Natural Earth (public domain). Coastline only.",
    "features": [{"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [r]}}
                 for r in land_rings],
}, separators=(",", ":")))
print(f"{GEO / 'land.json'}  {(GEO / 'land.json').stat().st_size // 1024} KB")
