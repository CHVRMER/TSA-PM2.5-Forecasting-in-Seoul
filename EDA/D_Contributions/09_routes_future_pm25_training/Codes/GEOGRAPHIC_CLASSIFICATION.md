# Geographic passage classification

`classify_geographic_routes.py` uses geographic administrative polygons, not
measurement-station proximity circles. It reads only 2021–2022 arrival origins
at 500 m AGL from the original trajectory table. Each of 5,840 arrivals has
hourly modeled positions at ages 1–48 hours. Neither the arrival position at
age 0 nor the segment from age 0 to age 1 is used.

## Definitions

- Primary passage: at least one connecting line between modeled hourly
  positions intersects a region. Boundary touches count. The line is linear
  in longitude/latitude; unresolved subhourly air movement is not observed.
- Windows 01–48, 07–48 and 13–48 use 48/42/36 positions and 47/41/35 segments.
- `visited_points` is a sensitivity based on at least one hourly position
  inside/on the polygon. A region can be crossed between two positions without
  either endpoint falling inside; therefore segment and point rates differ.
- An arrival can visit multiple regions. Passage percentages do not sum to
  100%. `point_hours` counts hourly positions, not continuous residence time.
- The domain uses South Korean seven broad administrative groups, six named
  mainland Chinese regions plus other mainland China, Japan, Mongolia,
  Russia, North Korea, other land, and sea/geographically unclassified.
- The 13 border-proxy Chinese city observations are not a geographic category.
  Russia passage does not establish that Russian PM monitoring data exist.
- The analysis never estimates PM mass, emissions, source apportionment,
  country contribution, or causal transport of a measured surface plume.

## Geometry safeguards

Polygon holes are retained. Invalid polygon inputs are repaired. A fixed order
resolves any overlapping source interiors; the overlap diagnostic records the
degree-squared geometry change (not a physical area). Shared boundaries are
assigned once for point classification; segment passages can belong to several
regions. Antimeridian lines take the short branch rather than crossing Asia.

Other land uses only non-focus country features from the global country file.
It does not interpret coarse-minus-detailed shoreline slivers as another
country. Source discrepancies, unresolved islands and the sea are combined in
the explicit `SEA_OR_UNCLASSIFIED` category. Do not label this as verified sea.

## Boundary sources and attribution

- KOR ADM1: Natural Earth via geoBoundaries, 2021, Public Domain.
- CHN ADM1: geoBoundaries/Wikimedia Commons, 2019, Public Domain. Exact source
  member names are retained; its `Guangzhou Province` label is in other China.
  Taiwan, Hong Kong and Macau are excluded from the mainland-China grouping.
- Japan ADM0: Ministry of Land, Infrastructure, Transport and Tourism,
  National Land Numerical Information Download Site, via geoBoundaries, 2022,
  Creative Commons Attribution 4.0 International. Boundaries were repaired,
  dissolved, and overlap-clipped for this analysis. License:
  https://creativecommons.org/licenses/by/4.0/
- Mongolia/Russia/North Korea and other countries: reused Natural Earth 1:110m
  country polygons, Public Domain. Its precise boundary epoch is unavailable;
  the exact reused file is pinned by SHA256.

This is a fixed geographic classification using mixed-source reference dates,
not a reconstruction of every administrative/coastal change in 2021–2022 and
not a statement about sovereignty. GeoBoundaries API responses and pinned
downloaded GeoJSONs are kept in `assets/`. Exact URLs and SHA256 hashes are in
`Results/geographic_classification_manifest.json`.

## Run

From Git Bash with dependencies installed:

```bash
python classify_geographic_routes.py --trajectories '/c/Users/kimjy/Desktop/Time Series Analysis/Project/Revised_Project/Dataset/Trajectories/trajectory_points_2021_2024.parquet'
```

The script defaults to its enclosing EDA directory for outputs. The original
trajectory dataset is read only. `origin_visits.parquet` keeps all arrival ×
region × window rows, including false/zero nonvisits. The embedded tests check
23 known locations, polygon holes, boundary touches, crossing without an
endpoint, antimeridian handling, complete point assignment and exact row keys.
