# PPT diagrams (Mermaid reference)

Reference sources for the graphics in `PPT_BRIEF.md`. Paste each block into https://mermaid.live, export SVG or PNG, then restyle in the deck (SIH blue, red highlights, icons). Use these as the structure. Do not ship the raw Mermaid look.

Labels use only facts from `../PRD.md` and `../TECHNICAL_SPEC.md`. No measured numbers.

---

## Slide 2: icon hub (solution at a glance)

Redraw as: centre icon, six icons around it, like the winning deck's shield graphic.

```mermaid
flowchart LR
    C(("RegimeRain<br/>India rainfall<br/>post-processing"))
    A["Regime Classifier<br/>active / break / depression<br/>coastal / orographic"]
    B["Per-Regime<br/>Bias Correction"]
    D["Heavy-Rain Probability<br/>calibrated"]
    E["District Table<br/>and Map"]
    F["Six-Metric<br/>Verification Report"]
    G["Forecast-Time<br/>Inputs Only<br/>no leakage"]
    A --- C
    B --- C
    D --- C
    E --- C
    F --- C
    G --- C
    classDef hub fill:#1f6fb5,stroke:#0b3c6b,color:#fff,stroke-width:2px;
    classDef node fill:#e8e8f4,stroke:#6b5b95,color:#111;
    class C hub;
    class A,B,D,E,F,G node;
```

---

## Slide 2: problem in one picture (optional, one-line story)

Why one curve fails. Use as a small strip under the pills if space allows.

```mermaid
flowchart LR
    subgraph OLD["Conventional"]
        direction LR
        O1["All days<br/>active, break,<br/>depression, coastal"] --> O2["ONE global<br/>correction curve"] --> O3["Wrong for the<br/>days that matter"]
    end
    subgraph NEW["Proposed"]
        direction LR
        N1["Forecast"] --> N2["Predict<br/>regime"] --> N3["Curve built for<br/>THAT regime"] --> N4["Sharper on heavy<br/>and very heavy rain"]
    end
    OLD ~~~ NEW
    style OLD fill:#ffe4e6,stroke:#e11d48
    style NEW fill:#dcfce7,stroke:#16a34a
```

---

## Slide 3: main technical flow (the big one)

Mirrors the winning deck: inputs top-left, model box, branches, outputs, feedback loop.

```mermaid
flowchart TB
    IN["<b>Inputs</b><br/>IFS HRES / NOAA GFS forecast<br/>IMD rain (CHIRPS fallback) · IBTrACS<br/>MJO RMM · terrain + coast masks"]
    ING["<b>Ingestion</b><br/>clip to India box 6.5-38.5N, 66.5-100E<br/>0.25 deg grid · mm/day · IMD rain day 03-03 UTC"]
    LAB["<b>Regime labelling</b> (offline, training only)<br/>active/break index · depression flag<br/>coastal + orographic masks"]
    CLS["<b>Regime classifier</b><br/>multi-class LightGBM<br/>forecast-time features only"]
    PROB(["Probability per regime"])
    QM["<b>Branch A: Per-regime<br/>quantile mapping</b><br/>curves per regime x terrain x zone x lead<br/>blended by regime probability · GPD tail"]
    EXC["<b>Branch B: Exceedance model</b><br/>heavy 64.5 · very heavy 115.6 mm/day<br/>isotonic calibration"]
    CORR(["Corrected rainfall grid"])
    PEX(["P(heavy), P(very heavy) grid"])
    AGG["<b>District aggregation</b><br/>area-weighted mean and max"]
    OUT(["<b>District table + map</b>"])
    VER["<b>Verification</b><br/>RMSE · ETS · CSI · POD · FAR · FSS<br/>raw vs corrected, both thresholds<br/>reliability diagram"]
    REP(["<b>Verification report</b>"])
    FB["<b>Feedback loop</b><br/>flag weak / low-sample regimes<br/>refit curves + classifier<br/>re-verify"]

    IN --> ING --> LAB --> CLS --> PROB
    PROB --> QM --> CORR
    PROB --> EXC --> PEX
    CORR --> AGG
    PEX --> AGG
    AGG --> OUT
    CORR --> VER
    PEX --> VER
    VER --> REP
    REP --> FB
    FB -.-> CLS
    FB -.-> QM

    classDef io fill:#dbeafe,stroke:#1d4ed8,color:#111;
    classDef proc fill:#e8e8f4,stroke:#6b5b95,color:#111;
    classDef out fill:#dcfce7,stroke:#16a34a,color:#111;
    classDef loop fill:#fef3c7,stroke:#d97706,color:#111;
    class IN,ING io;
    class LAB,CLS,QM,EXC,AGG,VER proc;
    class PROB,CORR,PEX,OUT,REP out;
    class FB loop;
```

---

## Slide 3: regime classifier detail (optional inset)

Shows the "forecast-time only" point and the two regime axes (PRD Rev B §10.1).

```mermaid
flowchart LR
    subgraph TRAIN["Training only (hindsight labels)"]
        direction TB
        L1["Depression<br/>IBTrACS centre within 500 km"]
        L4["Active / Break<br/>core-zone rain anomaly ±1 sigma, 3+ days"]
        L5["Normal<br/>otherwise"]
        L1 -->|"overrides"| L4 --> L5
    end
    subgraph INFER["Inference (forecast time only)"]
        direction TB
        F1["Forecast rainfall field"]
        F2["850 hPa wind"]
        F3["Moisture flux convergence"]
        F4["MJO phase + amplitude"]
        F5["Terrain / coast descriptors, day of year"]
    end
    TRAIN -->|"synoptic labels"| M["LightGBM<br/>multi-class"]
    INFER -->|"features"| M
    M --> P(["Probability vector<br/>active / break / depression / normal"])
    G["Geographic class (fixed per cell)<br/>orographic / coastal / plains"] --> O
    P --> O["Regime = synoptic x geographic<br/>selects the correction curves"]
    style TRAIN fill:#fee2e2,stroke:#dc2626
    style INFER fill:#dcfce7,stroke:#16a34a
```

---

## Slide 3: leakage-proof validation (optional inset)

Leave-one-monsoon-out. Season names are placeholders: use the real seasons in the training window.

```mermaid
flowchart LR
    S["Historical Jun-Sep seasons<br/>Y1, Y2, ... Yn"] --> F1["Fold 1<br/>hold out Y1<br/>train on rest"]
    S --> F2["Fold 2<br/>hold out Y2<br/>train on rest"]
    S --> FN["Fold n<br/>hold out Yn<br/>train on rest"]
    F1 --> E["Fit classifier + QM curves<br/>+ exceedance model<br/>on training seasons"]
    F2 --> E
    FN --> E
    E --> T["Score the held-out<br/>season only"]
    T --> R["Six metrics per fold<br/>+ pooled<br/>raw vs corrected"]
    classDef n fill:#e8e8f4,stroke:#6b5b95,color:#111;
    class S,F1,F2,FN,E,T,R n;
```

---

## Slide 4: feasibility risk map (optional graphic behind or beside the table)

Risk to mitigation, one row per table row.

```mermaid
flowchart LR
    R1["Rare regimes<br/>noisy curve"] --> M1["Report sample count<br/>per curve<br/>top-2 blend if time allows"]
    R2["NCUM not public<br/>IMD data unconfirmed"] --> M2["Validate on HRES + ERA5<br/>CHIRPS truth<br/>NCUM adapter interface"]
    R3["Small gains<br/>hindsight leakage"] --> M3["Leave-one-monsoon-out<br/>leakage unit test<br/>report failures"]
    R4["Boundary licence<br/>daily run speed"] --> M4["GADM + attribution<br/>no retraining in daily mode"]
    classDef risk fill:#fecaca,stroke:#dc2626,color:#111;
    classDef fix fill:#dcfce7,stroke:#16a34a,color:#111;
    class R1,R2,R3,R4 risk;
    class M1,M2,M3,M4 fix;
```

---

## Slide 5: who gets what (impact graphic)

Optional replacement for one bottom visual. Flow of value from pipeline to users.

```mermaid
flowchart LR
    P["RegimeRain<br/>pipeline"] --> T["District table + map<br/>corrected rain<br/>heavy-rain probability"]
    P --> V["Verification report<br/>six metrics<br/>per-regime error"]
    T --> U1["District disaster<br/>management officers"]
    T --> U2["State water, flood,<br/>agri, urban agencies"]
    V --> U3["NCMRWF / IMD<br/>forecasters"]
    V --> U4["Model developers<br/>and researchers"]
    classDef p fill:#1f6fb5,stroke:#0b3c6b,color:#fff;
    classDef o fill:#e8e8f4,stroke:#6b5b95,color:#111;
    classDef u fill:#dcfce7,stroke:#16a34a,color:#111;
    class P p;
    class T,V o;
    class U1,U2,U3,U4 u;
```

---

## Slide 5: IMD threshold ladder

```mermaid
flowchart BT
    A["Daily rainfall (mm)"] --> B["Heavy: 64.5 mm/day or more"] --> C["Very heavy: 115.6 mm/day or more"]
    classDef a fill:#dbeafe,stroke:#1d4ed8,color:#111;
    classDef b fill:#fde68a,stroke:#d97706,color:#111;
    classDef c fill:#fecaca,stroke:#dc2626,color:#111;
    class A a;
    class B b;
    class C c;
```

---

## Slide 5: illustrative curve concept (not measured)

Mermaid cannot plot curves. Draw this one in the deck tool. Spec:

- X axis: forecast rainfall. Y axis: corrected rainfall.
- One flat grey line labelled "single global curve".
- Three coloured lines, each with a different slope: "active", "break", "depression".
- Caption in small text: "Illustrative, not measured."

The only Mermaid-friendly version is a comparison, below.

```mermaid
quadrantChart
    title Illustrative: correction need by regime (not measured)
    x-axis "Low rainfall impact" --> "High rainfall impact"
    y-axis "Small model error" --> "Large model error"
    quadrant-1 "Correct hardest"
    quadrant-2 "Watch"
    quadrant-3 "Leave alone"
    quadrant-4 "Heavy but reliable"
    "Depression": [0.85, 0.85]
    "Orographic": [0.75, 0.7]
    "Active": [0.6, 0.45]
    "Coastal": [0.5, 0.55]
    "Break": [0.2, 0.3]
```

Positions on the quadrant chart are made-up illustration coordinates. Keep the "not measured" caption. If the team cannot defend the placement, drop this graphic.

---

## Slide 6: conventional vs proposed (visual for the survey table)

```mermaid
flowchart LR
    subgraph CONV["Conventional"]
        direction TB
        C1["One global curve"] --> C2["No regime awareness"] --> C3["Tails clipped"] --> C4["RMSE only"] --> C5["Gridded file"]
    end
    subgraph PROP["Proposed"]
        direction TB
        P1["Curve per regime"] --> P2["Classify first"] --> P3["Tail extrapolation +<br/>calibrated P(heavy)"] --> P4["Six metrics, leakage-proof"] --> P5["District table + map"]
    end
    CONV ==>|"upgrade"| PROP
    style CONV fill:#ffe4e6,stroke:#e11d48
    style PROP fill:#dcfce7,stroke:#16a34a
```

---

## Export notes

- Render at 1920x1080 canvas. Export SVG so it stays sharp in the PDF.
- Keep text at 14 pt or larger after scaling. If a diagram gets too dense, use the optional insets on separate visuals, not one crowded slide.
- Replace box text with icons where the winning deck did (user, database, shield, alert). Keep the same colour roles: blue inputs, lilac processing, green outputs, amber loops, red risks.
- Slide budget stays at 6. Insets replace text, not add slides.
