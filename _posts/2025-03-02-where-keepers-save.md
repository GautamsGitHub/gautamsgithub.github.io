---
layout: post
title:  "Where shots go relative to keepers"
date:   2025-03-02 12:01:00 +0000
---

I will make use of statsbomb data to see which shots get saved by the keeper
dependin on the shot placement relative to the keeper.


```python
from statsbombpy import sb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


```python
free_comps = sb.competitions()

# wc_matches = sb.matches(competition_id=9, season_id=281)
# bl_matches = sb.matches(competition_id=9, season_id=281)

#euro24
eu_matches = sb.matches(competition_id=55, season_id=282)

```

    C:\Users\gaut\AppData\Local\Programs\Python\Python312\Lib\site-packages\statsbombpy\api_client.py:21: NoAuthWarning: credentials were not supplied. open data access only
      warnings.warn(
    


```python
# with pd.option_context('display.max_rows', None):
#     display(free_comps)
```


```python
large_shot_set = pd.concat(
    [
        sb.events(match_id=match_id, split=True)["shots"]
        for match_id in eu_matches["match_id"]
    ],
    axis=0,
    ignore_index=True
)
large_gk_set = pd.concat(
    [
        sb.events(match_id=match_id, split=True)["goal_keepers"]
        for match_id in eu_matches["match_id"]
    ],
    axis=0,
    ignore_index=True
)
```

    C:\Users\gaut\AppData\Local\Programs\Python\Python312\Lib\site-packages\statsbombpy\api_client.py:21: NoAuthWarning: credentials were not supplied. open data access only
      warnings.warn(
    C:\Users\gaut\AppData\Local\Programs\Python\Python312\Lib\site-packages\statsbombpy\api_client.py:21: NoAuthWarning: credentials were not supplied. open data access only
      warnings.warn(
    


```python
gk_events = large_gk_set.set_index("id")
gk_events = gk_events[[
    "player_id",
    "player",
    "timestamp",
    "location",
    "goalkeeper_body_part",
    "goalkeeper_outcome",
    "goalkeeper_technique",
    "goalkeeper_position"
]].rename(columns={
    "player_id": "keeper_id",
    "player": "keeper_name",
    "timestamp": "gk_timestamp",
    "location": "gk_location",
})

shot_events = large_shot_set[[
    "timestamp",
    "shot_technique",
    "location",
    "related_events",
    "shot_outcome",
    "shot_end_location",
    "shot_body_part"
]].rename(columns={
    "timestamp": "shot_timestamp",
    "location": "shooter_location",
})

```


```python
shots_and_gks = pd.merge(
    gk_events,
    shot_events.explode("related_events"),
    how="inner",
    left_on="id",
    right_on="related_events"
).groupby("related_events").first().reset_index(drop=True)
```


```python
def shot_relative_to_gk(shot_start, shot_end, gk_position):
    flipping = np.array([120, 80])
    sef = flipping - shot_end
    ssf = flipping - shot_start
    k_to_e = sef - gk_position
    k_to_s = ssf - gk_position
    s_to_e = sef - ssf
    # distance along shot keeper attempts block
    daskab = (
        np.sum(np.square(s_to_e))
        + np.sum(np.square(k_to_s))
        - np.sum(np.square(k_to_e))
    ) / (2 * np.sum(np.square(s_to_e)))
    gk_interacts_at = daskab * s_to_e + ssf
    srk = gk_interacts_at - gk_position
    distance = np.linalg.norm(srk)
    directed_distance = distance * np.sign(srk[1])
    return directed_distance
```


```python
shot_relative_to_gk(
    np.array([100, 40]),
    np.array([120, 45]),
    np.array([1, 40])
)
```




    -4.608176875690325




```python
shots_and_gks["shot_end_location_2D"] = shots_and_gks["shot_end_location"].map(
    lambda l: l[:2]
)
shots_and_gks["shot_end_height"] = shots_and_gks["shot_end_location"].map(
    lambda l: l[2] if len(l) > 2 else pd.NA
)
shots_and_gks[
    ["shooter_location",
     "shot_end_location_2D",
     "gk_location"]
] = shots_and_gks[[
    "shooter_location",
    "shot_end_location_2D",
    "gk_location"]].apply(np.array)
```


```python
shots_and_gks["shot_relative_to_gk"] = shots_and_gks.apply(
    lambda row: shot_relative_to_gk(
        row["shooter_location"],
        row["shot_end_location_2D"],
        row["gk_location"]
    ),
    axis='columns'
)
```


```python
shots_and_gks["goal_numeric"] = shots_and_gks.apply(
    lambda r: 1.0 if r["shot_outcome"] == "Goal" else 0.0,
    axis='columns'
)
```


```python
gk_faces = shots_and_gks[
    (
        shots_and_gks["shot_outcome"] == "Goal"
    ) | (
        shots_and_gks["shot_outcome"] == "Saved"
    )
].copy()
```


```python
keeper_ids = large_gk_set.groupby("player_id").first()["player"]
keeper_ids
```




    player_id
    3468             Jordan Pickford
    3711             Martin Dúbravka
    3761                Mike Maignan
    3815           Kasper Schmeichel
    4127           David Raya Martin
    5550                 Yann Sommer
    5570                Manuel Neuer
    5669           Wojciech Szczęsny
    6378                   Jan Oblak
    7036        Gianluigi Donnarumma
    7789            Thomas Strakosha
    8240               Koen Casteels
    8524               Péter Gulácsi
    9731               Patrick Pentz
    11508           Łukasz Skorupski
    11748        Unai Simón Mendibil
    15731           Predrag Rajković
    16531          Dominik Livaković
    21390               Andriy Lunin
    22379                 Angus Gunn
    28040            Jindřich Staněk
    28242     Florin Constantin Niţă
    30310           Fehmi Mert Günok
    30442             Altay Bayındır
    32975       Diogo Meireles Costa
    37274            Bart Verbruggen
    37330            Anatolii Trubin
    46046                Matěj Kovář
    102371      Giorgi Mamardashvili
    Name: player, dtype: object




```python
PICKFORD_ID = 3468
SOMMER_ID = 5550
SIMON_ID = 11748
RAYA_ID = 4127
MAMARDASHVILLI_ID = 102371
CASTEELS_ID = 8240
```


```python
GK_COLOURS = {
    "Goal": "black",
    "Both Hands": "coral",
    "Left Hand": "red",
    "Right Hand": "gold",
    "Right Foot": "green",
    "Left Foot": "blue",
    "Head": "pink",
    "Chest": "purple"
}
```


```python
our_gk_id = PICKFORD_ID
our_gk_name = keeper_ids.loc[our_gk_id]
our_gk_faces = gk_faces[gk_faces["keeper_id"] == our_gk_id]
```


```python
def graph_gk(ax, the_gk_faces, the_gk_name, legend=False):
    ax.spines["left"].set_position("zero");
    ax.spines["right"].set_position("zero");
    ax.spines["top"].set_color("none");
    grouped_saves = the_gk_faces[
        the_gk_faces["shot_outcome"] == "Saved"
    ].groupby("goalkeeper_body_part")
    for part, shots in grouped_saves:
        ax.scatter(
            shots["shot_relative_to_gk"],
            shots["shot_end_height"],
            marker='x',
            c=GK_COLOURS[part],
            label=f"Saved by {part}",
        );
    goals = the_gk_faces[the_gk_faces["shot_outcome"] == "Goal"]
    ax.scatter(
        goals["shot_relative_to_gk"],
        goals["shot_end_height"],
        marker='o',
        c=GK_COLOURS["Goal"],
        label="Goal"
    );
    ax.set_title(f"Shots faced by {the_gk_name}");
    if legend:
        ax.legend(
            loc="center",
            bbox_to_anchor=(0.5, -0.3)
        );
```

## Keeper maps

Here are maps of shots and goals against some keepers from the perspective of the keeper looking at the striker.
The body part the save was made with is shown by colour.

```python
our_keepers = [PICKFORD_ID, SIMON_ID, SOMMER_ID]
fig, axs = plt.subplots(
    nrows=len(our_keepers),
    figsize=(8, 4*len(our_keepers)),
    layout="constrained"
)

for i, the_gk_id in enumerate(our_keepers):
    the_gk_name = keeper_ids.loc[the_gk_id]
    the_gk_faces = gk_faces[gk_faces["keeper_id"] == the_gk_id]
    graph_gk(axs[i], the_gk_faces, the_gk_name, legend=False)

# print(labels_handles)
fig.legend(loc="outside right");

```


    
![maps of saves by some keepers](/assets/images/keeper_save_maps.png)
    


## Scoring rates in quadrants


```python
def partition_shot_quadrants(across, height):
    hc = "high" if height > 0.9 else "low"
    ac = "right" if across > 0.0 else "left"
    return (hc + " " + ac)
```


```python
gk_faces["quadrant"] = gk_faces.apply(
    lambda r: partition_shot_quadrants(
        r["shot_relative_to_gk"],
        r["shot_end_height"]
    ),
    axis='columns'
)
```


```python
keepers_in_quadrants = gk_faces.groupby(["keeper_id", "quadrant"]).agg(
    total=("goal_numeric", "count"),
    goals=("goal_numeric", "sum")
)
keepers_in_quadrants["quadrant_save_rate"] = 1 - (
    keepers_in_quadrants["goals"] / keepers_in_quadrants["total"])
```


```python
quad_totals = keepers_in_quadrants.unstack(level="quadrant")["total"]
quad_goals = keepers_in_quadrants.unstack(level="quadrant")["goals"]
```


```python
quad_goals.sum(axis=0) / quad_totals.sum(axis=0)
```




    quadrant
    high left     0.220183
    high right    0.321839
    low left      0.266667
    low right     0.333333
    dtype: float64



Keepers concede less on their left side.


```python
quad_totals.sum(axis=0)
```


The shots keepers face are about even between right and left.

    quadrant
    high left     109.0
    high right     87.0
    low left      120.0
    low right     126.0
    dtype: float64


