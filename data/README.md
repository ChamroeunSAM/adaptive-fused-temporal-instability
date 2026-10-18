# Data

The crash records are from the Highways Accident Information Management System (HAIMS) of Thailand's Department of
Highways (DOH), 2016–2023, 144,658 crashes. They are not distributed with this repository. Requests for the data can be
made to the Department of Highways, Thailand, or to the author.

## Expected file

`data/raw/crashes_thailand_2016_2023.csv`, UTF-8, one row per crash. A different location can be given with the
environment variable `AFTI_RAW_CSV`. Columns:

| Column | Content |
|---|---|
| `date` | crash date, `d/m/yyyy` text or an Excel serial number (only the year is used) |
| `Injury` | most severe injury in the crash: 1 no injury (PDO), 2 minor, 3 serious, 4 fatal |
| `Injury_binary` | 1 if `Injury` is 3 or 4, else 0 (the outcome) |
| all other columns | 0/1 indicators (84 columns) |

The indicator columns, in file order, are:

`motor_1, motor_>1, car_1, car_>1, pickup_1, pickup_>1, truck_6_1, truck_6_>1, truck_10_1, truck_10_>1, truck_>10_1,
truck_>10_>1, bike, three_wheels, van, large_bas, predestrain, weekend, songkran, new_year, holiday, Central, Eastern,
Northeastern, Northern, Southern, Western, Daytime, Evening, Midnight/early morning, Peak hours, rear_end_same_lane,
off_left_hit_object, mounts_traffic_island, off_right_hit_object, sideswipe_parallel, off_left, off_hit_object_right_bend,
off_hit_object_left_bend, off_right, head_on, out_of_control, off_left_bend, off_right_bend, speeding, cutting_in_too_close,
falling_asleep, vehicle_defect, drunk_driving, running_red_light, main_carriageway, frontage_road, entry_exit_main,
under_maintenance, under_construction, no_median, painted_median, raised_earth_median, depressed_median, barrier_median,
one_way_traffic, normal_curve, sharp_curve, concrete_surface, crest, sag, on_grade, t_intersection, plus_intersection,
y_intersection, grade_separated_ramps, uturn_no_lane, uturn_decel_only, uturn_accel_only, uturn_both_lanes,
access_public_commercial, access_education_govt, access_private, bridge, wet_road, pothole_road, raining, dark_no_lighting,
dark_with_lighting`

The `_1` and `_>1` vehicle columns indicate one, or two or more, vehicles of that type. `predestrain` (pedestrian
involved) keeps the spelling of the source file. Descriptions of every indicator used in the models are in
`src/labels.py`.

## Processed data

`python pipeline/s1_prepare_data.py` writes the four subsets to `data/processed/` as NumPy archives. Each contains:

- `X`: the constant and the candidate indicators;
- `y`: the outcome;
- `t`: the period index;
- `names`: the column names.

It also writes `subsets.json`, which lists the indicators excluded in each subset and the reason for each exclusion.
