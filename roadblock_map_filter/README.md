# roadblock_map_filter

ROS 2 static-obstacle map filter for competition roadblocks. It consumes
single-frame measurements already aligned to the visual timestamp and
transformed into `map`, then publishes a stable persistent map.

## Interfaces

- Input: `/navigation/roadblock_measurements_map`
  (`roadblock_interfaces/msg/RoadblockArray`, RELIABLE/VOLATILE, depth 10)
- Output: `/navigation/roadblock_map_filtered`
  (`roadblock_interfaces/msg/RoadblockArray`, RELIABLE/TRANSIENT_LOCAL, depth 1)
- Reset: `/roadblock_map_filter/reset` (`std_srvs/srv/Trigger`)

The input `Roadblock.id` is frame-local. The filter allocates stable track IDs
starting at 1. Output is sorted by stable track ID.

## V3.1 algorithm

- first-pass deterministic greedy one-to-one association matches confirmed
  tracks at up to 0.15 m;
- second-pass deterministic greedy one-to-one reacquisition considers only
  measurements and confirmed tracks still unmatched after the first pass.
  A distance above 0.15 m and at most 0.25 m produces
  `REACQUIRE_SUSPECT`, preserving the old track identity without moving its
  stable position, adding history, or creating a new track;
- only measurements still unmatched after confirmed-track association and
  reacquisition may associate with tentative tracks. This protects an existing
  confirmed identity from a closer but not-yet-confirmed candidate;
- 3-of-5 independent-frame confirmation for new tentative tracks;
- before an unmatched measurement creates a tentative track, suppress it as
  `SUPPRESS_NEAR_CONFIRMED` when it is within 0.18 m of any confirmed track.
  This check includes confirmed tracks already used by the frame's one-to-one
  association; tentative tracks do not participate;
- coordinate-wise median over the five latest accepted positions;
- separate association and update gates: ACCEPT updates history, while SUSPECT
  consumes the association without moving the stable point;
- confirmed tracks persist until explicit reset or node restart.

V3.1 deliberately does **not** subscribe to or use yaw or vehicle pose. It has no
Kalman filter, TTL, field-of-view/near-blind logic, automatic relocation or
deletion, confirmed-track merge, anchor drift limit, distance-weighted fusion,
measurement quality model, global yaw compensation, or motion model.

The default gates remain distinct: association is 0.15 m, confirmed updates are
0.07 m, tentative confirmation is 0.08 m, and new-track suppression is 0.18 m.
The second-pass reacquisition gate is 0.25 m and applies only to confirmed
tracks left unmatched by the first pass. A confirmed track already matched in
the frame cannot be reacquired by another measurement; that measurement
continues to V2 suppression and tentative creation logic.
Suppression decisions are written to CSV with the raw measurement, nearest
confirmed track ID, distance, and unchanged confirmed-track state.

The processing priority is: confirmed normal association, confirmed
reacquisition, tentative association, suppression, then new tentative creation.

## Run

```bash
source /opt/tros/humble/setup.bash
source /root/intelligent_car_ws/install/setup.bash
ros2 launch roadblock_map_filter roadblock_map_filter.launch.py
```

Reset the complete persistent map:

```bash
ros2 service call /roadblock_map_filter/reset std_srvs/srv/Trigger "{}"
```

CSV debugging is enabled by default. Each node start creates
`roadblock_filter_YYYYMMDD_HHMMSS.csv` under
`/root/intelligent_car_ws/test_logs/roadblock_map_filter`. CSV failures disable
logging but do not stop filtering.
