#!/usr/bin/env python3
"""ROS 2 node wrapping the pure static-roadblock filter core."""

import csv
from datetime import datetime
import math
from pathlib import Path
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from roadblock_interfaces.msg import Roadblock, RoadblockArray
from std_srvs.srv import Trigger

from roadblock_map_filter.filter_core import (
    DecisionEvent,
    Measurement,
    RoadblockMapFilterCore,
)


CSV_COLUMNS = (
    'recv_time',
    'measurement_stamp_sec',
    'measurement_stamp_nanosec',
    'raw_id',
    'raw_x',
    'raw_y',
    'matched_track_id',
    'track_state_before',
    'stable_x_before',
    'stable_y_before',
    'distance_m',
    'decision',
    'stable_x_after',
    'stable_y_after',
    'track_state_after',
)


class RoadblockMapFilterNode(Node):
    def __init__(self):
        super().__init__('roadblock_map_filter')
        self.declare_parameter('association_gate_m', 0.15)
        self.declare_parameter('update_gate_m', 0.07)
        self.declare_parameter('candidate_confirm_gate_m', 0.08)
        self.declare_parameter('new_track_suppression_gate_m', 0.18)
        self.declare_parameter('reacquire_gate_m', 0.25)
        self.declare_parameter('history_size', 5)
        self.declare_parameter('tentative_window_frames', 5)
        self.declare_parameter('tentative_required_hits', 3)
        self.declare_parameter('publish_rate_hz', 2.0)
        self.declare_parameter(
            'input_topic', '/navigation/roadblock_measurements_map'
        )
        self.declare_parameter(
            'output_topic', '/navigation/roadblock_map_filtered'
        )
        self.declare_parameter('enable_csv_log', True)
        self.declare_parameter(
            'csv_log_dir',
            '/root/intelligent_car_ws/test_logs/roadblock_map_filter',
        )

        self.core = RoadblockMapFilterCore(
            association_gate_m=self._float_parameter('association_gate_m'),
            update_gate_m=self._float_parameter('update_gate_m'),
            candidate_confirm_gate_m=self._float_parameter(
                'candidate_confirm_gate_m'
            ),
            new_track_suppression_gate_m=self._float_parameter(
                'new_track_suppression_gate_m'
            ),
            reacquire_gate_m=self._float_parameter('reacquire_gate_m'),
            history_size=self._int_parameter('history_size'),
            tentative_window_frames=self._int_parameter(
                'tentative_window_frames'
            ),
            tentative_required_hits=self._int_parameter(
                'tentative_required_hits'
            ),
        )
        self.publish_rate_hz = max(
            0.1, self._float_parameter('publish_rate_hz')
        )
        self.input_topic = str(self.get_parameter('input_topic').value)
        self.output_topic = str(self.get_parameter('output_topic').value)
        self.latest_output_stamp: Optional[Tuple[int, int]] = None
        self.warned_zero_stamp = False
        self.warned_empty_frame = False
        self.csv_file = None
        self.csv_writer = None
        self.csv_path = None
        self._open_csv_log()

        input_qos = QoSProfile(depth=10)
        input_qos.reliability = ReliabilityPolicy.RELIABLE
        input_qos.durability = DurabilityPolicy.VOLATILE
        output_qos = QoSProfile(depth=1)
        output_qos.reliability = ReliabilityPolicy.RELIABLE
        output_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.filtered_pub = self.create_publisher(
            RoadblockArray, self.output_topic, output_qos
        )
        self.measurement_sub = self.create_subscription(
            RoadblockArray,
            self.input_topic,
            self.measurement_callback,
            input_qos,
        )
        self.reset_service = self.create_service(
            Trigger, '/roadblock_map_filter/reset', self.reset_callback
        )
        self.publish_timer = self.create_timer(
            1.0 / self.publish_rate_hz, self.publish_filtered_map
        )
        self.publish_filtered_map()
        self.get_logger().info(
            'Static roadblock map filter ready: '
            f'input={self.input_topic}, output={self.output_topic}, '
            f'association={self.core.association_gate_m:.3f}m, '
            f'update={self.core.update_gate_m:.3f}m, '
            f'confirm={self.core.candidate_confirm_gate_m:.3f}m, '
            f'new-track-suppression='
            f'{self.core.new_track_suppression_gate_m:.3f}m, '
            f'reacquire={self.core.reacquire_gate_m:.3f}m, '
            f'history={self.core.history_size}, '
            f'confirmation={self.core.tentative_required_hits}-of-'
            f'{self.core.tentative_window_frames}'
        )
        if self.csv_path is not None:
            self.get_logger().info(f'CSV log: {self.csv_path}')

    def _float_parameter(self, name: str) -> float:
        return float(self.get_parameter(name).value)

    def _int_parameter(self, name: str) -> int:
        return int(self.get_parameter(name).value)

    def _open_csv_log(self) -> None:
        if not bool(self.get_parameter('enable_csv_log').value):
            return
        try:
            log_dir = Path(str(self.get_parameter('csv_log_dir').value))
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.csv_path = log_dir / f'roadblock_filter_{timestamp}.csv'
            self.csv_file = self.csv_path.open('w', newline='', encoding='utf-8')
            self.csv_writer = csv.DictWriter(
                self.csv_file, fieldnames=CSV_COLUMNS
            )
            self.csv_writer.writeheader()
            self.csv_file.flush()
        except OSError as exc:
            self.csv_file = None
            self.csv_writer = None
            self.csv_path = None
            self.get_logger().warning(f'CSV logging disabled: {exc}')

    @staticmethod
    def _stamp_key(msg: RoadblockArray) -> Optional[Tuple[int, int]]:
        sec = int(msg.header.stamp.sec)
        nanosec = int(msg.header.stamp.nanosec)
        return None if sec == 0 and nanosec == 0 else (sec, nanosec)

    def _write_event(
        self,
        event: DecisionEvent,
        stamp_sec: int,
        stamp_nanosec: int,
    ) -> None:
        if self.csv_writer is None:
            return
        measurement = event.measurement
        self.csv_writer.writerow({
            'recv_time': datetime.now().isoformat(timespec='milliseconds'),
            'measurement_stamp_sec': stamp_sec,
            'measurement_stamp_nanosec': stamp_nanosec,
            'raw_id': '' if measurement is None else measurement.raw_id,
            'raw_x': '' if measurement is None else measurement.x,
            'raw_y': '' if measurement is None else measurement.y,
            'matched_track_id': (
                '' if event.matched_track_id is None
                else event.matched_track_id
            ),
            'track_state_before': event.track_state_before,
            'stable_x_before': self._csv_number(event.stable_x_before),
            'stable_y_before': self._csv_number(event.stable_y_before),
            'distance_m': self._csv_number(event.distance_m),
            'decision': event.decision,
            'stable_x_after': self._csv_number(event.stable_x_after),
            'stable_y_after': self._csv_number(event.stable_y_after),
            'track_state_after': event.track_state_after,
        })
        self.csv_file.flush()

    @staticmethod
    def _csv_number(value):
        return '' if value is None else f'{float(value):.9f}'

    def measurement_callback(self, msg: RoadblockArray) -> None:
        frame_id = str(msg.header.frame_id)
        if frame_id and frame_id != 'map':
            self.get_logger().warning(
                f'Ignoring measurement frame {frame_id!r}; expected map'
            )
            return
        if not frame_id and not self.warned_empty_frame:
            self.warned_empty_frame = True
            self.get_logger().warning(
                'Measurement frame_id is empty; accepting it as map in V3'
            )

        stamp_key = self._stamp_key(msg)
        if stamp_key is None and not self.warned_zero_stamp:
            self.warned_zero_stamp = True
            self.get_logger().warning(
                'Measurement timestamp is zero; processing callbacks in order '
                'without timestamp deduplication'
            )
        measurements = [
            Measurement(int(item.id), float(item.x), float(item.y))
            for item in msg.obstacles
        ]
        result = self.core.process_frame(measurements, stamp_key)
        for event in result.events:
            self._write_event(
                event,
                int(msg.header.stamp.sec),
                int(msg.header.stamp.nanosec),
            )
            if event.decision == 'CONFIRM':
                self.get_logger().info(
                    f'Confirmed roadblock track {event.matched_track_id} at '
                    f'({event.stable_x_after:.3f}, '
                    f'{event.stable_y_after:.3f})'
                )
            elif event.decision == 'TENTATIVE_EXPIRE':
                self.get_logger().info(
                    f'Expired tentative track {event.matched_track_id}'
                )
            elif event.decision in ('ACCEPT', 'SUSPECT'):
                self.get_logger().debug(
                    f'{event.decision} track={event.matched_track_id} '
                    f'distance={event.distance_m:.4f}m'
                )
            elif event.decision == 'SUPPRESS_NEAR_CONFIRMED':
                self.get_logger().debug(
                    f'SUPPRESS_NEAR_CONFIRMED raw_id='
                    f'{event.measurement.raw_id} '
                    f'nearest_track={event.matched_track_id} '
                    f'distance={event.distance_m:.4f}m'
                )
            elif event.decision == 'REACQUIRE_SUSPECT':
                self.get_logger().debug(
                    f'REACQUIRE_SUSPECT raw_id={event.measurement.raw_id} '
                    f'track={event.matched_track_id} '
                    f'distance={event.distance_m:.4f}m'
                )
            elif event.decision == 'INVALID_MEASUREMENT':
                self.get_logger().warning(
                    f'Ignoring non-finite measurement id='
                    f'{event.measurement.raw_id}'
                )

        if result.duplicate_frame:
            return
        if stamp_key is None:
            now_msg = self.get_clock().now().to_msg()
            self.latest_output_stamp = (
                int(now_msg.sec), int(now_msg.nanosec)
            )
        else:
            self.latest_output_stamp = stamp_key
        self.publish_filtered_map()

    def publish_filtered_map(self) -> None:
        output = RoadblockArray()
        output.header.frame_id = 'map'
        if self.latest_output_stamp is None:
            output.header.stamp = self.get_clock().now().to_msg()
        else:
            output.header.stamp.sec = self.latest_output_stamp[0]
            output.header.stamp.nanosec = self.latest_output_stamp[1]
        for track in self.core.confirmed_tracks():
            item = Roadblock()
            item.id = track.track_id
            item.x = track.stable_x
            item.y = track.stable_y
            output.obstacles.append(item)
        self.filtered_pub.publish(output)

    def reset_callback(self, _request, response):
        removed = self.core.reset()
        self.latest_output_stamp = None
        self._write_event(DecisionEvent('RESET', None, None), 0, 0)
        self.publish_filtered_map()
        response.success = True
        response.message = f'Cleared {removed} roadblock track(s)'
        self.get_logger().info(response.message)
        return response

    def destroy_node(self):
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RoadblockMapFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
