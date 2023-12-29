from typing import Dict, Text

import numpy as np
import random

from highway_env import utils
from highway_env.envs.common.action import Action
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, SineLane, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import Obstacle
from highway_env.vehicle.behavior import LinearVehicle, AggressiveVehicle, DefensiveVehicle, IDMVehicle


class MergeEnv(AbstractEnv):

    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {"type": "Kinematics"},
                "action": {
                    "type": "DiscreteMetaAction",
                },
                "lanes_count": 4,
                "vehicles_count": 50,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 40,  # [s]
                "ego_spacing": 2,
                "vehicles_density": 1,
                "collision_reward": -1,  # The reward received when colliding with a vehicle.
                "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                # zero for other lanes.
                "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                # lower speeds according to config["reward_speed_range"].
                "lane_change_reward": 0,  # The reward received at each lane change action.
                "reward_speed_range": [20, 30],
                "normalize_reward": True,
                "offroad_terminal": False,
            }
        )
        return config


    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [
                    self.config["collision_reward"],
                    self.config["high_speed_reward"] + self.config["right_lane_reward"],
                ],
                [0, 1],
            )
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = (
            self.vehicle.target_lane_index[2]
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index[2]
        )
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road),
        }

    # def default_config(cls) -> dict:
    #     cfg = super().default_config()
    #     cfg.update(
    #         {

    #             #"observation": {"type": "Lidar"},
    #             # "observation": {
    #             #     "type": "LidarObservation",
    #             #     "cells": 128,
    #             #     "maximum_range": 64,
    #             #     "normalise": True,
    #             # },
    #             "observation": {"type": "Kinematics"},
    #             "action":{"type": "DiscreteMetaAction"},
    #             "collision_reward": -100,
    #             "right_lane_reward": 1,
    #             "high_speed_reward": .04,
    #             "reward_speed_range": [20, 30],
    #             # "merging_speed_reward": -5,
    #             "normalize_reward": True,
    #             #"end_reward": 100,
    #             #"lane_change_reward": -0.005,
    #         }
    #     )
    #     return cfg

    # def _reward(self, action: Action) -> float:
    #     """
    #     The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
    #     :param action: the last action performed
    #     :return: the corresponding reward
    #     """
    #     rewards = self._rewards(action)
    #     reward = sum(
    #         self.config.get(name, 0) * reward for name, reward in rewards.items()
    #     )
    #     if self.config["normalize_reward"]:
    #         reward = utils.lmap(
    #             reward,
    #             [
    #                 self.config["collision_reward"], #+ self.config["merging_speed_reward"],
    #                 self.config["high_speed_reward"] + self.config["right_lane_reward"],
    #             ],
    #             [0, 1],
    #         )
    #     reward *= rewards["on_road_reward"]

    #     if self.vehicle.position[0] > 2000:
    #         reward += 100

    #     reward += (self.time * 0.8)

    #     return reward

    # def _rewards(self, action: Action) -> Dict[Text, float]:
    #     neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
    #     lane = (
    #         self.vehicle.target_lane_index[2]
    #         if isinstance(self.vehicle, ControlledVehicle)
    #         else self.vehicle.lane_index[2]
    #     )
    #     # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
    #     forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
    #     scaled_speed = utils.lmap(
    #         forward_speed, self.config["reward_speed_range"], [0, 1]
    #     )
    #     rewards =  {
    #         "collision_reward": float(self.vehicle.crashed),
    #         "right_lane_reward": lane / max(len(neighbours) - 1, 1),
    #         "high_speed_reward": np.clip(scaled_speed, 0, 1),
    #         "on_road_reward": float(self.vehicle.on_road),
    #         # "merging_speed_reward": sum(  # Altruistic penalty
    #         #     (vehicle.target_speed - vehicle.speed) / vehicle.target_speed
    #         #     for vehicle in self.road.vehicles
    #         #     if vehicle.lane_index == ("b", "c", 2)
    #         #     and isinstance(vehicle, ControlledVehicle)
    #         # ),
    #     }

    #     return rewards

    def _is_terminated(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        # print("crash" + str(self.vehicle.crashed))
        # print("over" + str(self.vehicle.position[0] > 1500))
        return self.vehicle.crashed or bool(self.vehicle.position[0] > 2000)

    def _is_truncated(self) -> bool:
        return False

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        net = RoadNetwork()

        ends = [150, 80, 80, 2100]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH, 2 * StraightLane.DEFAULT_WIDTH]  # Three lanes
        line_type = [[c, s], [s, s], [n, c]]

        for i in range(3):  # Three lanes
            net.add_lane(
                "a",
                "b",
                StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]),
            )
            net.add_lane(
                "b",
                "c",
                StraightLane(
                    [sum(ends[:2]), y[i]],
                    [sum(ends[:3]), y[i]],
                    line_types=line_type[i],
                ),
            )
            net.add_lane(
                "c",
                "d",
                StraightLane(
                    [sum(ends[:3]), y[i]],
                    [sum(ends), y[i]],
                    line_types=line_type[i],
                ),
            )

        amplitude = 2.25
        # Adjusted the starting position of the merging lane
        ljk = StraightLane(
            [0, 6.5 + 20], [ends[0] - 10, 20], line_types=[c, c], forbidden=True
        )
        lkb = SineLane(
            ljk.position(ends[0] - 10, -amplitude),
            ljk.position(sum(ends[:2]), -amplitude),
            amplitude,
            2 * np.pi / (2 * (ends[1] - 5)),  # Adjusted frequency
            np.pi / 2,
            line_types=[c, c],
            forbidden=True,
        )
        lbc = StraightLane(
            lkb.position(ends[1] + 10, 0 ),
            lkb.position(ends[1] + 10, 0) + [ends[2], 0],# take off the 100
            line_types=[n, c],
            forbidden=True,
        )
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)

        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road
    
    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        road = self.road
        ego_vehicle = self.action_type.vehicle_class(
            road, road.network.get_lane(("j", "k", 0)).position(30, 0), speed=10
        )
        ego_vehicle.target_speed = 55
        road.vehicles.append(ego_vehicle)

        # Other vehicles type
        other_vehicle_types = [
            #LinearVehicle,
            #AggressiveVehicle,
            IDMVehicle,
            #DefensiveVehicle,
            # Add more vehicle types as needed
        ]
        num_highway_cars = 100
        # Generate unique positions for highway cars with lane index
        highway_positions = set()

        while len(highway_positions) < num_highway_cars:
            position = self.np_random.uniform(0, 2000)  # Adjust the range based on your road length
            lane_index = random.randint(0, 2)

            # Ensure that each two cars in the same lane have the space of three cars between them
            while any(
                abs(pos[0] - position) <= random.randint(20,25) and pos[1] == lane_index
                for pos in highway_positions
            ):
                position = self.np_random.uniform(0, 2000)

            highway_positions.add((position, lane_index))

        for positionIndex, lane_index in highway_positions:
            lane = road.network.get_lane(("a", "b", lane_index))
            position = lane.position(positionIndex, 0)

            # Randomly choose a vehicle type and speed
            vehicle_type = np.random.choice(other_vehicle_types)

            speed = 10 * (positionIndex * 0.004)  # Adjust the speed range as needed

            highwayCar = vehicle_type(road, position, speed=speed)
            highwayCar.target_speed = 55

            highwayCar.randomize_behavior()

            road.vehicles.append(highwayCar)


        merging_v = self.np_random.choice(other_vehicle_types)(
            road, road.network.get_lane(("j", "k", 0)).position(110, 0), speed=50
        )
        merging_v.target_speed = 55
        road.vehicles.append(merging_v)

        merging_v2 = self.np_random.choice(other_vehicle_types)(
            road, road.network.get_lane(("j", "k", 0)).position(15, 0), speed=15
        )
        merging_v2.target_speed = 55
        road.vehicles.append(merging_v2)

        self.vehicle = ego_vehicle