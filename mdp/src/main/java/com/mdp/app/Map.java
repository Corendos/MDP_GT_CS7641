package com.mdp.app;

import burlap.domain.singleagent.gridworld.GridWorldRewardFunction;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;

public class Map {
    int[][] map;
    int width;
    int height;

    private final double goalReward = 1.0;
    private final double frozenCellReward = -0.04;
    private final double holeReward = -1.0;

    Map(int[][] mapLayout) {
        width = mapLayout[0].length;
        height = mapLayout.length;

        map = mapLayout.clone();
    }

    public void fillRewardFunctionAndTerminalFunction(
        GridWorldRewardFunction rf,
        GridWorldTerminalFunction tf) {
        for (int y = 0;y < height;y++) {
            for (int x = 0;x < width;x++) {
                if (map[y][x] == 0) {
                    rf.setReward(x, y, frozenCellReward);
                } else if (map[y][x] == 1) {
                    rf.setReward(x, y, holeReward);
                } else if (map[y][x] == 2) {
                    rf.setReward(x, y, goalReward);
                    tf.markAsTerminalPosition(x, y);
                }
            }   
        }
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }
}