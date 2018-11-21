package com.mdp.app;

import burlap.behavior.policy.GreedyQPolicy;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldRewardFunction;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.statehashing.simple.SimpleHashableStateFactory;

/**
 * Hello world!
 */
public final class LargeAppProfiling {
    private LargeAppProfiling() {
    }

    public static final int WIDTH = 4;
    public static final int HEIGHT = 4;

    public static final int[][] map = new int[][]{
        {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
        {1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0},
        {1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0},
        {0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0},
        {1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
        {0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0},
        {0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0},
        {0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1},
        {0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0},
        {0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0},
        {1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
        {0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
        {1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0},
        {1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0},
        {1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0},
        {1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1},
        {1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1},
        {1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1},
        {1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1},
        {1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1},
        {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 2}
    };

    /**
     * Main function
     * @param args The arguments of the program.
     */
    public static void main(String[] args) {
        /* Initialization */
        int height = map.length;
        int width = map[0].length;

        GridWorldDomain gw = new GridWorldDomain(width, height);
        gw.setProbSucceedTransitionDynamics(0.9);

        GridWorldRewardFunction rf = new GridWorldRewardFunction(width, height);
        for (int y = 0;y < height;y++) {
            for (int x = 0; x < width; x++) {
                if (map[y][x] == 1) {
                    rf.setReward(x, y, -1.0);
                } else if (map[y][x] == 0) {
                    rf.setReward(x, y, -0.04);
                } else if (map[y][x] == 2) {
                    rf.setReward(x, y, 1.0);
                }
            }
        }
        TerminalFunction tf = new GridWorldTerminalFunction(23, 23);

        gw.setRf(rf);
        gw.setTf(tf);
        
        OOSADomain domain = gw.generateDomain();
        SimpleHashableStateFactory hashingFactory = new SimpleHashableStateFactory();

        GridWorldState state = new GridWorldState(new GridAgent(0, 0, "Start"));

        ValueIteration vi = new ValueIteration(domain, 0.99, hashingFactory, 0.01, 1000);
        PolicyIteration pi = new PolicyIteration(domain, 0.99, hashingFactory, 0.01, 1000, 1000);


        vi.toggleDebugPrinting(false);
        long cumulatedComputeTime = 0;
        for (int i = 0;i < 10;i++) {
            vi = new ValueIteration(domain, 0.99, hashingFactory, 0.01, 1000);

            long begin = System.nanoTime();
            vi.planFromState(state);
            long end = System.nanoTime();
            cumulatedComputeTime += (end - begin);
        }
        double computeTime = (float)cumulatedComputeTime / (1e9f * 1000f);
        System.out.println("Average computing time for Value Iteration: " + computeTime);

        cumulatedComputeTime = 0;
        for (int i = 0;i < 10;i++) {
            pi = new PolicyIteration(domain, 0.99, hashingFactory, 0.01, 1000, 1000);
            
            long begin = System.nanoTime();
            pi.planFromState(state);
            long end = System.nanoTime();
            cumulatedComputeTime += (end - begin);
        }
        computeTime = (float)cumulatedComputeTime / (1e9f * 1000f);
        System.out.println("Average computing time for Policy Iteration: " + computeTime);
    }
}
