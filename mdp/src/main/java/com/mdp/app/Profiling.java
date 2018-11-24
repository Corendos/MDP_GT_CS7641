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
public final class Profiling {
    private Profiling() {
    }

    public static final double[][] map = new double[][]{
        {-0.04, -0.04,  -0.04,  -0.04},
        {-0.04, -1.0,   -0.04,  -1.0},
        {-0.04, -1.0,   -0.04,  -1.0},
        {-0.04, -0.04,  -0.04,   1.0},
    };

    public static final int[][] smallMapLayout = new int[][]{
        {0, 0, 0, 0},
        {0, 1, 0, 1},
        {0, 1, 0, 1},
        {0, 0, 0, 2},
    };

    public static final int[][] largeMapLayout = new int[][]{
        {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
        {1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0},
        {1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0},
        {0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0},
        {1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
        {0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0},
        {0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0},
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

    private static Map smallMap;
    private static Map largeMap;

    private static double DISCOUNT_FACTOR = 0.99;
    private static double MAX_DELTA = 0.01;
    private static int MAX_ITERATIONS = 1000;

    /**
     * Main function
     * @param args The arguments of the program.
     */
    public static void main(String[] args) {
        /* Initialization */
        smallMap = new Map(smallMapLayout);
        largeMap = new Map(largeMapLayout);

        Map map = smallMap;

        GridWorldDomain gw = new GridWorldDomain(map.getWidth(), map.getHeight());
        gw.setProbSucceedTransitionDynamics(0.9);

        GridWorldRewardFunction rf = new GridWorldRewardFunction(map.getWidth(), map.getHeight());
        GridWorldTerminalFunction tf = new GridWorldTerminalFunction();

        map.fillRewardFunctionAndTerminalFunction(rf, tf);
        gw.setRf(rf);
        gw.setTf(tf);
        
        OOSADomain domain = gw.generateDomain();
        SimpleHashableStateFactory hashingFactory = new SimpleHashableStateFactory();

        GridWorldState state = new GridWorldState(new GridAgent(0, 0, "Start"));

        ValueIteration vi = new ValueIteration(domain, DISCOUNT_FACTOR, hashingFactory, MAX_DELTA, MAX_ITERATIONS);
        PolicyIteration pi = new PolicyIteration(domain, DISCOUNT_FACTOR, hashingFactory, MAX_DELTA, 1000, MAX_ITERATIONS);


        vi.toggleDebugPrinting(false);
        long cumulatedComputeTime = 0;
        for (int i = 0;i < 1000;i++) {
            vi = new ValueIteration(domain, DISCOUNT_FACTOR, hashingFactory, MAX_DELTA, MAX_ITERATIONS);

            long begin = System.nanoTime();
            vi.planFromState(state);
            long end = System.nanoTime();
            cumulatedComputeTime += (end - begin);
        }
        double computeTime = (float)cumulatedComputeTime / (1e9f * 1000f);
        System.out.println("Average computing time for Value Iteration: " + computeTime);

        cumulatedComputeTime = 0;
        for (int i = 0;i < 1000;i++) {
            pi = new PolicyIteration(domain, DISCOUNT_FACTOR, hashingFactory, MAX_DELTA, 1000, MAX_ITERATIONS);
            
            long begin = System.nanoTime();
            pi.planFromState(state);
            long end = System.nanoTime();
            cumulatedComputeTime += (end - begin);
        }
        computeTime = (float)cumulatedComputeTime / (1e9f * 1000f);
        System.out.println("Average computing time for Policy Iteration: " + computeTime);
    }
}
