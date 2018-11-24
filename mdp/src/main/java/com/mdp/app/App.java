package com.mdp.app;

import burlap.behavior.policy.GreedyQPolicy;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldRewardFunction;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.statehashing.simple.SimpleHashableStateFactory;

public final class App {
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

    private App() {}

    /**
     * Main function
     * @param args The arguments of the program.
     */
    public static void main(String[] args) {
        smallMap = new Map(smallMapLayout);
        largeMap = new Map(largeMapLayout);

        Map map = largeMap;
        MapWriter mapWriter = new MapWriter();
        
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
        GreedyQPolicy viPolicy = vi.planFromState(state);

        PolicyIteration pi = new PolicyIteration(domain, DISCOUNT_FACTOR, hashingFactory, MAX_DELTA, 1000, MAX_ITERATIONS);
        GreedyQPolicy piPolicy = pi.planFromState(state);

        vi.toggleDebugPrinting(true);

        MapWriter.Direction[][] viDirections = new MapWriter.Direction[map.getHeight()][map.getWidth()];
        MapWriter.Direction[][] piDirections = new MapWriter.Direction[map.getHeight()][map.getWidth()];

        for (int y = map.getHeight() - 1;y >= 0;y--) {
            for (int x = 0; x < map.getWidth(); x++) {
                GridWorldState currentState = new GridWorldState(x, y);
                if (viPolicy.action(currentState).actionName() == "east") {
                    viDirections[y][x] = MapWriter.Direction.RIGHT;
                } else if (viPolicy.action(currentState).actionName() == "west") {
                    viDirections[y][x] = MapWriter.Direction.LEFT;
                } else if (viPolicy.action(currentState).actionName() == "north") {
                    viDirections[y][x] = MapWriter.Direction.UP;
                } else if (viPolicy.action(currentState).actionName() == "south") {
                    viDirections[y][x] = MapWriter.Direction.DOWN;
                }
            }
        }

        for (int y = map.getHeight() - 1;y >= 0;y--) {
            for (int x = 0; x < map.getWidth(); x++) {
                GridWorldState currentState = new GridWorldState(x, y);
                if (piPolicy.action(currentState).actionName() == "east") {
                    piDirections[y][x] = MapWriter.Direction.RIGHT;
                } else if (piPolicy.action(currentState).actionName() == "west") {
                    piDirections[y][x] = MapWriter.Direction.LEFT;
                } else if (piPolicy.action(currentState).actionName() == "north") {
                    piDirections[y][x] = MapWriter.Direction.UP;
                } else if (piPolicy.action(currentState).actionName() == "south") {
                    piDirections[y][x] = MapWriter.Direction.DOWN;
                }
            }
        }

        mapWriter.write(viDirections, "output/vi.png");
        mapWriter.write(piDirections, "output/pi.png");
    }
}
