package com.mdp.app;

import burlap.behavior.policy.GreedyQPolicy;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldRewardFunction;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridLocation;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.auxiliary.common.ConstantStateGenerator;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.statehashing.simple.SimpleHashableStateFactory;

/**
 * Hello world!
 */
public final class LargeApp {
    private LargeApp() {
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
        GreedyQPolicy policy = vi.planFromState(state);

        PolicyIteration pi = new PolicyIteration(domain, 0.99, hashingFactory, 0.01, 1000, 1000);
        GreedyQPolicy piPolicy = pi.planFromState(state);

        vi.toggleDebugPrinting(false);
        
        System.out.println("Value Iteration");
        System.out.println("+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+");
        for (int y = height - 1;y >= 0;y--) {
            System.out.print("|");
            for (int x = 0; x < width; x++) {
                GridWorldState currentState = new GridWorldState(x, y);
                if (policy.action(currentState).actionName() == "east") {
                    System.out.print(" > |");
                } else if (policy.action(currentState).actionName() == "west") {
                    System.out.print(" < |");
                } else if (policy.action(currentState).actionName() == "north") {
                    System.out.print(" ^ |");
                } else if (policy.action(currentState).actionName() == "south") {
                    System.out.print(" v |");
                } else {
                    System.out.print(" x |");
                }
            }
            System.out.println();
            System.out.println("+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+");
        }
        
        System.out.println("Policy Iteration" + Integer.toString(pi.getTotalValueIterations()));
        System.out.println("+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+");
        for (int y = height - 1;y >= 0;y--) {
            System.out.print("|");
            for (int x = 0; x < width; x++) {
                GridWorldState currentState = new GridWorldState(x, y);
                if (policy.action(currentState).actionName() == "east") {
                    System.out.print(" > |");
                } else if (policy.action(currentState).actionName() == "west") {
                    System.out.print(" < |");
                } else if (policy.action(currentState).actionName() == "north") {
                    System.out.print(" ^ |");
                } else if (policy.action(currentState).actionName() == "south") {
                    System.out.print(" v |");
                } else {
                    System.out.print(" x |");
                }
            }
            System.out.println();
            System.out.println("+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+");
        }
    }
}
